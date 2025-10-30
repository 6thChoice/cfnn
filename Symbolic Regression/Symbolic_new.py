# symbolic_regression_cofrnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import symbols, simplify, expand, latex, sympify, nsimplify, cancel, factor, collect
import pandas as pd
from typing import List, Tuple, Dict, Optional
import warnings
import copy
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

class PolynomialTerm(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.degree = degree
        self.projection = nn.Linear(input_dim, output_dim)
        self.coeffs = nn.Parameter(torch.randn(output_dim, degree + 1) * 0.1)

    def forward(self, x):
        z = torch.tanh(self.projection(x))
        powers = [torch.ones_like(z)]
        for d in range(1, self.degree + 1):
            powers.append(powers[-1] * z)
        z_powered = torch.stack(powers, dim=-1)
        output = torch.sum(z_powered * self.coeffs, dim=2)
        return output

class CFNet(nn.Module):
    def __init__(self, input_dim, output_dim, depth, poly_degree):
        super().__init__()
        self.output_dim = output_dim
        self.terms = nn.ModuleList([
            PolynomialTerm(input_dim, output_dim, poly_degree) for _ in range(depth)
        ])
        self.raw_betas = nn.ParameterList([
            nn.Parameter(torch.full((output_dim,), 0.1)) for _ in range(depth - 1)
        ])

    def forward(self, x):
        if not self.terms:
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)
        output = F.softplus(self.terms[-1](x)) + 1.0
        for i in range(len(self.terms) - 2, -1, -1):
            beta = F.softplus(self.raw_betas[i])
            output = self.terms[i](x) + beta / output
        return output

class EnsembleResCoFrNet(nn.Module):
    """集成模型：支持候选模型的选择性加入机制"""
    def __init__(self, input_dim, output_dim, shallow_depth, poly_degree, learning_rate):
        super().__init__()
        self.models = nn.ModuleList()
        self.candidate_model = None
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shallow_depth = shallow_depth
        self.poly_degree = poly_degree

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if not self.models and self.candidate_model is None:
            return torch.zeros(x.shape[0], self.output_dim, device=x.device)
        total_output = torch.zeros(x.shape[0], self.output_dim, device=x.device)
        for model in self.models:
            total_output += self.learning_rate * model(x)
        if self.candidate_model is not None:
            total_output += self.learning_rate * self.candidate_model(x)
        return total_output

    def create_candidate(self):
        """创建一个新的候选模型用于训练"""
        self.candidate_model = CFNet(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            depth=self.shallow_depth,
            poly_degree=self.poly_degree
        )
        return self.candidate_model

    def promote_candidate(self):
        """若候选模型有效，则将其采纳为正式模型"""
        if self.candidate_model is not None:
            self.models.append(self.candidate_model)
            self.candidate_model = None
    
    def discard_candidate(self):
        """丢弃无效的候选模型"""
        self.candidate_model = None

    def freeze_for_candidate_training(self):
        """冻结所有正式模型，只训练候选模型"""
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False
        if self.candidate_model is not None:
            for param in self.candidate_model.parameters():
                param.requires_grad = True

class SymbolicRegressionCoFrNet:
    """基于连分式网络的符号回归"""
    
    def __init__(self, config=None):
        self.default_config = self.get_default_config()
        if config:
            self.config = {**self.default_config, **config}
        else:
            self.config = self.default_config
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ensemble = None
        self.symbolic_expressions = []
        
    @staticmethod
    def get_default_config():
        return {
            'input_dim': 1,
            'output_dim': 1,
            'shallow_depth': 2,
            'polynomial_degree': 3,
            'boosting_learning_rate': 0.3,
            'num_models_max': 3,
            'learning_rate_adam': 0.01,
            'weight_decay': 0.01,
            'epochs_per_model': 200,
            'early_stopping_patience': 20,
            'sparsity_threshold': 0.05,
            'simplification_threshold': 0.1,  # 用于激进简化的阈值
            'noise_level': 0.001,
            'gradient_clip': 1.0
        }
    
    def generate_data(self, func_str: str, n_samples: int = 1000, 
                     x_range: Tuple[float, float] = (-2, 2)) -> Tuple[np.ndarray, np.ndarray]:
        """生成训练数据"""
        x = np.linspace(x_range[0], x_range[1], n_samples).reshape(-1, 1)
        
        x_sym = symbols('x')
        func = sympify(func_str)
        func_lambda = lambda val: float(func.subs(x_sym, val))
        
        y = np.array([func_lambda(xi[0]) for xi in x]).reshape(-1, 1)
        
        if self.config['noise_level'] > 0:
            noise = np.random.normal(0, self.config['noise_level'] * np.std(y), y.shape)
            y += noise
        
        return x, y
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, verbose=True):
        """训练集成模型（使用选择性加入机制）"""
        # 创建集成模型
        self.ensemble = EnsembleResCoFrNet(
            input_dim=self.config['input_dim'],
            output_dim=self.config['output_dim'],
            shallow_depth=self.config['shallow_depth'],
            poly_degree=self.config['polynomial_degree'],
            learning_rate=self.config['boosting_learning_rate']
        ).to(self.device)
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)
        
        best_val_loss = float('inf')
        best_ensemble_state = None
        
        for attempt in range(self.config['num_models_max']):
            if verbose:
                print(f"\n训练候选模型 {attempt + 1}/{self.config['num_models_max']}...")
            
            # 创建候选模型
            candidate = self.ensemble.create_candidate().to(self.device)
            self.ensemble.freeze_for_candidate_training()
            
            optimizer = optim.AdamW(
                candidate.parameters(), 
                lr=self.config['learning_rate_adam'],
                weight_decay=self.config['weight_decay']
            )
            
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=10, factor=0.5, min_lr=1e-5
            )
            
            best_candidate_loss = float('inf')
            best_candidate_state = None
            no_improve = 0
            
            # 训练候选模型
            for epoch in range(self.config['epochs_per_model']):
                self.ensemble.train()
                optimizer.zero_grad()
                
                outputs = self.ensemble(X_train_t)
                loss = F.mse_loss(outputs, y_train_t)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    candidate.parameters(), 
                    self.config['gradient_clip']
                )
                optimizer.step()
                
                # 验证
                self.ensemble.eval()
                with torch.no_grad():
                    val_outputs = self.ensemble(X_val_t)
                    val_loss = F.mse_loss(val_outputs, y_val_t).item()
                
                scheduler.step(val_loss)
                
                if val_loss < best_candidate_loss:
                    best_candidate_loss = val_loss
                    best_candidate_state = copy.deepcopy(candidate.state_dict())
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve >= self.config['early_stopping_patience']:
                    if verbose:
                        print(f"  早停于epoch {epoch+1}")
                    break
                
                if verbose and (epoch + 1) % 50 == 0:
                    print(f"  Epoch {epoch+1}: Loss = {loss:.6f}, Val Loss = {val_loss:.6f}")
            
            # 恢复最佳候选状态
            if best_candidate_state:
                candidate.load_state_dict(best_candidate_state)
            
            # 评估是否采纳候选模型
            self.ensemble.eval()
            with torch.no_grad():
                val_outputs_with_candidate = self.ensemble(X_val_t)
                val_loss_with_candidate = F.mse_loss(val_outputs_with_candidate, y_val_t).item()
            
            if val_loss_with_candidate < best_val_loss:
                self.ensemble.promote_candidate()
                best_val_loss = val_loss_with_candidate
                best_ensemble_state = copy.deepcopy(self.ensemble.state_dict())
                if verbose:
                    print(f"  ✓ 候选模型被采纳 (Val Loss: {val_loss_with_candidate:.6f})")
            else:
                self.ensemble.discard_candidate()
                if verbose:
                    print(f"  ✗ 候选模型被丢弃")
                    print(f"  停止添加更多模型")
                break
        
        # 恢复最佳集成状态
        if best_ensemble_state:
            self.ensemble.load_state_dict(best_ensemble_state)
            # 确保没有候选模型
            self.ensemble.candidate_model = None
    
    def aggressive_simplify(self, coeffs: np.ndarray, degree: int):
        """激进的多项式简化，去除不重要的项"""
        x = symbols('x')
        
        # 找到最大系数（按绝对值）
        abs_coeffs = np.abs(coeffs)
        max_coeff = np.max(abs_coeffs)
        
        if max_coeff == 0:
            return 0
        
        # 计算每个系数的相对重要性
        relative_importance = abs_coeffs / max_coeff
        
        # 构建简化的表达式
        expr = 0
        terms_added = []
        
        for i, (coeff, importance) in enumerate(zip(coeffs, relative_importance)):
            power = degree - i
            
            # 只保留相对重要性超过阈值的项
            if importance >= self.config['simplification_threshold']:
                # 如果系数接近整数，使用整数
                if abs(coeff - round(coeff)) < 0.1:
                    simplified_coeff = round(coeff)
                elif abs(coeff) > 10:  # 大系数四舍五入
                    simplified_coeff = round(coeff, 1)
                else:
                    # 尝试找到简单的分数表示
                    simplified_coeff = float(nsimplify(coeff, rational=True, tolerance=0.05))
                    # 如果分数太复杂，使用小数
                    if len(str(simplified_coeff)) > 8:
                        simplified_coeff = round(coeff, 3)
                
                if simplified_coeff != 0:
                    if power == 0:
                        expr += simplified_coeff
                        terms_added.append(f"{simplified_coeff}")
                    elif power == 1:
                        if simplified_coeff == 1:
                            expr += x
                            terms_added.append("x")
                        elif simplified_coeff == -1:
                            expr -= x
                            terms_added.append("-x")
                        else:
                            expr += simplified_coeff * x
                            terms_added.append(f"{simplified_coeff}*x")
                    else:
                        if simplified_coeff == 1:
                            expr += x**power
                            terms_added.append(f"x**{power}")
                        elif simplified_coeff == -1:
                            expr -= x**power
                            terms_added.append(f"-x**{power}")
                        else:
                            expr += simplified_coeff * x**power
                            terms_added.append(f"{simplified_coeff}*x**{power}")
        
        # 如果没有保留任何项，至少保留最重要的项
        if expr == 0 and len(coeffs) > 0:
            max_idx = np.argmax(abs_coeffs)
            coeff = coeffs[max_idx]
            power = degree - max_idx
            
            if abs(coeff - round(coeff)) < 0.1:
                coeff = round(coeff)
            else:
                coeff = round(coeff, 3)
            
            if power == 0:
                expr = coeff
            elif power == 1:
                expr = coeff * x if coeff not in [1, -1] else (x if coeff == 1 else -x)
            else:
                expr = coeff * x**power if coeff not in [1, -1] else (x**power if coeff == 1 else -x**power)
        
        # 最终简化
        try:
            expr = simplify(expr)
            # 尝试因式分解
            expr_factored = factor(expr)
            # 如果因式分解后更简单，使用它
            if len(str(expr_factored)) < len(str(expr)):
                expr = expr_factored
        except:
            pass
        
        return expr
    
    def analyze_and_decompose(self, X_test: np.ndarray):
        """分析与分解：对每个子模型进行IPS分析"""
        self.symbolic_expressions = []
        
        print("\n" + "="*60)
        print("IPS分析结果:")
        print("="*60)
        
        for idx, model in enumerate(self.ensemble.models):
            # IPS多项式逼近
            model.eval()
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            
            with torch.no_grad():
                y_pred = model(X_tensor).cpu().numpy().flatten()
            
            X_flat = X_test.flatten()
            
            # 使用AIC选择最佳阶数
            best_degree = 1
            best_aic = float('inf')
            best_coeffs = None
            
            for degree in range(1, min(self.config['polynomial_degree'] + 3, 10)):
                try:
                    coeffs = np.polyfit(X_flat, y_pred, degree)
                    poly_pred = np.polyval(coeffs, X_flat)
                    
                    mse = np.mean((y_pred - poly_pred) ** 2)
                    aic = len(X_flat) * np.log(mse + 1e-10) + 2 * (degree + 1)
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_degree = degree
                        best_coeffs = coeffs
                except:
                    break
            
            # 激进简化
            symbolic_expr = self.aggressive_simplify(best_coeffs, best_degree)
            
            print(f"\n子模型 {idx + 1}:")
            print(f"  原始多项式阶数: {best_degree}")
            print(f"  简化后的符号形式: {symbolic_expr}")
            
            self.symbolic_expressions.append({
                'model_idx': idx,
                'coefficients': best_coeffs,
                'degree': best_degree,
                'symbolic': symbolic_expr
            })
        
        print("="*60)
        return self.symbolic_expressions
    
    def combine_symbolic_expressions(self):
        """组合所有子模型的符号形式"""
        if not self.symbolic_expressions:
            return None
        
        x = symbols('x')
        combined = 0
        
        print("\n组合过程:")
        for i, expr_dict in enumerate(self.symbolic_expressions):
            weighted_expr = self.config['boosting_learning_rate'] * expr_dict['symbolic']
            print(f"  {self.config['boosting_learning_rate']} × ({expr_dict['symbolic']}) = {simplify(weighted_expr)}")
            combined += weighted_expr
        
        # 最终简化
        combined = simplify(expand(combined))
        
        # 尝试进一步简化
        try:
            # 收集同类项
            combined = collect(combined, symbols('x'))
            
            # 如果结果是多项式，尝试激进简化
            poly = combined.as_poly(symbols('x'))
            if poly:
                coeffs = [float(c) for c in poly.all_coeffs()]
                degree = poly.degree()
                combined_simplified = self.aggressive_simplify(np.array(coeffs), degree)
                if combined_simplified != 0:
                    combined = combined_simplified
        except:
            pass
        
        return combined
    
    def run_experiment(self, formula: str, experiment_name: str = "experiment"):
        """符号回归实验"""
        print(f"\n{'='*80}")
        print(f"连分式符号回归实验: {experiment_name}")
        print(f"目标公式: {formula}")
        print(f"{'='*80}")
        
        # 1. 生成数据
        print("\n1. 生成数据...")
        X, y = self.generate_data(formula, n_samples=800)
        
        # 分割数据
        n_train = int(len(X) * 0.6)
        n_val = int(len(X) * 0.2)
        
        indices = np.random.permutation(len(X))
        X_train = X[indices[:n_train]]
        y_train = y[indices[:n_train]]
        X_val = X[indices[n_train:n_train+n_val]]
        y_val = y[indices[n_train:n_train+n_val]]
        X_test = X[indices[n_train+n_val:]]
        y_test = y[indices[n_train+n_val:]]
        
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  验证集: {len(X_val)} 样本")
        print(f"  测试集: {len(X_test)} 样本")
        
        # 2. 训练
        print("\n2. 训练EnsembleResCoFrNet...")
        self.train_ensemble(X_train, y_train, X_val, y_val)
        print(f"\n最终采纳的模型数量: {len(self.ensemble.models)}")
        
        # 3. 分析与分解
        self.analyze_and_decompose(X_test)
        
        # 4. 符号化与组合
        print("\n" + "="*60)
        print("符号表达式组合:")
        print("="*60)
        combined_expr = self.combine_symbolic_expressions()
        
        print("\n" + "="*60)
        print("最终结果:")
        print("="*60)
        print(f"发现的公式: {combined_expr}")
        print(f"真实公式:   {formula}")
        print("="*60)
        
        # 5. 评估
        self.ensemble.eval()
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            y_pred = self.ensemble(X_test_t).cpu().numpy()
        
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        
        print(f"\n性能指标:")
        print(f"  MSE:  {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  R²:   {r2:.6f}")
        
        # 6. 可视化
        self.visualize_results(X_test, y_test, y_pred, combined_expr, formula, experiment_name)
        
        return {
            'formula': formula,
            'discovered': str(combined_expr),
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'num_models': len(self.ensemble.models),
            'symbolic_expressions': self.symbolic_expressions
        }
    
    def visualize_results(self, X_test, y_test, y_pred, discovered_formula, true_formula, name):
        """创建可视化结果"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        colors = sns.color_palette("husl", n_colors=max(len(self.ensemble.models) + 2, 5))
        
        # 1. 数据拟合
        ax = axes[0, 0]
        sorted_idx = np.argsort(X_test.flatten())
        ax.scatter(X_test[sorted_idx], y_test[sorted_idx], 
                  alpha=0.5, label='True Data', s=20, color=colors[0])
        ax.plot(X_test[sorted_idx], y_pred[sorted_idx], 
               'r-', label='CoFrNet Prediction', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Data Fitting')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 子模型贡献
        ax = axes[0, 1]
        x_range = np.linspace(X_test.min(), X_test.max(), 200).reshape(-1, 1)
        x_tensor = torch.FloatTensor(x_range).to(self.device)
        
        for idx, model in enumerate(self.ensemble.models):
            with torch.no_grad():
                y_sub = self.config['boosting_learning_rate'] * model(x_tensor).cpu().numpy()
            ax.plot(x_range, y_sub, label=f'Model {idx+1}', 
                   color=colors[idx+2], linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('x')
        ax.set_ylabel('Contribution')
        ax.set_title('Submodel Contributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 残差图
        ax = axes[1, 0]
        residuals = y_test.flatten() - y_pred.flatten()
        ax.scatter(y_pred.flatten(), residuals, alpha=0.6, s=20, color=colors[0])
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.grid(True, alpha=0.3)
        
        # 4. Q-Q图
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'CoFrNet Symbolic Regression: {name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = f'cofrnet_symbolic_{name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图表已保存至: {save_path}")
        plt.show()

def run_all_experiments():
    """运行所有实验"""
    test_cases = [
        ("x**2 + 2*x + 1", "quadratic"),
        ("sin(x) + 0.5*x", "sine_linear"),
        ("exp(-x**2/2)", "gaussian"),
        ("1/(1 + x**2)", "lorentzian"),
        ("x**3 - 2*x**2 + x", "cubic")
    ]
    
    # 针对不同函数的配置
    configs = {
        "quadratic": {
            'shallow_depth': 1,
            'polynomial_degree': 3,
            'num_models_max': 2,
            'boosting_learning_rate': 0.5,
            'sparsity_threshold': 0.05,
            'simplification_threshold': 0.1
        },
        "sine_linear": {
            'shallow_depth': 2,
            'polynomial_degree': 5,
            'num_models_max': 3,
            'boosting_learning_rate': 0.3,
            'sparsity_threshold': 0.03,
            'simplification_threshold': 0.08
        },
        "gaussian": {
            'shallow_depth': 2,
            'polynomial_degree': 6,
            'num_models_max': 2,
            'boosting_learning_rate': 0.4,
            'sparsity_threshold': 0.04,
            'simplification_threshold': 0.1
        },
        "lorentzian": {
            'shallow_depth': 2,
            'polynomial_degree': 4,
            'num_models_max': 2,
            'boosting_learning_rate': 0.4,
            'sparsity_threshold': 0.04,
            'simplification_threshold': 0.1
        },
        "cubic": {
            'shallow_depth': 3,  
            'polynomial_degree': 6,  
            'num_models_max': 3,  
            'boosting_learning_rate': 0.3,  
            'learning_rate_adam': 0.005,  
            'epochs_per_model': 300,  
            'sparsity_threshold': 0.03,  
            'simplification_threshold': 0.05,  
            'noise_level': 0.0005  
        }
    }
    
    results = []
    
    for func_str, name in test_cases:
        print(f"\n\n{'#'*80}")
        print(f"# 实验: {name.upper()}")
        print(f"{'#'*80}")
        
        config = configs.get(name, {})
        sr_model = SymbolicRegressionCoFrNet(config)
        
        try:
            result = sr_model.run_experiment(func_str, name)
            results.append(result)
        except Exception as e:
            print(f"实验 {name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印总结
    print("\n\n" + "="*80)
    print("实验总结")
    print("="*80)
    
    if results:
        summary_data = []
        for r in results:
            summary_data.append({
                'Function': r['formula'],
                'Discovered': r['discovered'],
                'R²': f"{r['r2']:.4f}",
                'RMSE': f"{r['rmse']:.4f}",
                'Models': r['num_models']
            })
        
        df_results = pd.DataFrame(summary_data)
        print(df_results.to_string(index=False))
        
        # 创建总结图表
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # R²对比
        ax = axes[0]
        names = [r['formula'].split('(')[0][:15] for r in results]
        r2_values = [r['r2'] for r in results]
        colors = sns.color_palette("viridis", n_colors=len(results))
        
        bars = ax.bar(range(len(names)), r2_values, color=colors, alpha=0.8)
        ax.set_xlabel('Test Function')
        ax.set_ylabel('R² Score')
        ax.set_title('Model Performance (R²)')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        ax.axhline(y=1, color='green', linestyle='--', alpha=0.5)
        
        for bar, val in zip(bars, r2_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom')
        
        # RMSE对比
        ax = axes[1]
        rmse_values = [r['rmse'] for r in results]
        bars = ax.bar(range(len(names)), rmse_values, color=colors, alpha=0.8)
        ax.set_xlabel('Test Function')
        ax.set_ylabel('RMSE')
        ax.set_title('Model Error (RMSE)')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        
        for bar, val in zip(bars, rmse_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                   f'{val:.3f}', ha='center', va='bottom')
        
        plt.suptitle('CoFrNet Symbolic Regression Summary', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('cofrnet_symbolic_summary.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    return results

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    results = run_all_experiments()