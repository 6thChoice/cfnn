"""Common-budget Energy Efficiency compatibility audit."""
import argparse,json,time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from fair_baselines import build_mlp,build_gaussian_rbf,build_pykan,build_cofrnet,build_rational_nn,build_siren
from eis_models import CFNNHybridEIS

FAMILIES=['CFNN-Hybrid','MLP','Gaussian RBF','KAN','CoFrNet-Standard','Rational activation NN','SIREN']; BUDGETS=[64,256,512]; SEEDS=[42,123,456,789,1024]
CSV=Path(__file__).with_name('energy_efficiency.csv')
def build(f,b,s):
    if f=='CFNN-Hybrid': return CFNNHybridEIS(8,1,max(1,b//40),3),{}
    if f=='MLP': return build_mlp(8,1,max(4,b//11),2)
    if f=='Gaussian RBF': return build_gaussian_rbf(8,1,max(4,b//10))
    if f=='KAN': return build_pykan(8,1,max(3,b//20),grid=5,seed=s)
    if f=='CoFrNet-Standard': return build_cofrnet(8,1,max(1,b//20),3)
    if f=='Rational activation NN': return build_rational_nn(8,1,max(4,b//11),2)
    if f=='SIREN': return build_siren(8,1,max(4,b//11),2)
def fit(m,x,y,vx,vy,epochs,seed):
    torch.manual_seed(seed); opt=torch.optim.AdamW(m.parameters(),lr=3e-3,weight_decay=1e-5); loss=nn.MSELoss(); best=1e99; be=1; stale=0
    for ep in range(epochs):
        m.train(); opt.zero_grad(); z=loss(m(x),y); z.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),5); opt.step(); m.eval()
        with torch.no_grad(): q=loss(m(vx),vy).item()
        if q<best: best,be,stale=q,ep+1,0
        else: stale+=1
        if stale>=50: break
    return be,best
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='experiment_refine/energy_common_budget_results/raw.json'); ap.add_argument('--epochs',type=int,default=200); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args(); seeds=SEEDS[:1] if a.smoke else SEEDS
    df=pd.read_csv(CSV); X=df[[c for c in df if c.upper().startswith('X')][:8]].values.astype('float32'); y=df[[c for c in df if c.upper()=='Y1'][0]].values.astype('float32'); rec=[]; out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text('[]')
    for s in seeds:
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=s); Xtr,Xv,ytr,yv=train_test_split(Xtr,ytr,test_size=.1,random_state=s); xs=StandardScaler().fit(Xtr); ym,ys=ytr.mean(0),ytr.std(0)+1e-6
        to=lambda z: torch.from_numpy(xs.transform(z).astype('float32')); xt,xv,xe=to(Xtr),to(Xv),to(Xte); yt=torch.from_numpy(((ytr-ym)/ys).astype('float32')).reshape(-1,1); yv=torch.from_numpy(((yv-ym)/ys).astype('float32')).reshape(-1,1); ye=torch.from_numpy(((yte-ym)/ys).astype('float32')).reshape(-1,1)
        for f in FAMILIES:
            for b in BUDGETS:
                t=time.time(); m,_=build(f,b,s); ep,vm=fit(m,xt,yt,xv,yv,a.epochs,s); fm,_=build(f,b,s+1000003); opt=torch.optim.AdamW(fm.parameters(),lr=3e-3,weight_decay=1e-5); loss=nn.MSELoss()
                for _ in range(ep): opt.zero_grad(); z=loss(fm(torch.cat([xt,xv])),torch.cat([yt,yv])); z.backward(); torch.nn.utils.clip_grad_norm_(fm.parameters(),5); opt.step()
                with torch.no_grad(): pred=fm(xe).numpy()
                r={'family':f,'budget':b,'seed':s,'actual_parameters':sum(p.numel() for p in fm.parameters() if p.requires_grad),'test_r2':float(r2_score(ye.numpy(),pred)),'selection_val_mse':vm,'selected_epochs':ep,'seconds':time.time()-t,'metadata':{'target':'heating load Y1','input_scaler':'train only','target_scaler':'train only'}}; rec.append(r); out.write_text(json.dumps(rec,indent=2)); print(f,b,s,r['test_r2'],flush=True)
if __name__=='__main__': main()
