# 连分网络，可学习项为多项式函数
# 由于观察到深度过深时会产生梯度爆炸的问题，在ElementwiseCFN_Activation内设置clamp_max和clamp_min对输出进行限制

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_polynomial_basis(x: torch.Tensor, num_basis_functions: int) -> torch.Tensor:
    """
    Computes polynomial basis function values for input tensor x.
    Args:
        x: Input tensor, shape (batch_size, 1).
        num_basis_functions: Number of basis functions.

    Returns:
        Basis values tensor, shape (batch_size, num_basis_functions).
    """
    if num_basis_functions <= 0:
        return torch.empty(x.shape[0], 0, dtype=x.dtype, device=x.device)

    # Create polynomial basis: [x^0, x^1, x^2, ..., x^(num_basis_functions-1)]
    # Ensure x is 2D (batch_size, 1)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    elif x.ndim > 2 or x.shape[-1] != 1:
        raise ValueError(f"Input to get_polynomial_basis must be (batch_size, 1) or (batch_size,) but got {x.shape}")

    # Compute powers element-wise across the batch
    # Squeeze x to (batch_size,) for power calculation, then stack
    basis_values = torch.stack([x.squeeze(-1)**i for i in range(num_basis_functions)], dim=1) # Shape (batch_size, num_basis_functions)

    return basis_values

# --- Differentiable Learnable Function Module (uses coefficients for a basis) ---
# This represents a single learnable function f(scalar) -> scalar
class DifferentiableLearnableFunction(nn.Module):
    def __init__(self, num_basis_functions: int):
        """
        Represents a trainable function using learned coefficients for basis functions.
        Args:
            num_basis_functions: Number of basis functions / coefficients to learn.
        """
        super().__init__()
        self.num_basis_functions = num_basis_functions
        # These are the learnable parameters (coefficients for the basis functions)
        # Initialize with smaller values might help stability
        self.coeffs = nn.Parameter(torch.randn(num_basis_functions) * 0.01) # Using smaller initialization

        # Note: We conceptually use basis functions here (polynomials in the placeholder)
        # If using B-splines, basis parameters (knots, degree) would be fixed hyperparameters
        # and the actual B-spline basis evaluation needs a PyTorch implementation.


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the learnable function for input x.
        Args:
            x: Input tensor, shape (batch_size, 1).

        Returns:
            Output tensor, shape (batch_size, 1).
        """
        # Get basis function values for the input x
        # Using polynomial basis placeholder
        basis_values = get_polynomial_basis(x, self.num_basis_functions) # Shape (batch_size, num_basis_functions)

        # Linear combination of basis functions with learned coefficients
        # output = sum(coeffs_i * basis_i(x)) -> batch matrix multiplication
        # (batch_size, num_basis_functions) @ (num_basis_functions,) -> (batch_size,)
        output = torch.matmul(basis_values, self.coeffs)

        # Reshape output to (batch_size, 1) to maintain dimension for potential stacking
        output = output.unsqueeze(-1) # Shape (batch_size, 1)

        return output

# --- Element-wise Continued Fraction Activation Module ---
# This module applies the continued fraction calculation to each element of the input tensor.
class ElementwiseCFN_Activation(nn.Module):
    def __init__(self, depth: int, num_basis_functions_per_term: int):
        """
        Applies a trainable continued fraction function element-wise.

        Args:
            depth: Fixed recursion depth of the continued fraction.
            num_basis_functions_per_term: Number of basis functions/coefficients
                                        for each a_n(.) and b_n(.) term.
        """
        super().__init__()
        self.depth = depth
        self.num_basis_functions_per_term = num_basis_functions_per_term

        # Create ModuleLists to hold the trainable terms (a_n(scalar)->scalar, b_n(scalar)->scalar)
        # Each term uses the DifferentiableLearnableFunction
        # a_n(x) terms: a_0, a_1, ..., a_depth (depth + 1 terms)
        self.a_terms = nn.ModuleList([
            DifferentiableLearnableFunction(num_basis_functions_per_term)
            for _ in range(depth + 1)
        ])

        # b_n(x) terms: b_1, b_2, ..., b_depth (depth terms)
        self.b_terms = nn.ModuleList([
            DifferentiableLearnableFunction(num_basis_functions_per_term)
            for _ in range(depth)
        ])

        # Define clamping limits for numerical stability
        self.clamp_min = -1e0 # Minimum value for intermediate results
        self.clamp_max = 1e0  # Maximum value for intermediate results


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the value of the fixed-depth continued fraction for each element of the input tensor.
        Args:
            x: Input tensor, shape (batch_size, num_features).

        Returns:
            Output tensor, shape (batch_size, num_features).
        """
        # Store original shape to reshape back at the end
        original_shape = x.shape
        batch_size, num_features = original_shape

        # Flatten the input to apply the scalar function logic element-wise
        # Shape becomes (batch_size * num_features, 1)
        x_flat = x.view(-1, 1)

        # Evaluate the bottom term: a_depth(x_flat)
        # Each a_term/b_term takes (batch_size*num_features, 1) and returns (batch_size*num_features, 1)
        result = self.a_terms[self.depth](x_flat) # Shape (batch_size * num_features, 1)

        # Clamp the initial result
        result = torch.clamp(result, self.clamp_min, self.clamp_max)


        # Evaluate upwards recursively/iteratively
        # Iterate from a_{depth-1} down to a_0
        for i in range(self.depth -1, -1, -1):
            a_val = self.a_terms[i](x_flat) # Shape (batch_size * num_features, 1)
            # b_terms index i corresponds to b_{i+1}(x) in the standard CF notation
            b_val = self.b_terms[i](x_flat) # Shape (batch_size * num_features, 1)

            # Clamp a_val and b_val
            a_val = torch.clamp(a_val, self.clamp_min, self.clamp_max)
            b_val = torch.clamp(b_val, self.clamp_min, self.clamp_max)

            # Handle potential division by zero or very small numbers
            # Add epsilon for numerical stability during training
            epsilon = 1e-8 # Keep epsilon


            # Perform the continued fraction step: a_i(x) + b_{i+1}(x) / (a_{i+1}(x) + ...)
            # Add epsilon to the denominator to prevent division by zero
            denominator = result + epsilon * torch.sign(result) # Add epsilon in a sign-preserving way
            # Or a simpler version: denominator = result + epsilon # But can change sign near zero

            # Ensure denominator is not zero after adding epsilon if result was -epsilon
            denominator = torch.where(torch.abs(denominator) < epsilon, torch.tensor(epsilon, device=result.device, dtype=result.dtype) * torch.sign(result), denominator)
            # Handle case where result was exactly 0 -> denominator is epsilon * sign(0) which is 0. Make it epsilon
            denominator = torch.where(denominator == 0, torch.tensor(epsilon, device=result.device, dtype=result.dtype), denominator)


            term_to_add = b_val / denominator # Shape (batch_size * num_features, 1)

            # Clamp the term before adding to prevent explosion
            term_to_add = torch.clamp(term_to_add, self.clamp_min, self.clamp_max)


            result = a_val + term_to_add # Shape (batch_size * num_features, 1)

            # Clamp the result for the next iteration
            result = torch.clamp(result, self.clamp_min, self.clamp_max)


        # Reshape the result back to the original input shape
        result = result.view(original_shape) # Shape (batch_size, num_features)

        return result


# --- Multi-Layer Network using Elementwise CFN Activation ---
class MultiLayerCFN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list,
                cfn_depth: int, cfn_num_basis_functions_per_term: int):
        """
        A multi-layer network using ElementwiseCFN_Activation.

        Args:
            input_dim: Dimension of the input features.
            output_dim: Dimension of the output.
            hidden_dims: A list of integers specifying the number of features in hidden layers.
            cfn_depth: Recursion depth for the Continued Fraction Activation.
            cfn_num_basis_functions_per_term: Number of basis functions for each term
                                            within the CFN Activation.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.cfn_depth = cfn_depth
        self.cfn_num_basis_functions_per_term = cfn_num_basis_functions_per_term

        layers = []
        # Input layer
        in_features = input_dim
        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(in_features, hidden_dim))
            # CFN Activation layer
            for i in range(hidden_dim):
                layers.append(ElementwiseCFN_Activation(cfn_depth, cfn_num_basis_functions_per_term))
            in_features = hidden_dim

        # Output layer (no activation after the last linear layer for classification)
        layers.append(nn.Linear(in_features, output_dim))

        # Combine layers into a Sequential model
        self.net = nn.Sequential(*layers)

        # Initialize weights with smaller values might help stability
        # Note: Model weights are also initialized based on a random seed
        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Use Kaiming initialization suitable for ReLU-like activations (CFN is not ReLU, but often a decent starting point)
                nn.init.kaiming_uniform_(m.weight, a=0.01, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.apply(init_weights)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        Args:
            x: Input tensor, shape (batch_size, input_dim).

        Returns:
            Output tensor, shape (batch_size, output_dim).
        """
        # Ensure input shape is correct (batch_size, input_dim)
        if x.ndim != 2 or x.shape[-1] != self.input_dim:
            raise ValueError(f"Input tensor must have shape (batch_size, {self.input_dim}), but got {x.shape}")

        return self.net(x)
    
if __name__=="__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    INPUT_DIM = 28 * 28 
    OUTPUT_DIM = 10 # Number of classes in MNIST (0-9)

    num_epochs = 20 # Reduced epochs for a quicker example run on MNIST
    batch_size = 100
    learning_rate = 0.001 # Adjusted learning rate

    HIDDEN_DIMS = [2]
    CFN_DEPTH = 3
    CFN_NUM_BASIS_FUNCTIONS_PER_TERM = 3
    CLIP_GRAD_MAX_NORM = 1.0

    # --- Data Loading and Preprocessing for MNIST ---
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert PIL image to tensor (H*W*C) in [0, 1]
        # We don't normalize here for simplicity, CFN might prefer [0, 1] range inputs directly
        # If you add normalization later, ensure your CFN learnable functions can handle the new range.
        # transforms.Normalize((0.5,), (0.5,)) # Example normalization
    ])

    # Download and load the training dataset
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=True,
                                            transform=transform,
                                            download=True)

    # Download and load the test dataset
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                            train=False,
                                            transform=transform,
                                            download=True)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=batch_size,
                            shuffle=False)


    # --- Model Setup ---
    model = MultiLayerCFN(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        cfn_depth=CFN_DEPTH,
        cfn_num_basis_functions_per_term=CFN_NUM_BASIS_FUNCTIONS_PER_TERM
    ).to(device) # Move model to the configured device

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss() # Changed to CrossEntropyLoss for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # --- Training Loop ---
    print("Model structure:")
    print(model)
    print(f"Total number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    print("\nStarting training on MNIST...")
    start_time = time.time() # 记录训练开始时间

    train_losses = []
    test_losses = []
    test_accuracies = []

    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            # Move tensors to the configured device
            images = images.reshape(-1, INPUT_DIM).to(device) # Flatten images and move to device
            labels = labels.to(device) # Move labels to device

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Check for NaN loss
            if torch.isnan(loss):
                print(f"Epoch {epoch+1}, Step {i+1}: Loss is NaN. Stopping training.")
                break # Stop epoch or training if NaN occurs

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_MAX_NORM)

            optimizer.step()

            running_loss += loss.item()

            # Print training progress periodically
            if (i + 1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Train Loss: {loss.item():.6f}')

        # Calculate average training loss for the epoch
        epoch_train_loss = running_loss / total_steps
        train_losses.append(epoch_train_loss)

        # --- Evaluation on Test set ---
        # Set model to evaluation mode
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            running_test_loss = 0.0
            for images, labels in test_loader:
                images = images.reshape(-1, INPUT_DIM).to(device) # Flatten images and move to device
                labels = labels.to(device) # Move labels to device

                outputs = model(images)
                test_loss = criterion(outputs, labels)
                running_test_loss += test_loss.item()

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_test_loss = running_test_loss / len(test_loader)
            accuracy = 100 * correct / total

            test_losses.append(epoch_test_loss)
            test_accuracies.append(accuracy)

            print(f'Epoch [{epoch+1}/{num_epochs}], Test Loss: {epoch_test_loss:.6f}, Test Accuracy: {accuracy:.2f} %')


    end_time = time.time() # 记录训练结束时间
    print("Training finished.")

    total_time = end_time - start_time # 计算总训练时间
    print(f"Total training time: {total_time:.4f} seconds") # 打印总训练时间

    print(f"Final Test Loss: {test_losses[-1]:.6f}")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f} %")

    # --- Visualization ---
    print("\nGenerating visualization...")

    # Plot training and test loss over epochs
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1) # Plot loss on the left
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', marker='o', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Test Loss over Epochs (MNIST)")
    plt.legend()
    plt.grid(True)
    plt.yscale('log') # Use a log scale for loss

    # Plot test accuracy over epochs
    plt.subplot(1, 2, 2) # Plot accuracy on the right
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy', marker='o', color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy over Epochs (MNIST)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 100) # Accuracy is between 0 and 100%


    plt.tight_layout() # Adjust layout to prevent overlap
    plt.show()