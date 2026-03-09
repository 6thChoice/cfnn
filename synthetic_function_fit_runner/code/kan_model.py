"""
KAN (Kolmogorov-Arnold Networks) - Practical implementation for function fitting.
Simplified version using learnable piecewise linear activations for speed.
"""
import torch
import torch.nn as nn
import numpy as np


class PiecewiseLinearKAN(nn.Module):
    """
    Practical KAN implementation using learnable piecewise linear activations.
    Much faster than B-spline version while keeping learnable activation concept.
    """
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 64], num_knots=8):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_knots = num_knots

        # Build network
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(LearnableActivationLayer(dims[i], dims[i + 1], num_knots))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LearnableActivationLayer(nn.Module):
    """
    Layer with learnable piecewise linear activation functions.
    Similar to KAN concept but much more efficient.
    """
    def __init__(self, in_features, out_features, num_knots=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_knots = num_knots

        # Linear weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Learnable activation: piecewise linear function
        # Represented by knots at uniform positions
        self.activation_knots = nn.Parameter(torch.randn(out_features, in_features, num_knots) * 0.1)

        # Knot positions (fixed)
        self.register_buffer('knot_positions', torch.linspace(-3, 3, num_knots))

    def forward(self, x):
        # Apply learnable activation to each input
        # x: (batch, in_features)
        # activation_knots: (out_features, in_features, num_knots)

        # For each (out, in) pair, apply the learnable function
        # Use linear interpolation between knots

        batch_size = x.shape[0]

        # Clamp and prepare
        x_clamped = torch.clamp(x, -3, 3)  # (batch, in_features)

        # Compute interpolated activation values
        # For efficiency: use a simplified approach
        # Find which interval each x falls into
        activations = self._interpolate_activation(x_clamped)  # (batch, in_features, num_knots)

        # Weighted sum: (batch, in, num_knots) * (out, in, num_knots) -> (batch, out)
        weighted = activations.unsqueeze(1) * self.activation_knots.unsqueeze(0)  # (batch, out, in, num_knots)
        activated = weighted.sum(dim=(2, 3))  # (batch, out)

        # Standard linear pass
        linear_out = torch.matmul(x, self.weight.t()) + self.bias

        return linear_out + activated * 0.5  # Combine both

    def _interpolate_activation(self, x):
        """
        Linear interpolation of activation values at knot positions.
        Returns interpolated values at x for each knot pattern.
        Simplified: just return basis functions.
        """
        batch_size, in_features = x.shape

        # Compute triangular basis functions (tent functions) at knot positions
        # For each x, compute weight at each knot
        activations = torch.zeros(batch_size, in_features, self.num_knots, device=x.device)

        for i, pos in enumerate(self.knot_positions):
            # Distance-based weight (triangular/tent function)
            dist = torch.abs(x - pos)
            spacing = self.knot_positions[1] - self.knot_positions[0] if len(self.knot_positions) > 1 else 1.0
            weight = torch.clamp(1 - dist / spacing, 0, 1)
            activations[:, :, i] = weight

        return activations


# Alias for compatibility
KAN = PiecewiseLinearKAN


def get_kan_hidden_dim(target_params, input_dim=3, output_dim=1, grid_size=8, spline_order=3, num_hidden_layers=2):
    """Calculate hidden dimension to match target parameters."""
    # Each connection: weight + bias contribution + activation_knots
    params_per_connection = 1 + grid_size  # simplified

    def count_params(hidden_dim):
        total = 0
        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            total += out_d * in_d * params_per_connection + out_d
        return total

    low, high = 1, 512
    best_h = 8
    while low <= high:
        mid = (low + high) // 2
        if count_params(mid) < target_params:
            best_h = mid
            low = mid + 1
        else:
            high = mid - 1
    return best_h


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test
    model = KAN(input_dim=3, output_dim=1, hidden_dims=[32, 32], num_knots=8)
    print(f"Params: {count_parameters(model)}")

    # Speed test
    import time
    x = torch.randn(128, 3)

    # Warmup
    for _ in range(10):
        _ = model(x)

    start = time.time()
    for _ in range(100):
        _ = model(x)
    elapsed = time.time() - start
    print(f"Time per batch: {elapsed/100*1000:.2f} ms")

    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
