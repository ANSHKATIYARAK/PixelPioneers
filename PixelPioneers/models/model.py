import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Variance Stabilizing Transform (VST) Modules
# =====================================================================

class LogVST(nn.Module):
    """
    Stabilizes multiplicative speckle noise by mapping the image
    into log-space: v = log(y + delta)
    """
    def __init__(self, delta=1e-3):
        super().__init__()
        self.delta = delta

    def forward(self, y):
        # Allow values slightly below 0 (speckle can push intensities out-of-range)
        return torch.log(torch.clamp(y + self.delta, min=1e-5))


class InvLogVST(nn.Module):
    """
    Maps the reconstructed image back to the intensity domain:
    x = exp(v) - delta
    """
    def __init__(self, delta=1e-3):
        super().__init__()
        self.delta = delta

    def forward(self, v):
        return torch.exp(v) - self.delta


# =====================================================================
# 2. Learnable Physical Operators
# =====================================================================

class LearnableBlurOperator(nn.Module):
    """
    Convolves the input with a learnable blur kernel initialized as a Gaussian.
    Enforces non-negativity and energy conservation (kernel sums to 1).
    """
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size
        
        # Initialize with Gaussian kernel
        ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / torch.sum(kernel)
        
        # Raw kernel parameters that can be updated during training
        self.raw_kernel = nn.Parameter(kernel.view(1, 1, kernel_size, kernel_size))

    def get_kernel(self):
        # Enforce non-negativity using clamp
        kernel_pos = torch.clamp(self.raw_kernel, min=0.0)
        # Normalize to conserve energy (sum of weights = 1.0)
        return kernel_pos / (torch.sum(kernel_pos) + 1e-8)

    def forward(self, x):
        kernel = self.get_kernel()
        padding = self.kernel_size // 2
        x_pad = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        return F.conv2d(x_pad, kernel)

    def forward_transpose(self, grad):
        """Conjoint transpose operator (convolution with spatially flipped kernel)"""
        kernel = self.get_kernel()
        kernel_flipped = torch.flip(kernel, dims=[2, 3])
        padding = self.kernel_size // 2
        grad_pad = F.pad(grad, (padding, padding, padding, padding), mode='reflect')
        return F.conv2d(grad_pad, kernel_flipped)


class DifferentiableDownsampleOperator(nn.Module):
    """
    Simulates 2x spatial downsampling differentiably using average pooling.
    Includes its mathematically exact transpose operator for upsampling gradients.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 2x downsampling via average pooling
        return F.avg_pool2d(x, kernel_size=2, stride=2)

    def forward_transpose(self, y):
        # Mathematically exact transpose of 2x average pooling:
        # Duplicates pixels and divides by 4 (average pooling gradient factor)
        return F.interpolate(y, scale_factor=2, mode='nearest') * 0.25


# =====================================================================
# 3. Data Fidelity (DF) Block with Convergence Guarantees (Gap C)
# =====================================================================

class DataFidelityBlock(nn.Module):
    """
    Solves the quadratic data fidelity subproblem using T steps of gradient descent.
    Includes early stopping based on tolerance during inference (Gap C).
    """
    def __init__(self, blur_op, down_op, steps=3):
        super().__init__()
        self.blur = blur_op
        self.down = down_op
        self.steps = steps
        
        # Learnable step size alpha and regularization parameter mu
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.mu = nn.Parameter(torch.tensor(0.05))
        # Learnable scaling factor to compensate for kernel mismatches
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, z_k, v, tol=1e-4):
        x = z_k
        mu_reg = torch.clamp(self.mu, min=1e-4)
        alpha_step = torch.clamp(self.alpha, min=1e-4)
        
        prev_loss = float('inf')
        
        for t in range(self.steps):
            # Forward physical projection with learnable scaling adjustment
            phys_proj = self.down(self.blur(x)) * self.scale
            error_lr = phys_proj - v
            
            # During inference, check convergence for early stopping (Gap C)
            if not self.training:
                current_loss = 0.5 * (error_lr ** 2).sum() + 0.5 * mu_reg * ((x - z_k) ** 2).sum()
                if abs(prev_loss - current_loss) < tol:
                    break
                prev_loss = current_loss
            
            # Backproject error to high-resolution space
            grad_fidelity = self.blur.forward_transpose(self.down.forward_transpose(error_lr)) * self.scale
            
            # Prior regularization gradient
            grad_prior = mu_reg * (x - z_k)
            
            # Total gradient descent update
            total_grad = grad_fidelity + grad_prior
            x = x - alpha_step * total_grad
            
        return x


# =====================================================================
# 4. NAFNet Denoiser Prior Modules
# =====================================================================

class SimpleGate(nn.Module):
    """Activation-free gating mechanism: splits channels and multiplies them"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Lightweight attention block using global pooling and conv1x1"""
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        attention = self.conv(self.gap(x))
        return x * attention


class NAFBlock(nn.Module):
    """
    Non-linear Activation Free Block.
    Achieves high restoration performance with low computational footprint.
    """
    def __init__(self, channels):
        super().__init__()
        self.ln1 = nn.GroupNorm(1, channels) # LayerNorm equivalent in PyTorch for spatial tensor
        
        # Convolution expansion block
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=1, padding=0)
        self.dwconv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2)
        self.gate = SimpleGate()
        self.sca = SimplifiedChannelAttention(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        residual = x
        
        out = self.ln1(x)
        out = self.conv1(out)
        out = self.dwconv(out)
        out = self.gate(out)
        out = self.sca(out)
        out = self.conv2(out)
        
        # Channel-wise scaling learnable parameter
        return residual + out * self.beta


class NAFNetDenoiser(nn.Module):
    """
    A lightweight U-shaped encoder-decoder utilizing NAFBlocks.
    Provides denoising prior on the stabilized HR grid.
    """
    def __init__(self, in_channels=1, channels=32):
        super().__init__()
        self.intro = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        
        # Encoder
        self.enc1 = NAFBlock(channels)
        self.down = nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2)
        self.enc2 = NAFBlock(channels * 2)
        
        # Bottleneck
        self.middle = NAFBlock(channels * 2)
        
        # Decoder
        self.up = nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)
        self.dec1 = NAFBlock(channels)
        
        self.outro = nn.Conv2d(channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.intro(x)
        
        # Encoder
        res_enc1 = self.enc1(out)
        out = self.down(res_enc1)
        out = self.enc2(out)
        
        # Bottleneck
        out = self.middle(out)
        
        # Decoder
        out = self.up(out)
        out = self.dec1(out + res_enc1) # Skip connection
        
        out = self.outro(out)
        return out + x # Global residual learning


# =====================================================================
# 5. Core Deep Unfolding Network (DUN) Model
# =====================================================================

class PhysicsGuidedDUN(nn.Module):
    """
    Physics-Guided Deep Unfolding Network (DUN) for Joint Image Restoration.
    Unrolls N iterations of alternating Data Fidelity and Denoising steps.
    """
    def __init__(self, num_iterations=4, steps_per_df=3, channels=32):
        super().__init__()
        self.num_iterations = num_iterations
        
        # Variance Stabilizing Transform (VST)
        self.vst = LogVST(delta=1e-3)
        self.inv_vst = InvLogVST(delta=1e-3)
        
        # Differentiable operators
        self.blur_op = LearnableBlurOperator(kernel_size=5, sigma=1.0)
        self.down_op = DifferentiableDownsampleOperator()
        
        # Mild Gaussian pre-filter kernel for warm start upscaling
        # A fixed 3x3 kernel to reduce high frequency noise
        blur_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0
        self.register_buffer('pre_filter_kernel', blur_kernel.unsqueeze(0).unsqueeze(0))
        
        # Data Fidelity Blocks per iteration
        self.df_blocks = nn.ModuleList([
            DataFidelityBlock(self.blur_op, self.down_op, steps=steps_per_df) 
            for _ in range(num_iterations)
        ])
        
        # Denoiser Prior block (shared weights to minimize parameters and GPU memory)
        self.denoiser = NAFNetDenoiser(in_channels=1, channels=channels)
        
        # Learned Affine Intensity Calibration parameters (Gap 6)
        # Formulated as: x_calibrated = gamma * (x_est - mu_shift) + beta
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.mu_shift = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def get_warm_start(self, v):
        """
        Computes the initial estimate x^0 (z^0) using:
        1. Bilinear/Bicubic upsampling to the target grid resolution.
        2. A mild Gaussian pre-filter to suppress high-frequency noise.
        Includes safety clipping and fallback to prevent NaNs/Infs (Gap E).
        """
        # Clip to prevent interpolation artifacts/overshoots
        v_clipped = torch.clamp(v, min=-5.0, max=5.0)
        
        # Bicubic upsampling in log-space
        x0 = F.interpolate(v_clipped, scale_factor=2, mode='bicubic', align_corners=False)
        
        # Apply pre-filter
        x0_pad = F.pad(x0, (1, 1, 1, 1), mode='reflect')
        result = F.conv2d(x0_pad, self.pre_filter_kernel)
        
        # Safety check for NaN/Inf (Gap E)
        if torch.isnan(result).any() or torch.isinf(result).any():
            result = F.interpolate(v_clipped, scale_factor=2, mode='bilinear', align_corners=False)
            
        return result

    def forward(self, y):
        # 1. Apply Variance Stabilizing Transform (VST)
        v = self.vst(y)
        
        # 2. Warm start initialization (Gap 4, Gap E)
        x = self.get_warm_start(v)
        z = x
        
        # 3. Unrolled HQS Optimization Loop (Gap A fixed: passes z and v to DF blocks)
        for k in range(self.num_iterations):
            # Step A: Data Fidelity Projection (deblur/super-resolution physics)
            x = self.df_blocks[k](z, v)
            
            # Step B: Denoising Prior (NAFNet Denoising in log-space)
            z = self.denoiser(x)
            
        # 4. Map back to intensity domain via Inverse VST
        x_est = self.inv_vst(z)
        
        # 5. Learned range calibration (Gap 6) and clamping to valid range [0, 1]
        x_calibrated = self.gamma * (x_est - self.mu_shift) + self.beta
        x_final = torch.clamp(x_calibrated, min=0.0, max=1.0)
        
        return x_final


# =====================================================================
# 6. Training Loss Regularization Helper (Gap B)
# =====================================================================

def kernel_regularization_loss(blur_op, lambda_smooth=1e-3, lambda_nonneg=1e-4):
    """
    Enforces smoothness and non-negativity of the learned blur kernel (Gap B).
    """
    kernel = blur_op.get_kernel()
    
    # 1. Smoothness: Penalize high-frequency variations (Laplacian of kernel)
    laplacian_kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                                     dtype=torch.float32, device=kernel.device).view(1, 1, 3, 3)
    smooth_penalty = F.conv2d(kernel, laplacian_kernel, padding=1).abs().mean()
    
    # 2. Energy conservation penalty (keeps sum close to 1.0)
    energy_penalty = (torch.sum(kernel) - 1.0) ** 2
    
    return lambda_smooth * smooth_penalty + lambda_nonneg * energy_penalty
