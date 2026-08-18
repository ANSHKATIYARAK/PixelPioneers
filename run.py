import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Safe dynamic import of PhysicsGuidedDUN from either root or models/
try:
    from models.model import PhysicsGuidedDUN
except ImportError:
    try:
        from model import PhysicsGuidedDUN
    except ImportError:
        PhysicsGuidedDUN = None

# Metrics imports with safe fallbacks
try:
    from skimage.metrics import structural_similarity as compute_ssim
    from skimage.metrics import peak_signal_noise_ratio as compute_psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = False  # Bypassed to prevent Hub network checks from hanging in firewalled environments
except ImportError:
    LPIPS_AVAILABLE = False

# Check if onnxruntime is available
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


# =====================================================================
# Helper Functions for Image Loading and Saving (Robust Formats)
# =====================================================================

def load_image_as_tensor(path):
    """
    Loads an image from path. Supports:
    1. Numpy arrays (.npy)
    2. Standard images (.png, .jpg, .jpeg, .tif, .tiff)
    Returns: torch.Tensor of shape (1, 1, H, W) normalized to appropriate float ranges.
    """
    ext = os.path.splitext(path)[1].lower()
    
    if ext == '.npy':
        arr = np.load(path)
        # Ensure there are no NaN or Inf values in the degraded input array
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Squeeze extra dimensions to handle shapes like (H, W), (1, H, W), or (C, H, W)
        if arr.ndim == 3:
            if arr.shape[0] in [1, 3]:  # C, H, W
                arr = arr[0]  # Take first channel
            elif arr.shape[2] in [1, 3]:  # H, W, C
                arr = arr[:, :, 0]
        elif arr.ndim == 4:
            if arr.shape[1] in [1, 3]:
                arr = arr[0, 0]
        tensor = torch.from_numpy(arr).float()
    else:
        from PIL import Image
        img = Image.open(path).convert('L')  # Ensure grayscale (1-channel)
        arr = np.array(img, dtype=np.float32) / 255.0  # Normalize PNG/TIF to [0, 1]
        tensor = torch.from_numpy(arr)
        
    while tensor.ndim < 4:
        tensor = tensor.unsqueeze(0)
        
    if tensor.ndim > 4:
        tensor = tensor.view(1, 1, tensor.shape[-2], tensor.shape[-1])
        
    return tensor


def save_tensor_as_image(tensor, path):
    """
    Saves restored tensor to path. Supports:
    1. Numpy arrays (.npy)
    2. Standard images (.png, .jpg, .jpeg, .tif, .tiff)
    """
    ext = os.path.splitext(path)[1].lower()
    if isinstance(tensor, torch.Tensor):
        arr = tensor.squeeze().cpu().numpy()
    else:
        # NumPy array (from ONNX)
        arr = np.squeeze(tensor)
        
    # Ensure there are no NaN or Inf values in the restored output array
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Strictly clamp outputs to [0.0, 1.0] range as required
    arr = np.clip(arr, 0.0, 1.0)
    
    if ext == '.npy':
        # Grayscale array with shape (H, W)
        np.save(path, arr)
    else:
        from PIL import Image
        # Scale range [0, 1] back to standard 0-255 uint8 range
        arr_uint8 = (arr * 255.0).astype(np.uint8)
        img = Image.fromarray(arr_uint8, mode='L')
        img.save(path)


# =====================================================================
# Metrics Computation Logic
# =====================================================================

def compute_quality_metrics(pred_tensor, target_tensor, lpips_model=None):
    """
    Computes SSIM, PSNR, and LPIPS between two tensors (range [0, 1]).
    Correctly formats grayscale images for LPIPS by repeating channels and scaling to [-1, 1].
    """
    if isinstance(pred_tensor, torch.Tensor):
        pred = pred_tensor.squeeze().cpu().numpy()
    else:
        pred = np.squeeze(pred_tensor)
        
    if isinstance(target_tensor, torch.Tensor):
        target = target_tensor.squeeze().cpu().numpy()
    else:
        target = np.squeeze(target_tensor)
        
    results = {}
    
    if SKIMAGE_AVAILABLE:
        results['ssim'] = compute_ssim(target, pred, data_range=1.0)
        results['psnr'] = compute_psnr(target, pred, data_range=1.0)
    else:
        results['ssim'] = 0.0
        results['psnr'] = 0.0
        
    # LPIPS evaluation logic (VGG backend expects 3-channel input in range [-1, 1])
    if LPIPS_AVAILABLE and lpips_model is not None:
        p_tensor = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0) if isinstance(pred, np.ndarray) else pred_tensor.squeeze().unsqueeze(0).unsqueeze(0).cpu()
        t_tensor = torch.from_numpy(target).unsqueeze(0).unsqueeze(0) if isinstance(target, np.ndarray) else target_tensor.squeeze().unsqueeze(0).unsqueeze(0).cpu()
        
        # Duplicate 1-channel to 3-channel and scale [0, 1] to [-1, 1]
        p_3ch = p_tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0
        t_3ch = t_tensor.repeat(1, 3, 1, 1) * 2.0 - 1.0
        
        with torch.no_grad():
            lpips_val = lpips_model(p_3ch, t_3ch).item()
        results['lpips'] = lpips_val
    else:
        results['lpips'] = -1.0  # -1 flags LPIPS was skipped
        
    return results


# =====================================================================
# Main Inference Execution
# =====================================================================

def run_evaluation(model_path, input_dir, output_dir, gt_dir=None, num_iterations=3, steps_per_df=1, channels=24):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    is_onnx = model_path.endswith('.onnx')
    
    # 1. Load the model session / weights
    print(f"Loading model checkpoint from: {model_path}")
    if is_onnx:
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime library not found. Cannot load ONNX model.")
        # Configure providers to run ONNX on GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        try:
            ort_session = ort.InferenceSession(model_path, providers=providers)
        except Exception:
            # Fallback to CPU execution provider only
            ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = ort_session.get_inputs()[0].name
    else:
        if PhysicsGuidedDUN is None:
            raise ImportError("Could not import PhysicsGuidedDUN from model.py. Ensure model.py is in the root or models/ directory.")
        model = torch.load(model_path, map_location=device)
        if isinstance(model, dict):
            state_dict = model
            model = PhysicsGuidedDUN(num_iterations=num_iterations, steps_per_df=steps_per_df, channels=channels)
            model.load_state_dict(state_dict)
        model = model.to(device).eval()
        for param in model.parameters():
            param.requires_grad = False
            
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize LPIPS if target metrics are requested and packages are installed
    lpips_fn = None
    if gt_dir and LPIPS_AVAILABLE:
        print("Initializing LPIPS model (VGG network)...")
        try:
            lpips_fn = lpips.LPIPS(net='vgg', verbose=False).eval()
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize LPIPS: {e}. Only SSIM and PSNR will be computed.")
            
    # List files
    valid_exts = ('.npy', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)])
    
    if len(files) == 0:
        print(f"No valid image files found in {input_dir}")
        return
        
    print(f"Processing {len(files)} degraded images...")
    metrics_log = []
    
    for i, file_name in enumerate(files):
        in_path = os.path.join(input_dir, file_name)
        out_path = os.path.join(output_dir, file_name)
        
        # Load degraded input
        img_tensor = load_image_as_tensor(in_path)
        
        # Inference pass
        if is_onnx:
            # Convert PyTorch tensor to NumPy float32 array for ONNX Runtime input
            inp_numpy = img_tensor.numpy().astype(np.float32)
            restored_out = ort_session.run(None, {input_name: inp_numpy})[0]
        else:
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                restored_out = model(img_tensor)
            
        # Save output image
        save_tensor_as_image(restored_out, out_path)
        
        # Metric logging if ground truth is present
        if gt_dir:
            gt_path = os.path.join(gt_dir, file_name)
            if os.path.exists(gt_path):
                gt_tensor = load_image_as_tensor(gt_path)
                metrics = compute_quality_metrics(restored_out, gt_tensor, lpips_fn)
                metrics_log.append(metrics)
                
                # Print progress update
                ssim_str = f"SSIM: {metrics['ssim']:.4f}" if 'ssim' in metrics else ""
                psnr_str = f"PSNR: {metrics['psnr']:.2f} dB" if 'psnr' in metrics else ""
                lpips_str = f"LPIPS: {metrics['lpips']:.4f}" if metrics.get('lpips', -1) >= 0 else ""
                print(f"[{i+1}/{len(files)}] {file_name} | {ssim_str} | {psnr_str} | {lpips_str}".strip(' |'))
            else:
                print(f"[{i+1}/{len(files)}] {file_name} processed (GT file not found at {gt_path})")
        else:
            print(f"[{i+1}/{len(files)}] {file_name} processed")
            
    # Output average results
    if metrics_log and SKIMAGE_AVAILABLE:
        avg_ssim = np.mean([m['ssim'] for m in metrics_log])
        avg_psnr = np.mean([m['psnr'] for m in metrics_log])
        print(f"\n========================================")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        
        valid_lpips = [m['lpips'] for m in metrics_log if m['lpips'] >= 0]
        if valid_lpips:
            avg_lpips = np.mean(valid_lpips)
            print(f"Average LPIPS: {avg_lpips:.4f}")
        print(f"========================================")
    elif gt_dir and not SKIMAGE_AVAILABLE:
        print("\n⚠️ metrics computation skipped: 'scikit-image' library not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KLA Semicon 2026 Image Restoration - run.py")
    parser.add_argument('--model', default=None, help='Path to the trained model checkpoint (.onnx, .pt, or .pth)')
    parser.add_argument('--model_path', default=None, help='Alternative flag for path to the trained model checkpoint')
    parser.add_argument('--input_dir', default=None, help='Directory containing the degraded low-resolution images')
    parser.add_argument('--output_dir', default=None, help='Directory where restored output images will be saved')
    parser.add_argument('--gt_dir', default=None, help='Optional directory containing ground truth images for metric scoring')
    
    # Model architecture configuration parameters (RTX 5050 GPU optimized defaults for PyTorch path)
    parser.add_argument('--channels', type=int, default=24, help='Model channels')
    parser.add_argument('--num_iterations', type=int, default=3, help='Unrolled iterations')
    parser.add_argument('--steps_per_df', type=int, default=1, help='Data Fidelity inner steps')
    
    # Add support for positional arguments for maximum compatibility
    parser.add_argument('positional_args', nargs='*', help='Positional arguments: [input_dir] [output_dir]')
    
    args = parser.parse_args()
    
    # 1. Resolve input_dir and output_dir
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if args.positional_args:
        if len(args.positional_args) >= 1 and not input_dir:
            input_dir = args.positional_args[0]
        if len(args.positional_args) >= 2 and not output_dir:
            output_dir = args.positional_args[1]
            
    if not input_dir or not output_dir:
        parser.error("Both input_dir and output_dir are required (either as flags or as positional arguments).")
        
    # 2. Resolve model_path (with default fallback locations, preferring ONNX weights if ONNX runtime is installed)
    model_path = args.model or args.model_path
    if not model_path:
        possible_paths = []
        if ONNX_AVAILABLE:
            possible_paths.extend([
                "models/model.onnx",
                "model.onnx",
                "./models/model.onnx"
            ])
        possible_paths.extend([
            "models/best_model.pt",
            "best_model.pt",
            "./models/best_model.pt"
        ])
        
        for p in possible_paths:
            if os.path.exists(p):
                model_path = p
                print(f"No model path specified. Automatically resolved to default checkpoint: {model_path}")
                break
                
    if not model_path:
        # Fallback search inside models/
        if os.path.exists("models"):
            extensions = (".onnx", ".pt", ".pth") if ONNX_AVAILABLE else (".pt", ".pth")
            files = [f for f in os.listdir("models") if f.endswith(extensions)]
            if files:
                model_path = os.path.join("models", files[0])
                print(f"No model path specified. Automatically resolved to first model in models/: {model_path}")
                
    if not model_path:
        parser.error("Model checkpoint not found in default locations. Please specify --model or --model_path.")
        
    run_evaluation(
        model_path=model_path,
        input_dir=input_dir,
        output_dir=output_dir,
        gt_dir=args.gt_dir,
        num_iterations=args.num_iterations,
        steps_per_df=args.steps_per_df,
        channels=args.channels
    )
