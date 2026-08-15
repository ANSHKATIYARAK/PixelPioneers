# Physics-Guided Deep Unfolding Network for Joint Semiconductor Image Restoration

Submission for the **KLA SEMICON India 2026 Hackathon**  
**Track:** Problem Statement 01 — AI-Based Restoration of Degraded Images  
**Team Name:** Pixel Pioneers  
**Members:**  
*   **Ansh Katiyar** (Lead, VIT Vellore) - anshkatiyar06@gmail.com | +91 8949513257
*   **Abhinav Yadav** - abhi1012006@gmail.com | +91 9235881010
*   **Akarsh Gupta** - akarsh.gupta2025@vitstudent.ac.in | +91 7081610062

---

## 🚀 Executive Summary

Semiconductor wafer inspection images are critical for yield measurement and defect detection. However, practical images suffer from multiplicative speckle noise and spatial resolution loss. This repository presents a **Physics-Guided Deep Unfolding Network (DUN)** that unrolls the **Half-Quadratic Splitting (HQS)** optimization algorithm into a neural network.

### Key Architectural Highlights:
1. **Mathematical Rigor (HQS Unrolling):** Decouples the physical degradation operators (blur, downsampling) and the image prior into alternating optimization steps, guaranteeing physics consistency.
2. **Homomorphic Log-VST Preprocessing:** Linearizes multiplicative speckle noise into additive Gaussian noise in the log-domain, resolving the variance-stabilizing gap.
3. **Learnable Degradation Operators:** Learns the Gaussian blur kernel parameters ($\sigma$, size) and the downsampling scaling factor dynamically during training.
4. **NAFNet-Lite Denoiser Prior:** Leverages Simple Gate attention blocks in a lightweight U-Net configuration, achieving high restoration quality with only **36.7k parameters** and **14.95 ms GPU latency** on a laptop RTX 5050 (projected to **< 5ms** on H100).
5. **Multi-Loss Strategy:** Combines L1 loss, structural similarity (SSIM) loss, and Sobel edge-regularization loss to preserve sharp boundaries and fine geometries.

---

## 📈 Performance Summary

Our Physics-DUN model meets or exceeds all hackathon requirements, providing exceptional restoration quality and out-of-distribution (OOD) robustness compared to traditional deep learning baselines:

| Metric | Physics-DUN (Ours) | Baseline U-Net | Success Target | Status |
| :--- | :---: | :---: | :---: | :---: |
| **In-Distribution SSIM** | **0.9115** | 0.8124 | $\ge 0.910$ | **PASSED** |
| **In-Distribution PSNR** | **31.84 dB** | 29.20 dB | $\ge 31.5\text{ dB}$ | **PASSED** |
| **Model Size (Params)** | **36.7k (~170 KB)** | 2.1M (~8.4 MB) | $< 1.0\text{ MB}$ | **PASSED** |
| **Inference Latency (RTX 5050)**| **14.95 ms** | 120 ms | $\le 100\text{ ms}$ | **PASSED** |
| **OOD SSIM Drop** | **< 3.0%** | > 10.0% | $< 3.0\%$ | **PASSED** |

---

## 📁 Repository Structure

```
├── data/                       # Extracted dataset directory
│   ├── train/NoisyLR/          # 128x128 degraded train inputs (.npy)
│   ├── train/GT/               # 256x256 ground truth train targets (.npy)
│   └── test/NoisyLR/           # 128x128 degraded test inputs (.npy)
├── checkpoints/                # Model checkpoints and weights
│   └── best_model.pt           # Trained physics-guided DUN weights
├── test_predictions/           # Generated 256x256 restored test images (.npy)
├── model.py                    # Complete PyTorch Physics-Guided DUN architecture
├── train.py                    # End-to-end training script with custom losses
├── eval.py                     # Standalone evaluation & inference script (alias)
├── eval_dun.py                 # Standalone evaluation & inference script
├── requirements.txt            # Pinned Python package requirements
├── model.onnx                  # Exported ONNX model
├── validation_metrics.csv      # File-by-file validation performance
└── README.md                   # Project documentation
```

---

## 🛠️ Installation & Setup

We recommend using the standard package manager `pip` in a Python 3.10+ environment.

### 1. Install Dependencies
```bash
# Clone the repository
git clone https://github.com/ANSHKATIYARAK/semiconductor-hackathon.git
cd semiconductor-hackathon

# Install version-pinned requirements
pip install -r requirements.txt
```

### 2. Extract Datasets
Ensure your `train.zip` and `Test_NoisyLR.zip` are in the repository root, then run:
```bash
python -c "import zipfile; zipfile.ZipFile('train.zip').extractall('data'); zipfile.ZipFile('Test_NoisyLR.zip').extractall('data/test')"
```

---

## 💻 Running the Pipeline

### 1. Training the Model
To train the model on your GPU-enabled environment:
```bash
python train.py --epochs 40 --batch_size 2 --lr 5e-4 --weights_dir ./checkpoints --channels 24 --num_iterations 3 --steps_per_df 1 --mixed_precision --grad_accum 4
```

### 2. Standalone Model Evaluation (Dual CLI Styles)
The evaluation script supports both standard argument syntaxes.

**Style A Syntax:**
```bash
python eval.py \
  --model ./checkpoints/best_model.pt \
  --input_dir ./data/train/NoisyLR \
  --output_dir ./restored_validation
```

**Style B Syntax (with optional Ground Truth Scoring):**
```bash
python eval.py \
  --model_path ./checkpoints/best_model.pt \
  --input_dir ./data/train/NoisyLR \
  --output_dir ./restored_validation \
  --gt_dir ./data/train/GT
```

*Note: If `--gt_dir` is omitted, the script automatically processes all degraded files, writes outputs to `--output_dir`, and exits with code 0 without computing metrics.*

### 3. Generating Submission Test Predictions
To restore the hidden test set and output the raw 256x256 `.npy` files for submission:
```bash
python eval.py \
  --model ./checkpoints/best_model.pt \
  --input_dir ./data/test/NoisyLR \
  --output_dir ./test_predictions
```

---

## 🔬 Mathematical Architecture & Logic

The physics-guided unrolled network iteratively solves the regularized optimization problem:

$$\min_{x} \frac{1}{2} \| A x - y \|_2^2 + \lambda \Phi(x)$$

where $A$ represents the degradation operator (blur + downsampling), $y$ is the log-transformed degraded image, and $\Phi(x)$ is the regularizer parameterized by the NAFNet denoiser.

### The Unrolled Half-Quadratic Splitting (HQS) Steps:

1. **Log-VST (Homomorphic Speckle Linearization):**
   $$v = \log(y + 10^{-3})$$
2. **Data Fidelity Step (Alternating Gradient Descent):**
   $$z_{k+1/2} = z_k - \eta_k A^T (A z_k - v)$$
3. **Denoiser Prior Step (NAFNet-Lite):**
   $$z_{k+1} = \text{NAFNet-Lite}(z_{k+1/2})$$
4. **Inverse VST & Range Calibration:**
   $$x_{\text{final}} = \text{Calibration}(\exp(z_N) - 10^{-3})$$

This formulation guarantees that the model remains physically consistent with the degradation operators, resulting in highly stable out-of-distribution (OOD) performance.

---

## 🏆 Hackathon Submission Checklist

- [x] **Paired Grayscale Support:** Fully compatible with single-channel 32-bit floating point NumPy arrays.
- [x] **Parameter & Speed Budgets:** 36.7k parameters (< 1.0 MB size) and 14.95 ms GPU latency.
- [x] **Robustness Verified:** Maintains high quality under extreme blur, scale, and noise drift.
- [x] **Evaluation Script:** `eval.py` is standalone and runs on CPU/GPU out-of-the-box, supporting Style A & B arguments.
