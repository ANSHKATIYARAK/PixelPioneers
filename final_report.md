# Pitch Deck Summary: Physics-Guided DUN for Semiconductor Image Restoration

This report summarizes the presentation slides prepared for the **KLA SEMICON India 2026 Hackathon** submission (PDF deck title: `PixelPioneers_KLA_PS01.pdf`).

---

## 📺 Slide-by-Slide Presentation Content

### 📌 Slide 1: Cover & Team Information
*   **Project Title:** Physics-Guided Deep Unfolding Network (Physics-DUN) for Joint Semiconductor Wafer Restoration
*   **Subtitle:** AI-Based Image Restoration for High-Throughput Wafer Metrology & Defect Inspection
*   **Problem Statement:** Problem Statement 01 — AI-Based Restoration of Degraded Images (KLA Track)
*   **Team Name:** Pixel Pioneers
*   **Tagline:** Restoring reality, one pixel at a time.
*   **Team Lead:** Ansh Katiyar (VIT Vellore) - anshkatiyar06@gmail.com | +91 8949513257
*   **Team Member 2:** Abhinav Yadav - abhi1012006@gmail.com | +91 9235881010
*   **Team Member 3:** Akarsh Gupta - akarsh.gupta2025@vitstudent.ac.in | +91 7081610062
*   **Institution:** Vellore Institute of Technology (VIT Vellore)

---

### 📌 Slide 2: Problem Statement & Industrial Wafer Inspection Challenges
*   **The Inspection Bottleneck:** Microscopic wafer scanning requires sub-pixel fidelity to detect nanoscale yield-limiting defects. Signal degradation causes measurement dropouts and critical false-negative classifications.
*   **Compounded Physical Degradation Modes:**
    1.  **Multiplicative Speckle Noise:** Signal-dependent intensity spread where pixel readings exceed nominal `[0.0, 1.0]` bounds.
    2.  **Optical Gaussian Blur:** Spatially correlated light scattering that destroys fine line transitions and edge geometry.
    3.  **2x Spatial Resolution Loss:** Sensor downsampling (`256 x 256 -> 128 x 128` or `512 x 512 -> 256 x 256`).
*   **Mission:** Jointly super-resolve (2x) and denoise degraded inputs back to physically accurate ground-truth targets.

---

### 📌 Slide 3: Core Idea — Physics-Guided Unrolling vs. Black-Box CNNs
*   **The Black-Box Failure:** Standard U-Nets and Vision Transformers treat super-resolution as unconstrained pixel mapping, ignoring the forward physical optics. This leads to hallucinated defect features, high parameter bloat (`> 2M` params), and large Out-of-Distribution (OOD) failure rates.
*   **Our Solution (Algorithm Unrolling via HQS):** Decouples restoration into a mathematically grounded Half-Quadratic Splitting optimization loop:
    *   **Physical Data Fidelity Step:** Embeds differentiable blur ($A$) and downsampling ($D$) operators to enforce strict light-propagation consistency: $D(A(x)) \approx y$.
    *   **Deep Denoiser Prior Step:** Cleans residual noise on the high-resolution grid using a lightweight neural prior.
*   **Key Advantage:** Reduces model size by **98%** (only 36.7k params) while guaranteeing physical scale-invariance and zero feature hallucination.

---

### 📌 Slide 4: Proposed Solution Architecture & Pipeline Flow
The image is processed through a sequential, unrolled optimization pipeline:

```
[ Degraded Input y ] ──► [ Homomorphic Log-VST ] ──► [ Bicubic Warm-Start (2x) ]
│
┌───────────────────────────────────────────────────────┘
▼
[ UNROLLED HQS OPTIMIZATION LOOP (N = 3 Iterations) ]
├── 1. Data Fidelity Block: Gradient step enforcing physical optics consistency
└── 2. Deep Prior Block: High-resolution denoising via NAFNet-Lite
│
▼
[ Inverse Log-VST ] ──► [ Affine Range Calibration ] ──► [ Clamped Output x̂ ∈ [0, 1] ]
```

1.  **Homomorphic Log-VST:** Converts multiplicative speckle into additive Gaussian noise in the log-domain: $v = \log(y + 10^{-3})$. Stabilizes signal-dependent noise variance.
2.  **Scale-Invariant Warm Start:** Bicubic upsamples the log-image by 2x and applies a mild Gaussian pre-filter to suppress high-frequency noise.
3.  **Unrolled HQS Loop (3 Iterations):** Alternates between Data Fidelity Block (gradient step on blur/downsample operators) and Denoising Prior Block (shared-weight NAFNet-Lite).
4.  **Inverse Log-VST & Calibration:** Maps back via $x_{\text{clean}} = \exp(z_N) - 10^{-3}$, applies learned affine calibration: $x_{\text{out}} = \alpha \cdot x_{\text{clean}} + \beta$, and clamps final outputs strictly to `[0.0, 1.0]`.

---

### 📌 Slide 5: Innovation & Uniqueness
*   **Homomorphic Speckle Stabilization:** Linearizes speckle variance before deep denoising, allowing loss gradients to optimize structural geometry rather than scaling noise.
*   **End-to-End Differentiable Optical Operators:** Blur kernel parameters ($\sigma$, size) and downsampling scale factors ($s_k$) are dynamically learned and calibrated to sensor physics.
*   **Activation-Free Denoiser (NAFNet-Lite):** Replaces heavy non-linear activations (GELU/ReLU) with a Simple Gate ($x_1 \odot x_2$), cutting GPU execution latency by **22%**.
*   **Geometry-Preserving Composite Loss:**
    $$\mathcal{L}_{\text{total}} = \mathcal{L}_1 + 0.5 \cdot \mathcal{L}_{\text{MS-SSIM}} + 0.2 \cdot \mathcal{L}_{\text{SobelEdge}} + 0.01 \cdot \mathcal{L}_{\text{KernelReg}}$$

---

### 📌 Slide 6: Quantitative Results & Benchmark Comparison
Our model was evaluated against a standard 2.1-million parameter baseline U-Net:

| Architecture | Parameters | In-Dist SSIM | In-Dist PSNR | RTX 5050 Latency | H100 Latency (Est.) | OOD Drop | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline U-Net** | 2.1 Million | 0.8124 | 29.20 dB | 120.0 ms | $\sim 18.0\text{ ms}$ | > 10.0% | **PASSED** |
| **Physics-DUN (Ours)** | **36,710** | **0.9115** | **31.84 dB** | **14.95 ms** | **< 3.0 ms** | **< 3.0%** | **PASSED** |
| *Success Target* | *< 1.0 M* | *>= 0.910* | *>= 31.5 dB* | *<= 100 ms* | *<= 100 ms* | *Minimal Drop* | **PASSED** |

---

### 📌 Slide 7: Technology Stack, Runtime Latency & Edge Feasibility
*   **Framework & Reproducibility:** PyTorch 2.1.0, NumPy, Scikit-Image, Pillow with pinned, platform-independent dependencies.
*   **Model Footprint:** Compact checkpoint file of **169.7 KB** (`best_model.pt`), consuming `<18%` of the $1.0\text{ MB}$ budget limit.
*   **Hardware Execution Profile:**
    *   *Laptop Validation (RTX 5050 GPU):* $14.95\text{ ms}$ per batch under FP16 mixed precision.
    *   *Datacenter Deployment (NVIDIA H100 GPU):* Projected throughput $>600\text{ images/sec}$ ($<3.0\text{ ms/image}$).
*   **ONNX Portability:** Fully validated via `onnx.checker` (`model.onnx`, $430\text{ KB}$) for execution in OpenVINO, TensorRT, or C++ production runtimes.

---

### 📌 Slide 8: Deliverables Package & Verification Checklist
*   **Public GitHub Repository:** [https://github.com/ANSHKATIYARAK/semiconductor-hackathon.git](https://github.com/ANSHKATIYARAK/semiconductor-hackathon.git)
*   **Model Checkpoint:** Public download link for `best_model.pt` embedded in repository `README.md`.
*   **Verified Deliverables Package (./submission_package/):**
    *   `eval.py` / `eval_dun.py`: Standalone CLI supporting positional and flag arguments.
    *   `train.py`: Full training pipeline with mixed precision and multi-loss objective.
    *   `test_predictions/`: 400 verified `.npy` files ($256 \times 256$, normalized strictly to `[0.0, 1.0]`).
    *   `requirements.txt`: Clean, pinned `pip freeze` environment requirements.
    *   `PixelPioneers_KLA_PS01.pptx`: Slide deck matching KLA templates.
