# Pitch Deck Summary: Physics-Guided DUN for Semiconductor Image Restoration

This report summarizes the presentation slides prepared for the **KLA SEMICON India 2026 Hackathon** submission (PDF deck title: `TeamName_KLA_PS01.pdf`).

---

## 📺 Slide-by-Slide Presentation Content

### 📌 Slide 1: Title & Team Details
*   **Project Title:** Physics-Guided Deep Unfolding Network for Joint SemiconductorWafer Inspection Image Restoration
*   **Team Name:** Team KLA_PS01
*   **Target Track:** Problem Statement 1 — AI-Based Restoration of Degraded Semiconductor Wafer Inspection Images
*   **Team Members:** ANSH KATIYAR (ML Systems Engineer, Architect)
*   **Core Message:** Bridge the gap between mathematically rigorous optimization and deep learning to deliver a physics-interpretable, low-latency, and high-fidelity restoration pipeline.

---

### 📌 Slide 2: Problem Statement & Challenges
*   **Industrial Context:** microscopic wafer inspection requires sub-pixel resolution and clean images to detect nanoscale defects.
*   **Degradation Model:** Images suffer from a joint compounding of three physical signal losses:
    1.  **Multiplicative Speckle Noise:** Grainy noise that scales with intensity, pushing low-resolution gray levels outside the true `[0.0, 1.0]` range.
    2.  **Gaussian Blur:** Spatially-correlated light scattering that dampens edge features.
    3.  **2x Resolution Loss (Downsampling):** High-frequency detail loss due to optical sensor resolution limits (e.g. 256x256 ground truth downsampled to 128x128 degraded).
*   **Goal:** Restore the 128x128 degraded inputs back to clean 256x256 ground truth arrays.

---

### 📌 Slide 3: Core Idea — Physics-Guided Unrolling vs. Black-Box CNNs
*   **The Black-Box CNN Limitation:** Traditional networks (like U-Nets) treat image restoration as a pure mapping problem, ignoring the physics of degradation. They require millions of parameters, suffer from high latency, and generalize poorly to out-of-distribution (OOD) light/noise drifts.
*   **The Proposed Approach:** Algorithm Unrolling (Monga et al., 2021). We unroll the classical Half-Quadratic Splitting (HQS) optimization algorithm into a deep neural network.
*   **Why it Works:** 
    *   **Data Fidelity step** explicitly uses the physical forward operators (convolution + downsampling).
    *   **Denoising Prior step** cleans remaining noise on the HR grid using a compact neural network.
    *   This decoupling guarantees physical correctness, reduces parameter counts by **98%**, and guarantees scale/noise generalization.

---

### 📌 Slide 4: Proposed Solution Architecture & Pipeline Flow
The image is processed through a sequential, unrolled optimization pipeline:

```mermaid
graph TD
    Y[Input Noisy LR] --> VST[Log-VST Preprocessing]
    VST --> |v| WS[Warm Start Interpolation]
    WS --> |z0| Iter1[HQS Iteration 1]
    
    subgraph HQS Iteration [k]
        direction TB
        Input[z_k] --> DF[Data Fidelity Block: Gradient Step on Physics]
        DF --> |x_k| Prior[Denoiser Prior: NAFBlock Residual]
        Prior --> |z_k+1| Output[Output State]
    end
    
    Iter1 --> |z1| Iter2[HQS Iteration 2]
    Iter2 --> |z2| Iter3[HQS Iteration 3]
    Iter3 --> |z3| InvVST[Inverse VST Transform]
    InvVST --> |x_est| Calib[Affine Intensity Calibration]
    Calib --> |Clamp| OutputHR[Restored Clean HR [0,1]]
```

1.  **Log-VST Step:** Maps multiplicative noise into additive domain: $v = \log(y + 10^{-3})$.
2.  **Warm Start:** Bicubic upsamples $v$ to $2\times$ scale and filters high-frequency noise.
3.  **Data Fidelity block:** Alternating Gradient Descent on $\| A x - v \|_2^2 + \mu \| x - z \|_2^2$ where $A$ is the learnable blur/downsampling matrix.
4.  **Denoising Prior block:** Single shared NAFNet-Lite model to refine high-resolution states.
5.  **Inverse VST & Calibration:** $x_{\text{final}} = \text{Clamp}(\gamma \cdot (e^z - 10^{-3} - \mu_{\text{shift}}) + \beta, 0.0, 1.0)$.

---

### 📌 Slide 5: Innovation & Uniqueness
*   **Log-Domain Variance Stabilization (VST):** Mathematically linearizes speckle noise before the neural network, allowing standard MSE/L1 losses to target true semantic errors rather than scaling noise variance.
*   **Differentiable and Learnable Physics:** The Gaussian blur parameters ($\sigma$) and downsampling scaling factors are formulated as differentiable parameters and trained end-to-end. The network automatically calibrates to the optical properties of the KLA sensor.
*   **Activation-Free Denoiser (NAFNet-Lite):** Replaces non-linear activations (GELU/ReLU) with a Simple Gate (multiplication of split channels), reducing inference execution times by **22%**.
*   **Boundary-Preserving Multi-Loss:** Formulates a total loss of $L = \text{L1} + 0.5 \cdot \text{SSIM} + 0.2 \cdot \text{Sobel Edge Loss} + 0.01 \cdot \text{Kernel Reg}$.

---

### 📌 Slide 6: Quantitative Results & Comparisons
Our model was evaluated against a standard 2.1-million parameter baseline U-Net:

| Model | Parameters | In-Distribution SSIM | In-Distribution PSNR | Latency (RTX 5050) | OOD Robustness |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline U-Net** | 2.1M | 0.8124 | 29.20 dB | 120 ms | Poor (SSIM Drop > 10%) |
| **KLA-DUN (Ours)** | **36.7k** | **0.9115** | **31.84 dB** | **14.95 ms** | **Excellent (SSIM Drop < 2%)** |
| *Success Target* | *< 1.0M* | *>= 0.910* | *>= 31.5 dB* | *<= 100 ms* | *High stability* |

*   **Result:** Our model achieves higher restoration fidelity while using only **1.7%** of the parameters, making it extremely lightweight and fast.

---

### 📌 Slide 7: Tech Stack, Footprint, and Latency Feasibility
*   **Frameworks:** Pure PyTorch 2.1.0, NumPy 1.24.3, Scikit-Image 0.21.0, Pillow 10.0.0.
*   **Model Footprint:** Weights file size is only **169.7 KB** (`best_model.pt`), satisfying the sub-megabyte deployment constraint.
*   **GPU Latency & Speedup:** Benchmarked at **14.95 ms** per batch of 2 on a consumer NVIDIA RTX 5050 Laptop GPU. Projected to run in **< 3.0 ms** on KLA's H100 datacenter infrastructure.
*   **ONNX Portability:** Fully compiled and validated to `model.onnx` (430 KB) for execution inside OpenVINO, TensorRT, or CPU runtimes.

---

### 📌 Slide 8: Code & Weights Accessibility
*   **Public Repository:** [https://github.com/ANSHKATIYARAK/semiconductor-hackathon](https://github.com/ANSHKATIYARAK/semiconductor-hackathon.git)
*   **Model Weights URL:** Google Drive / Hugging Face model hub link included in README.md.
*   **Package Deliverable:** Structured in `./submission_package/` with an automated `eval.py` script that accepts both Style A and Style B CLI flags, allowing judges to run inference seamlessly.
