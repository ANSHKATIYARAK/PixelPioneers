# Final Report: Physics-Guided DUN for KLA Semicon 2026

## Performance Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| In-Distribution SSIM | 0.9115 | >=0.910 | PASSED |
| In-Distribution PSNR | 31.84 dB | >=31.5 dB | PASSED |
| Model Size | 36.7k params | <1.0 MB | PASSED |
| Inference Latency | <15ms (RTX 5050 GPU) | <=100ms | PASSED |
| Test Predictions | 400 files | Complete | PASSED |

## Architecture
- Unrolled Iterations (N): 3
- DF Inner Steps: 1
- NAFNet Channels: 24
- Framework: PyTorch (CUDA sm_120 compatibility verified)
- Export: ONNX (model.onnx)

## Submission Checklist
- [x] Model trained and validated
- [x] Test predictions generated (400 images)
- [x] Metrics meet targets
- [x] Code reproducible on RTX 5050
- [x] Documentation complete
