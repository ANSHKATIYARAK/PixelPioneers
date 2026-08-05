# Physics-Guided Deep Unfolding Network - KLA Semicon 2026

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py --epochs 40 --batch_size 2 --channels 24 --num_iterations 3 --steps_per_df 1 --mixed_precision --grad_accum 4
```

### Evaluation
```bash
python eval_dun.py --model best_model.pt --input_dir ./data/train/NoisyLR --output_dir ./restored --gt_dir ./data/train/GT --channels 24 --num_iterations 3 --steps_per_df 1
```

### Test Prediction
```bash
python eval_dun.py --model best_model.pt --input_dir ./data/test/NoisyLR --output_dir ./test_predictions --channels 24 --num_iterations 3 --steps_per_df 1
```

## Results
- SSIM: 0.9115+
- PSNR: 31.5+ dB
- Latency: ~14.9 ms (RTX 5050 GPU)
- Size: 36.7k parameters
