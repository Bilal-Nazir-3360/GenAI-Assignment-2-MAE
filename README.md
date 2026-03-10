# GenAI-Assignment-2-MAE
# Self-Supervised Image Representation Learning using Masked Autoencoders (MAE)

## Assignment No. 2 | Generative AI
### National University of Computer and Emerging Sciences

## Objective
Design and implement a self-supervised Masked Autoencoder (MAE) system that learns 
meaningful visual representations by reconstructing images with 75% of input patches masked.

## Model Architecture

### Encoder - ViT Base (B/16)
| Property           | Value       |
|--------------------|-------------|
| Patch Size         | 16 x 16     |
| Image Size         | 224 x 224   |
| Hidden Dimension   | 768         |
| Transformer Layers | 12          |
| Attention Heads    | 12          |
| Parameters         | ~86 Million |

### Decoder - ViT Small (S/16)
| Property           | Value       |
|--------------------|-------------|
| Patch Size         | 16 x 16     |
| Hidden Dimension   | 384         |
| Transformer Layers | 12          |
| Attention Heads    | 6           |
| Parameters         | ~22 Million |


## Results

| Metric          | Value    |
|-----------------|----------|
| Best Train Loss | 0.2125   |
| Best Val Loss   | 0.2121   |
| PSNR            | 21.82 dB |
| SSIM            | 0.6948   |
| Training Epochs | 20       |

## Dataset
- TinyImageNet
- 100,000 training images
- 10,000 validation images
- 200 classes

## Training Setup
| Property          | Value     |
|-------------------|-----------|
| Platform          | Kaggle    |
| Accelerator       | GPU T4 x2 |
| Batch Size        | 32        |
| Optimizer         | AdamW     |
| Scheduler         | Cosine LR |
| Mixed Precision   | Yes       |
| Gradient Clipping | Yes       |
| Mask Ratio        | 75%       |


## Implementation Details

### Part 1 - Patchification and Masking
- Split 224x224 image into 16x16 patches (196 total)
- Randomly retain 25% visible patches (49 patches)
- Mask remaining 75% patches (147 patches)
- Maintain proper positional ordering for reconstruction

### Part 2 - Forward Pass
- Pass visible patches to encoder
- Project encoder output to decoder dimension
- Append learnable mask tokens
- Apply decoder
- Reconstruct pixel patches
- Convert patches back to image space

### Part 3 - Training
- MSE loss computed only on masked patches
- Visible patches ignored during loss calculation
- Mixed precision training with torch.amp
- AdamW optimizer with cosine learning rate scheduler


## Project Structure
```
GenAI-Assignment-2-MAE/
│
├── AI_ASS02_XXF_YYYY.ipynb    # Main notebook
├── README.md                   # This file
```


## How to Run

### On Kaggle
```
1. Open Kaggle notebook
2. Add TinyImageNet dataset
3. Enable GPU T4 x2 accelerator
4. Run all cells in order
```

### Locally
```bash
pip install torch torchvision numpy matplotlib
pip install scikit-image gradio tqdm pillow
jupyter notebook AI_ASS02_XXF_YYYY.ipynb
```

---

## Gradio App
The notebook includes a Gradio app that:
- Accepts image upload
- Allows masking ratio selection (10% to 90%)
- Displays Original, Masked Input, and Reconstruction in real time

---

## Requirements
```
torch
torchvision
numpy
matplotlib
scikit-image
gradio
tqdm
pillow
```

---

## Submitted by
- Name: Bilal Chohan
- Roll No: 22F-3360
- Batch: 22
- Course: Generative AI 
- Semester: Spring 2026
- University: NUCES
```

