# Physics-Informed Few-Shot PV Forecasting Framework

This project implements a modular, physics-constrained framework for multi-site PV power forecasting, specifically designed to handle the "seasonal distribution shift" problem using only 1 month of target site data.

## 1. Architectural Philosophy

### A. Physical Grounding (Physics-Informed ML)
Instead of treating PV power as an abstract sequence, we anchor the model to the **Clear-sky GHI** (Global Horizontal Irradiance). 
- **Physical Upper Bound**: Predictions are strictly penalized if they exceed the theoretical solar energy envelope for the site's GPS coordinates.
- **Tools**: Powered by `pvlib` and our custom `CPT Loss` (Consistent Power Transport).

### B. Latent Decoupling (Site Fingerprinting)
We decouple **Transient Weather** from **Steady-State Site Physics**.
- **Codebook**: A memory bank of "Physical Prototypes" (e.g., high-efficiency utility sites vs. shaded rooftops) learned during pre-training.
- **Few-Shot Adaptation**: A new site is "fingerprinted" by querying this codebook using its 1-month of data, avoiding the overfitting associated with traditional fine-tuning.

### C. Stochastic Augmentation (Diffusion)
We use a **Conditional Diffusion Model** to generate synthetic data.
- **Diversity**: Unlike VAEs/GANs, diffusion captures the complex, stochastic nature of weather patterns.
- **Bridging the Gap**: 11 months of synthetic data are generated to provide the downstream forecaster with a complete seasonal cycle.

---

## 2. Module Overview

| File | Module | Responsibility |
| :--- | :--- | :--- |
| `dataset.py` | **Data Pipeline** | Loads NWP/Power, injects `pvlib` Clear-sky GHI, and applies time-encodings. |
| `memory.py` | **Site Codebook** | Learns global site archetypes and extracts latent fingerprints via Attention. |
| `generator.py` | **Diffusion Gen** | 1D ResNet-based DDPM that generates power sequences conditioned on NWP and Identity. |
| `cpt_loss.py` | **Physics Loss** | Custom ReLU-based penalty for physical bound violations and temporal smoothness. |
| `pipeline.py` | **Orchestrator** | Coordinates the "Fingerprint -> Generation -> Bounding" workflow. |
| `forecasting.py`| **Transformer** | A Temporal Transformer that trains on mixed Real+Synthetic data for forecasting. |
| `config.py` | **Central Config** | Unified hyperparameters and dimensions for the entire project. |

---

## 3. Execution Guide

### Step 0: Environment Setup
Ensure you have `pixi` installed.
```bash
pixi install
```

### Step 1: Stage 1 - Source Pre-training
Learn the global site prototypes and generator weights using data from source sites.
```python
# Execute training loop in train_stage1.py
from train_stage1 import train_stage1
# train_stage1(loader, epochs=50, device="cuda")
```
- **Inputs**: Years of data from multiple source sites.
- **Objective**: Minimize Noise MSE + CPT Physics Loss.

### Step 2: Stage 2 - Few-Shot Adaptation & Generation
Adapt to a new site and fill the 11-month data gap.
1. **Extract Fingerprint**: Pass the new site's 1-month data through `memory.py` to get the `site_latent`.
2. **Generate Synthetic Year**: Pass the `site_latent` and future NWP through `pipeline.py` to generate 11 months of synthetic PV power.

### Step 3: Stage 3 - Downstream Forecaster Training
Train the final forecasting model using the augmented dataset.
```python
# Use the training loop in forecasting.py
from forecasting import train_with_augmentation
# train_with_augmentation(model, real_loader, syn_loader, synthetic_ratio=0.5)
```
- **Result**: A transformer forecaster that understands 12 months of seasonal patterns despite only seeing 1 month of real data.
