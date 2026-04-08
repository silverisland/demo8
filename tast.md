# Role & Objective
You are a Principal AI/Deep Learning Engineer specializing in Physics-Informed Machine Learning (PIML) and Spatiotemporal Time Series Generation. 
Your task is to build a Few-Shot Cross-Site PV (Photovoltaic) Power Generation and Forecasting framework using Python and PyTorch.

# Problem Context & Innovation
We are performing multi-site PV power forecasting. We have rich historical data for multiple source sites, but a new target site only has 1 month of data. Due to extreme seasonal distribution shifts (e.g., smooth PV in sunny autumns, volatile PV in rainy springs), traditional few-shot fine-tuning fails.

**Our Core Innovations (Must be implemented):**
1. **Clear-sky GHI as Physical Anchor**: Using `pvlib` to calculate theoretical clear-sky GHI based on coordinates/time, acting as an absolute physical upper bound.
2. **Consistent Power Transport (CPT) Constraint**: Treating PV generation as a physical energy transport process. We will design a custom loss function to heavily penalize generated power that exceeds the GHI bound or violates monotonicity w.r.t NWP irradiance.
3. **Site Latent Codebook (Prototype Memory)**: Decoupling "Weather Fluctuations" (transient) from "Site Physical Efficiency" (steady-state). We will learn a Codebook of typical site fingerprints during pre-training. For the new site, we will use its 1-month data to query the Codebook (via Attention) to extract its unique weighted physical fingerprint without overfitting.

# Step-by-Step Implementation Guide

Please write modular, OOP-based PyTorch code step by step. Do NOT write all code in one file; separate them logically.

## Step 1: Physics-Informed Data Pipeline (`dataset.py`)
- Create a PyTorch `Dataset` for multi-site PV data (NWP, Power, Timestamps).
- **Physics Injection**: Write a utility function using `pvlib.location` and `pvlib.irradiance` to compute `clearsky_GHI` for each timestamp based on site `(lat, lon)`. Concatenate this GHI into the feature tensor.
- **Time Features**: Implement Sin/Cos positional encoding for hour-of-day and month-of-year.
- **Output Shape**: Ensure `__getitem__` returns dicts like `{'nwp': [seq_len, dim], 'ghi': [seq_len, 1], 'power': [seq_len, 1], 'site_id': int}`.

## Step 2: Site Latent Codebook Module (`memory.py`)
- Implement a `SiteMemoryBank` (Codebook) module.
- **Pre-training mode**: It holds learnable embeddings for `N` typical site clusters.
- **Few-Shot Extractor**: Implement an Attention mechanism. It takes a small context window of real data (e.g., 1 month of NWP + Power) from an unknown site, processes it via a lightweight encoder to generate a Query, and attends over the Codebook to output a continuous, weighted `site_latent_vector`.

## Step 3: Physics-Informed Generator (`generator.py` & `cpt_loss.py`)
- Implement a Conditional Generator (choose Conditional CVAE or a lightweight Diffusion model).
- **Inputs**: 
  1. Condition: NWP sequence + Time Encodings + Clear-sky GHI.
  2. Latent: The `site_latent_vector` from Step 2.
- **Output**: Synthetic PV power sequence.
- **Consistent Power Transport (CPT) Loss**: Implement a custom `nn.Module` for the loss function:
  - `L_recon`: Standard MSE/MAE against ground truth power.
  - `L_physics_bound`: `ReLU(P_generated - GHI_clearsky)` to strictly penalize physical violations.
  - `L_smoothness`: Temporal difference penalty to prevent unnatural high-frequency jitter.
  - `Total Loss = L_recon + alpha * L_physics_bound + beta * L_smoothness`

## Step 4: Few-Shot Adaptation & Generation Engine (`pipeline.py`)
- Write a class `PVGenerationPipeline`.
- **Method `extract_fingerprint(target_site_1month_data)`**: Uses Step 2 to freeze the codebook and output the new site's latent vector.
- **Method `generate_full_year(target_site_latent, full_year_nwp)`**: Autoregressively or parallelly generates 12 months of synthetic PV data, strictly bounded by the year-round `pvlib` GHI calculation.

## Step 5: Downstream Forecasting Baseline (`forecasting.py`)
- Implement a lightweight Temporal Fusion Transformer (TFT) or PatchTST baseline.
- Provide a training loop that mixes the "1-month real data" and "11-months synthetic data" to train the forecasting model for the new site.

# Coding Constraints & Vibe
- Use type hinting (`from typing import ...`).
- Add clear comments for tensor shapes (e.g., `# [batch_size, seq_len, hidden_dim]`).
- The code must be robust to missing values or curtailment anomalies (e.g., drop sequences where `real_power == 0` but `GHI > 500` during training).
- Prioritize the implementation of `cpt_loss.py` and `memory.py` as they are the core mathematical innovations.