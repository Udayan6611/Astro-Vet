<div align="center">
  <h1>Astro-Vet</h1>
  <p><b>Bayesian 3D-CNN Framework for Exoplanet Candidate Vetting</b></p>
  
  [![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
  [![PyTorch](https://img.shields.io/badge/PyTorch-Enabled-EE4C2C.svg)](#)
  [![Status](https://img.shields.io/badge/Status-Prototype-orange.svg)](#)
</div>

---

## Overview

Astronomical surveys like NASA's Transiting Exoplanet Survey Satellite (TESS) generate vast amounts of time-series photometry. However, detecting a true exoplanet transit is difficult because planetary signals are shallow, noisy, and often obscured by stellar variability, background eclipsing binaries, and instrumental artifacts. 

**Astro-Vet** is an automated, hybrid machine-learning pipeline designed to vet these candidates. Rather than relying on a pure black-box classifier, Astro-Vet combines a spatiotemporal deep learning scanner with deterministic, physics-informed constraints (Kepler's Laws) to maximize detection efficiency while aggressively filtering out astrophysical false positives.

## Architecture

The system is built on a four-stage **Detection & Vetting** architecture:

1. **The Sentinel (Data Ingestion):** Queries the NASA MAST API via `lightkurve`, downloads high-cadence Target Pixel Files (TPFs), applies NASA's 'hard' quality bitmask to remove anomalies, normalizes pixel flux, and constructs a 3D spatiotemporal tensor (`Time × Height × Width`).
2. **AI Scanner (Bayesian 3D-CNN):** A PyTorch-based convolutional neural network that scans the 3D tensor to detect transit dips. It utilizes Monte Carlo Dropout to generate calibrated epistemic uncertainty scores for every prediction.
3. **Physics Validator (PINN):** A Physics-Informed Neural Network acting as a strict validator. It forces AI-flagged candidates to pass physical constraints, including Centroid Shift Analysis, Odd-Even Depth parity, and Secondary Eclipse checks.
4. **Backend Orchestration:** Designed to process thousands of light curves, the pipeline uses **Celery** and **Redis** for asynchronous batch processing, alongside **DVC** (Data Version Control) for strict model and dataset provenance.

## Project Roadmap

- [x] **Stage 1: The Sentinel (Current Stage)** 
  - Implement MAST API data ingestion.
  - Apply hard bitmask filtering and Z-score normalization.
  - Create PyTorch `Dataset` class (`TESSTensorDataset`) for dynamic 3D tensor conversion.
- [ ] **Stage 2: AI Scanner**
  - Build the 3D-CNN PyTorch architecture.
  - Implement Monte Carlo Dropout for uncertainty quantification.
- [ ] **Stage 3: Physics Validator (PINN)**
  - Integrate centroid-shift and Keplerian orbit verification constraints.
- [ ] **Stage 4: Pipeline Orchestration**
  - Containerize tasks with Celery/Redis for high-throughput sector processing.

## Getting Started

### Prerequisites
- Python 3.10 or higher
- `pip` and standard `venv`

### Installation

Clone the repository and set up your local environment:

```bash
git clone [https://github.com/sarthak11234/AstroVet.git](https://github.com/sarthak11234/AstroVet.git)
cd AstroVet
python -m venv venv

# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Usage 

Currently, the Sentinel module acts as our functional data extraction and preprocessing prototype. 

**1. Run the Visualizer:** 
Downloads a known TESS candidate (e.g., Pi Mensae c), cleans the data, and outputs the normalized 1D light curve alongside the 2D pixel heatmap.
```bash
python src/sentinel.py
```

**2. Run the Tensor Constructor:** 
Parses the cached `.fits` files and transforms the raw telescope telemetry into the normalized PyTorch tensor format `[Batch, Time, Height, Width]` required for the 3D-CNN.
```bash
python src/dataset.py
```

## Research & References

This project builds upon foundational methodologies from the astrophysics and deep learning communities:
* **TESS Data Vetting:** Yu et al. (2019) on automating triage and vetting of TESS candidates using deep learning.
* **Algorithm Sensitivity:** Hippke & Heller (2019) regarding optimized transit detection via TLS.
* **Validation Standards:** Guerrero et al. (2021) outlining the TESS Objects of Interest (TOI) Catalog physical checks.
* **Physics & Uncertainty ML:** Soboczenski et al. (2018) for Bayesian Deep Learning in astrophysics, and Raissi et al. (2019) for the integration of Physics-Informed Neural Networks.