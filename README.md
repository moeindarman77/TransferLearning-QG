# Fourier Analysis of Transfer Learning for Subgrid-Scale Models of Ocean Turbulence

[![Paper](https://img.shields.io/badge/Paper-IOP%20Publishing-blue)](.)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Moein Darman, Ashesh Chattopadhyay, Laure Zanna, and Pedram Hassanzadeh**

Research Group: [Hassanzadeh Group](http://pedram.rice.edu/team/)

---

## Table of Contents
* [Introduction](#introduction)
* [Key Findings](#key-findings)
* [Repository Structure](#repository-structure)
* [Requirements](#requirements)
* [Installation](#installation)
* [Usage](#usage)
    * [Data Generation](#data-generation)
    * [Training Base Neural Network](#training-base-neural-network)
    * [Transfer Learning](#transfer-learning)
    * [Spectral Analysis](#spectral-analysis)
    * [Online Simulations](#online-simulations)
* [Experiments](#experiments)
* [Reproducibility](#reproducibility)
* [Citation](#citation)
* [Acknowledgments](#acknowledgments)

---

## Introduction

This repository contains the code for the paper **"Fourier Analysis of the Physics of Transfer Learning for Data-Driven Subgrid-Scale Models of Ocean Turbulence"**.

Transfer learning (TL) is a powerful technique for enhancing neural network generalization to out-of-distribution data with minimal training samples from the target system. In this study, we:

1. **Employ a 9-layer CNN** to predict subgrid forcing in a two-layer ocean quasi-geostrophic (QG) system
2. **Examine generalization metrics** across different dynamical regimes (isotropic eddies and anisotropic jets)
3. **Perform Fourier analysis** of CNN kernels to reveal that they learn low-pass, Gabor, and high-pass filters
4. **Explain why NNs fail to generalize** by analyzing activation spectra throughout the network
5. **Demonstrate how TL corrects** the spectral underestimation by retraining only one layer

### Key Contributions

- **Spectral interpretation**: We show that learned CNN kernels act as spectral filters regardless of training data isotropy
- **Generalization failure mechanism**: Weights and biases from one dataset underestimate out-of-distribution sample spectra
- **Transfer learning physics**: By retraining one layer, spectral underestimation is corrected, enabling accurate predictions
- **Practical metrics**: We identify which offline metrics best predict generalization performance

---

## Key Findings

### 1. CNN Kernels as Spectral Filters
<img src="docs/kernel_filters.png" width="600">

CNNs learn combinations of:
- **Low-pass filters**: Capture large-scale dynamics
- **Gabor filters**: Detect oriented features
- **High-pass filters**: Capture small-scale variability

This filter distribution is **independent of whether training data are isotropic or anisotropic**.

### 2. Why Neural Networks Fail to Generalize
<img src="docs/activation_spectra.png" width="600">

When applied to out-of-distribution data, the learned weights and biases **underestimate the activation spectra** at each layer. This underestimation propagates through the network, resulting in output spectra that don't match the target system.

### 3. How Transfer Learning Fixes Generalization
<img src="docs/transfer_learning.png" width="600">

By **retraining only the first hidden layer** (ℓ=2) with 2-10% of target data:
- The spectral content is upshifted at that layer
- This adjustment propagates through subsequent layers
- Output spectra align with the target system's physics

---

## Repository Structure

```
TransferLearning-QG/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── cnn_qg.py                          # Main CNN training script
├── cnn_qg_TL.py                       # Transfer learning script
├── couple_FCNN.py                     # Online coupling with pyqg
├── runs_pyqg.py                       # Batch simulation runner
│
├── utils/                             # Utility functions
│   ├── utils.py                       # Dataset classes, plotting
│   ├── corr2.py                       # Correlation metrics
│   ├── post_proccess.py               # Analysis functions
│   └── ...
│
├── pyqg_parameterization_benchmarks/  # Benchmarking suite
│   ├── src/
│   │   └── pyqg_parameterization_benchmarks/
│   │       ├── neural_networks.py     # NN architectures
│   │       ├── coarsening_ops.py      # Filtering operators
│   │       ├── online_metrics.py      # Online evaluation
│   │       └── utils.py
│   └── notebooks/                     # Analysis notebooks
│       ├── dataset_description.ipynb
│       ├── neural_networks.ipynb
│       ├── online_metrics.ipynb
│       └── subgrid_forcing.ipynb
│
├── FiguresProduction_baby.ipynb       # Paper figures generation
├── interpretability.ipynb             # Spectral analysis notebook
├── online_coupling.ipynb              # Online simulation analysis
├── post_proccess.ipynb                # Results post-processing
│
└── results/                           # Saved model checkpoints & outputs
    ├── BestModelBasedOnTestLoss.pt
    ├── activation_dict.pkl
    ├── weights_dict.pkl
    └── ...
```

---

## Requirements

### Core Dependencies
- **Python** >= 3.6
- **PyTorch** >= 2.3.1
- **NumPy** >= 1.19
- **SciPy** >= 1.5
- **netCDF4** >= 1.5
- **matplotlib** >= 3.3
- **pyqg** >= 0.7.2 ([documentation](https://pyqg.readthedocs.io/))

### Additional Packages
- `scikit-learn` - for k-means clustering analysis
- `xarray` - for NetCDF data handling
- `prettytable` - for formatted output tables
- `ray[tune]` - for hyperparameter optimization (optional)

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/moeindarman77/TransferLearning-QG.git
cd TransferLearning-QG
```

### 2. Create a Virtual Environment (Recommended)
```bash
conda create -n tl-qg python=3.8
conda activate tl-qg
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install pyqg_parameterization_benchmarks
```bash
cd pyqg_parameterization_benchmarks
pip install -e .
cd ..
```

---

## Usage

### Data Generation

The study uses four different quasi-geostrophic configurations:

| Case | Configuration | Parameters | Description |
|------|---------------|------------|-------------|
| **Case 0** | Eddy | rd = 15 km, β⁰ = 1.5×10⁻¹¹ m⁻¹s⁻¹ | Isotropic eddies (Base system) |
| **Case 1** | Eddy (higher rd) | rd = 20 km, β¹ = β⁰ | Larger isotropic eddies |
| **Case 2** | 3-Jet | rd = 15 km, β² = 1/3 β⁰ | Three anisotropic jets |
| **Case 3** | 4-Jet | rd = 15 km, β³ = 2/5 β⁰ | Four anisotropic jets |

Generate high-resolution simulation data using `pyqg`:

```python
import pyqg
import numpy as np

# Example for Case 0 (Eddy configuration)
m = pyqg.QGModel(
    nx=256,                    # High-resolution grid
    L=1e6,                     # Domain size (1000 km)
    dt=3600.0,                 # 1 hour timestep
    rd=15000.0,                # Deformation radius
    beta=1.5e-11,              # Beta parameter
    rek=5.78e-7,               # Ekman drag
    U1=0.025,                  # Upper layer velocity
    U2=0.0                     # Lower layer velocity
)

# Run simulation
m.run_with_snapshots(dt=3600, tsave=3600)
```

See the notebooks in `pyqg_parameterization_benchmarks/notebooks/` for detailed data generation examples.

### Training Base Neural Network

Train a base neural network (BNN) on Case 0 data:

```bash
python cnn_qg.py --case 0 --epochs 100 --batch_size 8 --lr 1e-3
```

**Key Training Parameters:**
- **Architecture**: 9-layer CNN with 7 hidden layers (64 channels each, 5×5 kernels)
- **Input**: 4 channels (u₁, v₁, u₂, v₂) at 64×64 resolution
- **Output**: 2 channels (Πq₁, Πq₂) - subgrid forcing for each layer
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate scheduler
- **Training Data**: 40,000 samples from Case 0

The CNN architecture:

```python
class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()

        # Input layer: 4 channels → 64 channels
        self.conv_layer1 = nn.Conv2d(4, 64, kernel_size=5, padding="same")

        # Hidden layers: 64 → 64 (×7 layers)
        self.conv_layer2 = nn.Conv2d(64, 64, kernel_size=5, padding="same")
        self.conv_layer3 = nn.Conv2d(64, 64, kernel_size=5, padding="same")
        # ... (layers 4-7)

        # Output layer: 64 → 2 channels
        self.conv_layer8 = nn.Conv2d(64, num_classes, kernel_size=5, padding="same")

        # ReLU activation (applied after each layer except output)
        self.relu = nn.ReLU()
```

### Transfer Learning

Apply transfer learning to adapt the BNN to a target case (e.g., Case 2):

```bash
python cnn_qg_TL.py --base_model BestModelBasedOnTestLoss.pt \
                     --target_case 2 \
                     --retrain_layer 2 \
                     --data_percentage 0.02 \
                     --epochs 100
```

**Transfer Learning Strategy:**
1. **Initialize** with BNN₀ weights from Case 0
2. **Freeze** all layers except layer ℓ=2 (first hidden layer)
3. **Retrain** layer 2 with 2%, 5%, or 10% of target case data
4. **Evaluate** using offline and online metrics

This approach (TLNN⁰,ⁱ) achieves comparable performance to training from scratch with far less data.

### Spectral Analysis

Perform Fourier analysis of CNN kernels and activations:

```python
import numpy as np
import torch
from scipy import signal

# Load trained model
model = torch.load('BestModelBasedOnTestLoss.pt')

# Extract and analyze kernels from layer 2
layer2_weights = model.conv_layer2.weight.detach().cpu().numpy()

# Fourier transform of kernels
for i in range(64):
    for j in range(64):
        kernel = layer2_weights[i, j, :, :]

        # Zero-pad to 64×64
        kernel_padded = np.zeros((64, 64))
        kernel_padded[:5, :5] = kernel

        # Compute 2D FFT
        kernel_fft = np.fft.fft2(kernel_padded)
        kernel_spectrum = np.abs(np.fft.fftshift(kernel_fft))

        # Classify filter type (low-pass, Gabor, high-pass)
        # ... (see interpretability.ipynb)
```

See `interpretability.ipynb` and `FiguresProduction_baby.ipynb` for complete spectral analysis workflows.

### Online Simulations

Couple the trained CNN with pyqg for online (a posteriori) evaluation:

```python
from couple_FCNN import FCNNParameterization
import pyqg

# Load trained model
param = FCNNParameterization(
    model_path='BestModelBasedOnTestLoss.pt',
    device='cuda'
)

# Create low-resolution QG model with CNN parameterization
m = pyqg.QGModel(
    nx=64,                     # Low-resolution grid
    L=1e6,
    dt=3600.0,
    parameterization=param
)

# Run 10-year simulation
m.run_with_snapshots(dt=3600, tsave=3600*24, runtime=3600*24*365*10)
```

Evaluate online metrics:
- **Kinetic Energy (KE) Spectra**: Compare energy distribution across wavenumbers
- **Probability Density Functions (PDFs)**: Check if extreme events are captured
- **Time Series**: Validate temporal evolution of PV fields

See `online_coupling.ipynb` and `test_for_online_runs.ipynb` for detailed examples.

---

## Experiments

### Offline (A Priori) Evaluation

We use three key metrics to assess generalization:

1. **Root Mean Square Error (RMSE)**:
   ```
   RMSE_Πq = √[ (1/N) Σᵢ ||Πq_pred,i - Πq_true,i||₂² ] / √[ (1/N) Σᵢ ||Πq_true,i||₂² ]
   ```

2. **Correlation Coefficient (CC)**:
   ```
   CC_Πq = ⟨(Πq_pred - ⟨Πq_pred⟩)(Πq_true - ⟨Πq_true⟩)⟩ / [σ(Πq_pred) σ(Πq_true)]
   ```

3. **Spectrum RMSE** (most indicative of generalization):
   ```
   Spectrum RMSE = (1/Nₖ) Σₖ |[Π̂q_pred(k) - Π̂q_true(k)] / Π̂q_true(k)|
   ```

**Key Results:**
- **BNN⁰,⁰** (trained and tested on Case 0): Best in-distribution performance
- **BNN⁰,ⁱ** (trained on Case 0, tested on Case i): Poor generalization
- **TLNN⁰,ⁱ** (TL with 2-10% target data): Recovers near-optimal performance

| Model | CC (Πq₁) | RMSE (Πq₁) | Spectrum RMSE |
|-------|----------|-----------|---------------|
| BNN⁰,⁰ | 0.92 ± 0.01 | 0.58 ± 0.03 | 0.77 ± 0.21 |
| BNN⁰,² | 0.94 ± 0.01 | 0.59 ± 0.06 | 4.13 ± 0.77 |
| TLNN⁰,² (2%) | 0.94 ± 0.01 | 0.53 ± 0.03 | 0.93 ± 0.23 |
| TLNN⁰,² (10%) | 0.94 ± 0.01 | 0.53 ± 0.04 | 0.73 ± 0.17 |

### Online (A Posteriori) Evaluation

Coupled simulations demonstrate:
- **TLNN improves KE spectra** at high wavenumbers compared to BNN
- **Better PDF tails** for extreme events (Case 2)
- **Scale-selective dissipation** in the solver can mask some improvements

Run batch online experiments:
```bash
python runs_pyqg.py --config experiments/case1/config.json
```

---

## Reproducibility

All figures in the paper can be reproduced using `FiguresProduction_baby.ipynb`:

```bash
jupyter notebook FiguresProduction_baby.ipynb
```

**Main Figures:**
- **Figure 1**: CNN architecture and input/output spectra
- **Figure 2**: Physical characteristics of different cases (PV snapshots, velocity/forcing spectra)
- **Figure 3**: A priori metrics (CC, RMSE, Spectrum RMSE) across cases
- **Figure 4**: A posteriori evaluation (KE spectra, PDFs)
- **Figure 5**: Activation spectra across layers for BNN and TLNN
- **Figure 6**: Cluster centers of Fourier-transformed kernels
- **Figure 7**: Histogram analysis of kernel maxima locations

### Pretrained Models

Pretrained models are available in the `results/` directory:
- `BestModelBasedOnTestLoss.pt` - Base model trained on Case 0
- `final_model_after_training.pt` - Full training checkpoint
- `activation_dict.pkl` - Saved activation spectra
- `weights_dict.pkl` - Saved kernel weights

---

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@article{Darman2026TransferLearningQG,
  author = {Darman, Moein and Chattopadhyay, Ashesh and Zanna, Laure and Hassanzadeh, Pedram},
  title = {Fourier analysis of the physics of transfer learning for data-driven subgrid-scale models of ocean turbulence},
  journal = {Journal Name},
  year = {2026},
  publisher = {IOP Publishing},
  note = {In press}
}
```

**Related Publications:**
- Ross et al. (2023): [Benchmarking of machine learning ocean subgrid parameterizations](https://doi.org/10.1029/2022MS003258)
- Subel et al. (2023): [Explaining the physics of transfer learning in data-driven turbulence modeling](https://doi.org/10.1093/pnasnexus/pgad015)
- Guan et al. (2022): [Stable a posteriori LES of 2D turbulence using CNNs](https://doi.org/10.1016/j.jcp.2022.111090)

---

## Acknowledgments

We thank:
- **Karan Jakhar, Rambod Mojgani, and Hamid A. Pahlavan** for valuable feedback
- **Funding**: ONR award N000142012722, NSF grants AGS-2046309 and 2425667, Schmidt Sciences LLC
- **Computational Resources**: NSF ACCESS MTH240019, NCAR CISL UCSC0008/UCSC0009

**Contact:**
- Moein Darman: [moein.darman@rice.edu](mailto:mdarman@ucsc.edu)
- Pedram Hassanzadeh: [pedram@rice.edu](mailto:pedramh@uchicago.edu)
- GitHub: [https://github.com/moeindarman77/TransferLearning-QG](https://github.com/moeindarman77/TransferLearning-QG)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Additional Resources

- **pyqg Documentation**: [https://pyqg.readthedocs.io/](https://pyqg.readthedocs.io/)
- **Hassanzadeh Group**: [http://pedram.rice.edu/team/](http://pedram.rice.edu/team/)
- **M²LInES Project**: [https://m2lines.github.io/](https://m2lines.github.io/)
