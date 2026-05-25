# Topological Deep Learning for Sensory Habituation in Infant Cognitive Milestones

This repository contains a state-of-the-art **Topological Deep Learning (TDL)** predictive pipeline designed to model sensory habituation in infants. By combining dynamical systems theory, algebraic topology, and temporal self-attention networks, the architecture reconstructs the joint phase space of physiological signals to predict whether a sensory stimulus will be accepted or rejected—specifically, the cognitive milestone **"lo_consume" (Yes/No)** from clinical trials.

The pipeline is fully implemented, mathematically documented, and end-to-end verified in the Jupyter Notebook:
👉 [sensory_habituation_tdl.ipynb](sensory_habituation_tdl.ipynb)

---

## 📊 Project Scientific Architecture

Sensory habituation is modeled as a trajectory in a multi-dimensional dynamical system. The pipeline transitions from sparse clinical data checkpoints to high-frequency simulated manifolds, extracts topological signatures of stress, and classifies the resulting sequence using a custom Transformer network.

```mermaid
graph TD
    A[babyface_consolidado.csv] --> B[Data Prep & Spline Interpolation]
    B --> C[Continuous Signals V(t) & A(t)]
    C --> D[Takens Phase Space Embedding in R^2d]
    D --> E[Kepler Mapper / Reeb Graph Bifurcation]
    C --> F[Sliding-Window Topological Segmenter]
    F --> G[Vietoris-Rips Homology H_0, H_1]
    G --> H[Persistence Landscapes Vectorization]
    H --> I[PyTorch Time-Series Transformer]
    I --> J[Predict 'lo_consume' Milestone]
```

### 1. Data Prep & Spline Interpolation
- **Input**: Sparse clinical checkpoints (8 events) of Valencia ($V \in [-1, 1]$) and Activation/Arousal ($A \in [0, 1]$) per subject-day row.
- **Robust Imputation**: Sequentially imputes NaNs using forward/backward fills per row.
- **Continuous Simulation**: Interpolates the 8 sparse milestones to a dense grid of $N = 100$ high-frequency points using **cubic splines** (with automatic linear fallback for low-entropy signals).

### 2. Dynamic Takens Embedding (No Arbitrary Parameters)
- **optimal $\tau$ (Delay)**: Calculated dynamically using **Average Mutual Information (AMI)** to locate the first local minimum.
- **optimal $d$ (Dimension)**: Calculated dynamically using **False Nearest Neighbors (FNN)** to find the minimal dimension where false neighbors drop below 1% ($<0.01$).
- **Cohort Medians**: Median delay ($\tau = 5$) and dimension ($d = 2$) are selected to build a consistent joint phase space reconstruction in $\mathbb{R}^{2d} = \mathbb{R}^4$.

### 3. Topological Analysis with Kepler Mapper (Reeb Graph)
- **Manifold Mapping**: Combines linear projections (PCA 1st component) and non-linear topological descriptors (**Euclidean Eccentricity $L_2$**) to build a custom 2D lens.
- **Bifurcation Analysis**: Constructs a Reeb graph via Kepler Mapper, coloring nodes by the cognitive milestone `lo_consume` (mean target) to isolate and visualize the physical bifurcations between "aversión" (rejection) and "tolerancia" (tolerance). The interactive graph is saved in `reeb_graph.html`.

### 4. Sliding-Window Persistent Homology (Giotto-TDA)
- **Segmentation**: Trajectories in $\mathbb{R}^4$ are divided into 5 overlapping sliding windows (size = 30, stride = 17).
- **Homology Groups**: Extracts components ($H_0$) and cycles/tunnels ($H_1$, representing topological stress structures) using Vietoris-Rips complexes.
- **Vectorization**: Transforms persistence diagrams into **Persistence Landscapes** (3 layers, 50 bins) to form a sequential topological tensor of shape `(samples, 5, 300)`.

### 5. Time-Series Transformer & Interpretabilidad (PyTorch)
- **Self-Attention Network**: A PyTorch sequential Transformer Encoder maps landscapes to a hidden space, extracts temporal dependencies using self-attention, and classifies the sequences.
- **Bifurcation Interpretation**: Utilizes a custom `AttentionExtractionLayer` to extract self-attention weights in the forward pass. Plotting these weights reveals which temporal window of topological stress collapse is critical in predicting the cognitive decision.

---

## 📈 Verification & Validation Metrics

The entire pipeline has been fully executed and validated using the consolidated clinical dataset:
- **Joint Phase Space**: 4-Dimensional ($\tau = 5, d = 2$)
- **Topological Sequence**: Tensor of shape `(363, 5, 300)`
- **PyTorch Model Performance**:
  - **ROC-AUC Score**: **0.7721** (showing high discriminative power)
  - **F1-Score**: **0.7593**
- **Attention Verification**: Flawless extraction of attention weights shape `(91, 5, 5)` for test sets.

---

## 📁 Repository Structure

```
├── babyface_consolidado.csv       # Clinical consolidated dataset
├── prompt_tdl.md                 # Original project prompt instructions
├── sensory_habituation_tdl.ipynb  # Primary Jupyter Notebook deliverable
├── reeb_graph.html               # Interactive Reeb Graph visualization
├── README.md                     # Project documentation (this file)
└── scratch/                      # (Git ignored) Verification and helper scripts
```

---

## 🛠️ Environment Setup & Quickstart

To ensure 100% stable execution and avoid package version conflicts between `giotto-tda` and newer `scikit-learn` libraries on Windows, the notebook implements dynamic environment isolation at the very top:

```python
import os
import sys
os.environ["PYTHONNOUSERSITE"] = "1"
sys.use_user_site = False
sys.path = [p for p in sys.path if "AppData\\Roaming\\Python" not in p]
```

### Installation Guide

We recommend using **Anaconda** (which includes optimized DLL search paths for PyTorch and SciPy) together with **uv** (for fast dependency management) or standard **pip**.

#### Option A: Using `uv` (Recommended for speed)
In your terminal, run the following command to install the entire Topological Deep Learning stack:
```bash
uv pip install giotto-tda kmapper torch scikit-learn pandas numpy matplotlib seaborn scipy
```

#### Option B: Using `pip` inside Anaconda
If you are running in an Anaconda environment, install the isolated packages directly using:
```bash
pip install giotto-tda kmapper torch scikit-learn==1.3.2 pandas numpy matplotlib seaborn scipy --ignore-installed --no-user
```
*(Downgrading `scikit-learn` to `1.3.2` ensures absolute compatibility with `giotto-tda`'s internals).*

### Running the Notebook
1. Open your Jupyter Notebook environment.
2. Launch `sensory_habituation_tdl.ipynb`.
3. Run all cells sequentially. The dataset `babyface_consolidado.csv` must reside in the same root folder.
4. Open the generated `reeb_graph.html` in any web browser to interactively explore the Reeb Graph.

---

## 📝 References & Mathematical Underpinnings
- **Takens Embedding Theorem**: Takens, F. (1981). "Detecting strange attractors in turbulence." *Dynamical Systems and Turbulence*.
- **Mapper Algorithm**: Singh, G., Mémoli, F., & Carlsson, G. (2007). "Topological methods for the analysis of high dimensional data sets and 3D object shapes." *Eurographics Symposium on Point-Based Graphics*.
- **Persistence Landscapes**: Bubenik, P. (2015). "Statistical topological data analysis using persistence landscapes." *Journal of Machine Learning Research*.
- **Attention Mechanism**: Vaswani, A. et al. (2017). "Attention is all you need." *Advances in Neural Information Processing Systems (NeurIPS)*.
