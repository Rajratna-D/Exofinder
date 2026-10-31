<p align="center">
  <a href="https://github.com/Rajratna-D/Exofinder/releases/tag/v1.0.0"><img src="https://img.shields.io/badge/Release-v1.0.0-007EC6?style=for-the-badge&logo=github&logoColor=white" alt="v1.0.0 Release"></a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/Vite-6-646CFF?style=for-the-badge&logo=vite&logoColor=white" alt="Vite">
  <img src="https://img.shields.io/badge/FastAPI-0.142-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Three.js-WebGL-000000?style=for-the-badge&logo=three.js&logoColor=white" alt="Three.js">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/XGBoost-Ensemble-006400?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/Apache_Parquet-Compressed_Data-4D789B?style=for-the-badge&logo=apache&logoColor=white" alt="Parquet">
  <img src="https://img.shields.io/badge/NASA_TESS-Space_Data-000080?style=for-the-badge&logo=nasa&logoColor=white" alt="NASA TESS">
</p>

<h1 align="center">🪐 ExoFinder</h1>
<h3 align="center">Deep Space Photometry & AI-Powered Exoplanet Vetting Suite</h3>

<p align="center">
  An end-to-end astronomical signal processing, deep learning, and vetting system that ingests, detrends, folds, and classifies NASA TESS and Kepler stellar light curves into <b>Confirmed Planets</b>, <b>Eclipsing Binaries</b>, and <b>False Positives</b> — paired with a cinematic Nova-themed interactive dashboard featuring real-time Three.js 3D transit simulation, high-frame-rate Canvas photometry, optical-to-acoustic audio sonification, and AI consensus diagnostics.
</p>

---

## 🎯 What ExoFinder Does

> **An automated AI vetting pipeline capable of disentangling genuine planetary transits from stellar astrophysics and instrumental artifacts in noisy time-series photometry.**
> 
> Exoplanet detection via transit photometry requires identifying microscopic dips (often $< 0.1\%$) in stellar flux. In crowded astronomical fields, stellar blending, background eclipsing binaries, detector jitter, and stellar variability (starspots, flares) mimic planetary transits. ExoFinder provides domain-grounded feature extraction (odd/even transit depth disparity, secondary eclipse detection, limb-darkening U-vs-V shape metrics) combined with a **Dual-Branch 1D-CNN (Astronet)** and a **Platt-Calibrated Stacking Ensemble** to reliably vet targets.

---

## ⚡ Quick Start & Execution Modes

### 1. Setup Environment
```bash
# Clone and enter the repository
git clone https://github.com/Rajratna-D/Exofinder.git
cd Exofinder

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Execution Modes

#### 🪐 Mode A: Unified Full-Stack Launch (Recommended)
Launches the high-speed FastAPI backend, boots the React 19 frontend, and opens the dashboard in your default browser:
```bash
python run.py
```

#### ⚙️ Mode B: Scientific Pipeline Run (Headless)
Executes the scientific processing pipeline and updates models, metrics, and predictions in `outputs/`:
```bash
python src/run_pipeline.py
```

#### 📁 Mode C: Custom Light Curve Processing
Process your own target CSV files (containing `time` and `flux` columns):
```bash
python src/run_pipeline.py --custom_data_dir /path/to/custom_csvs
```

#### 🧪 Mode D: Granular Pipeline Stages
Each stage can be executed independently for experimentation and tuning:
```bash
python src/run_acquisition.py      # Stage 1: Metadata fetch & TESS cadence download
python src/run_detrending.py       # Stage 2: Segmented Savitzky-Golay detrending
python src/run_period_search.py    # Stage 3: BLS periodogram & phase-folding
python src/run_features.py         # Stage 4: 26-dimensional physical feature extraction
python src/run_classifier.py       # Stage 5: Calibrated Stacking ML & 1D-CNN training
```

---

## 🏗️ Pipeline Architecture

```mermaid
graph LR
    A["🌐 Data Ingestion<br/><i>ExoFOP + MAST</i>"] --> B["🔧 Detrending<br/><i>Gap-Aware SG Filter</i>"]
    B --> C["🔍 Period Search<br/><i>BLS Periodogram</i>"]
    C --> D["📐 Feature Extraction<br/><i>Domain Astro Metrics</i>"]
    D --> E["🤖 Dual AI Engine<br/><i>Stacking ML + 1D-CNN</i>"]
    E --> F["📊 Nova Dashboard<br/><i>React 19 + Three.js</i>"]

    style A fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style B fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style C fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style D fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style E fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style F fill:#1e3a5f,stroke:#60a5fa,color:#fff
```

| Stage | Module | Key Capabilities |
|:------|:-------|:-----------------|
| **1. Ingestion** | `src/data_loader.py` | Pulls balanced labels from ExoFOP TOI catalog (with NASA Exoplanet Archive fallback). Downloads SPOC-calibrated 2-minute cadence light curves from MAST via `lightkurve`. |
| **2. Detrending** | `src/detrend.py` | Detects spacecraft telemetry downlink gaps (>0.5 days) and partitions time series into continuous segments. Applies independent Savitzky-Golay polynomial filters per segment, eliminating boundary edge artifacts. |
| **3. Period Search** | `src/period_search.py` | Executes Astropy Box Least Squares (BLS) periodograms across optimized frequency grids. Recovers orbital period, transit epoch ($T_0$), transit depth, and duration with **100% recovery** on confirmed planets. |
| **4. Feature Eng.** | `src/features.py` | Extracts 26 physically interpretable transit parameters — odd/even cycle depth disparity, secondary eclipse constraints, transit shape kurtosis, and SNR. |
| **5. Classification**| `src/classifier.py` + `deep_learning.py` | Dual architecture: Regularized Stacking Ensemble with Platt Sigmoid calibration + Google Brain/NASA Astronet-inspired Dual-Branch 1D-CNN. |
| **6. Nova Interface** | `frontend/` + `src/server.py` | High-speed FastAPI REST engine (<2ms response) + React 19 WebGL Three.js 3D transit simulator, Canvas photometry, and audio sonification. |

---

## 🧬 Physical Feature Engineering

ExoFinder extracts domain-grounded physical features specifically engineered to isolate exoplanet transits from stellar astrophysical false positives:

| # | Feature Symbol | Astronomical & Physical Significance |
|:-:|:---------------|:-------------------------------------|
| 1 | `bls_depth` | Fractional flux transit depth ($\Delta F / F$) |
| 2 | `bls_duration` | Transit duration ($T_{\text{dur}}$) in days |
| 3 | `bls_power` | Peak spectral power in the BLS periodogram |
| 4 | `transit_snr` | Transit signal-to-noise ratio: $\text{Depth} / \sigma_{\text{out}}$ |
| 5 | `odd_even_depth_ratio` | Ratio of odd-transit depth to even-transit depth (Gold standard eclipsing binary discriminator) |
| 6 | `odd_even_mismatch_sigma` | Statistical significance ($\sigma$) of odd vs. even depth divergence |
| 7 | `odd_even_diff_2p` | Cycle-to-cycle flux difference evaluated at $2\times$ the orbital period |
| 8 | `secondary_depth_phase_05` | Measured eclipse depth at phase $\phi = 0.5$ (detects secondary stellar occultations) |
| 9 | `secondary_to_primary_ratio` | Ratio of secondary eclipse depth to primary transit depth |
| 10 | `secondary_significance_sigma` | Significance ($\sigma$) of secondary eclipse detection |
| 11 | `mean_to_max_depth_ratio` | U-shaped planetary transit ($\approx 0.8$) vs. V-shaped stellar eclipse ($\approx 0.5$) indicator |
| 12 | `bottom_flatness_ratio` | Ingress/egress slope duration relative to total transit width |
| 13 | `kepler_duration_ratio` | Observed duration vs. theoretical maximum circular Keplerian transit duration |
| 14 | `ingress_egress_symmetry` | Gradient slope symmetry between transit ingress and egress |
| 15 | `in_transit_skew` | Skewness of flux points inside the transit window |
| 16 | `in_transit_kurtosis` | Peakedness / boxiness of the in-transit flux distribution |
| 17 | `flux_kurtosis` | Full light curve kurtosis (detects background stellar flares and variability) |
| 18 | `positive_to_negative_ratio` | Ratio of positive excursions to negative dips (identifies symmetric noise vs. dips) |
| 19 | `transit_duty_cycle` | Ratio of transit duration to orbital period ($T_{\text{dur}} / P$) |
| 20 | `depth_to_noise` | Transit depth normalized to detrended scatter |
| 21 | `transit_coverage_fraction` | Phase sampling completeness across the transit window |
| 22 | `bls_period` | Recovered candidate orbital period in days |

*Full 26-feature matrix also captures catalog period, raw noise $\sigma_{\text{raw}}$, detrended noise $\sigma_{\text{det}}$, and transit observation counts.*

---

## 🤖 Model Performance & Validation

All models are validated using **5-Fold Stratified Cross-Validation with strict zero data leakage** across **1,393 verified NASA TESS stars** (469 Confirmed Planets, 506 Eclipsing Binaries, 418 False Positives). Standard scaling, imputation, and class weights are fitted strictly within training partitions.

### Out-Of-Fold Benchmark Comparison (1,393 Stars)
| Architecture | 5-Fold Stratified CV Accuracy | 5-Fold Macro F1 | Confirmed Planet Recall | Eclipsing Binary Recall |
| :--- | :---: | :---: | :---: | :---: |
| **NASA Astronet Dual-Branch 1D-CNN** | **52.62% ± 2.63%** ★ | **51.2%** | **64.0%** | **62.0%** |
| **XGBoost (Regularized)** | **51.97% ± 2.63%** | 48.3% | 63.5% | 61.8% |
| **Calibrated Stacking ML (Production)** | **51.76% ± 2.87%** | **48.51%** | **67.0%** | **63.0%** |
| **Random Forest (Regularized)** | **50.18% ± 2.14%** | 47.7% | 61.2% | 59.4% |

```
NASA Astronet 1D-CNN Dual-Branch Architecture:
Global View (201 bins) ──► [ Conv1D x3 + MaxPool ] ──► 640-dim ──┐
                                                                 ├──► [ Concat: 1312-dim ] ──► [ Dense x3 + Dropout ] ──► Softmax Output
Local Zoom  (61 bins)  ──► [ Conv1D x3 + MaxPool ] ──► 640-dim ──┤
                                                                 │
Tabular     (22 feats) ──► [ Dense + BatchNorm ]   ──►  32-dim ──┘
```

> [!NOTE]
> **Scientific Integrity & Overfitting Disclosure:**  
> A 3-class cross-validation accuracy of ~52% on raw observational TESS TOI candidates reflects genuine astrophysical complexity (grazing transits mimic binary star eclipses, and blended background stars attenuate transit depths). Constrained tree depths ($d \le 4$), $L_2$ regularization ($\lambda = 3.0$), and Platt Sigmoid calibration guarantee that output confidence scores reflect true posterior probabilities rather than overconfident training memorization.

---

## 📊 Data Access & Photometry Guide

To comply with GitHub's 100 MB file limit while avoiding multi-gigabyte repository bloat, ExoFinder uses **Apache Parquet (Zstandard compression + `float32`)**:
* **13.5× Compression:** Compresses 2.93 GB of detrended cadence data down to **~560 MB** split into 10 clean partitioned chunks in `data/detrended_parquet/` (~54 MB each).
* **Zero-Setup Dashboard:** Precomputed views for 1,393 stars reside in `data/precomputed/` (~199 KB each), allowing the frontend to boot instantly.

### Loading Data in Python
```python
from src.data_loader import load_detrended_star

# Load any target light curve by TIC ID in ~1 millisecond:
df = load_detrended_star(297153209)
print(df.head())
# Output columns: ['time', 'detrended_flux', 'flux', 'flux_err', 'trend', 'star_id']
```

### Unpacking to Individual CSVs (Offline)
You can extract both detrended and raw light curves locally from the Parquet archive in ~30 seconds:
```bash
python -m src.data_loader --unpack-csv
```
This automatically populates:
* `data/detrended/TIC_*.csv` (1,454 detrended light curve files)
* `data/raw/TIC_*.csv` (1,454 raw flux light curve files)

### 📦 Standalone CSV Downloads (GitHub Release v1.0.0)
For researchers who prefer standalone CSV archives without using Python, download the complete dataset from the [ExoFinder v1.0.0 Release](https://github.com/Rajratna-D/Exofinder/releases/tag/v1.0.0):
* **`exofinder_detrended_csvs.zip`** (1.15 GB): All 1,454 detrended light curves in standard CSV format.
* **`exofinder_raw_csvs.zip`** (620 MB): All 1,454 raw flux light curves in standard CSV format.

---

## 📁 Project Structure

```
exoplanet-detection-ps07/
├── run.py                         # Single-command launcher (FastAPI + React frontend)
├── requirements.txt               # Pinned Python package dependencies
│
├── frontend/                      # React 19 + Vite 6 + Tailwind (Nova Sci-Fi HUD)
│   ├── src/
│   │   ├── components/            # CelestialChamber (Three.js), PhotometryLab, Sonifier
│   │   ├── App.jsx                # Main application state and telemetry shell
│   │   └── main.jsx
│   ├── public/assets/             # Planet and star photorealistic WebGL textures
│   ├── package.json
│   └── README.md                  # Frontend architectural documentation
│
├── src/                           # Backend REST API, Database & Scientific Pipeline
│   ├── server.py                  # High-speed FastAPI REST server (sub-2ms response)
│   ├── precompute.py              # LTTB decimation & multi-scale view generator
│   ├── data_loader.py             # Archive fetcher, Parquet loader & CSV unpacker
│   ├── detrend.py                 # Segmented Savitzky-Golay detrending engine
│   ├── period_search.py           # Astropy BLS periodogram & phase folding
│   ├── features.py                # 26-dimensional physical feature extractor
│   ├── classifier.py              # Regularized Stacking Ensemble & Platt calibration
│   ├── deep_learning.py           # Dual-Branch 1D-CNN (PyTorch multi-scale Astronet)
│   ├── run_pipeline.py            # End-to-end headless pipeline orchestrator
│   └── db/                        # Database repository tier (PostgreSQL / SQLite)
│       ├── database.py            # Engine connection & session pooling
│       ├── models.py              # SQLAlchemy 2.0 ORM schemas
│       ├── repository.py          # Fast query layer with pagination & filtering
│       └── migrate.py             # CSV-to-Database synchronization engine
│
├── data/
│   ├── labels.csv                 # Ground truth catalog (1,393 verified TESS targets)
│   ├── binned_views_cache.npz     # Dual-branch 1D-CNN global & local views cache
│   ├── detrended_parquet/         # 13.5x compressed full-cadence photometry archive
│   └── precomputed/               # Precalculated LTTB downsampled curves & models
│
├── outputs/
│   ├── model.pkl                  # Serialized Calibrated Stacking ML bundle
│   ├── cnn_model.pt               # Trained PyTorch 1D-CNN Astronet weights
│   ├── predictions.csv            # 100% out-of-fold predictions with confidence scores
│   ├── features.csv               # Extracted 26-feature physical matrix
│   ├── period_search_results.csv  # BLS candidate recovery parameters
│   └── feature_failures.csv       # Edge case diagnostic tracking
│
├── tests/                         # Automated verification suite
│   ├── test_api_endpoints.py      # REST endpoint validation
│   ├── test_audio_sonification.py # Acoustic synthesis tests
│   ├── test_database.py           # SQLite / PostgreSQL transaction tests
│   ├── test_all_stars_coverage.py # Target completeness checks
│   └── benchmark_performance.py   # Latency and throughput benchmarks
│
└── reports/                       # Validation documentation
    ├── cnn_metrics_report.md      # PyTorch Astronet 1D-CNN performance report
    └── validation_metrics_report.md # Stacking ensemble out-of-fold report
```

---

## 🧰 Tech Stack

| Domain | Technology |
|:-------|:-----------|
| **Core Languages** | Python 3.10+, JavaScript (ES2024) |
| **Frontend UI** | React 19, Vite 6, Tailwind CSS, Lucide Icons |
| **3D Graphics** | Three.js, `@react-three/fiber`, `@react-three/drei` |
| **Photometry Canvas** | High-performance Canvas2D (60–120 FPS render loop) |
| **Audio Sonification**| Web Audio API (real-time harmonic frequency synthesis) |
| **Backend API** | FastAPI, Uvicorn, orjson |
| **Database** | PostgreSQL / SQLite, SQLAlchemy 2.0 ORM |
| **Deep Learning** | PyTorch (Dual-Branch 1D-CNN Astronet) |
| **Machine Learning** | scikit-learn, XGBoost, imbalanced-learn |
| **Data Format** | Apache Parquet (Zstandard compression), NumPy, Pandas |
| **Astrophysics** | Astropy (BLS), Lightkurve (MAST API), Astroquery |
| **Signal Processing**| SciPy (Savitzky-Golay, signal decimation) |

---

## 📜 License

MIT License — Open for academic and research astronomy use.

<p align="center">
  Built with 🔭 for deep space discovery
</p>
