# 🪐 ExoFinder Nova UI (Frontend Dashboard)

The **Nova UI** is an interactive, cinematic Sci-Fi HUD dashboard for deep space transit photometry and exoplanet candidate vetting. It pairs with the high-speed FastAPI backend to provide real-time 3D simulation, interactive light curve analysis, and AI explainability.

---

## 🚀 Key Features & Components

* **🪐 Celestial Chamber (`CelestialChamber.jsx`)**:
  * Physics-grounded **Three.js WebGL 3D Transit Simulator**.
  * Renders dynamic host stars with solar limb darkening, orbital paths, and transiting companions accurately scaled to BLS transit depths and periods.
* **📈 Photometry Lab (`PhotometryLab.jsx`)**:
  * 60–120 FPS high-frame-rate Canvas photometric visualizer.
  * Interactive switching between raw cadence, detrended time series, phase-folded transit profiles, and Mandel-Agol theoretical transit overlays.
* **🎵 Audio Sonification Player (`SonificationPlayer.jsx`)**:
  * Optical-to-acoustic sound synthesis powered by the Web Audio API.
  * Maps stellar flux dips to audible frequencies, allowing auditory detection of subtle exoplanet transit signatures.
* **🧠 AI Diagnostics HUD (`AIDiagnostics.jsx`)**:
  * Side-by-side consensus analysis comparing **Calibrated Stacking ML** and **NASA Astronet 1D-CNN**.
  * Visualizes odd/even transit depth symmetry, U-vs-V shape ratios, SNR metrics, and confidence distributions.
* **🔭 Target Catalog (`TargetCatalog.jsx`)**:
  * Instant search and filtering across verified NASA TESS targets by class (`confirmed_planet`, `eclipsing_binary`, `false_positive`), SNR, period, and transit depth.
  * Target bookmarking and persistent observational notes.

---

## 🛠️ Tech Stack

* **Framework:** React 19 + Vite 6
* **Styling:** Tailwind CSS + Sci-Fi Glassmorphism HUD Design System
* **3D Graphics:** Three.js, `@react-three/fiber`, `@react-three/drei`
* **Audio:** Web Audio API (real-time harmonic oscillator synthesis)
* **Icons:** Lucide React

---

## 💻 Development Commands

From the `frontend/` directory:

```bash
# Install dependencies
npm install

# Start Vite development server (http://localhost:5173)
npm run dev

# Build optimized production bundle
npm run build

# Preview production build locally
npm run preview
```

> **Unified Launcher:** You can also run the full stack (FastAPI backend + React frontend) simultaneously with a single command from the project root:  
> `python run.py`
