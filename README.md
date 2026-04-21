# 🌪️ Supercell Tornado Simulator

A physically-grounded supercell tornado simulation with real-time animated plots.

## Physics model
- **Boussinesq Navier–Stokes** vorticity equations
- **Vorticity tilting & stretching** from wind shear → updraft interaction
- **CAPE-driven buoyancy** for updraft intensity
- **Coriolis force** `f = 2Ω sin(φ)` — adjustable by latitude
- **Hydrostatic / cyclostrophic pressure deficit** → EF-scale estimate
- Full storm lifecycle: Organising → Mature → Tornadic → Dissipating

## Run locally

```bash
# 1. Navigate to the project folder
cd tornado_sim

# 2. Create a virtual environment
uv venv

# 3. Activate it
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 4. Install dependencies
uv pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Deploy to Streamlit Cloud (free)

1. Push this folder to a GitHub repo (public or private).
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New app** → select your repo → set main file to `app.py`.
4. Click **Deploy** — live URL in ~60 seconds.

## Controls (sidebar)
| Slider | Effect |
|---|---|
| Wind shear (m/s/km) | Controls vorticity tilting; higher = stronger mesocyclone |
| CAPE (J/kg) | Convective instability; drives updraft & vortex stretching |
| Moisture | Scales storm intensity and debris cloud |
| Latitude (°N) | Sets Coriolis parameter f; weaker near equator |
| Animation speed | Frames per second scaling |
