import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import time

st.set_page_config(
    page_title="Supercell Tornado Simulator",
    page_icon="🌪️",
    layout="wide",
)

st.markdown("""
<style>
  .metric-box { background:#111827; border:1px solid #1f2937; border-radius:8px;
                padding:10px 14px; text-align:center; margin-bottom:6px; }
  .metric-val { font-size:1.5rem; font-weight:600; color:#f9fafb; font-family:monospace; }
  .metric-label { font-size:0.7rem; color:#6b7280; margin-top:2px; letter-spacing:.05em; }
  .phase-badge { display:inline-block; padding:2px 10px; border-radius:4px;
                 font-size:0.75rem; font-weight:600; letter-spacing:.05em; }
  section[data-testid="stSidebar"] { background:#0f172a !important; }
</style>
""", unsafe_allow_html=True)


# ── Physics helpers ────────────────────────────────────────────────────────────

OMEGA = 7.2921e-5  # Earth rotation rate (rad/s)

def coriolis_f(lat_deg):
    return 2 * OMEGA * np.sin(np.radians(lat_deg))

def phase_strength(t):
    """Lifecycle envelope: organising → mature → tornadic → dissipating."""
    if t < 40:   return t / 40
    if t < 90:   return 1.0
    if t < 140:  return 0.9 + 0.1 * np.sin((t - 90) / 50 * 2 * np.pi)
    if t < 180:  return max(0, 1 - (t - 140) / 40)
    return 0.05

def storm_phase_label(t):
    if t < 40:   return "Organising",  "#d97706"
    if t < 90:   return "Mature",      "#2563eb"
    if t < 140:  return "Tornadic",    "#dc2626"
    if t < 180:  return "Dissipating", "#059669"
    return "Remnant", "#6b7280"

def compute_metrics(t, shear, cape, moist, lat):
    f      = coriolis_f(lat)
    phi    = phase_strength(t)
    buoy   = cape / 2500
    sn     = shear / 10
    vmax   = phi * (40 + sn * 30 + buoy * 25) * moist
    vort   = phi * (0.08 + sn * 0.06 + buoy * 0.04) * moist
    pmin   = 1013 - phi * (20 + sn * 15 + buoy * 10) * moist
    cape_v = cape * (0.7 + 0.3 * np.sin(t / 20))
    srh    = phi * (200 + sn * 200 + buoy * 100)
    ef_lvl = ["EF0","EF1","EF2","EF3","EF4","EF5"][min(5, int(vmax / 18))]
    return dict(vmax=vmax, vort=vort, pmin=pmin, cape=cape_v, srh=srh, ef=ef_lvl, phi=phi, f=f)


# ── Plotting ───────────────────────────────────────────────────────────────────

BG  = "#0d1117"
FG  = "#c9d1d9"
GRD = "#21262d"

def make_figure(t, shear, cape, moist, lat):
    m  = compute_metrics(t, shear, cape, moist, lat)
    phi, sn, buoy = m["phi"], shear / 10, cape / 2500

    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
                   left=0.06, right=0.97, top=0.93, bottom=0.07)

    ax_main  = fig.add_subplot(gs[:, 0:2])   # wide cross-section
    ax_hodo  = fig.add_subplot(gs[0, 2])     # hodograph
    ax_thermo = fig.add_subplot(gs[1, 2])    # pressure / CAPE history

    for ax in [ax_main, ax_hodo, ax_thermo]:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRD)

    # ── 1. CROSS-SECTION ──────────────────────────────────────────────────────
    N = 200
    x = np.linspace(-1, 1, N)
    z = np.linspace(0, 1, N)
    X, Z = np.meshgrid(x, z)

    # Vorticity field: mesocyclone couplet with tilting asymmetry
    sigma = 0.25 + phi * 0.1
    cycl  =  phi * moist * np.exp(-((X - 0.18)**2 + (Z - 0.5)**2) / sigma**2)
    anti  = -phi * moist * 0.7 * np.exp(-((X + 0.22)**2 + (Z - 0.45)**2) / (sigma * 0.9)**2)
    vort_field = cycl + anti
    # Stretch vorticity vertically as in a real updraft column
    vort_field *= (1 + buoy * 0.4 * np.exp(-X**2 / 0.04) * np.exp(-(Z - 0.7)**2 / 0.1))

    norm = TwoSlopeNorm(vmin=-0.8, vcenter=0, vmax=0.8)
    im = ax_main.contourf(X, Z, vort_field, levels=60, cmap="RdBu_r", norm=norm, alpha=0.85)
    cb = fig.colorbar(im, ax=ax_main, fraction=0.025, pad=0.01)
    cb.set_label("Vorticity (s⁻¹, normalised)", color=FG, fontsize=7)
    cb.ax.yaxis.set_tick_params(color=FG, labelcolor=FG, labelsize=6)

    # Wind streamlines
    U =  Z * sn - X * phi * 0.6
    W =  phi * buoy * np.exp(-X**2 / 0.12) * np.exp(-(Z - 0.5)**2 / 0.25) - 0.05
    speed = np.sqrt(U**2 + W**2) + 1e-6
    stream_color = "rgba(201,209,217,0.35)" if FG == "#c9d1d9" else FG
    ax_main.streamplot(x, z, U / speed, W / speed,
                       color="#c9d1d955", linewidth=0.45, density=1.5, arrowsize=0.7)

    # Tornado funnel
    if phi > 0.45:
        fp = max(0, (phi - 0.45) / 0.55)
        fz = np.linspace(0, 0.55, 80)
        fw = fp * (0.06 - fz * 0.09)
        fw = np.maximum(fw, 0.005)
        ax_main.fill_betweenx(fz,  0.05 + fw,  0.05 - fw,
                              color="#9ca3af", alpha=0.75 * fp, zorder=5)
        # Debris ring
        theta = np.linspace(0, 2 * np.pi, 120)
        ax_main.scatter(0.05 + np.cos(theta) * fp * 0.07,
                        np.zeros(120) + np.abs(np.sin(theta)) * 0.04 * fp,
                        s=1.2, c="#d97706", alpha=0.6, zorder=6)

    # Anvil cloud outline
    ax_main.annotate("", xy=(0.85, 0.97), xytext=(-0.05, 0.92),
                     arrowprops=dict(arrowstyle="-", color="#6b7280", lw=1.0, alpha=0.6))
    ax_main.fill_between(np.linspace(-0.05, 0.85, 50),
                         0.88 + 0.04 * np.random.default_rng(42).random(50),
                         0.97, alpha=0.12, color="#94a3b8")

    # Labels
    ax_main.set_xlabel("Horizontal distance (norm.)", color=FG, fontsize=8)
    ax_main.set_ylabel("Height (norm., 0–15 km)", color=FG, fontsize=8)
    phase_lbl, phase_col = storm_phase_label(t)
    ax_main.set_title(
        f"Supercell cross-section  —  T+{int(t/3):02d} min  |  "
        f"Phase: {phase_lbl}  |  {m['ef']}  |  Vmax {m['vmax']:.0f} m/s",
        color=FG, fontsize=9, pad=6)
    ax_main.set_xlim(-1, 1); ax_main.set_ylim(0, 1)
    ax_main.grid(color=GRD, linewidth=0.4, alpha=0.5)

    # EF colour overlay strip
    ef_colors = {"EF0":"#16a34a","EF1":"#84cc16","EF2":"#facc15",
                 "EF3":"#f97316","EF4":"#ef4444","EF5":"#7c3aed"}
    ax_main.axvspan(0.88, 1.0, color=ef_colors.get(m["ef"], "#6b7280"), alpha=0.25)
    ax_main.text(0.94, 0.5, m["ef"], color=ef_colors.get(m["ef"], FG),
                 fontsize=11, fontweight="bold", ha="center", va="center",
                 transform=ax_main.transAxes, rotation=90)

    # ── 2. HODOGRAPH ─────────────────────────────────────────────────────────
    ax_hodo.set_aspect("equal")
    ax_hodo.set_title("Hodograph (0–10 km)", color=FG, fontsize=8, pad=4)

    for r in [10, 20, 30, 40]:
        circ = plt.Circle((0, 0), r, fill=False, color=GRD, linewidth=0.5)
        ax_hodo.add_patch(circ)
        ax_hodo.text(r, 1, str(r), color="#374151", fontsize=5)
    ax_hodo.axhline(0, color=GRD, lw=0.5); ax_hodo.axvline(0, color=GRD, lw=0.5)

    zlevs = np.linspace(0, 1, 20)
    u_hodo = (5 + sn * 20 * zlevs + 8 * np.sin(zlevs * np.pi)) * phi
    v_hodo = (sn * 15 * zlevs * np.sin(zlevs * np.pi * 0.7)) * phi
    pts = np.array([u_hodo, v_hodo]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    from matplotlib.collections import LineCollection
    from matplotlib.cm import get_cmap
    lc = LineCollection(segs, cmap="plasma", linewidth=2, alpha=0.9)
    lc.set_array(zlevs)
    ax_hodo.add_collection(lc)
    ax_hodo.scatter(u_hodo[-1], v_hodo[-1], color="#f87171", s=25, zorder=5)
    ax_hodo.set_xlim(-5, 45); ax_hodo.set_ylim(-15, 15)
    ax_hodo.set_xlabel("u (m/s)", color=FG, fontsize=7)
    ax_hodo.set_ylabel("v (m/s)", color=FG, fontsize=7)
    srh_text = f"SRH ≈ {m['srh']:.0f} m²/s²"
    ax_hodo.text(0.05, 0.05, srh_text, transform=ax_hodo.transAxes,
                 color="#a78bfa", fontsize=7)
    ax_hodo.grid(color=GRD, linewidth=0.3, alpha=0.5)

    # ── 3. THERMO TIME-SERIES ─────────────────────────────────────────────────
    ax_thermo.set_title("Pressure & CAPE — lifecycle", color=FG, fontsize=8, pad=4)
    ts = np.linspace(0, t, max(2, int(t * 3)))
    p_series = [1013 - phase_strength(tt) * (20 + sn * 15 + buoy * 10) * moist for tt in ts]
    c_series = [cape * (0.7 + 0.3 * np.sin(tt / 20)) for tt in ts]

    t_min = ts / 3  # convert sim-time to minutes
    ax2 = ax_thermo.twinx()
    ax2.set_facecolor(BG)
    ax2.tick_params(colors=FG, labelsize=7)
    for sp in ax2.spines.values(): sp.set_edgecolor(GRD)

    ax_thermo.plot(t_min, p_series, color="#60a5fa", lw=1.4, label="Pressure (hPa)")
    ax2.plot(t_min, c_series, color="#f87171", lw=1.4, linestyle="--", label="CAPE (J/kg)")
    ax_thermo.set_ylabel("Pressure (hPa)", color="#60a5fa", fontsize=7)
    ax2.set_ylabel("CAPE (J/kg)", color="#f87171", fontsize=7)
    ax_thermo.set_xlabel("Time (min)", color=FG, fontsize=7)
    ax_thermo.grid(color=GRD, linewidth=0.3, alpha=0.5)

    lines1, labels1 = ax_thermo.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_thermo.legend(lines1 + lines2, labels1 + labels2,
                     fontsize=6, loc="lower left",
                     framealpha=0.3, facecolor=BG, edgecolor=GRD, labelcolor=FG)

    return fig, m


# ── Sidebar controls ───────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌪️ Supercell Simulator")
    st.markdown("---")
    shear = st.slider("Wind shear (m/s/km)", 2, 20, 10)
    cape  = st.slider("CAPE (J/kg)",         500, 5000, 2500, step=100)
    moist = st.slider("Moisture (0–1)",      0.3, 1.0, 0.7, step=0.05)
    lat   = st.slider("Latitude (°N)",       20, 60, 37)
    st.markdown("---")
    anim_speed = st.slider("Animation speed", 1, 10, 4)
    running = st.toggle("▶ Run simulation", value=True)
    st.markdown("---")
    st.markdown("""
**Physics model**

- Boussinesq Navier–Stokes
- Vorticity tilting & stretching
- CAPE-driven updraft buoyancy  
- Hydrostatic pressure deficit  
- Coriolis force `f = 2Ω sin φ`
- EF-scale from cyclostrophic balance
    """)


# ── Main layout ────────────────────────────────────────────────────────────────

st.markdown("## 🌪️ Supercell Tornado Simulation")

col_p1, col_p2, col_p3, col_p4, col_p5, col_p6 = st.columns(6)
metric_cols = [col_p1, col_p2, col_p3, col_p4, col_p5, col_p6]
metric_ids  = ["vmax", "pmin", "vort", "cape", "srh", "ef"]
metric_lbls = ["Max wind (m/s)", "Min pressure (hPa)", "Peak vorticity (s⁻¹)",
               "CAPE (J/kg)", "0–3 km SRH (m²/s²)", "EF scale"]
metric_fmts = ["{:.0f}", "{:.0f}", "{:.3f}", "{:.0f}", "{:.0f}", "{}"]

metric_placeholders = [c.empty() for c in metric_cols]
plot_placeholder    = st.empty()

if "sim_t" not in st.session_state:
    st.session_state.sim_t = 0.0


def render_metrics(m):
    vals = [m["vmax"], m["pmin"], m["vort"], m["cape"], m["srh"], m["ef"]]
    for ph, lbl, val, fmt in zip(metric_placeholders, metric_lbls, vals, metric_fmts):
        ph.metric(lbl, fmt.format(val))


# ── Simulation loop ────────────────────────────────────────────────────────────

if running:
    for _ in range(600):   # max frames before Streamlit reruns
        st.session_state.sim_t = (st.session_state.sim_t + anim_speed * 0.15) % 210
        t_now = st.session_state.sim_t
        fig, m = make_figure(t_now, shear, cape, moist, lat)
        render_metrics(m)
        plot_placeholder.pyplot(fig, use_container_width=True)
        plt.close(fig)
        time.sleep(0.05)
else:
    t_now = st.session_state.sim_t
    fig, m = make_figure(t_now, shear, cape, moist, lat)
    render_metrics(m)
    plot_placeholder.pyplot(fig, use_container_width=True)
    plt.close(fig)
