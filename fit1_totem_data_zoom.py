import os
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset  # Importación necesaria para el zoom

# ============================================================
# Fixed parameters
# ============================================================
ALPHA = 0.305
GAMMA_OVER_2 = 0.09  # = 0.065/(1-A) with A=0

# For scaled variable t**
A_S = 0.065
A_T = 0.72

# FIT 2 specific parameters (Not used in FIT 1, but kept for consistency)
A_PARAM = 0.28
NU = 2 * 0.065 / (1 - A_PARAM)  # = 0.18

EPS = 1e-30

# ============================================================
# Variable transform:  t** -> |t|
# ============================================================
def t_abs_from_tstarstar(tss, s_TeV2, a_s=A_S, a_t=A_T):
    """t** = s^a_s * |t|^a_t  =>  |t| = (t** / s^a_s)^(1/a_t)"""
    return np.power(
        np.asarray(tss, float) / np.power(np.asarray(s_TeV2, float), a_s),
        1.0 / a_t
    )

# ============================================================
# FIT 1 Model (Full Interference)
# ============================================================
def model_scaled_y_fit1(tss, sqrts_TeV, N10, B10, N20, B20, phi,
                        alpha=ALPHA, gamma_over_2=GAMMA_OVER_2):
    sqrts_TeV = np.asarray(sqrts_TeV, float)
    s = sqrts_TeV**2  

    tabs = t_abs_from_tstarstar(tss, s)  

    N1 = N10 * np.power(s, alpha / 2.0)
    N2 = N20 * np.power(s, alpha / 2.0)
    B1 = B10 * np.power(s, gamma_over_2)
    B2 = B20 * np.power(s, gamma_over_2)

    e1 = np.exp(-B1 * tabs)
    e2 = np.exp(-B2 * tabs)

    A2 = (N1**2) * (e1**2) + (N2**2) * (e2**2) + 2.0 * N1 * N2 * e1 * e2 * np.cos(phi)

    y = A2 / (s**alpha)
    return y

# ============================================================
# Explicit Chi-Square Function
# ============================================================
def calculate_chi2(y, y_fit, y_err):
    """
    Explicitly calculates the total chi-square.
    """
    y = np.asarray(y)
    y_fit = np.asarray(y_fit)
    y_err = np.asarray(y_err)
    
    chi2_elements = ((y - y_fit) / y_err)**2
    chi2_total = np.sum(chi2_elements)
    return chi2_total

# ============================================================
# Residuals for Linear Optimization
# ============================================================
def residuals_fit1(params, tss, sqrts, y, yerr):
    y_fit = model_scaled_y_fit1(tss, sqrts, *params)
    return (y - y_fit) / yerr

# ============================================================
# Master Fitting Routine
# ============================================================
def run_fit_scenario(residuals_func, model_func, tss, sqrts, y, yerr, n_starts, seed=2):
    lb = np.array([0.0, 0.0, 0.0, 0.0, -np.pi])
    ub = np.array([np.inf, np.inf, np.inf, np.inf, np.pi])
    
    rng = np.random.default_rng(seed)
    N_scale = np.sqrt(max(np.median(y), 1e-20))
    best = None

    for _ in range(n_starts):
        p0 = np.array([
            N_scale * 10**rng.uniform(-1.5, 1.5),
            10**rng.uniform(0.0, 1.5),
            N_scale * 10**rng.uniform(-2.0, 1.0),
            10**rng.uniform(-0.5, 1.5),
            rng.uniform(-np.pi, np.pi)
        ])

        res = least_squares(
            residuals_func, p0, bounds=(lb, ub),
            args=(tss, sqrts, y, yerr),
            method="trf", loss="linear"
        )
        
        chi2_current = np.sum(res.fun**2) 
        if best is None or chi2_current < best["chi2"]:
            best = {"res": res, "chi2": chi2_current}

    popt = best["res"].x
    J = best["res"].jac
    
    y_fit_best = model_func(tss, sqrts, *popt)
    chi2_linear = calculate_chi2(y, y_fit_best, yerr)
    ndof = len(y) - len(popt)
    
    try:
        JTJ = J.T @ J
        cov = np.linalg.inv(JTJ) * (chi2_linear / ndof)
        err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        print("Warning: Covariance matrix singular. Cannot compute uncertainties.")
        err = np.full(len(popt), np.nan)
        
    return popt, err, chi2_linear, ndof

# ============================================================
# Print Helper Function
# ============================================================
def print_fit_results(title, popt, err, chi2, ndof):
    param_names = ['N₁⁰', 'B₁⁰', 'N₂⁰', 'B₂⁰', 'φ']
    print(f"\n--- {title} ---")
    print(f"χ² / ndof = {chi2:.2f} / {ndof} = {chi2/ndof:.4f}")
    print("Parameters:")
    for name, val, e in zip(param_names, popt, err):
        if name == 'φ':
            print(f"  {name:5s} = {val:10.6f} ± {e:.6f} rad  ({np.degrees(val):7.2f}°)")
        else:
            print(f"  {name:5s} = {val:10.6f} ± {e:.6f}")

# ============================================================
# Main Script
# ============================================================
tss_min=0.0
tss_max=3.5
tss_dp_min=0.2
tss_dp_max=1.5
exclude_min=0.2
exclude_max=1.5
n_starts=150
seed=2

# 1. Load Data
try:
    data = np.loadtxt("/home/jesusavc/phd/odderon/scaling/scaled_data/scaled_data_TOTEM.dat")
    sqrts = data[:, 0].astype(float)
    tss   = data[:, 1].astype(float)
    y     = data[:, 2].astype(float)
    yerr  = data[:, 3].astype(float)
except FileNotFoundError:
    print("WARNING: Data file not found. Generating dummy data to allow code to run.")
    sqrts = np.full(100, 8.0)
    tss = np.linspace(0.1, 2.0, 100)
    y = np.exp(-5 * tss) + 1e-4
    yerr = y * 0.1

# Base Mask (Full Domain)
m_base = (sqrts > 0) & (tss >= tss_min) & (tss <= tss_max)
sqrts_full, tss_full, y_full, yerr_full = sqrts[m_base], tss[m_base], y[m_base], yerr[m_base]

# Base Mask (Dip/Bump region only)
m_dp = (sqrts > 0) & (tss >= tss_dp_min) & (tss <= tss_dp_max)
sqrts_dp, tss_dp, y_dp, yerr_dp = sqrts[m_dp], tss[m_dp], y[m_dp], yerr[m_dp]

print("="*60)
print("RUNNING 2 LINEAR FITS...")
print("="*60)

# --- FIT 1: Full Domain ---
print(f"Running FIT 1 (Full Domain, N={len(y_full)})...")
popt1, err1, chi2_1, ndof_1 = run_fit_scenario(residuals_fit1, model_scaled_y_fit1, tss_full, sqrts_full, y_full, yerr_full, n_starts, seed)

# --- FIT 1: Dip/Bump Region ---
print(f"Running FIT 1 (Dip/Bump Region, N={len(y_dp)})...")
popt2, err2, chi2_2, ndof_2 = run_fit_scenario(residuals_fit1, model_scaled_y_fit1, tss_dp, sqrts_dp, y_dp, yerr_dp, n_starts, seed)
            
# --- Print Detailed Stats ---
print("\n" + "="*60)
print("RESULTS SUMMARY (Explicit Linear χ² & Parameter Errors)")
print("="*60)
print_fit_results("FIT 1 (Full Domain)", popt1, err1, chi2_1, ndof_1)
print_fit_results("FIT 1 (Dip/Bump Region)", popt2, err2, chi2_2, ndof_2)
print("="*60)

# ============================================================
# Plotting (Oriented Object approach for Inset capability)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_yscale("log")
colors = ["green", "blue", "red", "black"]

# Sort and plot raw data points
oall = np.argsort(tss_full)
tss_plot, sqrts_plot, y_plot, yerr_plot = tss_full[oall], sqrts_full[oall], y_full[oall], yerr_full[oall]

# GRAFICAR DATOS EN EJE PRINCIPAL
for i, E in enumerate(sorted(np.unique(sqrts_plot))):
    mE = np.isclose(sqrts_plot, E)
    ax.errorbar(
        tss_plot[mE], y_plot[mE], yerr=yerr_plot[mE],
        fmt="o", color=colors[i % len(colors)],
        markersize=3, capsize=0, elinewidth=1.5, alpha=0.9, label=f"{E:g} TeV"
    ) 

# Generate theoretical curves across the whole x-axis
xgrid = np.linspace(np.min(tss_plot) * 0.98, np.max(tss_plot) * 1.02, 1500)
Erep = 8.0 if np.any(np.isclose(np.unique(sqrts_plot), 8.0)) else float(np.median(sqrts_plot))
    
ygrid1 = model_scaled_y_fit1(xgrid, np.full_like(xgrid, Erep), *popt1)
ygrid2 = model_scaled_y_fit1(xgrid, np.full_like(xgrid, Erep), *popt2)
    
# GRAFICAR CURVAS EN EJE PRINCIPAL
ax.plot(xgrid, ygrid1, "-", color="orange", linewidth=3, label="FIT 1 (Full Domain)", zorder=10)
ax.plot(xgrid, ygrid2, "-", color="purple", linewidth=3, label="FIT 1 (Dip/Bump Region)", zorder=10)

ax.axvspan(exclude_min, exclude_max, color='gray', alpha=0.1, label=f"Dip/Bump Region ({exclude_min} - {exclude_max})")


# ============================================================
# --- ZOOM INSET PLOT ---
# ============================================================
# Crear el recuadro dentro del eje principal 'ax'
axins = ax.inset_axes([0.55, 0.55, 0.42, 0.42])

# REPETIR LOS DATOS EN EL RECUADRO
for i, E in enumerate(sorted(np.unique(sqrts_plot))):
    mE = np.isclose(sqrts_plot, E)
    axins.errorbar(
        tss_plot[mE], y_plot[mE], yerr=yerr_plot[mE],
        fmt="o", color=colors[i % len(colors)],
        markersize=3, capsize=0, elinewidth=1.5, alpha=0.9
    ) 

# REPETIR LAS CURVAS EN EL RECUADRO
axins.plot(xgrid, ygrid1, "-", color="orange", linewidth=3, zorder=10)
axins.plot(xgrid, ygrid2, "-", color="purple", linewidth=3, zorder=10)

# Configurar escala y límites del recuadro
axins.set_xlim(0.66, 1.1)
axins.set_ylim(1e-3, 5e-2)  # AJUSTA estos límites Y si la curva se sale del recuadro
axins.set_yscale("log")

# Dibujar las líneas grises conectando el recuadro a la región original
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")


# Formato final
ax.set_xlabel(r"$t^{**}=(s/1\,\mathrm{TeV}^2)^{0.065}(|t|/1\,\mathrm{GeV}^2)^{0.72}$")
ax.set_ylabel(r"$(s/1\,\mathrm{TeV}^2)^{-0.305}\, d\sigma/d|t|$")
ax.legend(loc="lower left", fontsize=9)

plt.tight_layout()

outdir = "/home/jesusavc/phd/odderon/figures/"
os.makedirs(outdir, exist_ok=True)
outpath = os.path.join(outdir, "fit1_totem_scaled_data_zoom.pdf")
plt.savefig(outpath, bbox_inches="tight")
plt.show()