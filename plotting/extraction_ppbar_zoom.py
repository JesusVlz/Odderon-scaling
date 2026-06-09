import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# ===============================
# Total Amplitude Parameters (A)
# ===============================
PARAMS_STD_a = {'N10': 10.4,  'N20': 0.49, 'B10': 5.79, 'B20': 1.43, 'phi': 2.68,     'theta': -0.09}
PARAMS_STD_b = {'N10': 10.17, 'N20': 0.45, 'B10': 5.76, 'B20': 1.34, 'phi': -2.69,    'theta': -0.09}

# --- UNCERTAINTIES for STD ---
ERRORS_STD_a = {'N10': 0.04, 'N20': 0.01, 'B10': 0.17, 'B20': 0.014, 'phi': 0.01, 'theta': 0.0}
ERRORS_STD_b = {'N10': 0.04, 'N20': 0.01, 'B10': 0.17, 'B20': 0.014, 'phi': 0.01, 'theta': 0.0}

# ===============================
# Positive Signature Parameters (A+)
# ===============================
PARAMS_PLUS_a = {'N10': 10.22, 'N20': 0.49, 'B10': 5.92, 'B20': 1.43, 'phi': -1.95, 'theta': -0.09}
PARAMS_PLUS_b = {'N10': 10.09, 'N20': 0.39, 'B10': 5.73, 'B20': 1.26, 'phi': -2.88, 'theta': -0.09}

# --- UNCERTAINTIES for PLUS ---
ERRORS_PLUS_a = {'N10': 0.04, 'N20': 0.01, 'B10': 0.02, 'B20': 0.015, 'phi': 0.01, 'theta': 0.0}
ERRORS_PLUS_b = {'N10': 0.04, 'N20': 0.01, 'B10': 0.02, 'B20': 0.015, 'phi': 0.01, 'theta': 0.0}
# ===============================
# Regge parameters
# ===============================
alpha = 0.18
gamma_over_2 = 0.09

# For scaled variable t**
A_S = 0.065
A_T = 0.72

# ============================================================
# Variable transform:  t** -> |t|
# ============================================================
def t_abs_from_tstarstar(tss, s_TeV2, a_s=A_S, a_t=A_T):
    """t** = s^a_s * |t|^a_t  =>  |t| = (t** / s^a_s)^(1/a_t)"""
    return np.power(
        np.asarray(tss, float) / np.power(np.asarray(s_TeV2, float), a_s),
        1.0 / a_t
    )

def N_scaled(s, N0):
    return N0 * (s)**(alpha/2.0)

def B_scaled(s, B0):
    return B0 * (s)**(gamma_over_2)

def theta_tau(t, s, B1_plus_0):
    B1_plus_s = B_scaled(s, B1_plus_0)
    term = B1_plus_s * np.abs(t) * np.tan(gamma_over_2 * np.pi / 2.0)
    return term - (np.pi * alpha / 4.0)

def phi_tau(t, s, B1_plus_0, B2_plus_0, phi_plus):
    B1_plus_s = B_scaled(s, B1_plus_0)
    B2_plus_s = B_scaled(s, B2_plus_0)
    delta_B = B2_plus_s - B1_plus_s
    return phi_plus + delta_B * np.abs(t) * np.tan(gamma_over_2 * np.pi / 2.0)

def T_mag(t, s, N0, B0):
    N = N_scaled(s, N0)
    B = B_scaled(s, B0)
    return N * np.exp(-B * np.abs(t))

# ===============================
# Differential cross section (A_-)
# ===============================
def dsigma_ppbar(t, s, p_std, p_plus):
    T1 = T_mag(t, s, p_std["N10"], p_std["B10"])
    T2 = T_mag(t, s, p_std["N20"], p_std["B20"])
    
    T1_plus = T_mag(t, s, p_plus["N10"], p_plus["B10"])
    T2_plus = T_mag(t, s, p_plus["N20"], p_plus["B20"])

    phi = p_std["phi"]
    theta = p_std["theta"]
    
    th_tau = theta_tau(t, s, p_plus["B10"])
    ph_tau = phi_tau(t, s, p_plus["B10"], p_plus["B20"], p_plus["phi"])

    term1 = T1**2 + 4*T1_plus**2 - 4 * T1 * T1_plus * np.cos(th_tau - theta)
    term2 = T2**2 + 4*T2_plus**2 - 4 * T2 * T2_plus * np.cos(th_tau + ph_tau - theta - phi)
    term3 = 2 * T1 * T2 * np.cos(phi)
    term4 = 8 * T1_plus * T2_plus * np.cos(ph_tau)
    term5 = -4 * T1_plus * T2 * np.cos(th_tau - theta - phi)
    term6 = -4 * T1 * T2_plus * np.cos(theta - th_tau - ph_tau)

    return term1 + term2 + term3 + term4 + term5 + term6

# ===============================
# Differential cross section positive signature (A_+)
# ===============================
def dsigma_plus(t, s, p_plus):
    T1_plus = T_mag(t, s, p_plus["N10"], p_plus["B10"])
    T2_plus = T_mag(t, s, p_plus["N20"], p_plus["B20"])
    
    th_tau = theta_tau(t, s, p_plus["B10"])
    ph_tau = phi_tau(t, s, p_plus["B10"], p_plus["B20"], p_plus["phi"])

    term1 = T1_plus**2 
    term2 = T2_plus**2 
    term3 = 2*T1_plus*T2_plus * np.cos(ph_tau)
        
    return term1 + term2 + term3

# ===============================
# Monte Carlo Error Propagation
# ===============================
def get_error_band(t, s, p_std, p_plus, e_std, e_plus, n_samples=300):
    curves = []
    for _ in range(n_samples):
        samp_std = {k: np.random.normal(p_std[k], e_std[k]) for k in p_std}
        samp_plus = {k: np.random.normal(p_plus[k], e_plus[k]) for k in p_plus}
        curve = dsigma_ppbar(t, s, samp_std, samp_plus)
        curves.append(curve)
    curves = np.array(curves)
    lower_bound = np.percentile(curves, 16, axis=0)
    median_curve = np.percentile(curves, 50, axis=0)
    upper_bound = np.percentile(curves, 84, axis=0)
    return median_curve, lower_bound, upper_bound

def get_error_band_plus(t, s, p_plus, e_plus, n_samples=300):
    curves = []
    for _ in range(n_samples):
        samp_plus = {k: np.random.normal(p_plus[k], e_plus[k]) for k in p_plus}
        curve = dsigma_plus(t, s, samp_plus)
        curves.append(curve)
    curves = np.array(curves)
    lower_bound = np.percentile(curves, 16, axis=0)
    median_curve = np.percentile(curves, 50, axis=0)
    upper_bound = np.percentile(curves, 84, axis=0)
    return median_curve, lower_bound, upper_bound

# ===============================
# Load D0 ppbar data
# ===============================
try:
    d0_data = pd.read_csv('/home/jesusavc/phd/odderon/scaling/ppbar_data/D0_ppbar_data_1.96TeV.csv')
    t_star_d0 = d0_data['t_star'].values
    dsigma_d0 = d0_data['dsigma_dt'].values
    dsigma_err_d0 = d0_data['dsigma_dt_err'].values
    data_available = True
    print("D0 ppbar data loaded successfully!")
    print(f"  {len(t_star_d0)} √s = 1.96 TeV")
except FileNotFoundError:
    print("Warning: D0 data file not found. Plotting model only.")
    data_available = False

# ===============================
# Plotting
# ===============================
t = np.linspace(0.001, 1.6, 100)
sqrt_s = 1.96  # TeV
s = sqrt_s**2

db_min_starstar = 0.2 
db_max_starstar = 1.5
db_min_tabs = t_abs_from_tstarstar(db_min_starstar, s)  
db_max_tabs = t_abs_from_tstarstar(db_max_starstar, s)  

# Get median curves and error bands for Full Amplitude (-)
med_1, low_1, high_1 = get_error_band(t, s, PARAMS_STD_a, PARAMS_PLUS_a, ERRORS_STD_a, ERRORS_PLUS_a)
med_2, low_2, high_2 = get_error_band(t, s, PARAMS_STD_b, PARAMS_PLUS_b, ERRORS_STD_b, ERRORS_PLUS_b)

# Get median curves and error bands for Positive Signature (+)
med_plus_a, low_plus_a, high_plus_a = get_error_band_plus(t, s, PARAMS_PLUS_a, ERRORS_PLUS_a)
med_plus_b, low_plus_b, high_plus_b = get_error_band_plus(t, s, PARAMS_PLUS_b, ERRORS_PLUS_b)

# Set up figure and main axis explicitly
fig, ax = plt.subplots(figsize=(8,6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# --- MAIN PLOT ---
if data_available:
    ax.errorbar(t_star_d0, dsigma_d0, yerr=dsigma_err_d0,
                 fmt='s', color='black', markersize=4,
                 capsize=2, capthick=1.0, elinewidth=1.0,
                 label=r'D0  $\sqrt{s}=1.96$ TeV',
                 alpha=0.7, zorder=10)

ax.plot(t, med_1, linestyle='-', color=colors[0], label=r"$p\bar{p}$ (Full domain))")
ax.fill_between(t, low_1, high_1, color=colors[0], alpha=0.2)

ax.plot(t, med_2, linestyle='-', color=colors[1], label=r"$p\bar{p}$ (Only dip/bump)")
ax.fill_between(t, low_2, high_2, color=colors[1], alpha=0.2)

# Curve Plus A (Pomeron Only)
ax.plot(t, med_plus_a, linestyle=':', color=colors[2], label="Pomeron signature (Full domain)")
ax.fill_between(t, low_plus_a, high_plus_a, color=colors[2], alpha=0.2)

# Curve Plus B (Pomeron Only)
ax.plot(t, med_plus_b, linestyle=':', color=colors[3], label="Pomeron signature (Dip/Bump region)")
ax.fill_between(t, low_plus_b, high_plus_b, color=colors[3], alpha=0.2)

ax.axvspan(db_min_tabs, db_max_tabs, color='gray', alpha=0.1, label=f"Totem Dip/Bump ({db_min_tabs:.2f} - {db_max_tabs:.2f})")

ax.set_yscale("log")
ax.set_xlabel(r"$|t|$ (GeV$^2$)")
ax.set_ylabel(r"$d\sigma/d|t| \,[mb/GeV^2]$")
ax.legend(fontsize=9, loc='lower left', frameon=False)
#ax.set_xlim(0.170, 1.35)
ax.set_ylim(6e-4, 1e1)


# --- ZOOM INSET PLOT ---
# [x, y, width, height] of the inset in normalized coordinates (0 to 1)
# Adjust these numbers to move or resize the inset box within the main figure
axins = ax.inset_axes([0.55, 0.55, 0.42, 0.42])

# Plot exactly the same data on the inset
if data_available:
    axins.errorbar(t_star_d0, dsigma_d0, yerr=dsigma_err_d0,
                   fmt='s', color='black', markersize=4,
                   capsize=2, capthick=1.0, elinewidth=1.0, alpha=0.7, zorder=10)

axins.plot(t, med_1, linestyle='-', color=colors[0])
axins.fill_between(t, low_1, high_1, color=colors[0], alpha=0.2)

axins.plot(t, med_2, linestyle='-', color=colors[1])
axins.fill_between(t, low_2, high_2, color=colors[1], alpha=0.2)

# Curve Plus A (Pomeron Only)
axins.plot(t, med_plus_a, linestyle=':', color=colors[2])
axins.fill_between(t, low_plus_a, high_plus_a, color=colors[2], alpha=0.2)

# Curve Plus B (Pomeron Only)
axins.plot(t, med_plus_b, linestyle=':', color=colors[3])
axins.fill_between(t, low_plus_b, high_plus_b, color=colors[3], alpha=0.2)


# Set limits and scale for the inset to focus on the dip (0.6 < |t| < 1.0)
axins.set_xlim(0.56, 1.0)
axins.set_ylim(5e-3, 5e-2) # You may need to adjust these limits slightly to perfectly fit your data curve's height
axins.set_yscale("log")

# Optional: Disable tick labels on the inset if you want a cleaner look
# axins.set_xticklabels([])
# axins.set_yticklabels([])

# Draw connecting lines from the main plot to the zoomed box
# loc1, loc2 determine which corners are connected
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")

plt.tight_layout()

# Save plot
outdir = "/home/jesusavc/phd/odderon/figures/"
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, "extraction_ppbar_zoom.pdf"), dpi=150, bbox_inches="tight")
                
plt.show()