import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Fit Parameters (Dip/Bump Region - 1b & 2b)
# ============================================================
PARAMS_STD_b  = {'N10': 10.17, 'B10': 5.76, 'N20': 0.45, 'B20': 1.34, 'phi': -2.69, 'theta': -0.09}
PARAMS_PLUS_b = {'N10': 10.09, 'B10': 5.73, 'N20': 0.39, 'B20': 1.26, 'phi': -2.88, 'theta': -0.09}

ERRORS_STD_b = {'N10': 0.03, 'N20': 0.01, 'B10': 0.17, 'B20': 0.02, 'phi': 0.01, 'theta': 0.0}
ERRORS_PLUS_b = {'N10': 0.04, 'N20': 0.01, 'B10': 0.02, 'B20': 0.015, 'phi': 0.01, 'theta': 0.0}

# Regge / Amplitude parameters
alpha = 0.18
gamma_over_2 = 0.09

# Scaling parameters
A_S = 0.065
A_T = 0.72

# ============================================================
# Scaling Functions (Forward Mapping)
# ============================================================
def compute_tstarstar(t_GeV2, sqrts_TeV, a_s=A_S, a_t=A_T):
    s_TeV2 = sqrts_TeV**2
    return np.power(s_TeV2, a_s) * np.power(t_GeV2, a_t)

def N_scaled(s, N0): return N0 * (s)**(alpha/2.0)
def B_scaled(s, B0): return B0 * (s)**(gamma_over_2)

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

# ============================================================
# Cross Section Formulas
# ============================================================
def dsigma_pp(t, s, p_std):
    T1 = T_mag(t, s, p_std["N10"], p_std["B10"])
    T2 = T_mag(t, s, p_std["N20"], p_std["B20"])
    phi = p_std["phi"]
    return T1**2 + T2**2 + 2 * T1 * T2 * np.cos(phi)

def dsigma_minus(t, s, p_std, p_plus):
    T1 = T_mag(t, s, p_std["N10"], p_std["B10"])
    T2 = T_mag(t, s, p_std["N20"], p_std["B20"])
    
    T1_plus = T_mag(t, s, p_plus["N10"], p_plus["B10"])
    T2_plus = T_mag(t, s, p_plus["N20"], p_plus["B20"])

    phi = p_std["phi"]
    theta = p_std["theta"]
    
    th_tau = theta_tau(t, s, p_plus["B10"])
    ph_tau = phi_tau(t, s, p_plus["B10"], p_plus["B20"], p_plus["phi"])

    term1 = T1**2 + T1_plus**2 - 2 * T1 * T1_plus * np.cos(th_tau - theta)
    term2 = T2**2 + T2_plus**2 - 2 * T2 * T2_plus * np.cos(th_tau + ph_tau - theta - phi)
    term3 = 2 * T1 * T2 * np.cos(phi)
    term4 = 2 * T1_plus * T2_plus * np.cos(ph_tau)
    term5 = -2 * T1_plus * T2 * np.cos(th_tau - theta - phi)
    term6 = -2 * T1 * T2_plus * np.cos(theta - th_tau - ph_tau)

    return term1 + term2 + term3 + term4 + term5 + term6

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

# ============================================================
# Monte Carlo Error Propagation for Ratios
# ============================================================
def get_ratio_error_bands(t, s, p_std, p_plus, e_std, e_plus, n_samples=300):
    ratio_pp_samples = []
    ratio_ppbar_samples = []
    
    for _ in range(n_samples):
        # Sample parameters
        samp_std = {k: np.random.normal(p_std[k], e_std[k]) for k in p_std}
        samp_plus = {k: np.random.normal(p_plus[k], e_plus[k]) for k in p_plus}
        
        # Calculate cross sections for this random sample
        ds_pp = dsigma_pp(t, s, samp_std)
        ds_minus = dsigma_minus(t, s, samp_std, samp_plus)
        ds_ppbar = dsigma_ppbar(t, s, samp_std, samp_plus)
        
        # Calculate ratios for this sample
        ratio_pp_samples.append(ds_minus / ds_pp)
        ratio_ppbar_samples.append(ds_minus / ds_ppbar)
        
    ratio_pp_samples = np.array(ratio_pp_samples)
    ratio_ppbar_samples = np.array(ratio_ppbar_samples)
    
    # Calculate percentiles (1-sigma bounds)
    med_pp = np.percentile(ratio_pp_samples, 50, axis=0)
    low_pp = np.percentile(ratio_pp_samples, 16, axis=0)
    high_pp = np.percentile(ratio_pp_samples, 84, axis=0)
    
    med_ppbar = np.percentile(ratio_ppbar_samples, 50, axis=0)
    low_ppbar = np.percentile(ratio_ppbar_samples, 16, axis=0)
    high_ppbar = np.percentile(ratio_ppbar_samples, 84, axis=0)
    
    return med_pp, low_pp, high_pp, med_ppbar, low_ppbar, high_ppbar


# ============================================================
# Main Workflow
# ============================================================
sqrts = 1.96  # TeV
s_val = sqrts**2

# 1. Create the native |t| array
t_abs_all = np.linspace(0.01, 2.5, 400)

# 2. Evaluate physical ratios WITH ERRORS natively in |t|
(ratio_odderon_pp, ratio_pp_low, ratio_pp_high, 
 ratio_odderon_ppbar, ratio_ppbar_low, ratio_ppbar_high) = get_ratio_error_bands(
     t_abs_all, s_val, PARAMS_STD_b, PARAMS_PLUS_b, ERRORS_STD_b, ERRORS_PLUS_b, n_samples=300
 )

# 3. Scale the |t| array to t** for the x-axis plotting
t_starstar_all = compute_tstarstar(t_abs_all, sqrts)

# Define the valid Dip-Bump boundaries in the scaled variable t**
db_min_starstar = 0.2
db_max_starstar = 1.5

# Create Boolean Masks
mask_valid = (t_starstar_all >= db_min_starstar) & (t_starstar_all <= db_max_starstar)

# ============================================================
# TRUCO: Usar np.nan para cortar las líneas y evitar cruces
# ============================================================
# Arrays para las líneas sólidas (solo válidas, resto es NaN)
ratio_pp_solid = np.where(mask_valid, ratio_odderon_pp, np.nan)
ratio_pp_low_solid = np.where(mask_valid, ratio_pp_low, np.nan)
ratio_pp_high_solid = np.where(mask_valid, ratio_pp_high, np.nan)

ratio_ppbar_solid = np.where(mask_valid, ratio_odderon_ppbar, np.nan)
ratio_ppbar_low_solid = np.where(mask_valid, ratio_ppbar_low, np.nan)
ratio_ppbar_high_solid = np.where(mask_valid, ratio_ppbar_high, np.nan)

# Arrays para las líneas punteadas (solo inválidas, el centro es NaN)
ratio_pp_dotted = np.where(~mask_valid, ratio_odderon_pp, np.nan)
ratio_pp_low_dotted = np.where(~mask_valid, ratio_pp_low, np.nan)
ratio_pp_high_dotted = np.where(~mask_valid, ratio_pp_high, np.nan)

ratio_ppbar_dotted = np.where(~mask_valid, ratio_odderon_ppbar, np.nan)
ratio_ppbar_low_dotted = np.where(~mask_valid, ratio_ppbar_low, np.nan)
ratio_ppbar_high_dotted = np.where(~mask_valid, ratio_ppbar_high, np.nan)

# ============================================================
# Plotting
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

c_pp = '#1f77b4' # Azul
c_ppbar = '#d62728' # Rojo

# --- Plot Valid Region (Solid Lines & Stronger Fill) ---
ax.plot(t_starstar_all, ratio_pp_solid, '-', color=c_pp, lw=2.5, label=r'$Odderon / pp$')
ax.fill_between(t_starstar_all, ratio_pp_low_solid, ratio_pp_high_solid, color=c_pp, alpha=0.3, linewidth=0, edgecolor='none')

ax.plot(t_starstar_all, ratio_ppbar_solid, '-', color=c_ppbar, lw=2.5, label=r'$Odderon / p\bar{p}$')
ax.fill_between(t_starstar_all, ratio_ppbar_low_solid, ratio_ppbar_high_solid, color=c_ppbar, alpha=0.3, linewidth=0, edgecolor='none')


# --- Plot Extrapolated Region (Weak dotted lines & Weaker Fill) ---
ax.plot(t_starstar_all, ratio_pp_dotted, ':', color=c_pp, lw=2, alpha=0.4)
ax.fill_between(t_starstar_all, ratio_pp_low_dotted, ratio_pp_high_dotted, color=c_pp, alpha=0.1, linewidth=0, edgecolor='none')

ax.plot(t_starstar_all, ratio_ppbar_dotted, ':', color=c_ppbar, lw=2, alpha=0.4)
ax.fill_between(t_starstar_all, ratio_ppbar_low_dotted, ratio_ppbar_high_dotted, color=c_ppbar, alpha=0.1, linewidth=0, edgecolor='none')


# Highlight the dip/bump region
ax.axvspan(db_min_starstar, db_max_starstar, color='gray', alpha=0.1, 
           label=f"Dip/Bump Region ({db_min_starstar} < $t^{{**}}$ < {db_max_starstar})")

ax.set_xlabel(r"$t^{**}=(s/1\,\mathrm{TeV}^2)^{0.065}(|t|/1\,\mathrm{GeV}^2)^{0.72}$", fontsize=13)
ax.set_ylabel(rf"Differential Cross Section Ratio at $\sqrt{{s}} = {sqrts}$ TeV", fontsize=13)
ax.set_xlim(0, 1.6)
ax.set_ylim(0, 2.5) 

ax.legend(fontsize=10, loc='upper left', frameon=True)

plt.tight_layout()

# Save plot
outdir = "/home/jesusavc/phd/odderon/figures/"
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, "odderon_ratio_tstarstar.pdf"), dpi=150, bbox_inches="tight")

plt.show()