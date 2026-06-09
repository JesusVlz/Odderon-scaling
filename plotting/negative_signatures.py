import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ===============================
# Total Amplitude Parameters (A)
# ===============================
PARAMS_STD_a = {'N10': 10.4,  'N20': 0.49, 'B10': 5.79, 'B20': 1.43, 'phi': 2.68,     'theta': -0.09}
PARAMS_STD_b = {'N10': 10.17, 'N20': 0.45, 'B10': 5.76, 'B20': 1.34, 'phi': -2.69,    'theta': -0.09}

# --- UNCERTAINTIES for STD 
ERRORS_STD_a = {'N10': 0.04, 'N20': 0.01, 'B10': 0.17, 'B20': 0.02, 'phi': 0.01, 'theta': 0.0}
ERRORS_STD_b = {'N10': 0.03, 'N20': 0.01, 'B10': 0.17, 'B20': 0.02, 'phi': 0.01, 'theta': 0.0}

# ===============================
# Positive Signature Parameters (A+)
# ===============================
PARAMS_PLUS_a = {'N10': 10.22, 'N20': 0.49, 'B10': 5.92, 'B20': 1.43, 'phi': -1.95, 'theta': -0.09}
PARAMS_PLUS_b = {'N10': 10.09, 'N20': 0.39, 'B10': 5.73, 'B20': 1.26, 'phi': -2.88, 'theta': -0.09}

# --- UNCERTAINTIES for PLUS 
ERRORS_PLUS_a = {'N10': 0.04, 'N20': 0.01, 'B10': 0.02, 'B20': 0.01, 'phi': 0.01, 'theta': 0.0}
ERRORS_PLUS_b = {'N10': 0.03, 'N20': 0.01, 'B10': 0.02, 'B20': 0.02, 'phi': 0.01, 'theta': 0.0}

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

# ===============================
# Monte Carlo Error Propagation
# ===============================
def get_error_band(t, s, p_std, p_plus, e_std, e_plus, n_samples=300):
    curves = []
    for _ in range(n_samples):
        samp_std = {k: np.random.normal(p_std[k], e_std[k]) for k in p_std}
        samp_plus = {k: np.random.normal(p_plus[k], e_plus[k]) for k in p_plus}
        
        curve = dsigma_minus(t, s, samp_std, samp_plus)
        curves.append(curve)
        
    curves = np.array(curves)
    
    lower_bound = np.percentile(curves, 16, axis=0)
    upper_bound = np.percentile(curves, 84, axis=0)
    median_curve = np.percentile(curves, 50, axis=0)
    
    return median_curve, lower_bound, upper_bound

# ===============================
# Plotting
# ===============================
fig, ax = plt.subplots(figsize=(8, 6))

t = np.linspace(0.001, 2.8, 300) 

# Energies to loop through
energies_TeV = [0.5, 1.96, 5.0, 7.0, 10.0, 13.0]

colors = ["#008BFD",  '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Limites fijos en el sistema escalado t**
db_min_starstar = 0.2
db_max_starstar = 1.5

for sqrt_s, color in zip(energies_TeV, colors):
    s = sqrt_s**2
    
    # Calcular la mediana y las bandas de error para todo el rango t
    med, low, high = get_error_band(t, s, PARAMS_STD_b, PARAMS_PLUS_b, ERRORS_STD_b, ERRORS_PLUS_b)
    
    # 1. Encontrar los límites en |t| para esta energía específica
    t_bound_min = t_abs_from_tstarstar(db_min_starstar, s)
    t_bound_max = t_abs_from_tstarstar(db_max_starstar, s)
    
    # Crear una máscara para la región válida (dentro de los boundaries)
    mask_valid = (t >= t_bound_min) & (t <= t_bound_max)
    
    # 2. GRAFICAR EL FONDO GRIS (La curva completa extrapolada)
    ax.plot(t, med, linestyle='-', color='gray', alpha=0.4)
    ax.fill_between(t, low, high, color='gray', alpha=0.15, linewidth=0, edgecolor='none')
    
    # 3. GRAFICAR LA REGIÓN VÁLIDA (A color, superpuesta sobre el gris)
    med_valid = np.where(mask_valid, med, np.nan)
    low_valid = np.where(mask_valid, low, np.nan)
    high_valid = np.where(mask_valid, high, np.nan)
    
    ax.plot(t, med_valid, linestyle='-', color=color, lw=2, label=rf"$\sqrt{{s}}={sqrt_s}$ TeV")
    ax.fill_between(t, low_valid, high_valid, color=color, alpha=0.3, linewidth=0, edgecolor='none')
    
    # 4. LÍNEAS VERTICALES en los boundaries
    ax.axvline(x=t_bound_min, color=color, linestyle='--', alpha=0.6, linewidth=1.5)
    ax.axvline(x=t_bound_max, color=color, linestyle='--', alpha=0.6, linewidth=1.5)

# Formato del Gráfico
ax.set_yscale("log")
ax.set_xlabel(r"$|t|$ (GeV$^2$)", fontsize=13)
ax.set_ylabel(r"$d\sigma_{-}/d|t| \,[mb/GeV^2]$", fontsize=13)
ax.set_xlim(0, 2.2)

ax.legend(fontsize=9, loc='upper right', frameon=True)

plt.tight_layout()

# Guardar y mostrar
outdir = "/home/jesusavc/phd/odderon/figures/"
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, "odderon_xc_energies.pdf"), dpi=150, bbox_inches="tight")
                
plt.show()