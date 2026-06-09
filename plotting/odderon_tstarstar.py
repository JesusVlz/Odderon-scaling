import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# Parámetros del Ajuste (Región Dip/Bump - 1b & 2b)
# ============================================================
PARAMS_STD_b  = {'N10': 10.17, 'B10': 5.76, 'N20': 0.45, 'B20': 1.34, 'phi': -2.69, 'theta': -0.09}
PARAMS_PLUS_b = {'N10': 10.09, 'B10': 5.73, 'N20': 0.39, 'B20': 1.26, 'phi': -2.88, 'theta': -0.09}

# --- INCERTIDUMBRES ---
ERRORS_STD_b = {'N10': 0.03, 'N20': 0.01, 'B10': 0.17, 'B20': 0.02, 'phi': 0.01, 'theta': 0.0}
ERRORS_PLUS_b = {'N10': 0.04, 'N20': 0.01, 'B10': 0.02, 'B20': 0.015, 'phi': 0.01, 'theta': 0.0}

# Parámetros de Regge / Amplitud
alpha_regge = 0.18
gamma_over_2 = 0.09

# Parámetros de Escalado
A_S = 0.065
A_T = 0.72
ALPHA_SCALING = 0.305  # Exponente para escalar la sección eficaz transversal

# ============================================================
# Funciones de Escalado
# ============================================================
def compute_tstarstar(t_GeV2, sqrts_TeV, a_s=A_S, a_t=A_T):
    s_TeV2 = sqrts_TeV**2
    return np.power(s_TeV2, a_s) * np.power(t_GeV2, a_t)

def compute_scaled_y(dsigma_dt, sqrts_TeV, alpha=ALPHA_SCALING):
    s_TeV2 = sqrts_TeV**2
    return dsigma_dt / np.power(s_TeV2, alpha)

# Funciones de amplitud
def N_scaled(s, N0): return N0 * (s)**(alpha_regge/2.0)
def B_scaled(s, B0): return B0 * (s)**(gamma_over_2)

def theta_tau(t, s, B1_plus_0):
    B1_plus_s = B_scaled(s, B1_plus_0)
    term = B1_plus_s * np.abs(t) * np.tan(gamma_over_2 * np.pi / 2.0)
    return term - (np.pi * alpha_regge / 4.0)

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
# Sección Eficaz del Odderón (A_-)
# ============================================================
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

# ============================================================
# Propagación de Errores (Monte Carlo)
# ============================================================
def get_error_band_minus(t, s, p_std, p_plus, e_std, e_plus, n_samples=300):
    """
    Realiza un muestreo aleatorio de los parámetros dentro de su 
    incertidumbre (1-sigma) para calcular la banda de error del Odderón.
    """
    curves = []
    for _ in range(n_samples):
        # Muestreo gaussiano de los parámetros
        samp_std = {k: np.random.normal(p_std[k], e_std[k]) for k in p_std}
        samp_plus = {k: np.random.normal(p_plus[k], e_plus[k]) for k in p_plus}
        
        curve = dsigma_minus(t, s, samp_std, samp_plus)
        curves.append(curve)
        
    curves = np.array(curves)
    
    # Calcular los percentiles (banda 1-sigma central)
    lower_bound = np.percentile(curves, 16, axis=0)
    median_curve = np.percentile(curves, 50, axis=0)
    upper_bound = np.percentile(curves, 84, axis=0)
    
    return median_curve, lower_bound, upper_bound

# ============================================================
# Flujo Principal de Trabajo
# ============================================================
sqrts = 1.96  # TeV (Referencia Tevatron)
s_val = sqrts**2

# 1. Crear el array en |t| nativo
t_abs_all = np.linspace(0.01, 2.5, 500)

# 2. Evaluar la sección eficaz con INCERTIDUMBRES
med_ds, low_ds, high_ds = get_error_band_minus(t_abs_all, s_val, PARAMS_STD_b, PARAMS_PLUS_b, ERRORS_STD_b, ERRORS_PLUS_b, n_samples=300)

# 3. Escalar ambas variables para el gráfico
t_starstar_all = compute_tstarstar(t_abs_all, sqrts)
y_med_scaled = compute_scaled_y(med_ds, sqrts)
y_low_scaled = compute_scaled_y(low_ds, sqrts)
y_high_scaled = compute_scaled_y(high_ds, sqrts)

# ============================================================
# Máscaras y preparación visual (Truco NaN)
# ============================================================
# Limites de validez en t**
db_min_starstar = 0.2
db_max_starstar = 1.5
mask_valid = (t_starstar_all >= db_min_starstar) & (t_starstar_all <= db_max_starstar)

# Arrays para la región sólida (Válida)
y_med_solid = np.where(mask_valid, y_med_scaled, np.nan)
y_low_solid = np.where(mask_valid, y_low_scaled, np.nan)
y_high_solid = np.where(mask_valid, y_high_scaled, np.nan)

# Arrays para la región punteada (Extrapolación)
y_med_dotted = np.where(~mask_valid, y_med_scaled, np.nan)
y_low_dotted = np.where(~mask_valid, y_low_scaled, np.nan)
y_high_dotted = np.where(~mask_valid, y_high_scaled, np.nan)

# ============================================================
# Graficar
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

# --- REGIÓN VÁLIDA (Línea sólida y sombra más densa) ---
ax.plot(t_starstar_all, y_med_solid, '-', color='purple', lw=2.5, label=r'Odderon signature')
ax.fill_between(t_starstar_all, y_low_solid, y_high_solid, color='purple', alpha=0.3, linewidth=0, edgecolor='none')

# --- REGIÓN EXTRAPOLADA (Línea punteada y sombra tenue) ---
ax.plot(t_starstar_all, y_med_dotted, ':', color='purple', lw=2, alpha=0.5)
ax.fill_between(t_starstar_all, y_low_dotted, y_high_dotted, color='purple', alpha=0.1, linewidth=0, edgecolor='none')

# Sombrear visualmente la región Dip/Bump
ax.axvspan(db_min_starstar, db_max_starstar, color='gray', alpha=0.1, 
           label=f"Dip/Bump Region ({db_min_starstar} < $t^{{**}}$ < {db_max_starstar})",
           linewidth=0, edgecolor='none')

# Formato del gráfico
ax.set_yscale("log")
ax.set_xlabel(r"$t^{**}=(s/1\,\mathrm{TeV}^2)^{0.065}(|t|/1\,\mathrm{GeV}^2)^{0.72}$", fontsize=13)
ax.set_ylabel((r"$(s/1\,\mathrm{TeV}^2)^{-0.305}\, d\sigma/d|t|$"), fontsize=13)
ax.set_xlim(0, 1.6)

# Asegurar que los límites en y acomoden los errores (ajusta si es necesario)
# ax.set_ylim(1e-4, 1e1) 

ax.legend(fontsize=11, loc='upper right', frameon=False)
#ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()

# Guardar figura
outdir = "/home/jesusavc/phd/odderon/figures/"
os.makedirs(outdir, exist_ok=True)
plt.savefig(os.path.join(outdir, "odderon_cross_section_scaled_con_error.pdf"), dpi=150, bbox_inches="tight")

plt.show()