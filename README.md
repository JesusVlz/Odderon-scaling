# Odderon determination from Scaling in Elastic *pp* and *p̄p* Cross-Sections

Code and data for extracting evidence of a negative-signature (Odderon) component
of the elastic scattering amplitude from the scaling properties of *pp* and *p̄p*
differential cross-sections at LHC and Tevatron energies.

The analysis builds on the scaling law of Baldenegro, Royon & Staśto
([Phys. Lett. B **830** (2022) 137141](https://arxiv.org/abs/2204.08328)) and
applies the S-matrix Regge energy-to-phase relation to separate the positive
(Pomeron) and negative (Odderon) signature contributions.

## Physics summary

Elastic *pp* differential cross-sections measured by TOTEM at 2.76, 7, 8 and
13 TeV collapse onto a universal curve when expressed through the scaling
variable

```
t** = (s / 1 TeV²)^0.065 · (|t| / 1 GeV²)^0.72
```

with the rescaled cross-section `(s/1 TeV²)^(-0.305) dσ/d|t|`. The elastic
amplitude is fitted with a double-exponential form:

- **Fit 1** — full amplitude `A`, with a fixed interference phase `φ`
  (reproduces the dip/bump structure).
- **Fit 2** — positive-signature amplitude `A₊` only, where the Regge formalism
  imposes a *t*-dependent phase shift on `θ` and `φ`.

The Odderon amplitude is then obtained as `A₋ = A − A₊`, and the *p̄p*
prediction as `A_{p̄p} = 2A₊ − A`. These parameter-free predictions are compared
to the D0 *p̄p* measurement at 1.96 TeV.

## Repository structure

```
.
├── fit1_totem_data_zoom.py        # Fit 1: full interference amplitude A
├── fit2_totem_data_zoom.py        # Fit 2: positive-signature amplitude A+
├── scaling/                       # Processed data in the scaling framework
└── plotting/                      # Extraction and figure-production scripts
    ├── extraction_odd.py              # Odderon cross-section dσ_-/d|t| at 1.96 TeV
    ├── extraction_ppbar_zoom.py       # p̄p prediction vs. D0 data (Fit 1/Fit 2)
    ├── extraction_ppbar_phi.py        # Impact of the free sign of φ (Fit 1)
    ├── negative_signatures.py         # Odderon dσ_-/d|t| from 0.5 to 13 TeV
    ├── odderon_tstarstar.py           # Odderon cross-section in the t** variable
    └── ratios_tstarstar_first_method.py  # Odderon/pp and Odderon/p̄p ratios
```

## Requirements

- Python ≥ 3.9
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`

```bash
pip install numpy scipy matplotlib pandas
```

## Usage

Each script is standalone. The fitting scripts perform a multi-start
least-squares fit (TOTEM scaling plane) and produce a figure with a zoomed inset
of the dip/bump region:

```bash
python fit1_totem_data_zoom.py     # Fit 1
python fit2_totem_data_zoom.py     # Fit 2
```

The plotting scripts compute the Odderon and *p̄p* predictions from the fitted
parameters, with 1σ uncertainty bands obtained by Monte Carlo propagation of the
parameter errors:

```bash
python plotting/extraction_ppbar_zoom.py
python plotting/negative_signatures.py
python plotting/odderon_tstarstar.py
python plotting/ratios_tstarstar_first_method.py
```

## Configuration notes

- **Hard-coded paths.** Several scripts read data from and write figures to
  absolute paths under `/home/jesusavc/phd/odderon/...`. Before running, update
  the input data paths (e.g. the `np.loadtxt`/`pd.read_csv` calls) and the
  `outdir` figure directory to match your environment. The fitting scripts fall
  back to dummy data if the input file is not found.
- **Fixed scaling constants.** `α = 0.305` and `γ/2 ≈ 0.09` are imposed from the
  scaling fit; `θ = −0.09 rad` is fixed to satisfy `ρ(t=0) ≈ 0.10`.
- **Fit parameters** for both fits (full domain and dip/bump region only) are
  documented in the manuscript Table I / Appendix and are hard-coded as
  dictionaries in the `plotting/` scripts.

## References

1. C. Baldenegro, C. Royon, A. M. Staśto, *Scaling properties of elastic
   proton-proton scattering at LHC energies*, Phys. Lett. B **830** (2022)
   137141, [arXiv:2204.08328](https://arxiv.org/abs/2204.08328).
2. V. M. Abazov *et al.* (D0 and TOTEM), *Odderon Exchange from Elastic
   Scattering Differences between pp and p̄p Data at 1.96 TeV*, Phys. Rev. Lett.
   **127** (2021) 062003, [arXiv:2012.03981](https://arxiv.org/abs/2012.03981).
