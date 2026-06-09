#Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import least_squares
import pandas as pd


def plot_scaled_data_combined(
    fname_totem="/home/jesusavc/phd/odderon/scaling/scaled_data/scaled_data_TOTEM.dat",
    fname_d0="/home/jesusavc/phd/odderon/scaling/scaled_data/D0_data_1.96TeV_scaled.csv"
):
    """
    Plot TOTEM and D0 scaled data on the same figure.
    
    Args:
        fname_totem: path to TOTEM .dat file
        fname_d0: path to D0 .csv or .dat file
    """
    
    # ========================================================
    # Load TOTEM data
    # ========================================================
    data_totem = np.loadtxt(fname_totem)
    sqrts_totem = data_totem[:, 0]
    tau_totem   = data_totem[:, 1]
    y_totem     = data_totem[:, 2]
    yerr_totem  = data_totem[:, 3]
    
    # ========================================================
    # Load D0 data
    # ========================================================
    if fname_d0.endswith('.csv'):
        # CSV file with header
        data_d0 = np.loadtxt(fname_d0, delimiter=',', skiprows=1)
        sqrts_d0 = data_d0[:, 0]
        tau_d0   = data_d0[:, 2]  # tstarstar column
        y_d0     = data_d0[:, 3]  # y column
        yerr_d0  = data_d0[:, 4]  # y_err column
    else:
        # DAT file without header
        data_d0 = np.loadtxt(fname_d0)
        sqrts_d0 = data_d0[:, 0]
        tau_d0   = data_d0[:, 1]
        y_d0     = data_d0[:, 2]
        yerr_d0  = data_d0[:, 3]
    
    # ========================================================
    # Plot
    # ========================================================
    # Colors for TOTEM energies
    colors_totem = ["green", "blue", "red", "black"]
    
    plt.figure(figsize=(8, 6))
    plt.yscale("log")
    
    # Plot TOTEM data
    for i, E in enumerate(sorted(np.unique(sqrts_totem))):
        mE = np.isclose(sqrts_totem, E)
        plt.errorbar(
            tau_totem[mE], y_totem[mE], yerr=yerr_totem[mE],
            fmt="o",
            color=colors_totem[i % len(colors_totem)],
            markersize=3,
            capsize=0,
            elinewidth=1.5,
            alpha=0.9,
            label=f"TOTEM {E:g} TeV"
        )
    
    # Plot D0 data (use different marker style)
    for i, E in enumerate(sorted(np.unique(sqrts_d0))):
        mE = np.isclose(sqrts_d0, E)
        plt.errorbar(
            tau_d0[mE], y_d0[mE], yerr=yerr_d0[mE],
            fmt="v",
            color="orange",
            markersize=7,
            capsize=0,
            elinewidth=2,
            alpha=0.9,
            label=f"D0 {E:g} TeV",
            zorder=10
        )
    
    plt.xlabel(r"$t^{**}=(s/1\,\mathrm{TeV}^2)^{0.065}(|t|/1\,\mathrm{GeV}^2)^{0.72}$", fontsize=12)
    plt.ylabel(r"$(s/1\,\mathrm{TeV}^2)^{-0.305}\, d\sigma/d|t|$ (mb/GeV$^2$)", fontsize=12)
   # plt.title("Scaled differential cross sections: TOTEM + D0", fontsize=13)
    plt.legend(fontsize=10)
    #plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    outdir = "/home/jesusavc/phd/odderon/figures/"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "scaled_data_TOTEM_D0_combined.pdf")
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    print(f"Plot saved to: {outpath}")
    plt.show()


def plot_scaled_data_separate(
    fname_totem="/home/jesusavc/phd/odderon/scaling/scaled_data_TOTEM.dat",
    fname_d0="/home/jesusavc/phd/odderon/scaling/D0_data_1.96TeV_scaled.csv"
):
    """
    Plot TOTEM and D0 data in separate subplots side by side.
    """
    
    # Load TOTEM data
    data_totem = np.loadtxt(fname_totem)
    sqrts_totem = data_totem[:, 0]
    tau_totem   = data_totem[:, 1]
    y_totem     = data_totem[:, 2]
    yerr_totem  = data_totem[:, 3]
    
    # Load D0 data
    if fname_d0.endswith('.csv'):
        data_d0 = np.loadtxt(fname_d0, delimiter=',', skiprows=1)
        sqrts_d0 = data_d0[:, 0]
        tau_d0   = data_d0[:, 2]
        y_d0     = data_d0[:, 3]
        yerr_d0  = data_d0[:, 4]
    else:
        data_d0 = np.loadtxt(fname_d0)
        sqrts_d0 = data_d0[:, 0]
        tau_d0   = data_d0[:, 1]
        y_d0     = data_d0[:, 2]
        yerr_d0  = data_d0[:, 3]
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors_totem = ["green", "blue", "red", "black"]
    
    # ========================================================
    # Left panel: TOTEM
    # ========================================================
    ax1.set_yscale("log")
    for i, E in enumerate(sorted(np.unique(sqrts_totem))):
        mE = np.isclose(sqrts_totem, E)
        ax1.errorbar(
            tau_totem[mE], y_totem[mE], yerr=yerr_totem[mE],
            fmt="o",
            color=colors_totem[i % len(colors_totem)],
            markersize=3,
            capsize=0,
            elinewidth=1.5,
            alpha=0.9,
            label=f"{E:g} TeV"
        )
    
    ax1.set_xlabel(r"$t^{**}=(s/1\,\mathrm{TeV}^2)^{0.065}(|t|/1\,\mathrm{GeV}^2)^{0.72}$", fontsize=12)
    ax1.set_ylabel(r"$(s/1\,\mathrm{TeV}^2)^{-0.305}\, d\sigma/d|t|$", fontsize=12)
    ax1.set_title("TOTEM", fontsize=13)
    ax1.legend(fontsize=10)
    #ax1.grid(True, alpha=0.3)
    
    # ========================================================
    # Right panel: D0
    # ========================================================
    ax2.set_yscale("log")
    for i, E in enumerate(sorted(np.unique(sqrts_d0))):
        mE = np.isclose(sqrts_d0, E)
        ax2.errorbar(
            tau_d0[mE], y_d0[mE], yerr=yerr_d0[mE],
            fmt="s",
            color="orange",
            markersize=5,
            capsize=0,
            elinewidth=2,
            alpha=0.9,
            label=f"{E:g} TeV",
            zorder=10
        )
    
    ax2.set_xlabel(r"$t^{**}=(s/1\,\mathrm{TeV}^2)^{0.065}(|t|/1\,\mathrm{GeV}^2)^{0.72}$", fontsize=12)
    ax2.set_ylabel(r"$(s/1\,\mathrm{TeV}^2)^{-0.305}\, d\sigma/d|t|$", fontsize=12)
    ax2.set_title("D0", fontsize=13)
    ax2.legend(fontsize=10)
    #ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    outdir = "/home/jesusavc/phd/odderon/figures/"
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "scaled_data_TOTEM_D0_separate.pdf")
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    print(f"Plot saved to: {outpath}")
    plt.show()


if __name__ == "__main__":
    # Option 1: Combined plot (both on same axes)
    print("Creating combined plot...")
    plot_scaled_data_combined(
        fname_totem="/home/jesusavc/phd/odderon/scaling/scaled_data/scaled_data_TOTEM.dat",
        fname_d0="/home/jesusavc/phd/odderon/scaling/scaled_data/D0_data_1.96TeV_scaled.csv"
    )
    
    # Option 2: Separate subplots (side by side)
    print("\nCreating separate subplots...")
    plot_scaled_data_separate(
        fname_totem="/home/jesusavc/phd/odderon/scaling/scaled_data/scaled_data_TOTEM.dat",
        fname_d0="/home/jesusavc/phd/odderon/scaling/scaled_data/D0_data_1.96TeV_scaled.csv"
    )