"""
================================================================================
 OPEN CHANNEL HYDRAULICS ANALYSIS SUITE
 Trapezoidal Channel : Uniform Flow + Specific Energy + Gradually Varied Flow
================================================================================

Pipeline (executed in this exact order):

    [User Input Module]
            |
            v
    [Geometry & Uniform Flow Module]  --> yn, yc, Fr
            |
            v
    [Specific Energy Module]          --> E-y data, (E_min, yc) point
            |
            v
    [Gradually Varied Flow Module]    --> RK4 solution of dy/dx = (S0-Sf)/(1-Fr^2)
            |
            v
    [Final Dashboard Visualization]   --> Dual-panel Matplotlib plot

Trapezoidal section geometry (bottom width B, side slope z (H:V = z:1)):
    A(y) = (B + z*y) * y            Area
    T(y) = B + 2*z*y                Top width
    P(y) = B + 2*y*sqrt(1+z^2)      Wetted perimeter
    R(y) = A(y) / P(y)              Hydraulic radius
================================================================================
"""

import os
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


# ==============================================================================
# MODULE 1 : USER INPUT
# ==============================================================================
def get_user_input():
    """
    Collects the consolidated numerical input template.
    Press Enter on any prompt to accept the shown default value.
    Returns a dict with every parameter needed downstream.
    """

    def ask(prompt, default):
        raw = input(f"{prompt} [default = {default}]: ").strip()
        return float(raw) if raw else default

    print("=" * 70)
    print(" OPEN CHANNEL FLOW ANALYSIS -- INPUT TEMPLATE")
    print("=" * 70)

    print("\n-- Channel Geometry --")
    B = ask("Bottom width B (m)", 4.0)
    z = ask("Side slope z (H:V = z:1, use 0 for rectangular)", 1.5)
    S0 = ask("Bed slope S0 (m/m)", 0.001)

    print("\n-- Fluid / Flow Properties --")
    Q = ask("Discharge Q (m^3/s)", 20.0)
    n = ask("Manning's roughness coefficient n", 0.025)
    g = ask("Gravitational acceleration g (m/s^2)", 9.81)

    print("\n-- GVF Boundary Conditions --")
    y_start = ask("Known depth y_start at control structure (m)", 3.0)
    L = ask("Total channel stretch length L (m)", 1000.0)
    dx = ask("Step size dx (m)", 10.0)

    return dict(B=B, z=z, S0=S0, Q=Q, n=n, g=g,
                y_start=y_start, L=L, dx=dx)


# ==============================================================================
# MODULE 2 : GEOMETRY & UNIFORM FLOW
# ==============================================================================
def geometry_props(y, B, z):
    """Returns (A, T, P, R) for a trapezoidal section at depth y."""
    y = np.maximum(y, 1e-9)  # guard against non-positive depth
    A = (B + z * y) * y
    T = B + 2.0 * z * y
    P = B + 2.0 * y * np.sqrt(1.0 + z ** 2)
    R = A / P
    return A, T, P, R


def _manning_residual(y, Q, n, B, z, S0):
    """Residual of Manning's equation: Q - (1/n) A R^(2/3) S0^(1/2) = 0."""
    y = y[0] if hasattr(y, "__len__") else y
    A, T, P, R = geometry_props(y, B, z)
    Q_calc = (1.0 / n) * A * R ** (2.0 / 3.0) * S0 ** 0.5
    return [Q - Q_calc]


def calculate_yn(Q, n, B, z, S0, guess=1.0):
    """Solves Manning's equation for normal depth yn via fsolve (Newton-type)."""
    yn = fsolve(_manning_residual, x0=[guess], args=(Q, n, B, z, S0))[0]
    return float(yn)


def _critical_residual(y, Q, g, B, z):
    """Residual of critical flow condition: Q^2 T - g A^3 = 0."""
    y = y[0] if hasattr(y, "__len__") else y
    A, T, P, R = geometry_props(y, B, z)
    return [Q ** 2 * T - g * A ** 3]


def calculate_yc(Q, g, B, z, guess=1.0):
    """Solves the critical-depth condition Q^2*T/(g*A^3) = 1 via fsolve."""
    yc = fsolve(_critical_residual, x0=[guess], args=(Q, g, B, z))[0]
    return float(yc)


def froude_number(y, Q, g, B, z):
    """Fr = V / sqrt(g * A / T)  =  (Q/A) / sqrt(g * A / T)."""
    A, T, P, R = geometry_props(y, B, z)
    V = Q / A
    Fr = V / np.sqrt(g * A / T)
    return Fr


# ==============================================================================
# MODULE 3 : SPECIFIC ENERGY MAPPING
# ==============================================================================
def specific_energy(y, Q, g, B, z):
    """E(y) = y + Q^2 / (2 g A^2)."""
    A, T, P, R = geometry_props(y, B, z)
    return y + Q ** 2 / (2.0 * g * A ** 2)


def build_energy_curve(Q, g, B, z, yn, yc, n_points=400):
    """
    Builds the E-y dataset over y_range = [0.1*yc, 3*yn] (widened slightly
    to also comfortably bracket yc itself), and returns the (E_min, yc) point.
    """
    y_lo = 0.1 * yc
    y_hi = max(3.0 * yn, 3.0 * yc)  # ensure yc's branch is fully visible too
    y_range = np.linspace(y_lo, y_hi, n_points)
    E_range = specific_energy(y_range, Q, g, B, z)

    E_min = specific_energy(yc, Q, g, B, z)  # true minimum specific energy
    return y_range, E_range, (E_min, yc)


def alternate_depth(E_target, Q, g, B, z, guess):
    """Finds the other root y (alternate depth) sharing the same specific energy."""
    def residual(y):
        y = y[0] if hasattr(y, "__len__") else y
        return [specific_energy(y, Q, g, B, z) - E_target]
    y_alt = fsolve(residual, x0=[guess])[0]
    return float(y_alt)


# ==============================================================================
# MODULE 4 : GRADUALLY VARIED FLOW (GVF) -- RK4 ODE SOLVER
# ==============================================================================
def friction_slope(y, Q, n, B, z):
    """Sf = n^2 Q^2 / (A^2 R^(4/3))."""
    A, T, P, R = geometry_props(y, B, z)
    return (n ** 2 * Q ** 2) / (A ** 2 * R ** (4.0 / 3.0))


def dydx(x, y, Q, n, B, z, S0, g, fr_cap=0.985):
    """
    The dynamic GVF equation:  dy/dx = (S0 - Sf) / (1 - Fr^2)

    Near-critical flow (Fr -> 1) is singular; the denominator is clipped
    away from zero (fr_cap) to keep RK4 numerically stable and to flag
    the likely location of a hydraulic jump rather than diverging.
    """
    Sf = friction_slope(y, Q, n, B, z)
    Fr = froude_number(y, Q, g, B, z)
    Fr2 = min(Fr ** 2, fr_cap ** 2) if Fr < 1 else max(Fr ** 2, (2 - fr_cap) ** 2)
    denom = 1.0 - Fr2
    return (S0 - Sf) / denom


def rk4_step(x, y, h, func, args):
    """One classical 4th-order Runge-Kutta step of size h."""
    k1 = func(x, y, *args)
    k2 = func(x + h / 2.0, y + k1 * h / 2.0, *args)
    k3 = func(x + h / 2.0, y + k2 * h / 2.0, *args)
    k4 = func(x + h, y + k3 * h, *args)
    y_next = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_next


def solve_gvf(params, yn, yc):
    """
    Marches the water-surface profile along the channel using RK4.

    Direction logic (per spec):
      - If the flow at the control depth is SUBCRITICAL (Fr < 1), the control
        is downstream, so the profile is computed downstream -> upstream
        (backwater curve). We march with negative steps from x = L back to x = 0.
      - If the flow is SUPERCRITICAL (Fr > 1), the control is upstream, so the
        profile is computed upstream -> downstream, marching with positive
        steps from x = 0 to x = L.

    Returns (x_arr, y_arr, regime, curve_type) with x_arr strictly increasing
    (0 -> L) for plotting, regardless of the marching direction used.
    """
    Q, n, B, z, S0, g = (params[k] for k in ("Q", "n", "B", "z", "S0", "g"))
    y_start, L, dx = params["y_start"], params["L"], params["dx"]
    args = (Q, n, B, z, S0, g)

    Fr_start = froude_number(y_start, Q, g, B, z)
    subcritical = Fr_start < 1.0

    n_steps = int(round(L / dx))
    xs, ys = [], []

    if subcritical:
        # March upstream from x = L (where y_start applies) to x = 0
        x, y = L, y_start
        xs.append(x); ys.append(y)
        h = -dx
        for _ in range(n_steps):
            y = rk4_step(x, y, h, dydx, args)
            x = x + h
            if y <= 0 or x < 0:
                break
            xs.append(x); ys.append(y)
        xs, ys = xs[::-1], ys[::-1]  # reorder to x increasing (0 -> L)
        regime = "Subcritical (Fr < 1) -- control downstream, solved downstream to upstream"
    else:
        # March downstream from x = 0 (where y_start applies) to x = L
        x, y = 0.0, y_start
        xs.append(x); ys.append(y)
        h = dx
        for _ in range(n_steps):
            y = rk4_step(x, y, h, dydx, args)
            x = x + h
            if y <= 0 or x > L:
                break
            xs.append(x); ys.append(y)
        regime = "Supercritical (Fr > 1) -- control upstream, solved upstream to downstream"

    x_arr, y_arr = np.array(xs), np.array(ys)

    # Classify the curve type (M-family for mild/subcritical-normal channels,
    # S-family for steep/supercritical-normal channels) using yn vs yc vs y.
    y_ref = y_arr[-1] if subcritical else y_arr[0]
    if yn > yc:  # mild channel
        curve_type = "M1 (backwater, y > yn > yc)" if y_ref > yn else \
                     "M2 (drawdown, yc < y < yn)" if y_ref > yc else \
                     "M3 (rapidly varied, y < yc < yn)"
    else:  # steep channel
        curve_type = "S1 (backwater, y > yc > yn)" if y_ref > yc else \
                     "S2 (drawdown, yn < y < yc)" if y_ref > yn else \
                     "S3 (rapidly varied, y < yn < yc)"

    return x_arr, y_arr, regime, curve_type


# ==============================================================================
# MODULE 5 : MULTI-PANEL VISUALIZATION
# ==============================================================================
def plot_results(params, yn, yc, y_curve, E_curve, e_min_point,
                  x_profile, y_profile, regime, curve_type):
    Q, n, B, z, S0, g = (params[k] for k in ("Q", "n", "B", "z", "S0", "g"))
    L = params["L"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Open Channel Flow Analysis Dashboard", fontsize=14, fontweight="bold")

    # ---- Left panel : Specific Energy Diagram ----
    ax1.plot(E_curve, y_curve, color="steelblue", lw=2, label="E-y curve")
    E_min, y_c_pt = e_min_point
    ax1.scatter([E_min], [y_c_pt], color="red", zorder=5, label=f"Critical point (yc = {yc:.3f} m)")
    ax1.axvline(E_min, color="red", ls=":", lw=1)

    # Show normal-depth energy point and its alternate depth for reference
    E_at_yn = specific_energy(yn, Q, g, B, z)
    ax1.scatter([E_at_yn], [yn], color="darkorange", zorder=5, label=f"Normal depth (yn = {yn:.3f} m)")
    try:
        guess_alt = 3 * yc if yn < yc else 0.3 * yc
        y_alt = alternate_depth(E_at_yn, Q, g, B, z, guess=guess_alt)
        if y_alt > 0 and abs(y_alt - yn) > 1e-3:
            ax1.scatter([E_at_yn], [y_alt], facecolors="none", edgecolors="darkorange",
                        zorder=5, label=f"Alternate depth ({y_alt:.3f} m)")
    except Exception:
        pass

    ax1.set_xlabel("Specific Energy, E (m)")
    ax1.set_ylabel("Depth, y (m)")
    ax1.set_title("Specific Energy Diagram")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    # ---- Right panel : Water Surface Profile ----
    x_full = np.linspace(0, L, 200)
    bed = S0 * x_full
    # Anchor bed elevation so it sits below the shown depths (bed measured from x=0 = 0)
    ax2.plot(x_full, bed, color="saddlebrown", lw=2, label="Channel bed (S0 x)")
    ax2.plot(x_full, bed + yn, color="black", ls=":", lw=1.5, label=f"Normal depth line (yn = {yn:.3f} m)")
    ax2.plot(x_full, bed + yc, color="black", ls="--", lw=1.5, label=f"Critical depth line (yc = {yc:.3f} m)")

    bed_at_profile = S0 * x_profile
    ax2.plot(x_profile, bed_at_profile + y_profile, color="royalblue", lw=2.2,
              label=f"Water surface profile\n({curve_type})")

    ax2.set_xlabel("Channel length, x (m)")
    ax2.set_ylabel("Elevation (m)")
    ax2.set_title("Water Surface Profile (GVF)")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.text(0.5, 0.01, regime, ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
def main():
    params = get_user_input()
    Q, n, B, z, S0, g = (params[k] for k in ("Q", "n", "B", "z", "S0", "g"))

    # ---- Module 2 ----
    yn = calculate_yn(Q, n, B, z, S0)
    yc = calculate_yc(Q, g, B, z)
    Fr_n = froude_number(yn, Q, g, B, z)

    print("\n" + "=" * 70)
    print(" RESULTS -- UNIFORM FLOW & CRITICAL FLOW STATE")
    print("=" * 70)
    print(f" Normal depth   yn = {yn:.4f} m")
    print(f" Critical depth yc = {yc:.4f} m")
    print(f" Froude no. at yn  = {Fr_n:.4f}  ->  "
          f"{'Subcritical' if Fr_n < 1 else 'Supercritical'} normal flow")
    print(f" Channel classification: {'MILD (yn > yc)' if yn > yc else 'STEEP (yn < yc)' if yn < yc else 'CRITICAL (yn = yc)'}")

    # ---- Module 3 ----
    y_curve, E_curve, e_min_point = build_energy_curve(Q, g, B, z, yn, yc)

    # ---- Module 4 ----
    x_profile, y_profile, regime, curve_type = solve_gvf(params, yn, yc)
    print("\n" + "=" * 70)
    print(" RESULTS -- GRADUALLY VARIED FLOW PROFILE")
    print("=" * 70)
    print(f" {regime}")
    print(f" Curve type: {curve_type}")
    print(f" Profile computed over x = {x_profile[0]:.1f} m to {x_profile[-1]:.1f} m "
          f"({len(x_profile)} points)")

    # ---- Module 5 ----
    fig = plot_results(params, yn, yc, y_curve, E_curve, e_min_point,
                        x_profile, y_profile, regime, curve_type)

    # Save next to this script, wherever that is on the user's machine
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gvf_dashboard.png")
    fig.savefig(out_path, dpi=150)
    print(f"\n Dashboard saved to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()