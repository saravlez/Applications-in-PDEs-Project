#!/usr/bin/env python3
"""
Simple Equilibrium Analysis

Author: Sara Vélez Fuente
Created: July 2025
"""
import numpy as np

# Parameters Base
i0 = 0.5  # constant dimensionless inflow
K = 0.2
r_out = 0.1
r_in = 0.45
alpha = 0.01
v_d = 1.4
N_max = 6.0
beta = 0.005

A = (alpha * v_d * N_max) / r_out
B = beta / r_out
Ri = r_in / r_out

# Cubic equation for n*: (K*A/B)*n^3 - n^2 + n - Ri*i0 = 0
coeffs = [K * A / B, -1.0, 1.0, -Ri * i0]
roots = np.roots(coeffs)

real_roots = [root.real for root in roots if np.isclose(root.imag, 0, atol=1e-6)]
valid_roots = [n for n in real_roots if 0 <= n <= 1]

print("All roots:", roots)
print("Real roots:", real_roots)
print("Valid roots (0 <= n <= 1):", valid_roots)


def jacobian(n, p):
    # Partial derivatives
    df_dn = -(1 + K * p) * (1 - 2 * n)
    df_dp = -K * n * (1 - n)
    dg_dn = 2 * A * n + B * p
    dg_dp = -B * (1 - n)

    return np.array([[df_dn, df_dp], [dg_dn, dg_dp]])


for i, n_star in enumerate(valid_roots):
    p_star = (A * n_star ** 2) / (B * (1 - n_star))

    # Skip if n* = 1 (division by zero)
    if np.isclose(n_star, 1, atol=1e-6):
        print(f"\nEquilibrium {i + 1}: n* = {n_star:.4f}, p* = {p_star:.4f} is degenerate")
        continue

    # Analysis
    J = jacobian(n_star, p_star)
    trace = np.trace(J)
    det = np.linalg.det(J)

    # Stability
    stable = (trace < 0) and (det > 0)

    print(f"\nEquilibrium {i + 1}: n* = {n_star:.6f}, p* = {p_star:.6f}")
    print(f"Jacobian = \n{J}")
    print(f"Trace = {trace:.6f}, Determinant = {det:.6f}")
    print("Stability: Stable" if stable else "Stability: Unstable")
    print(f"Eigenvalues: {np.linalg.eigvals(J)}")

if not valid_roots:
    print("No valid equilibrium in [0,1] found.")