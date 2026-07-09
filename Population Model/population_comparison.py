#!/usr/bin/env python3
"""
Simulates crowd dynamics under multiple scenarios.

Models density and panic using ODEs with congestion feedback.
Compares effects of parameter variations on inflow, speed, and panic.
Generates plots for scenario-based analysis.

Author: Sara Vélez Fuente
Created: July 2025
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def congestion(N, params):
    """Congestion factor (0 when crushed, rate when free flow)"""
    return max(0.0, 1 - N / params['N_max'])


def actual_speed(N, params):
    """Walking speed based on congestion"""
    return params['v_d'] * congestion(N, params)


def inflow(N, t, params):
    """
    Composite inflow from multiple events, smoothed by logistic ramps.
    I(N,t) = sum_i [ rate_i * S_i(t) ] * congestion(N)
    """
    total = 0.0
    width = params.get('ramp_width', 120.0)
    for t0, t1, rate in params['inflow_events']:
        up = 1.0 / (1 + np.exp(-(t - t0) / width))
        down = 1.0 / (1 + np.exp((t - t1) / width))
        total += rate * up * down
    return total * congestion(N, params)


def derivatives(t, y, params):
    """Derivatives for the ODE system: y[0] = N and y[1] = P"""
    N, P = y
    I_t = inflow(N, t, params)
    outflow = params['r_out'] * N * congestion(N, params) * (1 + params['K'] * P)
    dN_dt = I_t - outflow
    velocity_deficit = params['v_d'] - actual_speed(N, params)
    dP_dt = params['alpha'] * N * velocity_deficit - params['beta'] * P * congestion(N, params)
    return [dN_dt, dP_dt]


def run_model(params, t_end=7200, dt=0.1):
    """ Simulates crowd density and panic dynamics """

    y0 = [0.5, 0.0]
    sol = solve_ivp(
        lambda t, y: derivatives(t, y, params),
        [0, t_end], y0,
        t_eval=np.arange(0, t_end + dt, dt),
        method='RK45', rtol=1e-5, atol=1e-8
    )
    t = sol.t
    N = sol.y[0]
    P = sol.y[1]
    v_actual = [actual_speed(N_i, params) for N_i in N]
    I_eff = [inflow(N_i, t_i, params) for t_i, N_i in zip(t, N)]
    return t, N, P, v_actual, I_eff


def main():
    # Define scenarios with varied parameters
    base_params = {
        'r_out': 0.1,  # Base outflow rate [1/s]
        'K': 0.2,  # Panic-induced outflow multiplier
        'alpha': 0.01,  # Panic generation coefficient
        'beta': 0.005,  # Panic decay rate
        'v_d': 1.4,  # Desired walking speed [m/s]
        'N_max': 6.0,  # Maximum sustainable density [ppl/m²]
        'inflow_events': [  # (start, end, base rate)
            (1800, 3600, 0.85),
            (5400, 5700, 0.45),
            (5800, 6100, 0.65)
        ],
        'ramp_width': 120.0   # seconds
    }

    scenarios = {
        'Base': base_params,
        'Higher r_in': {**base_params, 'inflow_events': [(1800, 3600, 1.5), (5400, 5700, 0.8), (5800, 6100, 0.9)]},
        'Lower r_out': {**base_params, 'r_out': 0.02},
        'Lower beta': {**base_params, 'beta': 0.001},
        'Higher alpha': {**base_params, 'alpha': 0.05},
        'Higher K': {**base_params, 'K': 0.5},
    }

    # Run all scenarios
    results = {}
    for name, params in scenarios.items():
        t, N, P, v_actual, I_eff = run_model(params)
        results[name] = {'t': t, 'N': N, 'P': P, 'v': v_actual, 'I': I_eff}

    # Colors
    palette = sns.color_palette('mako', n_colors=len(scenarios))
    scenario_colors = dict(zip(scenarios.keys(), palette))

    # Plot comparison of density
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data['t'], data['N'], label=name, color=scenario_colors[name])
    plt.axhline(base_params['N_max'], color='black', linestyle='--', label='Crush threshold')
    plt.xlabel('Time [s]')
    plt.ylabel('Density [ppl/m²]')
    plt.title('Density Comparison Across Scenarios')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('density_comparison.png', dpi=300)
    plt.show()

    # Plot comparison of normalized panic
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data['t'], data['P'] / np.max(data['P']), label=name, color=scenario_colors[name])
    plt.xlabel('Time [s]')
    plt.ylabel('Normalized Panic')
    plt.title('Panic Comparison Across Scenarios')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('panic_comparison.png', dpi=300)
    plt.show()

    # Plot comparison of inflow
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data['t'], data['I'], label=name, color=scenario_colors[name])
    plt.xlabel('Time [s]')
    plt.ylabel('Effective Inflow [ppl/m²·s]')
    plt.title('Inflow Comparison Across Scenarios')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('inflow_comparison.png', dpi=300)
    plt.show()

    # Plot comparison of speed
    plt.figure(figsize=(10, 6))
    for name, data in results.items():
        plt.plot(data['t'], data['v'], label=name, color=scenario_colors[name])
    plt.axhline(base_params['v_d'], color='black', linestyle=':', label='Desired speed')
    plt.xlabel('Time [s]')
    plt.ylabel('Speed [m/s]')
    plt.title('Speed Comparison Across Scenarios')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('speed_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
