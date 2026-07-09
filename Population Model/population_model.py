#!/usr/bin/env python3
"""
Crowd simulation for a high-density event.

Solves ODEs for density and panic with time-varying inflow.
Visualizes inflow, congestion, speed, and panic evolution.
Includes post-simulation analysis of critical thresholds.

Author: Sara Vélez Fuente
Created: July 2025
"""
import numpy as np
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
    width = params.get('ramp_width', 120.0)  # seconds for smooth transitions
    for t0, t1, rate in params['inflow_events']:
        up = 1.0 / (1 + np.exp(-(t - t0) / width))
        down = 1.0 / (1 + np.exp((t - t1) / width))
        total += rate * up * down
    return total * congestion(N, params)


def derivatives(t, y, params):
    """ Derivatives for the ODE system: y[0] = N and y[1] = P"""
    N, P = y
    outflow = params['r_out'] * N * congestion(N, params) * (1 + params['K'] * P)
    dN_dt = inflow(N, t, params) - outflow
    velocity_deficit = params['v_d'] - actual_speed(N, params)
    dP_dt = params['alpha'] * N * velocity_deficit - params['beta'] * P * congestion(N, params)
    return [dN_dt, dP_dt]


def crowd_density_model(t_end=7200, dt=0.1, show_plots=True):
    """ Simulates crowd density and panic dynamics """

    params = {
        'r_out': 0.1,  # Base outflow rate [1/s]
        'K': 0.2,  # Panic-induced outflow multiplier
        'alpha': 0.01,  # Panic generation coefficient
        'beta': 0.005,  # Panic decay rate
        'v_d': 1.4,  # Desired walking speed [m/s]
        'N_max': 6.0,  # Maximum sustainable density [ppl/m²]
        'T_inflow': (1800, 3600),  # Primary shading window
        'inflow_events': [  # (start, end, base rate)
            (1800, 3600, 0.85),  # Primary event
            (5400, 5700, 0.45),  # Secondary surge
            (5800, 6100, 0.65)  # Tertiary surge
        ],
        'ramp_width': 120.0  # seconds
    }

    # Initial conditions
    y0 = [0.5, 0.0]  # [N0, P0]

    # Solve ODE system
    sol = solve_ivp(
        lambda t, y: derivatives(t, y, params),
        [0, t_end], y0,
        t_eval=np.arange(0, t_end + dt, dt),
        method='RK45', rtol=1e-5, atol=1e-8
    )

    t = sol.t
    N = sol.y[0]
    P = sol.y[1]

    # Post-process
    v_actual = [actual_speed(N_i, params) for N_i in N]
    I_eff = [inflow(N_i, t_i, params) for t_i, N_i in zip(t, N)]
    congestion_level = [1 - congestion(N_i, params) for N_i in N]

    if show_plots:
        # First Figure: Inflow, Density, Panic
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Inflow
        axs[0].plot(t, I_eff, color='teal', linewidth=2, label='Effective inflow')
        axs[0].fill_between(t, 0, I_eff, color='teal', alpha=0.3)
        axs[0].set_ylabel('Inflow Rate\n[people/(m²·s)]')
        axs[0].set_title('Crowd Dynamics During High-Density Event')
        axs[0].grid(alpha=0.3)

        # Density
        axs[1].plot(t, N, color='royalblue', linewidth=2, label='Density')
        axs[1].fill_between(t, 0, N, color='royalblue', alpha=0.3)
        axs[1].axhline(params['N_max'], color='black', linestyle='--', label='Crush threshold')
        axs[1].set_ylabel('Density $N$\n[people/m²]')
        axs[1].set_ylim(0, params['N_max'] * 1.1)
        axs[1].legend()
        axs[1].grid(alpha=0.3)

        # Panic / np.max(P)
        axs[2].plot(t, P / np.max(P), color='palevioletred', linewidth=2, label='Normalized Panic level')
        axs[2].fill_between(t, 0, P / np.max(P), color='palevioletred', alpha=0.3)
        axs[2].axhline(0.6, color='black', linestyle=':', label='Critical threshold')
        axs[2].axhline(0.8, color='red', linestyle=':', label='Danger zone')
        axs[2].set_ylabel('Panic Level $P$')
        axs[2].set_xlabel('Time [seconds]')
        axs[2].legend()
        axs[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('crowd_dynamics_model.png', dpi=300)
        plt.show()

        # Second Figure: Velocity & Congestion
        fig, ax1 = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Velocity
        ax1[0].plot(t, v_actual, color='forestgreen', linewidth=2, label='Actual speed')
        ax1[0].axhline(params['v_d'], color='black', linestyle=':', label='Desired speed')
        ax1[0].set_ylabel('Speed [m/s]')
        ax1[0].legend(loc='upper right')
        ax1[0].grid(alpha=0.3)

        # Congestion
        ax1[1].plot(t, congestion_level, color='sienna', linestyle='--', label='Congestion level')
        ax1[1].axhline(1.0, color='black', linestyle=':', label='Crushed')
        ax1[1].set_ylabel('Congestion (0=free, 1=crushed)')
        ax1[1].set_xlabel('Time [seconds]')
        ax1[1].legend(loc='upper right')
        ax1[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('crowd_dynamics_model2.png', dpi=300)
        plt.show()

        # First Figure: Inflow, Density, Panic
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        # Inflow
        axs[0].plot(t, I_eff, color='teal', linewidth=2, label='Effective inflow')
        axs[0].fill_between(t, 0, I_eff, color='teal', alpha=0.3)
        axs[0].set_ylabel('Inflow Rate\n[people/(m²·s)]')
        axs[0].set_title('Crowd Dynamics During High-Density Event')
        axs[0].legend()
        axs[0].grid(alpha=0.3)

        # Density
        axs[1].plot(t, N, color='royalblue', linewidth=2, label='Density')
        axs[1].fill_between(t, 0, N, color='royalblue', alpha=0.3)
        axs[1].axhline(params['N_max'], color='black', linestyle='--', label='Crush threshold')
        axs[1].set_ylabel('Density $N$\n[people/m²]')
        axs[1].set_ylim(0, params['N_max'] * 1.1)
        axs[1].legend()
        axs[1].grid(alpha=0.3)

        # Panic / np.max(P)
        axs[2].plot(t, P, color='palevioletred', linewidth=2, label='Panic level')
        axs[2].fill_between(t, 0, P, color='palevioletred', alpha=0.3)
        axs[2].set_ylabel('Panic Level $P$')
        axs[2].set_xlabel('Time [seconds]')
        axs[2].legend()
        axs[2].grid(alpha=0.3)

        # Velocity and Congestion
        axs[3].plot(t, v_actual, color='forestgreen', linewidth=2, label='Actual speed')
        axs[3].axhline(params['v_d'], color='black', linestyle='--', label='Desired speed')
        axs[3].plot(t, congestion_level, color='sienna', linestyle='--', label='Congestion level')
        axs[3].axhline(1.0, color='black', linestyle=':', label='Crushed = 1')
        axs[3].set_xlabel('Time [seconds]')
        axs[3].set_ylabel('Speed [m/s] & Congestion')
        axs[3].legend(loc='upper right')
        axs[3].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('crowd_dynamics_model3.png', dpi=300)
        plt.show()

    return t, N, P


def main():
    # Simulation
    time, density, panic = crowd_density_model()

    # Post-simulation analysis
    crush_index = np.argmax(density > 6.0)  # First time density exceeds 6.0 ppl/m²
    popu_peak = np.max(density)
    if crush_index > 0:
        print(f"Crush conditions reached at t = {time[crush_index] / 60:.1f} minutes")
    print(f"Population density peak: {popu_peak:.2f} ({'critical' if popu_peak > 4.5 else 'subcritical'})")

    panic_peak = np.max(panic)
    print(f"Peak panic level: {panic_peak:.2f} ({'critical' if panic_peak > 0.6 else 'subcritical'})")


if __name__ == "__main__":
    main()
