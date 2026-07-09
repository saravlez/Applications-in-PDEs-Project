#!/usr/bin/env python3
"""
Compares crowd dynamics across different parameter scenarios

(dt = 0.05 -> faster, more wiggles in animation).

Runs multiple simulations varying key parameters:
- Directional bias
- Corridor width
- Agent responsiveness
- Repulsion strength
- Panic sensitivity
Generates comparative plots of density, panic, and speed.

Author: Sara Vélez Fuente
Created: July 2025
"""
import copy
import time
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patches as patches
from matplotlib.lines import Line2D

params = {
    'area_width': 4.0,
    'area_length': 40.0,
    'n_agents': 50,
    'ratio_right': 0.7,
    'v_d': 1.4,
    'tau': 0.5,
    'A': 2e3,
    'B': 0.08,
    'kappa_panic_speed': 1.0,
    'kappa_panic_rep': 1.0,
    'mass': 80.0,
    'radii': 0.1,
    'alpha': 0.01,
    'beta': 0.005,
    'N_max': 7.0,
    'dt': 0.05,
    't_end': 600.0,
    'inflow_events': [
        (30, 90, 25.0),
        (150, 210, 45.0),
        (270, 330, 2.0),
        (390, 450, 1.5)
    ],
    'ramp_width': 20.0,
    'velocity_tolerance': 0.01,
    'min_snapshot_interval': 10.0
}
params['area'] = params['area_width'] * params['area_length']


def inflow_rate(t, N, params):
    """Rate in agents per second, scaled by congestion"""
    total = 0.0
    width = params.get('ramp_width', 120.0)
    for t0, t1, rate in params['inflow_events']:
        up = 1.0 / (1 + np.exp(-(t - t0) / width))
        down = 1.0 / (1 + np.exp((t - t1) / width))
        total += rate * up * down
    return total * max(0.0, 1 - N / params['N_max'])


def spawn_agents(n_new, params):
    """Generate new agents at entrance or exit with random y"""
    # Split into right- and left-going
    n_right = np.random.binomial(n_new, params['ratio_right'])
    n_left = n_new - n_right

    # Right-going at x = 0
    position_r = np.zeros((n_right, 2))
    position_r[:, 1] = np.random.uniform(0, params['area_width'], n_right)
    velocity_r = np.zeros_like(position_r)
    direction_r = np.ones(n_right)  # Unit vector toward right exit

    # Left-going at x = area_length
    position_l = np.zeros((n_left, 2))
    position_l[:, 0] = params['area_length']
    position_l[:, 1] = np.random.uniform(0, params['area_width'], n_left)
    velocity_l = np.zeros_like(position_l)
    direction_l = -np.ones(n_left)  # Unit vector toward left exit

    # Concatenate
    positions = np.vstack((position_r, position_l))
    velocities = np.vstack((velocity_r, velocity_l))
    directions = np.concatenate((direction_r, direction_l))
    return positions, velocities, directions


def update_panic(P, N, params, dt):
    """Updates panic level using ODE model"""
    dP = (params['alpha'] * N * (params['v_d'] - params['v_d'] * max(0.0, 1 - N / params['N_max']))
          - params['beta'] * P * max(0.0, 1 - N / params['N_max']))

    # Using Euler integration
    return P + dP * dt


def compute_forces(positions, velocities, directions, P, params):
    """Computes total forces acting on each agent using Social Force Model"""
    N = positions.shape[0]  # Number of agents
    forces = np.zeros_like(positions)  # positions: N x 2

    # Panic-amplified parameters
    # v_d = params['v_d'] * (1 + params['kappa_panic_speed'] * P)  # panic increases velocity
    v_d = params['v_d'] * (1 - params['kappa_panic_speed'] * P)  # or panic decreases velocity
    A = params['A'] * (1 + params['kappa_panic_rep'] * P)
    B = params['B']

    # Direction vectors: rightward (+1,0) or leftward (-1,0)
    dir_vectors = np.zeros_like(positions)
    dir_vectors[:, 0] = directions

    # 1. Driving force toward desired velocity
    for i in range(N):
        desired = v_d * dir_vectors[i]
        forces[i] += params['mass'] * (desired - velocities[i]) / params['tau']

    # 2. Pairwise exponential repulsion (equal and opposite forces)
    for i in range(N):
        for j in range(i + 1, N):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance > 0:
                f_mag = A * np.exp((2 * params['radii'] - distance) / B)
                forces[i] += f_mag * (positions[i] - positions[j]) / distance
                forces[j] -= f_mag * (positions[i] - positions[j]) / distance

    # 3. Wall repulsion (top/bottom)
    for i in range(N):
        x, y = positions[i]
        forces[i, 1] += A * np.exp((params['radii'] - y) / B)  # Bottom wall at y = 0
        forces[i, 1] -= A * np.exp((params['radii'] - (params['area_width'] - y)) / B)  # Top wall at y = width

    return forces


def run_unified_simulation(params):
    """Run simulation once while collecting all necessary data for all plots"""
    dt = params['dt']
    n_steps = int(params['t_end'] / dt)
    next_frame = 0.0
    last_snapshot_time = -params['min_snapshot_interval']  # Initialize to allow first snapshot

    # Initialize agents
    positions = np.stack((
        np.random.uniform(0, params['area_length'], params['n_agents']),
        np.random.uniform(0, params['area_width'], params['n_agents'])
    ), axis=1)
    velocities = np.zeros_like(positions)
    panic = 0.0

    # Directions
    n_right = np.random.binomial(params['n_agents'], params['ratio_right'])
    n_left = params['n_agents'] - n_right
    directions = np.concatenate((np.ones(n_right), -np.ones(n_left)))

    # For static snapshots plot
    times = []
    avg_speeds = []
    snapshots = []

    # For animation
    positions_history = []
    directions_history = []
    times_history = []
    n_agents_history = []

    # For population model
    pop_times = []
    inflow_rates = []
    densities = []
    panic_levels = []
    speeds = []

    # Main simulation loop
    for step in range(n_steps):
        print(f'Timestep {step} / {n_steps}')
        t = step * dt
        N = len(positions) / params['area']

        # Record data for population model (every step)
        pop_times.append(t)
        inflow_rates.append(inflow_rate(t, N, params))
        densities.append(N)
        panic_levels.append(panic)
        current_speed = np.mean(np.linalg.norm(velocities, axis=1)) if len(velocities) > 0 else 0.0
        speeds.append(current_speed)

        # Record data for static snapshots plot (every step)
        times.append(t)
        avg_speeds.append(current_speed)

        # Take snapshot if velocity is almost zero and sufficient time has passed
        if (current_speed < params['velocity_tolerance'] and
                (t - last_snapshot_time) >= params['min_snapshot_interval']):
            snapshots.append((t, positions.copy(), directions.copy()))
            last_snapshot_time = t
            print(f"Snapshot taken at t = {t:.1f} s (avg speed = {current_speed:.4f} m/s)")

        # Capture frame for animation
        if t >= next_frame:
            positions_history.append(positions.copy())
            directions_history.append(directions.copy())
            times_history.append(t)
            n_agents_history.append(len(positions))
            next_frame += 0.5  # frame interval

        # Spawn new agents
        n_new = np.random.poisson(inflow_rate(t, N, params) * dt)
        if n_new > 0:
            new_p, new_v, new_d = spawn_agents(n_new, params)
            positions = np.vstack((positions, new_p))
            velocities = np.vstack((velocities, new_v))
            directions = np.concatenate((directions, new_d))

        # Update panic
        panic = update_panic(panic, N, params, dt)

        # Compute forces and move agents
        F = compute_forces(positions, velocities, directions, panic, params)
        velocities += (F / params['mass']) * dt
        positions += velocities * dt

        # Contain within walls
        positions[:, 1] = np.clip(positions[:, 1], 0, params['area_width'])

        # Remove agents who exited
        mask = (positions[:, 0] >= 0) & (positions[:, 0] <= params['area_length'])
        positions = positions[mask]
        velocities = velocities[mask]
        directions = directions[mask]

    return {
        'static': (times, avg_speeds, snapshots),
        'animation': (positions_history, directions_history, times_history, n_agents_history),
        'population': (pop_times, inflow_rates, densities, panic_levels, speeds)
    }


def plot_average_speed(times, avg_speeds, name):
    """Plot average speed over time in a separate figure"""
    plt.figure(figsize=(10, 6))
    plt.plot(times, avg_speeds, color='teal')
    plt.xlabel('Time (s)')
    plt.ylabel('Average Speed (m/s)')
    plt.title('Evolution of Average Speed')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{name}/average_speed_evolution.png', dpi=150)
    plt.close()


def plot_individual_snapshots(snapshots, name):
    """Create individual figures for each snapshot"""
    for idx, (t_snap, pos_snap, dir_snap) in enumerate(snapshots):
        fig, ax = plt.subplots(figsize=(8, 6))

        right_mask = dir_snap == 1
        left_mask = dir_snap == -1

        ax.scatter(pos_snap[right_mask, 0], pos_snap[right_mask, 1],
                   s=20, color='teal', alpha=0.7, label='Right-moving')
        ax.scatter(pos_snap[left_mask, 0], pos_snap[left_mask, 1],
                   s=20, color='crimson', alpha=0.7, label='Left-moving')

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title(f'Agent Positions at t = {t_snap:.1f} s\n(Avg Speed ≈ 0)')
        ax.set_xlim(0, params['area_length'])
        ax.set_ylim(0, params['area_width'])
        ax.grid(True)
        ax.legend(loc='upper right')

        # Add boundaries
        ax.plot([0, 0], [0, params['area_width']], 'k-', lw=2)
        ax.plot([params['area_length'], params['area_length']], [0, params['area_width']], 'k-', lw=2)
        ax.plot([0, params['area_length']], [0, 0], 'k-', lw=2)
        ax.plot([0, params['area_length']], [params['area_width'], params['area_width']], 'k-', lw=2)

        plt.tight_layout()
        plt.savefig(f'{name}/snapshot_{idx + 1}_t_{t_snap:.0f}s.png', dpi=150)
        plt.close(fig)  # Close figure to free memory


def create_animation(positions_history, directions_history, times_history, n_agents_history, name):
    """Create animation using precomputed data"""
    ani_fig, ani_ax = plt.subplots(figsize=(12, 6))
    ani_ax.set_xlim(-1, params['area_length'] + 1)
    ani_ax.set_ylim(-0.5, params['area_width'] + 0.5)
    ani_ax.set_xlabel('x (m)')
    ani_ax.set_ylabel('y (m)')
    ani_ax.set_title(f'Bidirectional Crowd Flow Simulation \n Scenario: {name}')
    ani_ax.grid(True)

    # Create thinner entrance/exit zones covering full height
    zone_width = 0.5
    entrance_color = 'royalblue'
    exit_color = 'green'

    # Left side zones
    entrance_right = patches.Rectangle(
        (-zone_width, 0), zone_width, params['area_width'], alpha=0.3, color=entrance_color)
    exit_left = patches.Rectangle(
        (-2 * zone_width, 0), zone_width, params['area_width'], alpha=0.3, color=exit_color)

    # Right side zones
    entrance_left = patches.Rectangle(
        (params['area_length'], 0), zone_width, params['area_width'], alpha=0.3, color=entrance_color)
    exit_right = patches.Rectangle(
        (params['area_length'] + zone_width, 0), zone_width, params['area_width'], alpha=0.3, color=exit_color)

    for patch in [entrance_right, exit_left, entrance_left, exit_right]:
        ani_ax.add_patch(patch)

    # Add boundary walls
    ani_ax.plot([0, 0], [0, params['area_width']], color='dimgray', linewidth=1)
    ani_ax.plot([params['area_length'], params['area_length']], [0, params['area_width']], color='dimgray', linewidth=1)
    ani_ax.plot([0, params['area_length']], [0, 0], color='dimgray', linewidth=1)
    ani_ax.plot([0, params['area_length']], [params['area_width'], params['area_width']], color='dimgray', linewidth=1)

    # Create scatter plot for agents
    scat = ani_ax.scatter([], [], s=10, alpha=0.7)

    # Create info text
    time_text = ani_ax.text(0.02, 0.95, '', transform=ani_ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    agent_text = ani_ax.text(0.02, 0.90, '', transform=ani_ax.transAxes, bbox=dict(facecolor='white', alpha=0.7))

    # Create custom handles for legends
    entrance_patch = patches.Patch(alpha=0.3, color=entrance_color, label='Entrance')
    exit_patch = patches.Patch(alpha=0.3, color=exit_color, label='Exit')
    right_patch = Line2D([0], [0], marker='o', color='w',
                         markerfacecolor='teal', markersize=8, label='Moving Right')
    left_patch = Line2D([0], [0], marker='o', color='w',
                        markerfacecolor='crimson', markersize=8, label='Moving Left')

    # Create unified legend outside the plot
    legend_handles = [entrance_patch, exit_patch, right_patch, left_patch]
    ani_ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.25, 1))

    # Adjust layout to make space for legend
    ani_fig.tight_layout(rect=[0, 0, 0.9, 1])

    # Animation functions
    def init():
        scat.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        agent_text.set_text('')
        return scat, time_text, agent_text

    def update(frame):
        pos = positions_history[frame]
        dirs = directions_history[frame]
        colors = ['teal' if d > 0 else 'crimson' for d in dirs]
        scat.set_offsets(pos)
        scat.set_color(colors)
        time_text.set_text(f'Time: {times_history[frame]:.1f} s')
        agent_text.set_text(f'Agents: {n_agents_history[frame]}')
        return scat, time_text, agent_text

    # Create animation
    ani = FuncAnimation(
        ani_fig,
        update,
        frames=len(positions_history),
        init_func=init,
        blit=True,
        interval=50,
        repeat=False
    )

    # Save animation
    writer = FFMpegWriter(fps=20, metadata=dict(artist='CrowdSim'), bitrate=1800)
    ani.save(f'{name}/bidirectional_crowd_flow2.mp4', writer=writer, dpi=100)


def plot_population_model(t, I_eff, N_sim, P_sim, V_sim, name):
    """Plot population model results using precomputed data"""
    P_sim = np.array(P_sim) 
    N_sim = np.array(N_sim) 

    # First Figure: Inflow, Density, Panic
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Inflow
    axs[0].set_title('Crowd Dynamics During High-Density Event')
    axs[0].plot(t, I_eff, color='teal', linewidth=2, label='Effective inflow')
    axs[0].fill_between(t, 0, I_eff, color='teal', alpha=0.3)
    axs[0].set_ylabel('Inflow Rate\n[ppl/(m²·s)]')
    axs[0].legend()
    axs[0].grid(alpha=0.3)

    # Density
    axs[1].plot(t, N_sim, color='royalblue', linewidth=2, label='Density')
    axs[1].fill_between(t, 0, N_sim, color='royalblue', alpha=0.3)
    axs[1].axhline(params['N_max'], color='black', linestyle='--', label='Crush threshold')
    axs[1].set_ylabel('Density $N$\n[ppl/m²]')
    axs[1].set_ylim(0, max(N_sim) * 1.1)
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    # Panic / np.max(P_sim)
    axs[2].plot(t, P_sim, color='palevioletred', linewidth=2, label='Panic level')
    axs[2].fill_between(t, 0, P_sim, color='palevioletred', alpha=0.3)
    axs[2].set_ylabel('Panic Level $P$')
    axs[2].set_xlabel('Time [seconds]')
    axs[2].legend()
    axs[2].grid(alpha=0.3)

    # Velocity and Congestion
    axs[3].plot(t, V_sim, color='forestgreen', linewidth=2, label='Actual speed')
    axs[3].axhline(params['v_d'], color='black', linestyle='--', label='Desired speed')
    axs[3].plot(t, 1 - (N_sim / params['N_max']), color='sienna', linestyle='--', label='Congestion level')
    axs[3].axhline(1.0, color='black', linestyle=':', label='Crushed = 1')
    axs[3].set_xlabel('Time [seconds]')
    axs[3].set_ylabel('Speed [m/s] & Congestion')
    axs[3].legend(loc='upper right')
    axs[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{name}/offlattice_crowd_dynamics.png', dpi=300)
    plt.show()


def main():
    # Build scenario dictionary by overriding one parameter
    scenarios = {
        'Base': params,
        'More Right‐Going': {**params, 'ratio_right': 0.9},
        'Narrow Channel': {**params, 'area_width': 2.0},
        'Faster Response': {**params, 'tau': 0.3},
        'Stronger Repulsion': {**params, 'A': 3000.0},
        'High Panic Sensitivity': {**params, 'kappa_panic_speed': 2.0},
    }
    all_pop = {}

    # Loop over scenarios
    start_total = time.time()
    for name, pset in scenarios.items():
        print(f"\n=== Running scenario: {name} ===")

        plt.close('all')
        os.makedirs(f'{name}', exist_ok=True)

        sim_params = copy.deepcopy(pset)
        sim_params['area'] = sim_params['area_width'] * sim_params['area_length']

        # Simulation
        start_time = time.time()
        results = run_unified_simulation(sim_params)
        print(f'Simulation took {((time.time() - start_time) / 60.0):.2f} minutes with dt = {params["dt"]}')
        print(f"Captured {len(results['static'][2])} snapshots")

        # Outputs
        times, avg_speeds, snapshots = results['static']
        anim_pos, anim_dirs, anim_times, anim_n_agents = results['animation']
        pop_t, pop_I, pop_N, pop_P, pop_V = results['population']

        # For later comparison
        all_pop[name] = (pop_t, pop_N, pop_P, pop_V)

        # Plot per‐scenario
        plot_average_speed(times, avg_speeds, name)
        plot_individual_snapshots(snapshots, name)
        create_animation(anim_pos, anim_dirs, anim_times, anim_n_agents, name)

    print(f'Simulation took {((time.time() - start_total) / 60.0):.2f} minutes with dt = {params["dt"]}')

    # Colors
    palette = sns.color_palette('mako', n_colors=len(scenarios))
    scenario_colors = dict(zip(scenarios.keys(), palette))

    # Plot comparison of density
    plt.figure(figsize=(10, 6))
    for name, (t, N, P, V) in all_pop.items():
        plt.plot(t, N, label=name, color=scenario_colors[name])
    plt.axhline(params['N_max'], color='k', linestyle='--', label='N_max')
    plt.xlabel('Time (s)')
    plt.ylabel('Density [ppl/m²]')
    plt.title('Density Across Scenarios')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('compare_density.png', dpi=300)
    plt.show()

    # Plot comparison of  panic
    plt.figure(figsize=(10, 6))
    for name, (t, N, P, V) in all_pop.items():
        plt.plot(t, P, label=name, color=scenario_colors[name])
    plt.xlabel('Time (s)')
    plt.ylabel('Panic Level P')
    plt.title('Panic Across Scenarios')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('compare_panic.png', dpi=300)
    plt.show()

    # Plot comparison of speed
    plt.figure(figsize=(10, 6))
    for name, (t, N, P, V) in all_pop.items():
        plt.plot(t, V, label=name, color=scenario_colors[name])
    plt.axhline(params['v_d'], color='k', linestyle='--', label='v_d')
    plt.xlabel('Time (s)')
    plt.ylabel('Average Speed [m/s]')
    plt.title('Speed Comparison Across Scenarios')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('compare_speed.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()



