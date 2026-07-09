#!/usr/bin/env python3
"""
Comparative analysis of lattice model scenarios.

Evaluates parameter variations:
- Panic sensitivity (generation/decay rates)
- Movement weights (density/randomness)
- Spatial constraints (alley width)
- Directional balance

Generates comparative plots across scenarios.

Author: Sara Vélez Fuente
Created: July 2025
"""
import os
import seaborn as sns
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import copy
from scipy.signal import convolve2d
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

params = {
    'area_width': 4.0,          # Width of the alley (m)
    'area_length': 40.0,        # Length of the alley (m)
    'cell_size': 0.4,           # Size of each grid cell (m)
    'dt': 0.1,                  # Simulation time step (s)
    't_end': 500.0,             # Total simulation time (s)
    'ratio_right': 0.7,         # Fraction of agents moving right
    'N_max': 9.0,               # Maximum density (agents per neighborhood)
    'alpha': 0.01,              # Panic generation rate
    'beta': 0.005,              # Panic dissipation rate
    'weights': {  # Movement decision weights
        'desire': 1.0,          # Weight for moving in intended direction
        'density': 0.8,         # Weight for avoiding crowded areas
        'panic': 0.5,           # Weight for panic-driven movement
        'random': 0.3           # Weight for random movement
    },
    'inflow_events': [  # Inflow events (start, end, rate in agents/s)
        (30, 90, 8.5),
        (150, 210, 4.0),
        (270, 330, 7.0)
    ],
    'ramp_width': 20.0  # Smoothing width for inflow transitions (s)
}

# Calculate grid dimensions from physical parameters
params['grid_width'] = int(params['area_width'] / params['cell_size'])
params['grid_length'] = int(params['area_length'] / params['cell_size'])
params['total_cells'] = params['grid_length'] * params['grid_width']


def debug(msg, DEBUG_MODE=True):
    if DEBUG_MODE:
        print(msg)


def inflow_rate(t, grid, params):
    """Calculate time-dependent inflow rate based on scheduled events"""
    total = 0.0
    width = params.get('ramp_width', 120.0)
    current_density = np.sum(grid != 0) / (params['total_cells']) if params['total_cells'] > 0 else 0

    for t0, t1, rate in params['inflow_events']:
        up = 1.0 / (1 + np.exp(-(t - t0) / width))
        down = 1.0 / (1 + np.exp((t - t1) / width))
        total += rate * up * down

    # N_max defined per 3×3 block, so /9 -> per‐cell maximum we want
    return total * max(0.0, 1 - current_density / (params['N_max'] / 9.0))


def initialize_grid(params):
    """Initialize the simulation grid and panic field"""
    grid = np.zeros((params['grid_length'], params['grid_width']), dtype=int)  # Either  0 +1 -1
    panic = np.zeros_like(grid, dtype=float)  # continuous value [0, 1]
    return grid, panic


def calculate_local_density(grid):
    """Compute local density using 3x3 neighborhood averaging"""
    # Neighborhood kernel
    kernel = np.ones((3, 3), dtype=float)

    # Value of cells to either 0 or 1
    occupancy = np.abs(grid).astype(float)

    # For each cell, sum values in its 3x3 neighbourhood (itself + 8)
    average = convolve2d(occupancy, kernel, mode='same', boundary='fill', fillvalue=0) / 9.0
    return average


def update_panic_field(panic, density, params):
    """Update the panic field based on local density: decay + generation"""
    new_panic = (1 - params['beta']) * panic + params['alpha'] * density  # (1 - beta) P + alpha rho
    return np.clip(new_panic, 0.0, 1.0)  # Panic bounds [0, 1]


def calculate_move_scores(i, j, grid, panic, density, params):
    """Calculate movement scores for an agent at position (i, j)"""
    w = params['weights']

    # Possible moves: right, left, up, down, stay
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (0, 0)]
    agent_direction = grid[i, j]  # +1 for right, -1 for left

    scores = []
    valid_moves = []  # (ni, nj) target positions

    for di, dj in moves:
        ni, nj = i + di, j + dj  # New position

        # Check if move is within bounds
        if not (0 <= ni < params['grid_length'] and 0 <= nj < params['grid_width']):
            continue

        # Check if target cell is empty (except when staying)
        if grid[ni, nj] != 0 and (di, dj) != (0, 0):
            continue

        # 1. Directional desire: +1 if moving in preferred direction
        if agent_direction > 0 and di > 0:  # Right-mover
            direction_score = 1.0
        elif agent_direction < 0 and di < 0:  # Left-mover
            direction_score = 1.0
        else:
            direction_score = 0.0

        # 2. Density avoidance: prefer less crowded cells
        density_score = 1.0 - min(density[ni, nj] / params['N_max'], 1.0)

        # 3. Panic influence: higher panic increases movement urgency
        panic_score = panic[ni, nj]

        # 4. Random factor: adds stochasticity
        random_score = np.random.uniform(0, 1)  # U(0,1)

        # Compute weighted total score
        total_score = (w['desire'] * direction_score + w['density'] * density_score
                       + w['panic'] * panic_score + w['random'] * random_score)

        valid_moves.append((ni, nj))
        scores.append(total_score)

    # Handle case with no valid moves -> stay
    if len(scores) == 0:
        return [], np.array([])  # empty valid_moves, probabilities

    # Normalize scores to probabilities
    scores = np.array(scores)
    if scores.sum() > 0:
        probabilities = scores / scores.sum()
    else:  # All scores zero, assign equal probability to each valid move.
        probabilities = np.ones_like(scores) / len(scores)

    return valid_moves, probabilities


def move_agents(grid, panic, density, params):
    """Perform synchronous agent movement with conflict resolution"""
    new_grid = np.zeros_like(grid)
    move_plans = {}  # To track move requests

    # Get positions of all agents
    agent_positions = np.argwhere(grid != 0)  # grid cells with values +1 -1

    # Plan moves
    for i, j in agent_positions:
        valid_moves, probabilities = calculate_move_scores(i, j, grid, panic, density, params)

        if len(valid_moves) > 0:
            # Choose a move based on probabilities
            ni, nj = valid_moves[np.random.choice(len(valid_moves), p=probabilities)]

            # Record the move plan
            if (ni, nj) not in move_plans:
                move_plans[(ni, nj)] = []
            move_plans[(ni, nj)].append((i, j, grid[i, j]))
        else:
            # If no valid moves, stay in place
            if (i, j) not in move_plans:
                move_plans[(i, j)] = []
            move_plans[(i, j)].append((i, j, grid[i, j]))

    # Resolve conflicts
    for target, agents in move_plans.items():
        ni, nj = target

        if len(agents) == 1:  # No conflict - move the agent
            i, j, direction = agents[0]
            new_grid[ni, nj] = direction
        else:  # Conflict - randomly select one agent to move
            chosen_idx = np.random.randint(len(agents))
            i, j, direction = agents[chosen_idx]
            new_grid[ni, nj] = direction

            # Other agents stay in place if possible
            for idx, (stay_i, stay_j, stay_direction) in enumerate(agents):
                if idx != chosen_idx and new_grid[stay_i, stay_j] == 0:
                    new_grid[stay_i, stay_j] = stay_direction

    return new_grid


def add_new_agents(grid, t, params):
    """Add new agents at boundaries based on inflow rate"""
    # Number of new agents
    n_new = np.random.poisson(inflow_rate(t, grid, params) * params['dt'])

    for _ in range(n_new):
        if np.random.random() < params['ratio_right']:
            # Right-moving agent at left boundary
            j = np.random.randint(params['grid_width'])
            if grid[0, j] == 0:  # Only if cell is empty
                grid[0, j] = 1
        else:
            # Left-moving agent at right boundary
            j = np.random.randint(params['grid_width'])
            if grid[-1, j] == 0:  # Only if cell is empty
                grid[-1, j] = -1

    return grid


def remove_exited_agents(grid):
    """Remove agents who have exited the simulation area"""
    grid[-1, :] = np.where(grid[-1, :] > 0, 0, grid[-1, :])  # Right-movers at right exit (-1)
    grid[0, :] = np.where(grid[0, :] < 0, 0, grid[0, :])  # Left-movers at left exit (0)
    return grid


def run_simulation(params, record_interval=1.0):
    """Run the lattice-based crowd simulation"""
    grid, panic = initialize_grid(params)

    # Data storage
    times = []
    densities = []  # Average densities
    panics = []  # Average panic levels
    grid_history = []
    panic_history = []

    # Calculate number of simulation steps
    n_steps = int(params['t_end'] / params['dt'])
    next_record = 0.0

    # Main simulation loop
    for step in range(n_steps):
        debug(f'Timestep {step} / {n_steps}')
        t = step * params['dt']

        # Add new agents
        grid = add_new_agents(grid, t, params)

        # Calculate local density
        density = calculate_local_density(grid)

        # Update panic field
        panic = update_panic_field(panic, density, params)

        # Move agents
        grid = move_agents(grid, panic, density, params)

        # Remove exited agents
        grid = remove_exited_agents(grid)

        # Record data at specified intervals
        if t >= next_record:
            times.append(t)
            densities.append(density.mean())
            panics.append(panic.mean())
            grid_history.append(grid.copy())
            panic_history.append(panic.copy())
            next_record += record_interval

    return times, densities, panics, grid_history, panic_history


def plot_time_series(times, densities, panics, num_agents, grid_history, params, name):
    """Plot results from lattice simulation with agent count"""
    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    # Inflow
    axs[0].set_title('Lattice Model: Crowd Dynamics')

    inflow_rates = []
    for i, t in enumerate(times):
        inflow_rates.append(inflow_rate(t, grid_history[i], params))

    axs[0].plot(times, inflow_rates, 'teal', linewidth=2, label='Effective inflow')
    axs[0].fill_between(times, 0, inflow_rates, color='teal', alpha=0.3)
    axs[0].set_ylabel('Inflow Rate\n(agents/s)')
    axs[0].legend(loc='upper right')
    axs[0].grid(alpha=0.3)

    # Density
    axs[1].plot(times, densities, 'royalblue', linewidth=2, label='Density')
    axs[1].fill_between(times, 0, densities, color='royalblue', alpha=0.3)
    axs[1].set_ylabel('Density\n(agents/m²)')
    axs[1].set_ylim(0, max(densities) * 1.1)
    axs[1].legend(loc='upper right')
    axs[1].grid(alpha=0.3)

    # Panic level
    axs[2].plot(times, panics, 'palevioletred', linewidth=2, label='Average Panic level')
    axs[2].fill_between(times, 0, panics, color='palevioletred', alpha=0.3)
    axs[2].set_ylabel('Panic Level')
    axs[2].legend(loc='upper right')
    axs[2].grid(alpha=0.3)

    # Agent count
    axs[3].plot(times, num_agents, 'forestgreen', linewidth=2, label='Agent Count')
    axs[3].fill_between(times, 0, num_agents, color='forestgreen', alpha=0.3)
    axs[3].set_ylabel('Number of Agents')
    axs[3].set_xlabel('Time [seconds]')
    axs[3].legend(loc='upper right')
    axs[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{name}/lattice_time_series.png', dpi=300)
    plt.close()


def create_animation(grid_history, panic_history, params, name, fps=5):
    """Create an animation of the simulation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Set up agent plot (left panel)
    ax1.set_title('Agent Positions')
    ax1.set_xlabel('Length (m)')
    ax1.set_ylabel('Width (m)')
    ax1.set_xlim(-1, params['area_length'] + 1)
    ax1.set_ylim(-0.5, params['area_width'] + 0.5)
    ax1.grid(True, linestyle=':', alpha=0.3)

    # Create scatter plots for each direction
    right_agents = ax1.scatter([], [], s=100, color='royalblue', edgecolor='k', alpha=0.8, label='Right-moving')
    left_agents = ax1.scatter([], [], s=100, color='crimson', edgecolor='k', alpha=0.8, label='Left-moving')
    ax1.legend(loc='upper right')

    # Add entrance/exit zones
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

    ax1.add_patch(entrance_right)
    ax1.add_patch(exit_right)
    ax1.add_patch(entrance_left)
    ax1.add_patch(exit_left)

    # Add walls
    ax1.plot([0, 0], [0, params['area_width']], 'k-', linewidth=2)
    ax1.plot([params['area_length'], params['area_length']], [0, params['area_width']], 'k-', linewidth=2)
    ax1.plot([0, params['area_length']], [0, 0], 'k-', linewidth=2)
    ax1.plot([0, params['area_length']], [params['area_width'], params['area_width']], 'k-', linewidth=2)

    # Add info text
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    agent_text = ax1.text(0.02, 0.90, '', transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    # Set up panic field plot (right panel)
    ax2.set_title('Panic Field Intensity')
    ax2.set_xlabel('Length (m)')
    ax2.set_ylabel('Width (m)')
    ax2.set_xlim(0, params['area_length'])
    ax2.set_ylim(0, params['area_width'])

    # Create initial panic field visualization
    panic_img = ax2.imshow(panic_history[0].T, cmap='hot', vmin=0, vmax=1,
                           extent=[0, params['area_length'], 0, params['area_width']],
                           aspect='auto', origin='lower')
    fig.colorbar(panic_img, ax=ax2, label='Panic Level')

    # Add info text for panic field
    panic_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))

    def update(frame):
        """Update function for animation frame"""
        grid = grid_history[frame]
        panic = panic_history[frame]

        # Get agent positions by direction
        right_positions = []
        left_positions = []
        agent_count = 0
        max_panic = 0

        # Convert grid positions to physical coordinates
        for i in range(params['grid_length']):
            for j in range(params['grid_width']):
                if grid[i, j] != 0:
                    # Calculate physical position (center of cell)
                    x = i * params['cell_size'] + params['cell_size'] / 2
                    y = j * params['cell_size'] + params['cell_size'] / 2

                    if grid[i, j] == 1:  # Right-mover
                        right_positions.append([x, y])
                    else:  # Left-mover
                        left_positions.append([x, y])

                    agent_count += 1
                    max_panic = max(max_panic, panic[i, j])

        # Update agent plots
        if right_positions:
            right_agents.set_offsets(right_positions)
        else:
            right_agents.set_offsets(np.empty((0, 2)))

        if left_positions:
            left_agents.set_offsets(left_positions)
        else:
            left_agents.set_offsets(np.empty((0, 2)))

        # Update panic field
        ax2.clear()
        panic_img = ax2.imshow(panic.T, cmap='hot', vmin=0, vmax=1,
                               extent=[0, params['area_length'], 0, params['area_width']],
                               aspect='auto', origin='lower')
        ax2.set_title('Panic Field Intensity')
        ax2.set_xlabel('Length (m)')
        ax2.set_ylabel('Width (m)')

        # Update info text
        time_text.set_text(f'Time: {10 * frame * params["dt"]:.1f}s')
        agent_text.set_text(f'Agents: {agent_count}')
        panic_text.set_text(f'Max Panic: {max_panic:.2f}')

        return right_agents, left_agents, time_text, agent_text, panic_text

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(grid_history), interval=1000 / fps, blit=False
    )

    plt.tight_layout()
    plt.savefig(f'{name}/lattice_simulation_snapshot.png', dpi=300)

    # Save animation
    writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
    ani.save(f'{name}/lattice_crowd_animation.mp4', writer=writer)
    debug("Animation saved as 'lattice_crowd_animation.mp4'")
    plt.close()

    return ani


def plot_grid_snapshot(grid, panic, params, name, time=None):
    """Plot a snapshot of the grid with cell structure visible"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Create custom colormap for panic field
    panic_cmap = LinearSegmentedColormap.from_list(
        'panic_cmap', ['#f7fbff', '#6baed6', '#2171b5', '#08306b'], N=256)

    # Physical dimensions
    area_length = params['area_length']
    area_width = params['area_width']
    cell_size = params['cell_size']

    # Create grid coordinates
    x = np.arange(0, area_length + cell_size, cell_size)
    y = np.arange(0, area_width + cell_size, cell_size)

    # Plot 1: Agent positions with grid
    ax1.set_title(f'Agent Positions{" at time " + str(time) + "s" if time is not None else ""}')
    ax1.set_xlabel('Length (m)')
    ax1.set_ylabel('Width (m)')
    ax1.set_xlim(0, area_length)
    ax1.set_ylim(0, area_width)

    # Draw grid cells (light grey background)
    for i in range(params['grid_length']):
        for j in range(params['grid_width']):
            rect = patches.Rectangle(
                (i * cell_size, j * cell_size),
                cell_size, cell_size,
                linewidth=0.5, edgecolor='#e0e0e0',
                facecolor='#f5f5f5', alpha=0.7
            )
            ax1.add_patch(rect)

    # Plot agents
    right_positions = []
    left_positions = []
    for i in range(params['grid_length']):
        for j in range(params['grid_width']):
            if grid[i, j] == 1:  # Right-mover
                x_pos = i * cell_size + cell_size / 2
                y_pos = j * cell_size + cell_size / 2
                right_positions.append([x_pos, y_pos])
            elif grid[i, j] == -1:  # Left-mover
                x_pos = i * cell_size + cell_size / 2
                y_pos = j * cell_size + cell_size / 2
                left_positions.append([x_pos, y_pos])

    if right_positions:
        ax1.scatter(*zip(*right_positions), s=100, color='royalblue', edgecolor='k', alpha=0.9, label='Right-moving')
    if left_positions:
        ax1.scatter(*zip(*left_positions), s=100, color='crimson', edgecolor='k', alpha=0.9, label='Left-moving')

    ax1.legend(loc='upper right')

    # Plot 2: Panic field with grid
    ax2.set_title(f'Panic Field{" at time " + str(time) + "s" if time is not None else ""}')
    ax2.set_xlabel('Length (m)')
    ax2.set_ylabel('Width (m)')
    ax2.set_xlim(0, area_length)
    ax2.set_ylim(0, area_width)

    # Create mesh for pcolormesh
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Plot panic field with grid overlay
    pc = ax2.pcolormesh(X, Y, panic, cmap=panic_cmap, vmin=0, vmax=1, shading='auto')

    # Add grid lines
    ax2.grid(True, color='#e0e0e0', linewidth=0.5, alpha=0.7)

    # Add colorbar
    cbar = fig.colorbar(pc, ax=ax2, label='Panic Level')
    cbar.solids.set_edgecolor("face")

    # Add agent positions on top of panic field
    if right_positions:
        ax2.scatter(*zip(*right_positions), s=40, color='royalblue', edgecolor='k', alpha=0.7)
    if left_positions:
        ax2.scatter(*zip(*left_positions), s=40, color='crimson', edgecolor='k', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'{name}/snapshot_t{str(time)}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_spatial_distribution(grid_history, panic_history, params, name):
    """Plot spatial distribution of density and panic at peak time"""
    # Find time with maximum number of agents
    agent_counts = [np.sum(grid != 0) for grid in grid_history]
    peak_idx = np.argmax(agent_counts)
    peak_time = 10 * peak_idx * params['dt']

    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 1])

    # Create custom colormaps
    density_cmap = LinearSegmentedColormap.from_list(
        'density_cmap', ['#f0f9e8', '#7bccc4', '#2b8cbe'], N=256)
    panic_cmap = LinearSegmentedColormap.from_list(
        'panic_cmap', ['#f7fbff', '#6baed6', '#2171b5', '#08306b'], N=256)

    # Calculate density field
    density = calculate_local_density(grid_history[peak_idx])

    # Plot 1: Density distribution
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(density.T, cmap=density_cmap,
                     extent=[0, params['area_length'], 0, params['area_width']],
                     origin='lower', aspect='auto', vmin=0, vmax=(density.max() * 1.1))
    ax1.set_title(f'Averaged Density Distribution at t={peak_time:.1f}s')
    ax1.set_xlabel('Length (m)')
    ax1.set_ylabel('Width (m)')
    cbar1 = fig.colorbar(im1, ax=ax1, label='Density [agents/m²]')
    cbar1.solids.set_edgecolor("face")

    # Add grid
    ax1.grid(True, color='#e0e0e0', linewidth=0.5, alpha=0.5)

    # Plot 2: Panic distribution
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(panic_history[peak_idx].T, cmap=panic_cmap, vmin=0, vmax=1,
                     extent=[0, params['area_length'], 0, params['area_width']],
                     aspect='auto', origin='lower')
    ax2.set_title(f'Panic Distribution at t={peak_time:.1f}s')
    ax2.set_xlabel('Length (m)')
    ax2.set_ylabel('Width (m)')
    cbar2 = fig.colorbar(im2, ax=ax2, label='Panic Level')
    cbar2.solids.set_edgecolor("face")

    # Add grid
    ax2.grid(True, color='#e0e0e0', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(f'{name}/lattice_spatial_distribution.png', dpi=300)
    plt.close()


def plot_flow_rate(times, grid_history, params, name):
    """Calculate and plot flow rate through exits over time"""
    # Calculate flow rate (agents exiting per time step)
    flow_rate = np.zeros(len(times))
    prev_agents = np.sum(grid_history[0] != 0)

    for i in range(1, len(times)):
        current_agents = np.sum(grid_history[i] != 0)
        # Flow rate = agents disappeared per second (assuming some exited)
        flow_rate[i] = max(0, (prev_agents - current_agents) / params['dt'])
        prev_agents = current_agents

    plt.figure(figsize=(10, 6))
    plt.plot(times, flow_rate, 'slateblue', linewidth=2)
    plt.fill_between(times, 0, flow_rate, color='slateblue', alpha=0.3)
    plt.title('Exit Flow Rate Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Flow Rate (agents/s)')
    plt.grid(alpha=0.3)
    plt.ylim(0, max(flow_rate) * 1.1)
    plt.tight_layout()
    plt.savefig(f'{name}/lattice_flow_rate.png', dpi=300)
    plt.close()


def main():
    # Build scenario dictionary by overriding one parameter per scenario
    scenarios = {
        'Base': params,
        'High Panic Sensitivity': {**params, 'alpha': 0.02},
        'Slow Panic Decay': {**params, 'beta': 0.001},
        'Strong Density Avoidance': {**params, 'weights': {**params['weights'], 'density': 1.2}},
        'Weak Density Avoidance': {**params, 'weights': {**params['weights'], 'density': 0.2}},
        'High Randomness': {**params, 'weights': {**params['weights'], 'random': 0.8}},
        'Narrow Alley': {**params, 'area_width': 2.0, 'grid_width': int(2.0 / params['cell_size'])},
        'Balanced Directions': {**params, 'ratio_right': 0.5},
    }

    # Container for all scenario outputs
    all_results = {}

    # Loop over scenarios
    for name, pset in scenarios.items():
        print(f"\n=== Running scenario: {name} ===")
        sim_params = copy.deepcopy(pset)

        # Recalculate derived params
        sim_params['grid_width'] = int(sim_params['area_width'] / sim_params['cell_size'])
        sim_params['grid_length'] = int(sim_params['area_length'] / sim_params['cell_size'])
        sim_params['total_cells'] = sim_params['grid_length'] * sim_params['grid_width']

        # Create output folder
        os.makedirs(name, exist_ok=True)

        # Run simulation
        start_time = time.time()
        times, densities, panics, grid_history, panic_history = run_simulation(sim_params, record_interval=1.0)
        print(f"Simulation for '{name}' took {(time.time() - start_time) / 60.0:.2f} min with dt = {params['dt']}")

        # Compute agent count and flow rate
        num_agents = [np.sum(g != 0) for g in grid_history]
        flow_rate = []
        prev = num_agents[0]
        for na in num_agents:
            flow_rate.append(max(0, (prev - na) / sim_params['dt']))
            prev = na

        # Store for later comparison
        all_results[name] = {
            'times': times,
            'density': densities,
            'panic': panics,
            'agents': num_agents,
            'flow_rate': flow_rate
        }

        # Per-scenario plotting
        peak_idx = np.argmax(num_agents)
        plot_grid_snapshot(grid_history[peak_idx], panic_history[peak_idx], sim_params, name, time=times[peak_idx])
        plot_spatial_distribution(grid_history, panic_history, sim_params, name)
        plot_flow_rate(times, grid_history, sim_params, name)
        create_animation(grid_history, panic_history, sim_params, name, fps=10)
        plt.close()

    # Choose colors
    palette = sns.color_palette('mako', n_colors=len(scenarios))
    scenario_colors = dict(zip(scenarios.keys(), palette))

    # 1. Density Comparison
    plt.figure(figsize=(10, 6))
    for name, res in all_results.items():
        plt.plot(res['times'], res['density'], label=name, color=scenario_colors[name])
    plt.xlabel('Time (s)')
    plt.ylabel('Density (agents/m²)')
    plt.title('Density Across Scenarios')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('compare_density_lattice.png', dpi=300)
    plt.show()

    # 2. Panic Comparison
    plt.figure(figsize=(10, 6))
    for name, res in all_results.items():
        plt.plot(res['times'], res['panic'], label=name, color=scenario_colors[name])
    plt.xlabel('Time (s)')
    plt.ylabel('Average Panic Level')
    plt.title('Panic Across Scenarios')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('compare_panic_lattice.png', dpi=300)
    plt.show()

    # 3. Agent Count Comparison
    plt.figure(figsize=(10, 6))
    for name, res in all_results.items():
        plt.plot(res['times'], res['agents'], label=name, color=scenario_colors[name])
    plt.xlabel('Time (s)')
    plt.ylabel('Number of Agents')
    plt.title('Agent Count Across Scenarios')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('compare_agents_lattice.png', dpi=300)
    plt.show()

    # 4. Flow Rate Comparison
    plt.figure(figsize=(10, 6))
    for name, res in all_results.items():
        plt.plot(res['times'], res['flow_rate'], label=name, color=scenario_colors[name])
    plt.xlabel('Time (s)')
    plt.ylabel('Flow Rate (agents/s)')
    plt.title('Exit Flow Rate Across Scenarios')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('compare_flow_rate_lattice.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
