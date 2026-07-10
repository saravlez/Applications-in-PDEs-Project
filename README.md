# Crowd Dynamics Simulation Project
### Modelling Crowd Dynamics to Prevent Crushes - Itaewon Halloween 2022
This repository contains three complementary mathematical models for simulating crowd dynamics, developed to reproduce and understand the fundamental mechanisms that led to the Itaewon Halloween crowd crush (October 29, 2022) in Seoul, South Korea.

#### Project Overview
The project implements three models to capture different aspects of crowd behaviour:
- Population ODE Model. A non-spatial system of ODEs that approximates overall crowd size and panic accumulation over time.
- Off-Lattice Particle Model. An agent-based simulation in continuous space where individuals interact via repulsive and panic-driven forces.
- Lattice (Cellular Automaton) Model. A discrete grid framework that emphasises how narrow geometry and synchronous movement rules can generate jams and crush phenomena.

**Animations**: View full animation gallery → [Animations.md](Animations.md)

### Repository Structure
```
Code/
├── Population Model
│   ├── population_model.py           # Main ODE simulation
│   ├── population_comparison.py      # Parameter sensitivity script
│   └── steady_states.py              # Equilibrium and stability analysis
├── Off Lattice Model
│   ├── off_lattice_bidirectional.py   # Main simulation script
│   ├── off_lattice_crush.py           # Overcrowded scenario (from Crush 0.05) 
│   ├── off_lattice_comparison.py      # Parameter comparison 
│   ├──bidirectional_crowd_flow.mp4    # Normal Animation  
│   └──bidirectional_crush_flow.mp4    # Crush Animation
└── Lattice Model
    ├── lattice_model.py               # Main cellular automaton simulation
    ├── lattice_crush.py               # Overcrowded scenario
    ├── lattice_comparison.py          # Parameter comparison 
    ├── lattice_crowd_animation.mp4    # Normal Animation 
    └── lattice_crush_animation.mp4    # Crush Animation
```

### Prerequisites
- NumPy
- SciPy
- Matplotlib
- Seaborn

### Contact 
This project is for academic and research purposes. Please feel free to email me at saravlezfue@gmail.com if you have any questions.
