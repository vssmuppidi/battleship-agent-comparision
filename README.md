# Autonomous Agents for Battleship

**Course:** Artificial Intelligence  
**Author:** Venkatashivasai Muppidi

---

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [File Descriptions](#file-descriptions)

---

## Requirements

- Python 3.10 or higher
- Operating System: macOS, Windows, or Linux

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pygame | 2.6.1 | Game visualization |
| numpy | latest | Numerical operations |
| scipy | latest | Statistical analysis |
| matplotlib | latest | Chart generation |
| pandas | latest | Data handling |

---

## Installation

1. Clone or download the repository:
```bash
cd battleship-ai
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install pygame numpy scipy matplotlib pandas
```

4. Verify installation:
```bash
python -c "import pygame, numpy, scipy, matplotlib, pandas; print('All dependencies installed successfully')"
```

---

## Project Structure

```
battleship-ai/
│
├── README.md                     # Project documentation (this file)
│
├── Core Game
│   └── battleship_game.py        # Game engine with Pygame GUI
│
├── Probability Agent
│   ├── probability_agent.py      # Agent implementation
│   └── game_with_agent.py        # GUI interface for agent
│
├── RL Agent
│   ├── rl_agent.py               # Q-Learning agent implementation
│   ├── train_rl_agent.py         # Training script
│   └── game_with_rl_agent.py     # GUI interface for agent
│
├── Experiments
│   ├── run_experiments.py        # Experimental evaluation script
│   └── generate_charts.py        # Visualization generator
│
├── models/
│   └── rl_agent.pkl              # Trained RL agent model
│
└── results/
    ├── probability_agent_results.csv
    ├── rl_agent_results.csv
    ├── summary_statistics.csv
    └── charts/
        ├── bar_chart_comparison.png
        ├── box_plot.png
        ├── histogram.png
        ├── accuracy_comparison.png
        ├── cumulative_wins.png
        └── violin_plot.png
```

---

## Usage

### 1. Play the Base Game (Manual Mode)

```bash
python battleship_game.py
```

**Controls:**
- Left-click: Place ships on your grid
- Right-click: Rotate ship orientation
- SPACE: Auto-place remaining ships
- H: Toggle probability heatmap
- R: Restart game

### 2. Run the Probability Agent

```bash
python game_with_agent.py
```

**Controls:**
- A: Toggle agent ON/OFF
- UP/DOWN: Adjust agent speed
- SPACE: Single-step mode
- H: Toggle heatmap display

### 3. Train the RL Agent

```bash
python train_rl_agent.py --episodes 5000
```

The trained model is saved to `models/rl_agent.pkl`.

### 4. Run the RL Agent

```bash
python game_with_rl_agent.py
```

Uses the same controls as the probability agent interface.

### 5. Run Experimental Evaluation

```bash
python run_experiments.py
```

Executes 1,000 games for each agent using identical ship configurations and outputs statistical analysis.

### 6. Generate Visualization Charts

```bash
python generate_charts.py
```

Creates comparison charts in the `results/charts/` directory.

---

## File Descriptions

| File | Description |
|------|-------------|
| `battleship_game.py` | Core game engine featuring a 10×10 grid, ship placement mechanics, attack system, hit/miss animations, and probability heatmap visualization |
| `probability_agent.py` | Probability-based agent using hunt/target modes, parity optimization, edge penalties, and ship density calculations |
| `game_with_agent.py` | Pygame interface for running and visualizing the probability agent |
| `rl_agent.py` | Q-Learning agent with state abstraction, epsilon-greedy exploration, and integrated hunt/target heuristics |
| `train_rl_agent.py` | Training script with configurable episodes, learning rate, discount factor, and evaluation metrics |
| `game_with_rl_agent.py` | Pygame interface for running and visualizing the RL agent |
| `run_experiments.py` | Experimental framework that runs both agents on identical configurations and performs statistical analysis |
| `generate_charts.py` | Generates publication-ready comparison charts using Matplotlib |
