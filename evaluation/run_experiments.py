"""
Experimental Evaluation Script for Battleship Agents

This module runs head-to-head experiments comparing the probability-based
agent against the Q-learning RL agent. It conducts a controlled experimental
evaluation using the same random ship configurations for both agents to
ensure fair comparison.

Experimental Design:
- Number of games: 1,000 per agent
- Random seed: 42 (for reproducibility)
- Same ship configurations used for both agents
- Metrics: shots to win, accuracy, statistical significance

Statistical Tests:
- Independent samples t-test for mean comparison
- Mann-Whitney U test for distribution comparison
- Cohen's d for effect size measurement

Output Files:
- results/probability_agent_results.csv: Per-game results for probability agent
- results/rl_agent_results.csv: Per-game results for RL agent
- results/summary_statistics.csv: Aggregated statistics and test results

Author: Venkatashivasai Muppidi
Course: Foundations of Artificial Intelligence
University: Northeastern University
"""

import numpy as np
import random
import time
import os
import csv
from typing import List, Tuple, Optional
from scipy import stats

from probability_agent import ProbabilityAgent
from rl_agent import RLAgent

# =============================================================================
# GAME CONSTANTS
# =============================================================================

GRID_SIZE = 10  # Standard 10x10 Battleship grid

# Standard Battleship fleet: (name, size)
SHIPS = [
    ('Carrier', 5),
    ('Battleship', 4),
    ('Cruiser', 3),
    ('Submarine', 3),
    ('Destroyer', 2),
]

SHIP_SIZES = [size for _, size in SHIPS]  # [5, 4, 3, 3, 2]
NUM_GAMES = 1000  # Number of games for statistical significance


# =============================================================================
# SIMPLE BOARD (for fast evaluation)
# =============================================================================


class SimpleBoard:
    """
    Lightweight game board for experiment evaluation.

    This is a simplified version optimized for fast simulation
    during experiments. It lacks GUI elements but provides
    the core game mechanics needed for evaluation.

    Grid values:
        0: Empty/water
        1: Ship (unhit)
        2: Hit
        3: Sunk
       -1: Miss
    """

    def __init__(self, grid_size: int = 10):
        """
        Initialize an empty board.

        Args:
            grid_size: Size of the board (default 10)
        """
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.ships: List[dict] = []

    def place_ships_randomly(self, ship_sizes: List[int]):
        """
        Randomly place all ships on the board.

        Args:
            ship_sizes: List of ship sizes to place [5, 4, 3, 3, 2]
        """
        # Reset the board
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.ships = []

        for size in ship_sizes:
            placed = False
            attempts = 0

            # Try random positions until valid placement found
            while not placed and attempts < 1000:
                row = random.randint(0, self.grid_size - 1)
                col = random.randint(0, self.grid_size - 1)
                horizontal = random.choice([True, False])

                if self._can_place(row, col, size, horizontal):
                    positions = self._place_ship(row, col, size, horizontal)
                    self.ships.append({
                        'size': size,
                        'positions': positions,
                        'hits': 0
                    })
                    placed = True

                attempts += 1

            # If placement failed, restart entire process
            if not placed:
                return self.place_ships_randomly(ship_sizes)

    def _can_place(self, row: int, col: int, size: int, horizontal: bool) -> bool:
        """
        Check if a ship can be placed at the given position.

        Args:
            row, col: Starting position
            size: Ship size
            horizontal: True for horizontal, False for vertical

        Returns:
            True if placement is valid
        """
        for i in range(size):
            if horizontal:
                c = col + i
                r = row
            else:
                c = col
                r = row + i

            # Check bounds
            if r >= self.grid_size or c >= self.grid_size:
                return False
            # Check for overlap
            if self.grid[r, c] != 0:
                return False

        return True

    def _place_ship(self, row: int, col: int, size: int, horizontal: bool) -> List[Tuple[int, int]]:
        """
        Place a ship on the board.

        Args:
            row, col: Starting position
            size: Ship size
            horizontal: True for horizontal, False for vertical

        Returns:
            List of (row, col) positions occupied by the ship
        """
        positions = []

        for i in range(size):
            if horizontal:
                c = col + i
                r = row
            else:
                c = col
                r = row + i

            self.grid[r, c] = 1  # Mark as ship
            positions.append((r, c))

        return positions

    def attack(self, pos: Tuple[int, int]) -> Tuple[bool, Optional[int], bool]:
        """
        Process an attack on the board.

        Args:
            pos: (row, col) attack coordinates

        Returns:
            Tuple of (hit, sunk_ship_size, game_over)
        """
        row, col = pos

        if self.grid[row, col] == 1:
            # Hit a ship
            self.grid[row, col] = 2

            for ship in self.ships:
                if pos in ship['positions']:
                    ship['hits'] += 1

                    if ship['hits'] == ship['size']:
                        # Ship is sunk
                        for p in ship['positions']:
                            self.grid[p[0], p[1]] = 3

                        all_sunk = all(s['hits'] == s['size'] for s in self.ships)
                        return True, ship['size'], all_sunk

                    return True, None, False

        elif self.grid[row, col] == 0:
            # Miss
            self.grid[row, col] = -1
            return False, None, False

        # Cell already attacked
        return False, None, False


# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================


def run_game(agent, board: SimpleBoard) -> dict:
    """
    Run a single game with the given agent.

    Args:
        agent: Agent to evaluate (ProbabilityAgent or RLAgent)
        board: Game board with ships placed

    Returns:
        Dictionary with game statistics (shots, hits, accuracy)
    """
    agent.reset()
    shots = 0
    hits = 0

    done = False
    while not done:
        # Get agent's target choice
        target = agent.choose_target()

        # Execute attack
        hit, sunk_size, done = board.attack(target)

        # Report result to agent
        agent.record_result(target, hit, sunk_size)

        # Track statistics
        shots += 1
        if hit:
            hits += 1

    accuracy = (hits / shots * 100) if shots > 0 else 0

    return {
        'shots': shots,
        'hits': hits,
        'accuracy': accuracy
    }


def run_experiments(num_games: int = NUM_GAMES):
    """
    Run the full experimental evaluation.

    Compares probability agent vs RL agent using the same ship
    configurations for fair comparison. Performs statistical tests
    and saves results to CSV files.

    Args:
        num_games: Number of games to run per agent

    Returns:
        Tuple of (prob_results, rl_results, prob_stats, rl_stats)
    """
    # Create results directory
    os.makedirs('results', exist_ok=True)

    # Initialize agents
    prob_agent = ProbabilityAgent(grid_size=GRID_SIZE, ship_sizes=SHIP_SIZES)

    rl_agent = RLAgent(grid_size=GRID_SIZE, ship_sizes=SHIP_SIZES)
    if os.path.exists('models/rl_agent.pkl'):
        rl_agent.load('models/rl_agent.pkl')
        rl_agent.epsilon = 0  # Disable exploration for evaluation
    else:
        print("WARNING: No trained RL model found. Using untrained agent.")

    # Storage for results
    prob_results = []
    rl_results = []

    # Print experiment header
    print("=" * 70)
    print("BATTLESHIP EXPERIMENT")
    print("=" * 70)
    print(f"Running {num_games} games per agent...")
    print("-" * 70)

    # Generate ship configurations with fixed seed for reproducibility
    # Both agents will play against the exact same configurations
    random.seed(42)
    np.random.seed(42)
    ship_configs = []
    for _ in range(num_games):
        board = SimpleBoard(GRID_SIZE)
        board.place_ships_randomly(SHIP_SIZES)
        # Save ship positions for reuse
        ship_positions = [(ship['size'], ship['positions'].copy()) for ship in board.ships]
        ship_configs.append(ship_positions)

    # Run Probability Agent experiments
    print("\nRunning Probability Agent...")
    start_time = time.time()

    for i, config in enumerate(ship_configs):
        # Reconstruct board from saved configuration
        board = SimpleBoard(GRID_SIZE)
        board.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        board.ships = []
        for size, positions in config:
            for r, c in positions:
                board.grid[r, c] = 1
            board.ships.append({'size': size, 'positions': positions, 'hits': 0})

        # Run game and collect results
        result = run_game(prob_agent, board)
        prob_results.append(result)

        # Progress update
        if (i + 1) % 200 == 0:
            print(f"  Completed {i + 1}/{num_games} games...")

    prob_time = time.time() - start_time
    print(f"  Done! Time: {prob_time:.1f}s")

    # Run RL Agent experiments
    print("\nRunning RL Agent...")
    start_time = time.time()

    for i, config in enumerate(ship_configs):
        # Reconstruct board from saved configuration
        board = SimpleBoard(GRID_SIZE)
        board.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        board.ships = []
        for size, positions in config:
            for r, c in positions:
                board.grid[r, c] = 1
            board.ships.append({'size': size, 'positions': positions, 'hits': 0})

        # Run game and collect results
        result = run_game(rl_agent, board)
        rl_results.append(result)

        # Progress update
        if (i + 1) % 200 == 0:
            print(f"  Completed {i + 1}/{num_games} games...")

    rl_time = time.time() - start_time
    print(f"  Done! Time: {rl_time:.1f}s")

    # ==========================================================================
    # STATISTICAL ANALYSIS
    # ==========================================================================

    # Extract metrics
    prob_shots = [r['shots'] for r in prob_results]
    prob_acc = [r['accuracy'] for r in prob_results]
    rl_shots = [r['shots'] for r in rl_results]
    rl_acc = [r['accuracy'] for r in rl_results]

    # Calculate summary statistics for probability agent
    prob_stats = {
        'agent': 'Probability',
        'games': num_games,
        'avg_shots': np.mean(prob_shots),
        'std_shots': np.std(prob_shots),
        'min_shots': np.min(prob_shots),
        'max_shots': np.max(prob_shots),
        'median_shots': np.median(prob_shots),
        'avg_accuracy': np.mean(prob_acc),
        'std_accuracy': np.std(prob_acc)
    }

    # Calculate summary statistics for RL agent
    rl_stats = {
        'agent': 'RL (Q-Learning)',
        'games': num_games,
        'avg_shots': np.mean(rl_shots),
        'std_shots': np.std(rl_shots),
        'min_shots': np.min(rl_shots),
        'max_shots': np.max(rl_shots),
        'median_shots': np.median(rl_shots),
        'avg_accuracy': np.mean(rl_acc),
        'std_accuracy': np.std(rl_acc)
    }

    # Statistical tests
    # Independent samples t-test (parametric)
    t_stat, p_value = stats.ttest_ind(prob_shots, rl_shots)

    # Mann-Whitney U test (non-parametric)
    u_stat, mann_whitney_p = stats.mannwhitneyu(prob_shots, rl_shots, alternative='two-sided')

    # Cohen's d effect size
    cohens_d = (np.mean(rl_shots) - np.mean(prob_shots)) / np.sqrt(
        (np.std(prob_shots) ** 2 + np.std(rl_shots) ** 2) / 2)

    # ==========================================================================
    # PRINT RESULTS
    # ==========================================================================

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\nPROBABILITY AGENT:")
    print(f"  Games played:     {prob_stats['games']}")
    print(f"  Average shots:    {prob_stats['avg_shots']:.2f} ± {prob_stats['std_shots']:.2f}")
    print(f"  Median shots:     {prob_stats['median_shots']:.1f}")
    print(f"  Min shots:        {prob_stats['min_shots']}")
    print(f"  Max shots:        {prob_stats['max_shots']}")
    print(f"  Average accuracy: {prob_stats['avg_accuracy']:.2f}%")

    print("\nRL AGENT (Q-Learning):")
    print(f"  Games played:     {rl_stats['games']}")
    print(f"  Average shots:    {rl_stats['avg_shots']:.2f} ± {rl_stats['std_shots']:.2f}")
    print(f"  Median shots:     {rl_stats['median_shots']:.1f}")
    print(f"  Min shots:        {rl_stats['min_shots']}")
    print(f"  Max shots:        {rl_stats['max_shots']}")
    print(f"  Average accuracy: {rl_stats['avg_accuracy']:.2f}%")

    print("\nSTATISTICAL ANALYSIS:")
    print(f"  Difference:       {rl_stats['avg_shots'] - prob_stats['avg_shots']:.2f} shots")
    print(f"  T-test p-value:   {p_value:.2e}")
    print(f"  Mann-Whitney p:   {mann_whitney_p:.2e}")
    print(f"  Cohen's d:        {cohens_d:.3f}")

    # Interpret significance level
    if p_value < 0.001:
        sig_level = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        sig_level = "very significant (p < 0.01)"
    elif p_value < 0.05:
        sig_level = "significant (p < 0.05)"
    else:
        sig_level = "not significant (p >= 0.05)"

    print(f"  Significance:     {sig_level}")

    # Interpret effect size (Cohen's d conventions)
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"

    print(f"  Effect size:      {effect_size}")

    print("=" * 70)

    # ==========================================================================
    # SAVE RESULTS TO CSV
    # ==========================================================================

    # Per-game results for probability agent
    with open('results/probability_agent_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['game', 'shots', 'hits', 'accuracy'])
        for i, r in enumerate(prob_results):
            writer.writerow([i + 1, r['shots'], r['hits'], r['accuracy']])

    # Per-game results for RL agent
    with open('results/rl_agent_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['game', 'shots', 'hits', 'accuracy'])
        for i, r in enumerate(rl_results):
            writer.writerow([i + 1, r['shots'], r['hits'], r['accuracy']])

    # Summary statistics
    with open('results/summary_statistics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'probability_agent', 'rl_agent'])
        writer.writerow(['games', prob_stats['games'], rl_stats['games']])
        writer.writerow(['avg_shots', f"{prob_stats['avg_shots']:.2f}", f"{rl_stats['avg_shots']:.2f}"])
        writer.writerow(['std_shots', f"{prob_stats['std_shots']:.2f}", f"{rl_stats['std_shots']:.2f}"])
        writer.writerow(['median_shots', f"{prob_stats['median_shots']:.1f}", f"{rl_stats['median_shots']:.1f}"])
        writer.writerow(['min_shots', prob_stats['min_shots'], rl_stats['min_shots']])
        writer.writerow(['max_shots', prob_stats['max_shots'], rl_stats['max_shots']])
        writer.writerow(['avg_accuracy', f"{prob_stats['avg_accuracy']:.2f}", f"{rl_stats['avg_accuracy']:.2f}"])
        writer.writerow(['t_test_p_value', f"{p_value:.2e}", ''])
        writer.writerow(['mann_whitney_p', f"{mann_whitney_p:.2e}", ''])
        writer.writerow(['cohens_d', f"{cohens_d:.3f}", ''])

    print("\nResults saved to:")
    print("  - results/probability_agent_results.csv")
    print("  - results/rl_agent_results.csv")
    print("  - results/summary_statistics.csv")

    return prob_results, rl_results, prob_stats, rl_stats


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    run_experiments(NUM_GAMES)