"""
Training Script for Q-Learning Battleship Agent

This module provides the training infrastructure for the Q-learning based
Battleship agent. It includes a simplified game environment for fast training,
the main training loop with progress logging, and evaluation functionality.

Training uses the following reward structure:
- Win (all ships sunk): +20.0
- Sink a ship: +5.0
- Hit a ship: +2.0
- Miss: -0.5

Hyperparameters:
- Learning rate (alpha): 0.15
- Discount factor (gamma): 0.95
- Epsilon start: 1.0, end: 0.05, decay: 0.995
- Training episodes: 5,000

Author: Venkatashivasai Muppidi
Course: Foundations of Artificial Intelligence
University: Northeastern University
"""

import numpy as np
import random
import time
from typing import List, Tuple, Optional
from collections import deque
import os

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

# Total cells occupied by ships (5+4+3+3+2 = 17)
TOTAL_SHIP_CELLS = sum(size for _, size in SHIPS)


# =============================================================================
# SIMPLE BOARD (for fast training)
# =============================================================================


class SimpleBoard:
    """
    Lightweight game board for training.

    This is a simplified version of the main game board, optimized for
    fast simulation during training. It lacks GUI elements and animations
    but provides the core game mechanics.

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
            - hit: True if attack hit a ship
            - sunk_ship_size: Size of ship if sunk, None otherwise
            - game_over: True if all ships are sunk
        """
        row, col = pos

        if self.grid[row, col] == 1:
            # Hit a ship
            self.grid[row, col] = 2  # Mark as hit

            # Find which ship was hit
            for ship in self.ships:
                if pos in ship['positions']:
                    ship['hits'] += 1

                    # Check if ship is sunk
                    if ship['hits'] == ship['size']:
                        # Mark all positions as sunk
                        for p in ship['positions']:
                            self.grid[p[0], p[1]] = 3

                        # Check if game is over (all ships sunk)
                        all_sunk = all(s['hits'] == s['size'] for s in self.ships)
                        return True, ship['size'], all_sunk

                    return True, None, False

        elif self.grid[row, col] == 0:
            # Miss (hit water)
            self.grid[row, col] = -1  # Mark as miss
            return False, None, False

        # Cell was already attacked (shouldn't happen with valid agent)
        return False, None, False


# =============================================================================
# TRAINING ENVIRONMENT
# =============================================================================


class TrainingEnvironment:
    """
    Environment wrapper for training the RL agent.

    Provides a step-based interface similar to OpenAI Gym style,
    with reward calculation and game state tracking.

    Reward Structure:
        - Win (all ships sunk): +20.0
        - Sink a ship: +5.0
        - Hit a ship: +2.0
        - Miss: -0.5
    """

    def __init__(self, grid_size: int = 10, ship_sizes: List[int] = None):
        """
        Initialize the training environment.

        Args:
            grid_size: Size of the game board
            ship_sizes: List of ship sizes
        """
        self.grid_size = grid_size
        self.ship_sizes = ship_sizes or [5, 4, 3, 3, 2]
        self.total_ship_cells = sum(self.ship_sizes)
        self.board = SimpleBoard(grid_size)
        self.reset()

    def reset(self):
        """Reset the environment for a new episode."""
        self.board.place_ships_randomly(self.ship_sizes)
        self.shots_fired = 0
        self.hits = 0
        self.ships_sunk = 0

    def step(self, action: Tuple[int, int]) -> Tuple[float, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: (row, col) coordinates to attack

        Returns:
            Tuple of (reward, done, info)
            - reward: Numerical reward for the action
            - done: True if game is over
            - info: Dictionary with additional information
        """
        hit, sunk_size, game_over = self.board.attack(action)

        # Update statistics
        self.shots_fired += 1
        if hit:
            self.hits += 1
        if sunk_size:
            self.ships_sunk += 1

        # Calculate reward based on outcome
        if game_over:
            reward = 20.0  # Win bonus
        elif sunk_size:
            reward = 5.0  # Sink bonus
        elif hit:
            reward = 2.0  # Hit bonus
        else:
            reward = -0.5  # Miss penalty

        # Additional info for logging/debugging
        info = {
            'hit': hit,
            'sunk_size': sunk_size,
            'shots': self.shots_fired,
            'hits': self.hits,
            'ships_sunk': self.ships_sunk
        }

        return reward, game_over, info


# =============================================================================
# TRAINING FUNCTION
# =============================================================================


def train(
        num_episodes: int = 5000,
        save_interval: int = 1000,
        log_interval: int = 100,
        model_path: str = "models/rl_agent.pkl"
):
    """
    Train the Q-learning agent.

    Args:
        num_episodes: Number of training episodes (games)
        save_interval: Save model every N episodes
        log_interval: Print progress every N episodes
        model_path: Path to save the trained model

    Returns:
        Trained RLAgent instance
    """
    # Create model directory if needed
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)

    # Initialize environment and agent
    ship_sizes = [size for _, size in SHIPS]
    env = TrainingEnvironment(GRID_SIZE, ship_sizes)
    agent = RLAgent(
        grid_size=GRID_SIZE,
        ship_sizes=ship_sizes,
        learning_rate=0.15,  # How quickly to update Q-values
        gamma=0.95,  # Discount factor for future rewards
        epsilon_start=1.0,  # Start with full exploration
        epsilon_end=0.05,  # Minimum exploration rate
        epsilon_decay=0.995  # Decay rate per episode
    )

    # Tracking metrics (rolling window of last 100 episodes)
    episode_shots = deque(maxlen=100)
    episode_accuracies = deque(maxlen=100)
    best_avg_shots = float('inf')

    print(f"Starting training for {num_episodes} episodes...")
    print("-" * 70)

    start_time = time.time()

    # Main training loop
    for episode in range(1, num_episodes + 1):
        # Reset for new episode
        env.reset()
        agent.reset()

        done = False

        # Play one complete game
        while not done:
            # Get current state
            state = agent._get_state_key()

            # Choose action (with exploration)
            target = agent.choose_target(training=True)
            action = agent.pos_to_action(target)

            # Execute action and get reward
            reward, done, info = env.step(target)

            # Update agent's internal state
            agent.record_result(target, info['hit'], info['sunk_size'])

            # Q-learning update
            next_state = agent._get_state_key()
            agent.update_q_value(state, action, reward, next_state, done)

        # End of episode: decay exploration rate
        agent.end_episode()

        # Track metrics
        episode_shots.append(env.shots_fired)
        accuracy = (env.hits / env.shots_fired * 100) if env.shots_fired > 0 else 0
        episode_accuracies.append(accuracy)

        # Logging
        if episode % log_interval == 0:
            avg_shots = np.mean(episode_shots)
            avg_accuracy = np.mean(episode_accuracies)
            elapsed = time.time() - start_time

            # Mark improvements
            improved = ""
            if avg_shots < best_avg_shots:
                best_avg_shots = avg_shots
                improved = " *"

            print(f"Episode {episode:5d} | "
                  f"Shots: {avg_shots:5.1f}{improved:2s} | "
                  f"Acc: {avg_accuracy:5.1f}% | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Q-states: {len(agent.q_table):5d} | "
                  f"Time: {elapsed:.0f}s")

        # Periodic save
        if episode % save_interval == 0:
            agent.save(model_path)

    # Final save
    agent.save(model_path)

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"Training complete! Total time: {total_time:.1f}s")
    print(f"Best average shots: {best_avg_shots:.1f}")
    print(f"Final model saved to: {model_path}")

    return agent


# =============================================================================
# EVALUATION FUNCTION
# =============================================================================


def evaluate(agent: RLAgent, num_games: int = 100) -> dict:
    """
    Evaluate a trained agent's performance.

    Runs multiple games with exploration disabled (greedy policy)
    to measure true performance.

    Args:
        agent: Trained RLAgent to evaluate
        num_games: Number of games to play

    Returns:
        Dictionary with evaluation statistics
    """
    ship_sizes = [size for _, size in SHIPS]
    env = TrainingEnvironment(GRID_SIZE, ship_sizes)

    all_shots = []
    all_accuracies = []

    # Disable exploration for evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0

    for game in range(num_games):
        env.reset()
        agent.reset()

        done = False
        while not done:
            target = agent.choose_target(training=False)
            _, done, info = env.step(target)
            agent.record_result(target, info['hit'], info['sunk_size'])

        all_shots.append(env.shots_fired)
        accuracy = (env.hits / env.shots_fired * 100) if env.shots_fired > 0 else 0
        all_accuracies.append(accuracy)

    # Restore original epsilon
    agent.epsilon = original_epsilon

    # Compile results
    results = {
        'games': num_games,
        'avg_shots': np.mean(all_shots),
        'std_shots': np.std(all_shots),
        'min_shots': np.min(all_shots),
        'max_shots': np.max(all_shots),
        'avg_accuracy': np.mean(all_accuracies),
    }

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Games played: {results['games']}")
    print(f"Average shots to win: {results['avg_shots']:.1f} ± {results['std_shots']:.1f}")
    print(f"Best game: {results['min_shots']} shots")
    print(f"Worst game: {results['max_shots']} shots")
    print(f"Average accuracy: {results['avg_accuracy']:.1f}%")
    print("=" * 60)

    return results


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    import argparse

    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Train RL Battleship Agent")
    parser.add_argument("--episodes", type=int, default=5000,
                        help="Number of training episodes")
    parser.add_argument("--save-interval", type=int, default=1000,
                        help="Save model every N episodes")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Print progress every N episodes")
    parser.add_argument("--model-path", type=str, default="models/rl_agent.pkl",
                        help="Path to save/load model")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate existing model instead of training")
    parser.add_argument("--eval-games", type=int, default=100,
                        help="Number of games for evaluation")

    args = parser.parse_args()

    if args.evaluate:
        # Load and evaluate existing model
        agent = RLAgent(grid_size=GRID_SIZE, ship_sizes=[5, 4, 3, 3, 2])
        agent.load(args.model_path)
        evaluate(agent, args.eval_games)
    else:
        # Train new model
        trained_agent = train(
            num_episodes=args.episodes,
            save_interval=args.save_interval,
            log_interval=args.log_interval,
            model_path=args.model_path
        )

        # Evaluate after training
        if trained_agent:
            print("\nEvaluating trained agent...")
            evaluate(trained_agent, num_games=100)