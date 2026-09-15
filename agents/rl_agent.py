"""
Reinforcement Learning Battleship Agent (Q-Learning)

This module implements an autonomous Battleship agent using tabular Q-learning.
The agent learns optimal targeting strategies through experience, maintaining
a Q-table that maps game states to action values.

The agent incorporates domain heuristics (hunt/target mode, center preference,
parity patterns) to bootstrap learning, which Q-learning then refines through
gameplay experience.

Key Features:
- Tabular Q-learning with state abstraction
- Epsilon-greedy exploration strategy
- Hunt/target mode logic for efficient ship destruction
- Position value initialization favoring center cells
- Pickle-based model persistence

Author: Venkatashivasai Muppidi
Course: Foundations of Artificial Intelligence
University: Northeastern University
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Dict
from enum import Enum
import pickle

# Flag for compatibility (originally used for PyTorch availability)
TORCH_AVAILABLE = True


class AgentMode(Enum):
    """
    Operating modes for the RL agent.

    HUNT: Searching for ships using learned Q-values
    TARGET: Focused destruction of a located ship
    """
    HUNT = 1
    TARGET = 2


class RLAgent:
    """
    A Battleship agent using tabular Q-learning with domain heuristics.

    The agent learns action values (Q-values) for state-action pairs through
    gameplay experience. States are abstracted to capture essential game
    information while keeping the state space manageable.

    Attributes:
        grid_size: Size of the game board (default 10x10)
        ship_sizes: List of ship sizes in the game
        learning_rate: Alpha - how quickly new info overrides old (0.1)
        gamma: Discount factor for future rewards (0.95)
        epsilon: Exploration rate (decays during training)
        q_table: Dictionary mapping state keys to Q-value arrays
        position_values: Initial position preferences (center bias)
        mode: Current operating mode (HUNT or TARGET)
    """

    def __init__(
            self,
            grid_size: int = 10,
            ship_sizes: List[int] = None,
            learning_rate: float = 0.1,
            gamma: float = 0.95,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.05,
            epsilon_decay: float = 0.995,
            **kwargs
    ):
        """
        Initialize the Q-learning agent.

        Args:
            grid_size: Size of the game board
            ship_sizes: List of ship sizes [5, 4, 3, 3, 2]
            learning_rate: Q-learning alpha parameter
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Multiplicative decay per episode
        """
        self.grid_size = grid_size
        self.ship_sizes = ship_sizes if ship_sizes else [5, 4, 3, 3, 2]
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-table: maps state strings to action-value arrays
        self.q_table: Dict[str, np.ndarray] = {}

        # Initialize position values with center preference
        self.position_values = self._init_position_values()

        # Training statistics
        self.episode_count = 0
        self.steps_done = 0
        self.device = "cpu"  # For compatibility with interface

        self.reset()
        print(f"RL Agent initialized (Q-Learning)")

    def _init_position_values(self) -> np.ndarray:
        """
        Initialize position value heuristics.

        Center cells are more valuable because ships have more valid
        placements covering them. Also applies parity bonus for
        checkerboard efficiency.

        Returns:
            2D array of position values (0-1 range)
        """
        values = np.zeros((self.grid_size, self.grid_size))
        center = self.grid_size // 2

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                # Distance-based value: cells closer to center are more valuable
                dist_from_center = abs(r - center) + abs(c - center)
                values[r, c] = 1.0 - (dist_from_center / (self.grid_size * 2))

                # Parity bonus: checkerboard pattern for efficient hunting
                if (r + c) % 2 == 0:
                    values[r, c] += 0.1

        return values

    def reset(self):
        """Reset the agent state for a new game (not training state)."""
        # Board state: 0=unknown, 1=hit, 2=miss
        self.board_state = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.hits: List[Tuple[int, int]] = []
        self.misses: List[Tuple[int, int]] = []
        self.sunk_positions: List[Tuple[int, int]] = []
        self.remaining_ships = self.ship_sizes.copy()
        self.mode = AgentMode.HUNT
        self.current_targets: List[Tuple[int, int]] = []

        # For Q-learning updates
        self.last_state = None
        self.last_action = None

    def _get_state_key(self) -> str:
        """
        Generate a compact state representation string.

        State abstraction keeps the state space manageable:
        - HUNT mode: encode hit/miss counts and ships remaining
        - TARGET mode: encode current target position and direction

        Returns:
            String key for the Q-table
        """
        if self.mode == AgentMode.HUNT:
            num_hits = len(self.hits)
            num_misses = len(self.misses)
            ships_left = len(self.remaining_ships)
            return f"HUNT_{num_hits}_{num_misses}_{ships_left}"
        else:
            # Target mode: encode target position and direction
            if self.current_targets:
                target = self.current_targets[0]
                direction = self._get_target_direction()
                return f"TARGET_{target[0]}_{target[1]}_{direction}"
            return f"TARGET_NONE"

    def _get_target_direction(self) -> str:
        """
        Determine the direction of current target hits.

        Returns:
            'HORIZONTAL', 'VERTICAL', or 'UNKNOWN'
        """
        if len(self.current_targets) < 2:
            return "UNKNOWN"

        rows = [t[0] for t in self.current_targets]
        cols = [t[1] for t in self.current_targets]

        if len(set(rows)) == 1:
            return "HORIZONTAL"
        elif len(set(cols)) == 1:
            return "VERTICAL"
        return "UNKNOWN"

    def _get_q_values(self, state_key: str) -> np.ndarray:
        """
        Get Q-values for a state, initializing if necessary.

        New states are initialized with position value heuristics
        to provide reasonable starting values.

        Args:
            state_key: String key for the state

        Returns:
            1D array of Q-values for all 100 actions
        """
        if state_key not in self.q_table:
            # Initialize with position value heuristics
            self.q_table[state_key] = self.position_values.copy().flatten()
        return self.q_table[state_key]

    def get_valid_actions(self) -> List[int]:
        """
        Get list of valid actions (unattacked cells).

        Returns:
            List of action indices (0-99) for valid cells
        """
        valid = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.board_state[row, col] == 0:  # Not yet attacked
                    valid.append(row * self.grid_size + col)
        return valid

    def action_to_pos(self, action: int) -> Tuple[int, int]:
        """
        Convert action index to board position.

        Args:
            action: Action index (0-99)

        Returns:
            (row, col) board coordinates
        """
        return (action // self.grid_size, action % self.grid_size)

    def pos_to_action(self, pos: Tuple[int, int]) -> int:
        """
        Convert board position to action index.

        Args:
            pos: (row, col) board coordinates

        Returns:
            Action index (0-99)
        """
        return pos[0] * self.grid_size + pos[1]

    def _get_adjacent_cells(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid unattacked adjacent cells.

        Args:
            pos: (row, col) center position

        Returns:
            List of valid adjacent positions
        """
        row, col = pos
        adjacent = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                if self.board_state[nr, nc] == 0:  # Not attacked
                    adjacent.append((nr, nc))
        return adjacent

    def _get_line_targets(self) -> List[Tuple[int, int]]:
        """
        Get cells that extend the current hit line.

        When multiple hits form a line, targets the cells at
        either end of that line.

        Returns:
            List of positions extending the hit line
        """
        if len(self.current_targets) < 2:
            return []

        targets = []
        rows = [t[0] for t in self.current_targets]
        cols = [t[1] for t in self.current_targets]

        if len(set(rows)) == 1:
            # Horizontal line - extend left and right
            row = rows[0]
            min_col = min(cols)
            max_col = max(cols)
            if min_col > 0 and self.board_state[row, min_col - 1] == 0:
                targets.append((row, min_col - 1))
            if max_col < self.grid_size - 1 and self.board_state[row, max_col + 1] == 0:
                targets.append((row, max_col + 1))

        elif len(set(cols)) == 1:
            # Vertical line - extend up and down
            col = cols[0]
            min_row = min(rows)
            max_row = max(rows)
            if min_row > 0 and self.board_state[min_row - 1, col] == 0:
                targets.append((min_row - 1, col))
            if max_row < self.grid_size - 1 and self.board_state[max_row + 1, col] == 0:
                targets.append((max_row + 1, col))

        return targets

    def choose_target(self, training: bool = False) -> Tuple[int, int]:
        """
        Select the next cell to attack.

        Uses target mode heuristics when actively hunting a ship,
        otherwise uses epsilon-greedy Q-value selection.

        Args:
            training: If True, use epsilon-greedy exploration

        Returns:
            (row, col) coordinates of the chosen target
        """
        valid_actions = self.get_valid_actions()

        if not valid_actions:
            return (0, 0)

        state_key = self._get_state_key()
        self.last_state = state_key

        # TARGET MODE: Use heuristics to destroy located ship
        if self.mode == AgentMode.TARGET and self.current_targets:
            # Try to extend along hit line
            line_targets = self._get_line_targets()
            if line_targets:
                if training and random.random() < self.epsilon * 0.3:
                    target = random.choice(line_targets)
                else:
                    target = line_targets[0]
                self.last_action = self.pos_to_action(target)
                self.steps_done += 1
                return target

            # Try adjacent to any current target hit
            for hit in self.current_targets:
                adjacent = self._get_adjacent_cells(hit)
                if adjacent:
                    if training and random.random() < self.epsilon * 0.3:
                        target = random.choice(adjacent)
                    else:
                        target = adjacent[0]
                    self.last_action = self.pos_to_action(target)
                    self.steps_done += 1
                    return target

            # No valid target mode moves, switch to hunt
            self.mode = AgentMode.HUNT
            self.current_targets = []

        # HUNT MODE: Epsilon-greedy Q-value selection
        if training and random.random() < self.epsilon:
            # Exploration: random valid action
            action = random.choice(valid_actions)
        else:
            # Exploitation: choose best Q-value action
            q_values = self._get_q_values(state_key)

            # Mask invalid actions with large negative value
            masked_q = np.full(self.grid_size * self.grid_size, -1e9)
            for a in valid_actions:
                masked_q[a] = q_values[a]

            action = int(np.argmax(masked_q))

        self.last_action = action
        self.steps_done += 1

        return self.action_to_pos(action)

    def record_result(
            self,
            pos: Tuple[int, int],
            hit: bool,
            sunk_ship_size: Optional[int] = None
    ):
        """
        Record the result of an attack and update agent state.

        Args:
            pos: (row, col) coordinates of the attack
            hit: True if the attack hit a ship
            sunk_ship_size: Size of ship if sunk, None otherwise
        """
        row, col = pos

        if hit:
            self.board_state[row, col] = 1  # Mark as hit

            if sunk_ship_size:
                # Ship was sunk - move hits to sunk_positions
                for t in self.current_targets:
                    if t not in self.sunk_positions:
                        self.sunk_positions.append(t)
                if pos not in self.sunk_positions:
                    self.sunk_positions.append(pos)

                self.current_targets = []

                # Update remaining ships
                if sunk_ship_size in self.remaining_ships:
                    self.remaining_ships.remove(sunk_ship_size)

                # Check if there are other hits to pursue
                if self.hits:
                    self.current_targets = [self.hits[0]]
                    self.mode = AgentMode.TARGET
                else:
                    self.mode = AgentMode.HUNT
            else:
                # Hit but not sunk - add to tracking
                if pos not in self.hits:
                    self.hits.append(pos)
                if pos not in self.current_targets:
                    self.current_targets.append(pos)
                self.mode = AgentMode.TARGET
        else:
            # Miss
            self.board_state[row, col] = 2  # Mark as miss
            if pos not in self.misses:
                self.misses.append(pos)

    def update_q_value(self, state: str, action: int, reward: float, next_state: str, done: bool):
        """
        Update Q-value using the Bellman equation.

        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]

        Args:
            state: Current state key
            action: Action taken
            reward: Reward received
            next_state: Resulting state key
            done: Whether the game ended
        """
        if state is None or action is None:
            return

        q_values = self._get_q_values(state)
        current_q = q_values[action]

        if done:
            # Terminal state: Q-value is just the reward
            target_q = reward
        else:
            # Non-terminal: include discounted future value
            next_q_values = self._get_q_values(next_state)
            valid_next = self.get_valid_actions()
            if valid_next:
                max_next_q = max(next_q_values[a] for a in valid_next)
            else:
                max_next_q = 0
            target_q = reward + self.gamma * max_next_q

        # Q-learning update
        q_values[action] = current_q + self.learning_rate * (target_q - current_q)

    def end_episode(self):
        """Called at the end of each training episode to decay epsilon."""
        self.episode_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """
        Save the trained model to a file.

        Args:
            filepath: Path to save the model (.pkl)
        """
        data = {
            'q_table': self.q_table,
            'position_values': self.position_values,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'steps_done': self.steps_done
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load a trained model from a file.

        Args:
            filepath: Path to the saved model (.pkl)
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.q_table = data['q_table']
        self.position_values = data['position_values']
        self.epsilon = data['epsilon']
        self.episode_count = data['episode_count']
        self.steps_done = data['steps_done']
        print(f"Model loaded from {filepath}")

    def get_stats(self) -> dict:
        """
        Get current agent statistics.

        Returns:
            Dictionary containing game and training statistics
        """
        total_shots = len(self.hits) + len(self.misses) + len(self.sunk_positions)
        total_hits = len(self.hits) + len(self.sunk_positions)

        return {
            'total_shots': total_shots,
            'hits': total_hits,
            'misses': len(self.misses),
            'accuracy': (total_hits / total_shots * 100) if total_shots > 0 else 0,
            'ships_remaining': len(self.remaining_ships),
            'ships_sunk': len(self.ship_sizes) - len(self.remaining_ships),
            'mode': self.mode.name,
            'epsilon': self.epsilon,
            'episode': self.episode_count
        }


# =============================================================================
# MAIN ENTRY POINT (for testing)
# =============================================================================


if __name__ == "__main__":
    agent = RLAgent()
    print("RLAgent created successfully!")
    print(f"Q-table size: {len(agent.q_table)}")
    target = agent.choose_target(training=True)
    print(f"First target: {target}")