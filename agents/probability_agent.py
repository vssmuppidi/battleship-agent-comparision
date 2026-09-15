"""
Probability-Based Battleship Agent

This module implements an autonomous Battleship agent that uses probability
calculations and domain-specific heuristics for targeting decisions. The agent
maintains a probability map estimating ship presence likelihood at each cell
and operates in two modes: hunt mode (searching for ships) and target mode
(destroying located ships).

Key Features:
- Probability-based cell selection using ship placement counting
- Hunt/target mode switching for efficient ship destruction
- Parity optimization (checkerboard pattern) for initial search
- Edge penalties, density bonuses, and miss cluster penalties
- Ship orientation tracking for informed targeting

Author: Venkatashivasai Muppidi
Course: Foundations of Artificial Intelligence
University: Northeastern University
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from enum import Enum


class AgentMode(Enum):
    """
    Operating modes for the probability agent.

    HUNT: Searching for ships using probability-based targeting
    TARGET: Focused destruction of a located ship
    """
    HUNT = 1
    TARGET = 2


class ProbabilityAgent:
    """
    An autonomous Battleship agent using probability calculations and heuristics.

    The agent calculates ship presence probability for each cell based on:
    1. Number of valid ship placements covering each cell
    2. Proximity to existing hits (target mode boost)
    3. Parity optimization (checkerboard pattern)
    4. Edge penalties and density bonuses
    5. Miss cluster avoidance

    Attributes:
        grid_size: Size of the game board (default 10x10)
        ship_sizes: List of remaining ship sizes to find
        remaining_ships: Ships that haven't been sunk yet
        mode: Current operating mode (HUNT or TARGET)
        hits: List of hit positions not yet part of a sunk ship
        misses: List of miss positions
        sunk_positions: List of positions belonging to sunk ships
        probability_map: 2D array of ship presence probabilities
        current_target_hits: Hits being tracked for current target
        smallest_remaining_ship: Size of smallest unsunk ship (for parity)
        sunk_orientations: Orientations of sunk ships (for prediction)
    """

    def __init__(self, grid_size: int = 10, ship_sizes: List[int] = None):
        """
        Initialize the probability agent.

        Args:
            grid_size: Size of the game board (default 10)
            ship_sizes: List of ship sizes [5, 4, 3, 3, 2] for standard game
        """
        self.grid_size = grid_size
        self.ship_sizes = ship_sizes if ship_sizes else [5, 4, 3, 3, 2]
        self.remaining_ships = self.ship_sizes.copy()
        self.mode = AgentMode.HUNT
        self.hits: List[Tuple[int, int]] = []
        self.misses: List[Tuple[int, int]] = []
        self.sunk_positions: List[Tuple[int, int]] = []
        self.probability_map = np.zeros((grid_size, grid_size))
        self.current_target_hits: List[Tuple[int, int]] = []
        self.smallest_remaining_ship = min(self.ship_sizes)
        self.sunk_orientations: List[str] = []

    def reset(self):
        """Reset the agent for a new game."""
        self.remaining_ships = self.ship_sizes.copy()
        self.mode = AgentMode.HUNT
        self.hits = []
        self.misses = []
        self.sunk_positions = []
        self.probability_map = np.zeros((self.grid_size, self.grid_size))
        self.current_target_hits = []
        self.smallest_remaining_ship = min(self.ship_sizes)
        self.sunk_orientations = []

    def get_attacked_cells(self) -> Set[Tuple[int, int]]:
        """
        Get all cells that have been attacked.

        Returns:
            Set of (row, col) coordinates of attacked cells
        """
        return set(self.hits + self.misses + self.sunk_positions)

    def get_unattacked_cells(self) -> List[Tuple[int, int]]:
        """
        Get all cells that haven't been attacked yet.

        Returns:
            List of (row, col) coordinates of unattacked cells
        """
        attacked = self.get_attacked_cells()
        unattacked = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) not in attacked:
                    unattacked.append((row, col))
        return unattacked

    def is_parity_cell(self, row: int, col: int) -> bool:
        """
        Check if a cell is on the parity pattern (checkerboard).

        Parity optimization: Ships of size N must cross cells where
        (row + col) % N == 0. For the smallest ship (size 2), this
        creates a checkerboard pattern that guarantees finding all ships
        while potentially skipping half the cells.

        Args:
            row, col: Cell coordinates

        Returns:
            True if cell is on the parity pattern
        """
        parity = self.smallest_remaining_ship
        return (row + col) % parity == 0

    def get_edge_penalty(self, row: int, col: int) -> float:
        """
        Calculate edge penalty for a cell.

        Ships have fewer valid placements near edges, so edge cells
        are statistically less likely to contain ships.

        Args:
            row, col: Cell coordinates

        Returns:
            Penalty multiplier (0.5 for corners, 0.75 for edges, 1.0 otherwise)
        """
        distance_from_edge = min(row, col, self.grid_size - 1 - row, self.grid_size - 1 - col)
        if distance_from_edge == 0:
            return 0.5  # Corner/edge cells
        elif distance_from_edge == 1:
            return 0.75  # Near-edge cells
        return 1.0  # Interior cells

    def get_density_bonus(self, row: int, col: int, attacked: Set[Tuple[int, int]]) -> float:
        """
        Calculate density bonus based on unexplored neighbors.

        Cells surrounded by more unexplored cells are more likely to
        contain ships, as there's more room for valid placements.

        Args:
            row, col: Cell coordinates
            attacked: Set of already-attacked cells

        Returns:
            Bonus multiplier (1.0 to 1.5 based on neighbor density)
        """
        unexplored_neighbors = 0
        # Check 5x5 neighborhood
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if (nr, nc) not in attacked:
                        unexplored_neighbors += 1
        return 1.0 + (unexplored_neighbors / 25.0) * 0.5

    def get_miss_cluster_penalty(self, row: int, col: int) -> float:
        """
        Calculate penalty for cells near miss clusters.

        Areas with many nearby misses are less likely to contain ships.

        Args:
            row, col: Cell coordinates

        Returns:
            Penalty multiplier (0.5 to 1.0 based on nearby misses)
        """
        miss_set = set(self.misses)
        nearby_misses = 0
        # Count misses in 5x5 neighborhood
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = row + dr, col + dc
                if (nr, nc) in miss_set:
                    nearby_misses += 1

        # Apply graduated penalty based on miss density
        if nearby_misses >= 6:
            return 0.5
        elif nearby_misses >= 4:
            return 0.7
        elif nearby_misses >= 2:
            return 0.85
        return 1.0

    def get_orientation_bonus(self, orientation: str) -> float:
        """
        Calculate bonus based on sunk ship orientations.

        If most sunk ships were horizontal, remaining ships are more
        likely to be vertical (and vice versa), assuming some boards
        favor certain orientations.

        Args:
            orientation: 'horizontal' or 'vertical'

        Returns:
            Bonus multiplier (0.8 to 1.2 based on observed orientations)
        """
        if not self.sunk_orientations:
            return 1.0

        horizontal_count = self.sunk_orientations.count('horizontal')
        vertical_count = self.sunk_orientations.count('vertical')
        total = horizontal_count + vertical_count

        if total == 0:
            return 1.0

        # Slightly favor the less common orientation
        if orientation == 'horizontal':
            if horizontal_count > vertical_count:
                return 0.8  # Already found many horizontal
            elif vertical_count > horizontal_count:
                return 1.2  # Expect more horizontal
        else:
            if vertical_count > horizontal_count:
                return 0.8  # Already found many vertical
            elif horizontal_count > vertical_count:
                return 1.2  # Expect more vertical
        return 1.0

    def update_probability_map(self):
        """
        Update the probability map based on current game state.

        For each remaining ship, counts valid placements at each cell.
        Applies heuristic bonuses/penalties for more accurate estimates.
        Higher probability indicates higher likelihood of ship presence.
        """
        self.probability_map = np.zeros((self.grid_size, self.grid_size))
        attacked = self.get_attacked_cells()
        hit_set = set(self.hits)

        # Count valid placements for each remaining ship
        for ship_size in self.remaining_ships:
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    # Check horizontal placement
                    if col + ship_size <= self.grid_size:
                        cells = [(row, col + i) for i in range(ship_size)]
                        blocked = False
                        hits_in_placement = 0

                        for cell in cells:
                            # Blocked if cell was attacked (miss or sunk)
                            if cell in attacked and cell not in hit_set:
                                blocked = True
                                break
                            # Count hits this placement would cover
                            if cell in hit_set:
                                hits_in_placement += 1

                        if not blocked:
                            # Weight by hits covered (strongly prefer placements covering hits)
                            weight = 1 + (hits_in_placement * 15)
                            orientation_bonus = self.get_orientation_bonus('horizontal')
                            weight *= orientation_bonus
                            for cell in cells:
                                if cell not in attacked:
                                    self.probability_map[cell[0], cell[1]] += weight

                    # Check vertical placement
                    if row + ship_size <= self.grid_size:
                        cells = [(row + i, col) for i in range(ship_size)]
                        blocked = False
                        hits_in_placement = 0

                        for cell in cells:
                            if cell in attacked and cell not in hit_set:
                                blocked = True
                                break
                            if cell in hit_set:
                                hits_in_placement += 1

                        if not blocked:
                            weight = 1 + (hits_in_placement * 15)
                            orientation_bonus = self.get_orientation_bonus('vertical')
                            weight *= orientation_bonus
                            for cell in cells:
                                if cell not in attacked:
                                    self.probability_map[cell[0], cell[1]] += weight

        # Apply heuristic adjustments
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if (row, col) not in attacked and self.probability_map[row, col] > 0:
                    edge_penalty = self.get_edge_penalty(row, col)
                    density_bonus = self.get_density_bonus(row, col, attacked)
                    miss_penalty = self.get_miss_cluster_penalty(row, col)
                    self.probability_map[row, col] *= edge_penalty * density_bonus * miss_penalty

        # Zero out attacked cells
        for pos in attacked:
            self.probability_map[pos[0], pos[1]] = 0

    def get_adjacent_cells(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get orthogonally adjacent cells (up, down, left, right).

        Args:
            pos: (row, col) center position

        Returns:
            List of valid adjacent (row, col) coordinates
        """
        row, col = pos
        adjacent = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                adjacent.append((nr, nc))
        return adjacent

    def get_valid_adjacent_cells(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get adjacent cells that haven't been attacked.

        Args:
            pos: (row, col) center position

        Returns:
            List of unattacked adjacent (row, col) coordinates
        """
        attacked = self.get_attacked_cells()
        return [adj for adj in self.get_adjacent_cells(pos) if adj not in attacked]

    def get_line_direction(self) -> Optional[str]:
        """
        Determine if current target hits form a line.

        Returns:
            'horizontal' if hits are in a row, 'vertical' if in a column,
            None if not enough hits or no clear direction
        """
        if len(self.current_target_hits) < 2:
            return None

        rows = [h[0] for h in self.current_target_hits]
        cols = [h[1] for h in self.current_target_hits]

        if len(set(rows)) == 1:
            return 'horizontal'  # All hits in same row
        elif len(set(cols)) == 1:
            return 'vertical'  # All hits in same column
        return None

    def get_target_mode_candidates(self) -> List[Tuple[int, int]]:
        """
        Get candidate cells for target mode attacks.

        In target mode, we focus on cells adjacent to or extending
        from the current hit sequence.

        Returns:
            List of (row, col) candidate positions
        """
        if not self.current_target_hits:
            return []

        attacked = self.get_attacked_cells()
        candidates = []

        if len(self.current_target_hits) == 1:
            # Single hit: try all four adjacent cells
            for adj in self.get_adjacent_cells(self.current_target_hits[0]):
                if adj not in attacked:
                    candidates.append(adj)
        else:
            # Multiple hits: extend along the determined direction
            direction = self.get_line_direction()
            sorted_hits = sorted(self.current_target_hits)

            if direction == 'horizontal':
                row = sorted_hits[0][0]
                min_col = min(h[1] for h in sorted_hits)
                max_col = max(h[1] for h in sorted_hits)
                # Try extending left and right
                if min_col > 0 and (row, min_col - 1) not in attacked:
                    candidates.append((row, min_col - 1))
                if max_col < self.grid_size - 1 and (row, max_col + 1) not in attacked:
                    candidates.append((row, max_col + 1))

            elif direction == 'vertical':
                col = sorted_hits[0][1]
                min_row = min(h[0] for h in sorted_hits)
                max_row = max(h[0] for h in sorted_hits)
                # Try extending up and down
                if min_row > 0 and (min_row - 1, col) not in attacked:
                    candidates.append((min_row - 1, col))
                if max_row < self.grid_size - 1 and (max_row + 1, col) not in attacked:
                    candidates.append((max_row + 1, col))
            else:
                # Hits don't form a clear line, try adjacent to all
                for hit in self.current_target_hits:
                    for adj in self.get_adjacent_cells(hit):
                        if adj not in attacked and adj not in candidates:
                            candidates.append(adj)

        return candidates

    def choose_target(self) -> Tuple[int, int]:
        """
        Select the next cell to attack.

        Uses target mode logic if actively hunting a ship,
        otherwise uses hunt mode with probability-based selection.

        Returns:
            (row, col) coordinates of the chosen target
        """
        unattacked = self.get_unattacked_cells()
        if not unattacked:
            return (0, 0)

        # Always update probability map for decision making
        self.update_probability_map()

        # TARGET MODE: Focus on destroying located ship
        if self.mode == AgentMode.TARGET and self.current_target_hits:
            candidates = self.get_target_mode_candidates()
            if candidates:
                # Choose candidate with highest probability
                best_candidate = max(candidates, key=lambda c: self.probability_map[c[0], c[1]])
                return best_candidate
            # No valid candidates, clear target and try other hits
            self.current_target_hits = []

        # Check if there are other unsunk hits to investigate
        if self.hits:
            for hit in self.hits[:]:
                valid_adj = self.get_valid_adjacent_cells(hit)
                if valid_adj:
                    # Found hit with valid adjacent cells, switch to target mode
                    self.current_target_hits = [hit]
                    self.mode = AgentMode.TARGET
                    best = max(valid_adj, key=lambda c: self.probability_map[c[0], c[1]])
                    return best
                else:
                    # No valid adjacent cells, remove from tracking
                    self.hits.remove(hit)

        # HUNT MODE: Search for new ships
        self.mode = AgentMode.HUNT
        self.current_target_hits = []

        # Prefer parity cells (checkerboard pattern) for efficiency
        parity_cells = []
        for cell in unattacked:
            row, col = cell
            if self.is_parity_cell(row, col):
                prob = self.probability_map[row, col]
                if prob > 0:
                    parity_cells.append((cell, prob))

        if parity_cells:
            # Sort by probability and choose highest
            parity_cells.sort(key=lambda x: x[1], reverse=True)
            return parity_cells[0][0]

        # Fallback: choose cell with highest probability
        best_prob = -1
        best_cell = unattacked[0]
        for cell in unattacked:
            prob = self.probability_map[cell[0], cell[1]]
            if prob > best_prob:
                best_prob = prob
                best_cell = cell
        return best_cell

    def record_result(self, pos: Tuple[int, int], hit: bool, sunk_ship_size: Optional[int] = None):
        """
        Record the result of an attack and update agent state.

        Args:
            pos: (row, col) coordinates of the attack
            hit: True if the attack hit a ship
            sunk_ship_size: Size of ship if one was sunk, None otherwise
        """
        if hit:
            # Add to hits if not already tracked
            if pos not in self.hits and pos not in self.sunk_positions:
                self.hits.append(pos)
            if pos not in self.current_target_hits:
                self.current_target_hits.append(pos)
            self.mode = AgentMode.TARGET

            if sunk_ship_size:
                # Ship was sunk - identify and remove its cells
                sunk_cells, orientation = self.identify_sunk_ship_cells(sunk_ship_size)
                if orientation:
                    self.sunk_orientations.append(orientation)

                # Move cells from hits to sunk_positions
                for cell in sunk_cells:
                    if cell in self.hits:
                        self.hits.remove(cell)
                    if cell in self.current_target_hits:
                        self.current_target_hits.remove(cell)
                    if cell not in self.sunk_positions:
                        self.sunk_positions.append(cell)

                # Update remaining ships
                if sunk_ship_size in self.remaining_ships:
                    self.remaining_ships.remove(sunk_ship_size)
                    if self.remaining_ships:
                        self.smallest_remaining_ship = min(self.remaining_ships)

                # Check if there are other hits to pursue
                if self.hits:
                    self.current_target_hits = [self.hits[0]]
                    self.mode = AgentMode.TARGET
                else:
                    self.current_target_hits = []
                    self.mode = AgentMode.HUNT
        else:
            # Miss - record position
            if pos not in self.misses:
                self.misses.append(pos)

    def identify_sunk_ship_cells(self, ship_size: int) -> Tuple[List[Tuple[int, int]], Optional[str]]:
        """
        Identify which cells belong to the just-sunk ship.

        Args:
            ship_size: Size of the ship that was sunk

        Returns:
            Tuple of (list of cell positions, orientation)
        """
        # If current target hits match ship size, use those
        if len(self.current_target_hits) == ship_size:
            direction = self.get_line_direction()
            return self.current_target_hits.copy(), direction

        # Otherwise, search for a connected line of hits
        all_hits = self.current_target_hits + self.hits
        unique_hits = list(dict.fromkeys(all_hits))

        for hit in unique_hits:
            # Try horizontal line
            horizontal_cells = [hit]
            for j in range(1, ship_size):
                next_cell = (hit[0], hit[1] + j)
                if next_cell in unique_hits:
                    horizontal_cells.append(next_cell)
                else:
                    break
            for j in range(1, ship_size):
                prev_cell = (hit[0], hit[1] - j)
                if prev_cell in unique_hits:
                    horizontal_cells.insert(0, prev_cell)
                else:
                    break
            if len(horizontal_cells) == ship_size:
                return horizontal_cells, 'horizontal'

            # Try vertical line
            vertical_cells = [hit]
            for j in range(1, ship_size):
                next_cell = (hit[0] + j, hit[1])
                if next_cell in unique_hits:
                    vertical_cells.append(next_cell)
                else:
                    break
            for j in range(1, ship_size):
                prev_cell = (hit[0] - j, hit[1])
                if prev_cell in unique_hits:
                    vertical_cells.insert(0, prev_cell)
                else:
                    break
            if len(vertical_cells) == ship_size:
                return vertical_cells, 'vertical'

        # Fallback: use most recent hits
        if len(self.current_target_hits) >= ship_size:
            return self.current_target_hits[-ship_size:], self.get_line_direction()
        return self.current_target_hits.copy(), self.get_line_direction()

    def get_stats(self) -> dict:
        """
        Get current agent statistics.

        Returns:
            Dictionary containing shots, hits, misses, accuracy, etc.
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
            'orientations_found': self.sunk_orientations.copy()
        }