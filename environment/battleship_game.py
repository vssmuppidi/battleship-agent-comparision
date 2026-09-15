"""
Battleship Game Engine

This module implements the core Battleship game logic including the game board,
ship placement, attack handling, and the Pygame-based graphical user interface.
It provides the foundation for both human play and AI agent evaluation.

Author: Venkatashivasai Muppidi
Course: Foundations of Artificial Intelligence
University: Northeastern University
"""

import pygame
import numpy as np
import random
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import time

# Initialize Pygame and its audio mixer
pygame.init()
pygame.mixer.init()

# =============================================================================
# GAME CONSTANTS
# =============================================================================

# Screen dimensions
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800

# Grid configuration
CELL_SIZE = 40  # Size of each cell in pixels
GRID_SIZE = 10  # 10x10 grid (standard Battleship)

# Grid positioning on screen
GRID_OFFSET_X = 50  # Player grid X position
GRID_OFFSET_Y = 140  # Both grids Y position
ENEMY_GRID_OFFSET_X = 750  # Enemy grid X position

# Color palette for the game interface
COLORS = {
    # Ocean colors
    'deep_ocean': (15, 35, 65),
    'ocean': (25, 55, 95),
    'ocean_light': (35, 75, 125),
    'grid_line': (45, 85, 135),
    # Ship colors
    'ship_hull': (75, 75, 80),
    'ship_deck': (95, 95, 100),
    'ship_accent': (60, 60, 65),
    # Hit/miss markers
    'hit_fire': (255, 100, 30),
    'hit_glow': (255, 150, 50),
    'miss_splash': (150, 200, 255),
    'miss_ring': (100, 150, 200),
    'sunk': (40, 40, 45),
    # Text colors
    'text_primary': (220, 220, 220),
    'text_secondary': (150, 160, 170),
    # Heatmap colors for probability visualization
    'heatmap_low': (25, 55, 95),
    'heatmap_high': (255, 80, 80),
    # UI panel colors
    'panel_bg': (20, 30, 50),
    'panel_border': (50, 70, 100),
    'button_normal': (40, 60, 90),
    'button_hover': (60, 80, 110),
    'button_active': (80, 100, 130),
}

# Ship definitions: (name, size, color)
# Standard Battleship fleet totaling 17 cells
SHIPS = [
    ('Carrier', 5, (180, 180, 190)),
    ('Battleship', 4, (160, 160, 170)),
    ('Cruiser', 3, (140, 140, 150)),
    ('Submarine', 3, (120, 130, 140)),
    ('Destroyer', 2, (100, 110, 120)),
]


# =============================================================================
# ENUMERATIONS
# =============================================================================


class CellState(Enum):
    """
    Represents the possible states of a cell on the game board.
    """
    EMPTY = 0  # Unexplored water
    SHIP = 1  # Contains a ship (only visible on player's board)
    HIT = 2  # Ship was hit at this location
    MISS = 3  # Attack missed (no ship)
    SUNK = 4  # Part of a sunk ship


class GamePhase(Enum):
    """
    Represents the current phase of the game.
    """
    PLACEMENT = 1  # Player is placing ships
    BATTLE = 2  # Active gameplay - attacking
    GAME_OVER = 3  # All ships sunk - game ended


class Orientation(Enum):
    """
    Ship placement orientation.
    """
    HORIZONTAL = 0  # Ship extends left to right
    VERTICAL = 1  # Ship extends top to bottom


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class Ship:
    """
    Represents a ship on the game board.

    Attributes:
        name: The ship's name (e.g., "Carrier", "Destroyer")
        size: Number of cells the ship occupies
        color: RGB tuple for rendering the ship
        positions: List of (row, col) coordinates the ship occupies
        hits: Number of times this ship has been hit
        orientation: Whether ship is placed horizontally or vertically
    """
    name: str
    size: int
    color: Tuple[int, int, int]
    positions: List[Tuple[int, int]] = field(default_factory=list)
    hits: int = 0
    orientation: Orientation = Orientation.HORIZONTAL

    @property
    def is_sunk(self) -> bool:
        """Returns True if the ship has been hit in all positions."""
        return self.hits >= self.size

    @property
    def is_placed(self) -> bool:
        """Returns True if the ship has been placed on the board."""
        return len(self.positions) == self.size


@dataclass
class Animation:
    """
    Base class for visual animations (explosions, splashes).

    Attributes:
        x, y: Screen coordinates for the animation center
        start_time: When the animation began
        duration: How long the animation lasts in seconds
    """
    x: float
    y: float
    start_time: float
    duration: float

    @property
    def progress(self) -> float:
        """Returns animation progress from 0.0 to 1.0."""
        elapsed = time.time() - self.start_time
        return min(1.0, elapsed / self.duration)

    @property
    def is_complete(self) -> bool:
        """Returns True if the animation has finished."""
        return self.progress >= 1.0


@dataclass
class ExplosionAnimation(Animation):
    """
    Animated explosion effect for hit markers.
    Creates particles that fly outward from the hit location.
    """
    particles: List[dict] = field(default_factory=list)

    def __post_init__(self):
        """Initialize random particles for the explosion effect."""
        for _ in range(20):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(50, 150)
            self.particles.append({
                'angle': angle,
                'speed': speed,
                'size': random.uniform(3, 8),
                'color': random.choice([
                    COLORS['hit_fire'],
                    COLORS['hit_glow'],
                    (255, 200, 100),
                    (255, 50, 0)
                ])
            })


@dataclass
class SplashAnimation(Animation):
    """
    Animated splash effect for miss markers.
    Creates expanding rings emanating from the miss location.
    """
    rings: List[dict] = field(default_factory=list)

    def __post_init__(self):
        """Initialize concentric rings for the splash effect."""
        for i in range(3):
            self.rings.append({
                'delay': i * 0.1,
                'max_radius': 15 + i * 8
            })


@dataclass
class GameStats:
    """
    Tracks game statistics during play.

    Attributes:
        total_shots: Number of attacks made
        hits: Number of successful hits
        misses: Number of missed attacks
        ships_sunk: Number of enemy ships destroyed
        turn_history: Record of all moves made
        start_time: When the game started
    """
    total_shots: int = 0
    hits: int = 0
    misses: int = 0
    ships_sunk: int = 0
    turn_history: List[dict] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    @property
    def accuracy(self) -> float:
        """Returns hit percentage (0-100)."""
        if self.total_shots == 0:
            return 0.0
        return (self.hits / self.total_shots) * 100

    @property
    def elapsed_time(self) -> float:
        """Returns seconds elapsed since game start."""
        return time.time() - self.start_time


# =============================================================================
# BOARD CLASS
# =============================================================================


class Board:
    """
    Represents a player's game board (10x10 grid).

    Handles ship placement, attack processing, and probability map calculation
    for the heatmap visualization.

    Attributes:
        grid: 2D numpy array storing cell states
        ships: List of Ship objects placed on this board
        is_player: True if this is the human player's board
        probability_map: 2D array for heatmap visualization
    """

    def __init__(self, is_player: bool = True):
        """
        Initialize an empty game board.

        Args:
            is_player: Whether this board belongs to the human player
        """
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.ships: List[Ship] = []
        self.is_player = is_player
        # Initialize uniform probability for heatmap
        self.probability_map = np.ones((GRID_SIZE, GRID_SIZE)) / (GRID_SIZE * GRID_SIZE)

    def can_place_ship(self, ship: Ship, start_pos: Tuple[int, int],
                       orientation: Orientation) -> bool:
        """
        Check if a ship can be legally placed at the given position.

        Args:
            ship: The ship to place
            start_pos: (row, col) starting position
            orientation: Horizontal or vertical placement

        Returns:
            True if placement is valid (within bounds and no overlap)
        """
        row, col = start_pos

        for i in range(ship.size):
            if orientation == Orientation.HORIZONTAL:
                c = col + i
                r = row
            else:
                c = col
                r = row + i

            # Check bounds
            if r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE:
                return False
            # Check for overlap with existing ships
            if self.grid[r, c] == CellState.SHIP.value:
                return False

        return True

    def place_ship(self, ship: Ship, start_pos: Tuple[int, int],
                   orientation: Orientation) -> bool:
        """
        Place a ship on the board if the position is valid.

        Args:
            ship: The ship to place
            start_pos: (row, col) starting position
            orientation: Horizontal or vertical placement

        Returns:
            True if ship was successfully placed
        """
        if not self.can_place_ship(ship, start_pos, orientation):
            return False

        row, col = start_pos
        ship.positions = []
        ship.orientation = orientation

        # Mark all cells occupied by this ship
        for i in range(ship.size):
            if orientation == Orientation.HORIZONTAL:
                c = col + i
                r = row
            else:
                c = col
                r = row + i

            self.grid[r, c] = CellState.SHIP.value
            ship.positions.append((r, c))

        self.ships.append(ship)
        return True

    def place_ships_randomly(self):
        """
        Randomly place all ships on the board.
        Used for enemy board setup and auto-placement feature.
        """
        # Reset the board
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.ships = []

        for name, size, color in SHIPS:
            ship = Ship(name=name, size=size, color=color)
            placed = False
            attempts = 0

            # Try random positions until valid placement found
            while not placed and attempts < 1000:
                row = random.randint(0, GRID_SIZE - 1)
                col = random.randint(0, GRID_SIZE - 1)
                orientation = random.choice([Orientation.HORIZONTAL, Orientation.VERTICAL])
                placed = self.place_ship(ship, (row, col), orientation)
                attempts += 1

    def receive_attack(self, pos: Tuple[int, int]) -> Tuple[CellState, Optional[Ship]]:
        """
        Process an attack on this board.

        Args:
            pos: (row, col) coordinates of the attack

        Returns:
            Tuple of (result_state, ship_if_hit)
        """
        row, col = pos

        if self.grid[row, col] == CellState.SHIP.value:
            # Hit a ship
            self.grid[row, col] = CellState.HIT.value

            # Find which ship was hit and update its hit count
            for ship in self.ships:
                if pos in ship.positions:
                    ship.hits += 1
                    if ship.is_sunk:
                        # Mark all ship cells as sunk
                        for r, c in ship.positions:
                            self.grid[r, c] = CellState.SUNK.value
                        return CellState.SUNK, ship
                    return CellState.HIT, ship

        elif self.grid[row, col] == CellState.EMPTY.value:
            # Missed - hit water
            self.grid[row, col] = CellState.MISS.value
            return CellState.MISS, None

        # Cell was already attacked
        return CellState.EMPTY, None

    def all_ships_sunk(self) -> bool:
        """Returns True if all ships on this board have been sunk."""
        return all(ship.is_sunk for ship in self.ships)

    def update_probability_map(self, remaining_ships: List[Ship]):
        """
        Update the probability heatmap based on remaining ships.

        Calculates likelihood of ship presence at each cell by counting
        valid ship placements and boosting probability near hits.

        Args:
            remaining_ships: Ships that haven't been sunk yet
        """
        self.probability_map = np.zeros((GRID_SIZE, GRID_SIZE))

        # Count valid placements for each remaining ship
        for ship in remaining_ships:
            if ship.is_sunk:
                continue

            for row in range(GRID_SIZE):
                for col in range(GRID_SIZE):
                    # Check horizontal placement
                    if col + ship.size <= GRID_SIZE:
                        valid = True
                        for i in range(ship.size):
                            cell = self.grid[row, col + i]
                            if cell in [CellState.MISS.value, CellState.SUNK.value]:
                                valid = False
                                break
                        if valid:
                            for i in range(ship.size):
                                self.probability_map[row, col + i] += 1

                    # Check vertical placement
                    if row + ship.size <= GRID_SIZE:
                        valid = True
                        for i in range(ship.size):
                            cell = self.grid[row + i, col]
                            if cell in [CellState.MISS.value, CellState.SUNK.value]:
                                valid = False
                                break
                        if valid:
                            for i in range(ship.size):
                                self.probability_map[row + i, col] += 1

        # Boost probability for cells adjacent to hits (target mode logic)
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.grid[row, col] == CellState.HIT.value:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                            if self.grid[nr, nc] == CellState.EMPTY.value:
                                self.probability_map[nr, nc] *= 3

        # Zero out already-attacked cells
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.grid[row, col] != CellState.EMPTY.value:
                    self.probability_map[row, col] = 0

        # Normalize to create probability distribution
        total = self.probability_map.sum()
        if total > 0:
            self.probability_map /= total


# =============================================================================
# RENDERER CLASS
# =============================================================================


class Renderer:
    """
    Handles all visual rendering for the game.

    Draws the ocean background, game grids, ships, markers,
    statistics panels, and animations.
    """

    def __init__(self, screen):
        """
        Initialize the renderer with font definitions.

        Args:
            screen: Pygame display surface to draw on
        """
        self.screen = screen
        self.fonts = {
            'title': pygame.font.Font(None, 48),
            'heading': pygame.font.Font(None, 36),
            'normal': pygame.font.Font(None, 28),
            'small': pygame.font.Font(None, 22),
        }
        self.animations: List[Animation] = []

    def draw_ocean_background(self):
        """Draw animated ocean background with wave effects."""
        self.screen.fill(COLORS['deep_ocean'])

        # Create subtle wave animation
        t = time.time()
        for y in range(0, SCREEN_HEIGHT, 20):
            wave_offset = math.sin(t * 0.5 + y * 0.02) * 5
            alpha = 30 + int(math.sin(t * 0.3 + y * 0.01) * 10)
            for x in range(0, SCREEN_WIDTH, 40):
                wave_x = x + wave_offset
                pygame.draw.circle(
                    self.screen,
                    (*COLORS['ocean_light'][:3],),
                    (int(wave_x), y),
                    2
                )

    def draw_grid(self, offset_x: int, offset_y: int, board: Board,
                  show_ships: bool = True, show_heatmap: bool = False,
                  label: str = ""):
        """
        Draw a game grid with all its contents.

        Args:
            offset_x, offset_y: Screen position for the grid
            board: The Board object to render
            show_ships: Whether to show ship positions
            show_heatmap: Whether to overlay probability heatmap
            label: Text label to display above the grid
        """
        # Draw grid label
        label_surface = self.fonts['heading'].render(label, True, COLORS['text_primary'])
        grid_width = GRID_SIZE * CELL_SIZE
        label_x = offset_x + (grid_width // 2) - (label_surface.get_width() // 2)
        self.screen.blit(label_surface, (label_x, offset_y - 60))

        # Draw column labels (A-J)
        for col in range(GRID_SIZE):
            letter = chr(ord('A') + col)
            text = self.fonts['small'].render(letter, True, COLORS['text_secondary'])
            x = offset_x + col * CELL_SIZE + CELL_SIZE // 2 - text.get_width() // 2
            self.screen.blit(text, (x, offset_y - 20))

        # Draw row labels (1-10)
        for row in range(GRID_SIZE):
            text = self.fonts['small'].render(str(row + 1), True, COLORS['text_secondary'])
            y = offset_y + row * CELL_SIZE + CELL_SIZE // 2 - text.get_height() // 2
            self.screen.blit(text, (offset_x - 25, y))

        # Draw each cell
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                x = offset_x + col * CELL_SIZE
                y = offset_y + row * CELL_SIZE
                cell = board.grid[row, col]

                # Determine cell color (heatmap or default ocean)
                if show_heatmap and cell == CellState.EMPTY.value:
                    prob = board.probability_map[row, col]
                    max_prob = board.probability_map.max() if board.probability_map.max() > 0 else 1
                    intensity = prob / max_prob
                    color = self._interpolate_color(
                        COLORS['heatmap_low'],
                        COLORS['heatmap_high'],
                        intensity
                    )
                else:
                    color = COLORS['ocean']

                # Draw cell background and border
                pygame.draw.rect(self.screen, color, (x, y, CELL_SIZE, CELL_SIZE))
                pygame.draw.rect(self.screen, COLORS['grid_line'],
                                 (x, y, CELL_SIZE, CELL_SIZE), 1)

                center_x = x + CELL_SIZE // 2
                center_y = y + CELL_SIZE // 2

                # Draw appropriate marker based on cell state
                if cell == CellState.HIT.value:
                    self._draw_hit_marker(center_x, center_y)
                elif cell == CellState.MISS.value:
                    self._draw_miss_marker(center_x, center_y)
                elif cell == CellState.SUNK.value:
                    self._draw_sunk_marker(center_x, center_y)
                elif cell == CellState.SHIP.value and show_ships:
                    self._draw_ship_cell(x, y, board, row, col)

    def _interpolate_color(self, color1, color2, t):
        """
        Linearly interpolate between two colors.

        Args:
            color1, color2: RGB tuples to interpolate between
            t: Interpolation factor (0.0 = color1, 1.0 = color2)

        Returns:
            Interpolated RGB tuple
        """
        return tuple(int(c1 + (c2 - c1) * t) for c1, c2 in zip(color1, color2))

    def _draw_ship_cell(self, x, y, board, row, col):
        """Draw a cell containing part of a ship."""
        for ship in board.ships:
            if (row, col) in ship.positions:
                idx = ship.positions.index((row, col))
                is_bow = idx == 0
                is_stern = idx == ship.size - 1

                # Draw ship hull
                pygame.draw.rect(self.screen, ship.color,
                                 (x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4))

                # Draw ship deck
                pygame.draw.rect(self.screen, COLORS['ship_deck'],
                                 (x + 6, y + 6, CELL_SIZE - 12, CELL_SIZE - 12))

                # Draw orientation line
                if ship.orientation == Orientation.HORIZONTAL:
                    pygame.draw.line(self.screen, COLORS['ship_accent'],
                                     (x + 4, y + CELL_SIZE // 2),
                                     (x + CELL_SIZE - 4, y + CELL_SIZE // 2), 2)
                else:
                    pygame.draw.line(self.screen, COLORS['ship_accent'],
                                     (x + CELL_SIZE // 2, y + 4),
                                     (x + CELL_SIZE // 2, y + CELL_SIZE - 4), 2)
                break

    def _draw_hit_marker(self, cx, cy):
        """Draw animated hit marker (fire/explosion effect)."""
        t = time.time()

        # Draw pulsing circles
        for i in range(3):
            radius = 12 - i * 3 + math.sin(t * 5 + i) * 2
            alpha = 200 - i * 50
            color = COLORS['hit_fire'] if i % 2 == 0 else COLORS['hit_glow']
            pygame.draw.circle(self.screen, color, (cx, cy), int(radius))

        # Draw X mark
        pygame.draw.line(self.screen, (255, 255, 255),
                         (cx - 8, cy - 8), (cx + 8, cy + 8), 3)
        pygame.draw.line(self.screen, (255, 255, 255),
                         (cx - 8, cy + 8), (cx + 8, cy - 8), 3)

    def _draw_miss_marker(self, cx, cy):
        """Draw animated miss marker (water splash effect)."""
        t = time.time()

        # Draw expanding rings
        for i in range(2):
            radius = 8 + i * 5 + math.sin(t * 3 + i) * 2
            pygame.draw.circle(self.screen, COLORS['miss_ring'],
                               (cx, cy), int(radius), 2)

        # Draw center splash
        pygame.draw.circle(self.screen, COLORS['miss_splash'], (cx, cy), 4)

    def _draw_sunk_marker(self, cx, cy):
        """Draw sunk ship marker (dark X)."""
        pygame.draw.rect(self.screen, COLORS['sunk'],
                         (cx - 15, cy - 15, 30, 30))
        pygame.draw.line(self.screen, (80, 0, 0),
                         (cx - 10, cy - 10), (cx + 10, cy + 10), 4)
        pygame.draw.line(self.screen, (80, 0, 0),
                         (cx - 10, cy + 10), (cx + 10, cy - 10), 4)

    def draw_statistics_panel(self, stats: GameStats, x: int, y: int,
                              width: int, height: int):
        """
        Draw the battle statistics panel.

        Args:
            stats: GameStats object with current statistics
            x, y: Panel position
            width, height: Panel dimensions
        """
        # Draw panel background
        pygame.draw.rect(self.screen, COLORS['panel_bg'],
                         (x, y, width, height), border_radius=10)
        pygame.draw.rect(self.screen, COLORS['panel_border'],
                         (x, y, width, height), 2, border_radius=10)

        # Draw title
        title = self.fonts['heading'].render("Battle Statistics", True,
                                             COLORS['text_primary'])
        self.screen.blit(title, (x + 15, y + 10))

        # Draw stat items
        stat_y = y + 50
        stat_items = [
            ("Total Shots", str(stats.total_shots)),
            ("Hits", str(stats.hits)),
            ("Misses", str(stats.misses)),
            ("Accuracy", f"{stats.accuracy:.1f}%"),
            ("Ships Sunk", f"{stats.ships_sunk}/5"),
            ("Time", f"{stats.elapsed_time:.0f}s"),
        ]

        for label, value in stat_items:
            label_surf = self.fonts['small'].render(label + ":", True,
                                                    COLORS['text_secondary'])
            self.screen.blit(label_surf, (x + 15, stat_y))

            value_surf = self.fonts['normal'].render(value, True,
                                                     COLORS['text_primary'])
            self.screen.blit(value_surf, (x + width - value_surf.get_width() - 15,
                                          stat_y))
            stat_y += 30

    def draw_ship_status(self, ships: List[Ship], x: int, y: int,
                         width: int, label: str):
        """
        Draw ship status panel showing health bars for each ship.

        Args:
            ships: List of Ship objects to display
            x, y: Panel position
            width: Panel width
            label: Panel title
        """
        height = len(ships) * 35 + 50

        # Draw panel background
        pygame.draw.rect(self.screen, COLORS['panel_bg'],
                         (x, y, width, height), border_radius=10)
        pygame.draw.rect(self.screen, COLORS['panel_border'],
                         (x, y, width, height), 2, border_radius=10)

        # Draw title
        title = self.fonts['normal'].render(label, True, COLORS['text_primary'])
        self.screen.blit(title, (x + 15, y + 10))

        # Draw each ship's status
        ship_y = y + 45
        for ship in ships:
            # Dim text for sunk ships
            color = COLORS['text_secondary'] if ship.is_sunk else COLORS['text_primary']
            name = self.fonts['small'].render(ship.name, True, color)
            self.screen.blit(name, (x + 15, ship_y))

            # Draw health bar
            bar_x = x + 120
            bar_width = width - 140
            bar_height = 12

            # Background bar
            pygame.draw.rect(self.screen, COLORS['panel_border'],
                             (bar_x, ship_y + 3, bar_width, bar_height),
                             border_radius=3)

            # Health bar (green to red based on damage)
            health_pct = 1 - (ship.hits / ship.size)
            if health_pct > 0:
                health_color = (
                    int(255 * (1 - health_pct)),  # Red increases as health decreases
                    int(255 * health_pct),  # Green decreases as health decreases
                    50
                )
                pygame.draw.rect(self.screen, health_color,
                                 (bar_x, ship_y + 3,
                                  int(bar_width * health_pct), bar_height),
                                 border_radius=3)

            ship_y += 35

    def add_explosion(self, x: int, y: int):
        """Add an explosion animation at the specified screen position."""
        self.animations.append(ExplosionAnimation(
            x=x, y=y, start_time=time.time(), duration=0.8
        ))

    def add_splash(self, x: int, y: int):
        """Add a splash animation at the specified screen position."""
        self.animations.append(SplashAnimation(
            x=x, y=y, start_time=time.time(), duration=0.6
        ))

    def update_animations(self):
        """Update and draw all active animations, removing completed ones."""
        for anim in self.animations[:]:
            if isinstance(anim, ExplosionAnimation):
                self._draw_explosion(anim)
            elif isinstance(anim, SplashAnimation):
                self._draw_splash(anim)

            if anim.is_complete:
                self.animations.remove(anim)

    def _draw_explosion(self, anim: ExplosionAnimation):
        """Draw explosion animation frame."""
        progress = anim.progress

        # Draw particles flying outward
        for particle in anim.particles:
            dist = particle['speed'] * progress
            px = anim.x + math.cos(particle['angle']) * dist
            py = anim.y + math.sin(particle['angle']) * dist
            size = particle['size'] * (1 - progress * 0.5)
            alpha = int(255 * (1 - progress))

            if size > 0:
                pygame.draw.circle(self.screen, particle['color'],
                                   (int(px), int(py)), int(size))

        # Draw initial flash
        if progress < 0.3:
            flash_size = 30 * (1 - progress / 0.3)
            pygame.draw.circle(self.screen, (255, 255, 200),
                               (int(anim.x), int(anim.y)), int(flash_size))

    def _draw_splash(self, anim: SplashAnimation):
        """Draw splash animation frame."""
        progress = anim.progress

        # Draw expanding rings
        for ring in anim.rings:
            ring_progress = max(0, progress - ring['delay'])
            if ring_progress > 0 and ring_progress < 1:
                radius = ring['max_radius'] * ring_progress
                alpha = int(200 * (1 - ring_progress))
                pygame.draw.circle(self.screen, COLORS['miss_splash'],
                                   (int(anim.x), int(anim.y)),
                                   int(radius), 2)


# =============================================================================
# MAIN GAME CLASS
# =============================================================================


class BattleshipGame:
    """
    Main game controller class.

    Manages game state, handles user input, and coordinates
    rendering and game logic.
    """

    def __init__(self):
        """Initialize the game window and game state."""
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Battleship - AI Battle Arena")
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)

        # Initialize game boards
        self.player_board = Board(is_player=True)
        self.enemy_board = Board(is_player=False)
        self.phase = GamePhase.PLACEMENT
        self.stats = GameStats()

        # UI state
        self.show_heatmap = True
        self.current_ship_index = 0
        self.dragging_ship = None
        self.drag_orientation = Orientation.HORIZONTAL

        # Replay system state
        self.replay_mode = False
        self.replay_index = 0

        # Set up enemy board with random ship placement
        self.enemy_board.place_ships_randomly()
        self._create_placement_ships()

    def _create_placement_ships(self):
        """Create ship objects for the placement phase."""
        self.placement_ships = []
        for name, size, color in SHIPS:
            self.placement_ships.append(Ship(name=name, size=size, color=color))

    def run(self):
        """Main game loop - handles events, updates, and rendering."""
        running = True

        while running:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event.pos, event.button)
                elif event.type == pygame.MOUSEMOTION:
                    self._handle_motion(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)

            # Update game state and render
            self._update()
            self._draw()
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS cap

        pygame.quit()

    def _handle_click(self, pos, button):
        """Route click events based on current game phase."""
        if self.phase == GamePhase.PLACEMENT:
            self._handle_placement_click(pos, button)
        elif self.phase == GamePhase.BATTLE:
            self._handle_battle_click(pos, button)

    def _handle_placement_click(self, pos, button):
        """Handle clicks during ship placement phase."""
        # Right-click rotates ship orientation
        if button == 3:
            if self.drag_orientation == Orientation.HORIZONTAL:
                self.drag_orientation = Orientation.VERTICAL
            else:
                self.drag_orientation = Orientation.HORIZONTAL
            return

        if self.current_ship_index >= len(self.placement_ships):
            return

        # Try to place ship at clicked position
        grid_pos = self._screen_to_grid(pos, GRID_OFFSET_X, GRID_OFFSET_Y)
        if grid_pos:
            ship = self.placement_ships[self.current_ship_index]
            if self.player_board.place_ship(ship, grid_pos, self.drag_orientation):
                self.current_ship_index += 1

                # Check if all ships placed
                if self.current_ship_index >= len(self.placement_ships):
                    self.phase = GamePhase.BATTLE
                    self.stats = GameStats()

    def _handle_battle_click(self, pos, button):
        """Handle clicks during battle phase - process attacks."""
        grid_pos = self._screen_to_grid(pos, ENEMY_GRID_OFFSET_X, GRID_OFFSET_Y)
        if grid_pos:
            row, col = grid_pos
            # Only attack unexplored cells
            if self.enemy_board.grid[row, col] in [CellState.EMPTY.value,
                                                   CellState.SHIP.value]:
                result, ship = self.enemy_board.receive_attack(grid_pos)
                self._process_attack_result(grid_pos, result, ship)

    def _process_attack_result(self, pos, result, ship):
        """
        Process the result of an attack and update game state.

        Args:
            pos: (row, col) of the attack
            result: CellState result of the attack
            ship: Ship object if one was hit
        """
        row, col = pos
        screen_x = ENEMY_GRID_OFFSET_X + col * CELL_SIZE + CELL_SIZE // 2
        screen_y = GRID_OFFSET_Y + row * CELL_SIZE + CELL_SIZE // 2

        self.stats.total_shots += 1

        # Trigger appropriate animation and update stats
        if result == CellState.HIT:
            self.stats.hits += 1
            self.renderer.add_explosion(screen_x, screen_y)
        elif result == CellState.SUNK:
            self.stats.hits += 1
            self.stats.ships_sunk += 1
            self.renderer.add_explosion(screen_x, screen_y)
        elif result == CellState.MISS:
            self.stats.misses += 1
            self.renderer.add_splash(screen_x, screen_y)

        # Record move in history
        self.stats.turn_history.append({
            'pos': pos,
            'result': result,
            'ship': ship.name if ship else None
        })

        # Update probability map for heatmap
        remaining = [s for s in self.enemy_board.ships if not s.is_sunk]
        self.enemy_board.update_probability_map(remaining)

        # Check for victory
        if self.enemy_board.all_ships_sunk():
            self.phase = GamePhase.GAME_OVER

    def _handle_motion(self, pos):
        """Handle mouse motion events (used for ship placement preview)."""
        pass

    def _handle_key(self, key):
        """Handle keyboard input."""
        if key == pygame.K_h:
            # Toggle heatmap display
            self.show_heatmap = not self.show_heatmap
        elif key == pygame.K_r and self.phase == GamePhase.GAME_OVER:
            # Reset game
            self._reset_game()
        elif key == pygame.K_SPACE and self.phase == GamePhase.PLACEMENT:
            # Auto-place remaining ships
            for i in range(self.current_ship_index, len(self.placement_ships)):
                ship = self.placement_ships[i]
                placed = False
                while not placed:
                    row = random.randint(0, GRID_SIZE - 1)
                    col = random.randint(0, GRID_SIZE - 1)
                    orientation = random.choice([Orientation.HORIZONTAL,
                                                 Orientation.VERTICAL])
                    placed = self.player_board.place_ship(ship, (row, col),
                                                          orientation)
            self.current_ship_index = len(self.placement_ships)
            self.phase = GamePhase.BATTLE
            self.stats = GameStats()

    def _screen_to_grid(self, pos, offset_x, offset_y) -> Optional[Tuple[int, int]]:
        """
        Convert screen coordinates to grid coordinates.

        Args:
            pos: (x, y) screen position
            offset_x, offset_y: Grid offset on screen

        Returns:
            (row, col) grid position or None if outside grid
        """
        x, y = pos
        col = (x - offset_x) // CELL_SIZE
        row = (y - offset_y) // CELL_SIZE

        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            return (row, col)
        return None

    def _reset_game(self):
        """Reset the game to initial state for a new game."""
        self.player_board = Board(is_player=True)
        self.enemy_board = Board(is_player=False)
        self.enemy_board.place_ships_randomly()
        self._create_placement_ships()
        self.current_ship_index = 0
        self.phase = GamePhase.PLACEMENT
        self.stats = GameStats()

    def _update(self):
        """Update game state (called each frame)."""
        pass

    def _draw(self):
        """Render the entire game screen."""
        # Draw background
        self.renderer.draw_ocean_background()

        # Draw title
        title = self.renderer.fonts['title'].render(
            "BATTLESHIP - AI Battle Arena", True, COLORS['text_primary'])
        self.screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 20))

        # Draw phase instructions
        phase_text = {
            GamePhase.PLACEMENT: "Place your ships (Right-click to rotate, Space to auto-place)",
            GamePhase.BATTLE: "Click enemy grid to attack! (H to toggle heatmap)",
            GamePhase.GAME_OVER: "Victory! Press R to play again"
        }
        phase_surf = self.renderer.fonts['normal'].render(
            phase_text[self.phase], True, COLORS['text_secondary'])
        self.screen.blit(phase_surf, (SCREEN_WIDTH // 2 - phase_surf.get_width() // 2, 55))

        # Draw player grid (always show ships)
        self.renderer.draw_grid(GRID_OFFSET_X, GRID_OFFSET_Y,
                                self.player_board, show_ships=True,
                                label="Your Fleet")

        # Draw enemy grid (hide ships, optionally show heatmap)
        self.renderer.draw_grid(ENEMY_GRID_OFFSET_X, GRID_OFFSET_Y,
                                self.enemy_board, show_ships=False,
                                show_heatmap=self.show_heatmap and self.phase == GamePhase.BATTLE,
                                label="Enemy Waters")

        # Draw ship placement preview
        if self.phase == GamePhase.PLACEMENT and self.current_ship_index < len(self.placement_ships):
            self._draw_placement_preview()

        # Draw statistics panel
        self.renderer.draw_statistics_panel(
            self.stats, ENEMY_GRID_OFFSET_X,
            GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE + 30,
            300, 220)

        # Draw ship status panels
        self.renderer.draw_ship_status(
            self.player_board.ships, GRID_OFFSET_X,
            GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE + 30,
            280, "Your Ships")

        if self.phase != GamePhase.PLACEMENT:
            self.renderer.draw_ship_status(
                self.enemy_board.ships, GRID_OFFSET_X + 300,
                                        GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE + 30,
                280, "Enemy Ships")

        # Draw heatmap legend
        if self.show_heatmap and self.phase == GamePhase.BATTLE:
            self._draw_heatmap_legend()

        # Update and draw animations
        self.renderer.update_animations()

    def _draw_placement_preview(self):
        """Draw ghost preview of ship being placed."""
        mouse_pos = pygame.mouse.get_pos()
        grid_pos = self._screen_to_grid(mouse_pos, GRID_OFFSET_X, GRID_OFFSET_Y)

        if grid_pos and self.current_ship_index < len(self.placement_ships):
            ship = self.placement_ships[self.current_ship_index]
            row, col = grid_pos
            valid = self.player_board.can_place_ship(ship, grid_pos,
                                                     self.drag_orientation)

            # Green for valid placement, red for invalid
            color = (100, 200, 100, 128) if valid else (200, 100, 100, 128)

            # Draw preview cells
            for i in range(ship.size):
                if self.drag_orientation == Orientation.HORIZONTAL:
                    c = col + i
                    r = row
                else:
                    c = col
                    r = row + i

                if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
                    x = GRID_OFFSET_X + c * CELL_SIZE
                    y = GRID_OFFSET_Y + r * CELL_SIZE

                    preview_surf = pygame.Surface((CELL_SIZE - 2, CELL_SIZE - 2))
                    preview_surf.fill(color[:3])
                    preview_surf.set_alpha(128)
                    self.screen.blit(preview_surf, (x + 1, y + 1))

            # Show ship name being placed
            name_text = self.renderer.fonts['normal'].render(
                f"Placing: {ship.name} ({ship.size} cells)",
                True, COLORS['text_primary'])
            self.screen.blit(name_text, (GRID_OFFSET_X, GRID_OFFSET_Y - 70))

    def _draw_heatmap_legend(self):
        """Draw the probability heatmap color legend."""
        x = ENEMY_GRID_OFFSET_X + GRID_SIZE * CELL_SIZE + 20
        y = GRID_OFFSET_Y
        width = 30
        height = 200

        # Draw title
        title = self.renderer.fonts['small'].render("Probability", True,
                                                    COLORS['text_primary'])
        self.screen.blit(title, (x, y - 25))

        # Draw gradient
        for i in range(height):
            t = i / height
            color = self.renderer._interpolate_color(
                COLORS['heatmap_high'], COLORS['heatmap_low'], t)
            pygame.draw.line(self.screen, color, (x, y + i), (x + width, y + i))

        # Draw border
        pygame.draw.rect(self.screen, COLORS['grid_line'],
                         (x, y, width, height), 1)

        # Draw labels
        high_label = self.renderer.fonts['small'].render("High", True,
                                                         COLORS['text_secondary'])
        low_label = self.renderer.fonts['small'].render("Low", True,
                                                        COLORS['text_secondary'])
        self.screen.blit(high_label, (x + width + 5, y))
        self.screen.blit(low_label, (x + width + 5, y + height - 15))


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    game = BattleshipGame()
    game.run()