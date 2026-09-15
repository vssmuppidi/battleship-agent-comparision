"""
Battleship Game with Probability Agent Integration

This module extends the base Battleship game to run with the probability-based
AI agent. It provides a visual interface for watching the agent play, with
controls for toggling agent automation, adjusting speed, and single-stepping
through moves.

The agent uses hunt/target mode logic and probability-based cell selection
to efficiently locate and destroy enemy ships.

Controls:
- A: Toggle agent on/off (auto-play vs manual observation)
- UP/DOWN: Adjust agent move delay (speed)
- SPACE: Single step (make one agent move)
- H: Toggle probability heatmap
- R: Restart game

Author: Venkatashivasai Muppidi
Course: Foundations of Artificial Intelligence
University: Northeastern University
"""

import pygame
import time
from battleship_game import (
    BattleshipGame, GamePhase, CellState, COLORS,
    ENEMY_GRID_OFFSET_X, GRID_OFFSET_Y, CELL_SIZE, GRID_SIZE,
    SCREEN_WIDTH, Orientation, SHIPS
)
from probability_agent import ProbabilityAgent


class BattleshipGameWithAgent(BattleshipGame):
    """
    Extended Battleship game with probability agent integration.

    Inherits from the base BattleshipGame and adds AI agent control,
    allowing the agent to play automatically or be stepped manually
    for observation and debugging.

    Attributes:
        agent: ProbabilityAgent instance making targeting decisions
        agent_enabled: Whether agent is playing automatically
        agent_delay: Seconds between automatic agent moves
        last_agent_move_time: Timestamp of last agent action
        ships_sunk_count: Running count of sunk enemy ships
    """

    def __init__(self):
        """Initialize the game with agent integration."""
        super().__init__()

        # Create the probability agent
        self.agent = ProbabilityAgent(
            grid_size=GRID_SIZE,
            ship_sizes=[size for _, size, _ in SHIPS]
        )

        # Agent control settings
        self.agent_enabled = False  # Start with agent disabled
        self.agent_delay = 0.5  # Half-second between moves
        self.last_agent_move_time = 0
        self.ships_sunk_count = 0

        # Auto-place player ships to skip placement phase
        self.auto_place_player_ships()

    def auto_place_player_ships(self):
        """
        Automatically place player ships and start battle phase.

        Since we're watching the agent play, we don't need manual
        ship placement. This skips directly to the battle phase.
        """
        self.player_board.place_ships_randomly()
        self.current_ship_index = len(self.placement_ships)
        self.phase = GamePhase.BATTLE

    def _handle_key(self, key):
        """
        Handle keyboard input with additional agent controls.

        Args:
            key: Pygame key code
        """
        # Call parent handler for standard controls (H, R, etc.)
        super()._handle_key(key)

        # Toggle agent auto-play
        if key == pygame.K_a:
            self.agent_enabled = not self.agent_enabled
            if self.agent_enabled:
                print("Agent ENABLED - watching AI play")
            else:
                print("Agent DISABLED - manual play")

        # Increase agent speed (decrease delay)
        if key == pygame.K_UP:
            self.agent_delay = max(0.1, self.agent_delay - 0.1)
            print(f"Agent delay: {self.agent_delay:.1f}s")

        # Decrease agent speed (increase delay)
        if key == pygame.K_DOWN:
            self.agent_delay = min(2.0, self.agent_delay + 0.1)
            print(f"Agent delay: {self.agent_delay:.1f}s")

        # Single step: make one agent move
        if key == pygame.K_SPACE and self.phase == GamePhase.BATTLE:
            self._agent_make_move()

    def _update(self):
        """
        Update game state each frame.

        When agent is enabled, automatically makes moves at the
        configured delay interval.
        """
        super()._update()

        # Auto-play: make agent move if enough time has passed
        if (self.agent_enabled and
                self.phase == GamePhase.BATTLE and
                time.time() - self.last_agent_move_time > self.agent_delay):
            self._agent_make_move()
            self.last_agent_move_time = time.time()

    def _count_sunk_ships(self):
        """
        Count the number of sunk enemy ships.

        Used to detect when a ship is sunk by comparing counts
        before and after an attack.

        Returns:
            Number of sunk ships
        """
        count = 0
        for ship in self.enemy_board.ships:
            if ship.is_sunk:
                count += 1
        return count

    def _agent_make_move(self):
        """
        Execute one agent move.

        Gets the agent's target choice, executes the attack,
        and reports the result back to the agent for learning.
        """
        # Only play during battle phase
        if self.phase != GamePhase.BATTLE:
            return

        # Check for game over
        if self.enemy_board.all_ships_sunk():
            self.phase = GamePhase.GAME_OVER
            return

        # Get agent's target selection
        target = self.agent.choose_target()
        row, col = target

        # Verify cell hasn't been attacked
        cell_state = self.enemy_board.grid[row, col]
        if cell_state in [CellState.EMPTY.value, CellState.SHIP.value]:
            # Count ships before attack to detect sinking
            ships_before = self._count_sunk_ships()

            # Execute the attack
            result, ship = self.enemy_board.receive_attack(target)
            self._process_attack_result(target, result, ship)

            # Check if a ship was sunk
            ships_after = self._count_sunk_ships()
            ship_was_sunk = ships_after > ships_before

            # Determine hit status and sunk ship info
            hit = result in [CellState.HIT, CellState.SUNK]
            sunk_size = None
            sunk_name = None

            if ship_was_sunk and ship:
                sunk_size = ship.size
                sunk_name = ship.name

            # Report result to agent for state updates
            self.agent.record_result(target, hit, sunk_size)

            # Print move information
            stats = self.agent.get_stats()
            if sunk_size:
                result_str = f"SUNK {sunk_name}!"
            elif hit:
                result_str = "HIT"
            else:
                result_str = "MISS"

            mode = self.agent.mode.name
            print(
                f"Agent attacks {chr(65 + col)}{row + 1}: {result_str} | Mode: {mode} | Accuracy: {stats['accuracy']:.1f}%")
        else:
            # Shouldn't happen with a well-functioning agent
            print(f"Agent tried to attack already-attacked cell {chr(65 + col)}{row + 1}, skipping...")

    def _reset_game(self):
        """Reset game state for a new game."""
        super()._reset_game()
        self.agent.reset()
        self.ships_sunk_count = 0
        self.auto_place_player_ships()

    def _draw(self):
        """
        Render the game screen with additional agent controls panel.
        """
        # Draw base game elements
        super()._draw()

        # Agent control panel instructions
        instructions = [
            "A: Toggle AI agent",
            "UP/DOWN: Adjust speed",
            "SPACE: Single step",
            "H: Toggle heatmap",
            "R: Restart"
        ]

        # Position below statistics panel
        y = GRID_OFFSET_Y + GRID_SIZE * CELL_SIZE + 250

        # Draw panel background
        pygame.draw.rect(self.screen, COLORS['panel_bg'],
                         (ENEMY_GRID_OFFSET_X, y, 300, 130), border_radius=10)
        pygame.draw.rect(self.screen, COLORS['panel_border'],
                         (ENEMY_GRID_OFFSET_X, y, 300, 130), 2, border_radius=10)

        # Draw panel title
        title = self.renderer.fonts['normal'].render("Controls", True, COLORS['text_primary'])
        self.screen.blit(title, (ENEMY_GRID_OFFSET_X + 15, y + 10))

        # Draw agent status indicator (green ON, red OFF)
        status = "ON" if self.agent_enabled else "OFF"
        status_color = (100, 255, 100) if self.agent_enabled else (255, 100, 100)
        status_text = self.renderer.fonts['small'].render(f"Agent: {status}", True, status_color)
        self.screen.blit(status_text, (ENEMY_GRID_OFFSET_X + 200, y + 10))

        # Draw control instructions
        for i, instruction in enumerate(instructions):
            text = self.renderer.fonts['small'].render(instruction, True, COLORS['text_secondary'])
            self.screen.blit(text, (ENEMY_GRID_OFFSET_X + 15, y + 35 + i * 18))


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    game = BattleshipGameWithAgent()
    game.run()