import numpy as np
import pygame
from typing import Optional, Tuple
from .game_state import GameState


class GameRenderer:
    def __init__(self, game_state: GameState, cell_size: int = 20, headless: bool = False):
        self.game_state = game_state
        self.cell_size = cell_size
        self.headless = headless
        self.width = game_state.width * cell_size
        self.height = game_state.height * cell_size
        self.screen = None
        self.clock = None
        self.closed = headless

        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width + 200, self.height))
            pygame.display.set_caption("Railroad Tycoon")
            self.clock = pygame.time.Clock()

        self.colors = {
            'plains': (200, 200, 100),
            'river': (100, 150, 200),
            'mountain': (150, 150, 150),
            'player0': (255, 0, 0),
            'player1': (0, 0, 255),
            'neutral': (100, 100, 100),
            'town': (255, 255, 0),
            'background': (50, 50, 50),
            'grid': (80, 80, 80),
        }

    def render(self) -> bool:
        """Draws one frame and pumps the window's event queue (required to keep it
        responsive/closable). Returns False if the window has been closed (by the
        user or a prior call), True otherwise - callers should stop rendering/stepping
        once this goes False."""
        if self.headless or self.closed:
            return False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False

        self.screen.fill(self.colors['background'])
        self._draw_map()
        self._draw_ui()
        pygame.display.flip()
        self.clock.tick(30)
        return True

    def _draw_map(self):
        inked_regions = self.game_state.inked_regions
        for y in range(self.game_state.height):
            for x in range(self.game_state.width):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)

                terrain = self.game_state.get_terrain(x, y)
                color_map = {0: self.colors['plains'], 1: self.colors['river'], 2: self.colors['mountain']}
                color = color_map.get(terrain, self.colors['background'])

                region = self.game_state.get_region(x, y)
                if region in inked_regions:
                    color = (30, 30, 30)

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.colors['grid'], rect, 1)

                if self.game_state.tracks[y, x] != -1:
                    track_owner = self.game_state.tracks[y, x]
                    track_color = self.colors['player0'] if track_owner == 0 else (
                        self.colors['player1'] if track_owner == 1 else self.colors['neutral']
                    )
                    pygame.draw.circle(
                        self.screen,
                        track_color,
                        (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2),
                        self.cell_size // 4,
                    )

        for town in self.game_state.towns:
            x, y = town.coord
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, self.colors['town'], rect)
            pygame.draw.rect(self.screen, (200, 200, 0), rect, 2)

    def _draw_ui(self):
        ui_x = self.game_state.width * self.cell_size + 10
        font = pygame.font.Font(None, 24)

        texts = [
            f"Turn: {self.game_state.turn}/{self.game_state.max_turns}",
            f"",
            f"Player 0: {self.game_state.scores[0]} pts",
            f"Player 1: {self.game_state.scores[1]} pts",
            f"",
            f"Paint: {self.game_state.paint_points[0]}/{self.game_state.paint_points[1]}",
            f"Disrupt: {self.game_state.disruption_points[0]}/{self.game_state.disruption_points[1]}",
        ]

        for i, text in enumerate(texts):
            if text:
                surf = font.render(text, True, (255, 255, 255))
                self.screen.blit(surf, (ui_x, 20 + i * 30))

    def close(self):
        if not self.closed and self.screen:
            pygame.quit()
        self.closed = True
        self.screen = None
