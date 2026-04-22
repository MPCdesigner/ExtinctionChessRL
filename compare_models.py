"""
Compare two AlphaZero checkpoints by having them play against each other.

    python compare_models.py --model1 ../models/az_iter70.pt --model2 ../models/az_iter30.pt --sims 50

Opens pygame. Play out an opening position, then press SPACE to watch the models play.
Two games are played from that position (M1 as white, then M1 as black).
Press SPACE after both games to set up another opening, or ESC to quit.
"""
import argparse
import sys
import threading
from copy import deepcopy

import pygame

from extinction_chess import ExtinctionChess, Color, Position, Move
from extinction_chess_gui import (
    ChessGUI, SIDEBAR_X, BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_SIZE,
    SQUARE_SIZE, WINDOW_WIDTH, WINDOW_HEIGHT, WHITE, BLACK,
)
from alphazero import AlphaZeroNet, AlphaZeroEvaluator, mcts_search


class CompareGUI(ChessGUI):
    """GUI for setting up opening positions, then watching models play live."""

    def __init__(self, game, eval1, eval2, sims, meta1, meta2):
        super().__init__(game)
        pygame.display.set_caption("Compare Models — Play an opening, then press SPACE")
        self.eval1 = eval1
        self.eval2 = eval2
        self.sims = sims
        self.meta1 = meta1
        self.meta2 = meta2

        # State machine: setup -> game_a -> game_a_done -> game_b -> results
        self.phase = "setup"
        self.frozen_state = None  # saved opening position

        # Live game state
        self.current_game_label = ""  # "Game A (M1=White)" etc.
        self.m1_is_white = True
        self.move_count = 0
        self.pending_move = None  # (move, who, visits) computed by background thread
        self.thinking = False
        self.game_a_result = 0
        self.game_b_result = 0
        self.game_a_log = []
        self.game_b_log = []
        self.current_log = []

        # Totals across rounds
        self.total_m1_wins = 0
        self.total_m2_wins = 0
        self.total_draws = 0
        self.round_num = 0

    def handle_click(self, pos):
        if self.phase != "setup":
            return
        super().handle_click(pos)

    def start_comparison(self):
        """Freeze the opening and start Game A."""
        self.frozen_state = deepcopy(self.game)
        self.round_num += 1
        print(f"\n=== Round {self.round_num} ===", flush=True)
        self._start_game("Game A", m1_is_white=True)

    def _start_game(self, label, m1_is_white):
        """Reset visible board to opening and begin a game."""
        self.game = deepcopy(self.frozen_state)
        self.m1_is_white = m1_is_white
        self.current_game_label = f"{label} (M1={'White' if m1_is_white else 'Black'})"
        self.move_count = 0
        self.current_log = []
        self.pending_move = None
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None
        color_str = "White" if m1_is_white else "Black"
        print(f"  {label} — M1 as {color_str}", flush=True)
        self.phase = "game_a" if "A" in label else "game_b"
        self._start_thinking()

    def _start_thinking(self):
        """Launch background thread to compute the next move."""
        if self.game.game_over or self.move_count >= 300:
            self._finish_current_game()
            return
        self.thinking = True
        self.pending_move = None
        thread = threading.Thread(target=self._compute_move, daemon=True)
        thread.start()

    def _compute_move(self):
        """Background: compute one move."""
        try:
            is_model1_turn = (self.game.current_player == Color.WHITE) == self.m1_is_white
            evaluator = self.eval1 if is_model1_turn else self.eval2
            who = "M1" if is_model1_turn else "M2"

            mv = mcts_search(self.game, evaluator, num_simulations=self.sims,
                             dirichlet_alpha=0, noise_weight=0,
                             tactical_shortcuts=False)
            if not mv:
                self.pending_move = (None, who, 0)
                return
            best = max(mv, key=lambda x: x[1])
            self.pending_move = (best[0], who, best[1])
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self.pending_move = (None, "??", 0)

    def apply_pending_move(self):
        """Apply the computed move to the visible board."""
        move, who, visits = self.pending_move
        self.pending_move = None
        self.thinking = False

        if move is None:
            self._finish_current_game()
            return

        self.move_count += 1
        self.current_log.append((who, str(move), visits))
        print(f"    move {self.move_count}: {who} plays {move} (visits={visits})", flush=True)
        self.game.make_move(move)
        self.last_move = move

        if self.game.game_over or self.move_count >= 300:
            self._finish_current_game()
        else:
            self._start_thinking()

    def _finish_current_game(self):
        """Handle end of current game."""
        if self.game.winner:
            model1_won = (self.game.winner == Color.WHITE) == self.m1_is_white
            result = 1 if model1_won else -1
        else:
            result = 0

        sym = "M1 wins" if result == 1 else ("M2 wins" if result == -1 else "Draw")
        print(f"  Result: {sym} ({self.move_count} moves)", flush=True)

        if self.phase == "game_a":
            self.game_a_result = result
            self.game_a_log = self.current_log[:]
            self.phase = "game_a_done"
            # Auto-start game B after a brief pause (handled in main loop)
        elif self.phase == "game_b":
            self.game_b_result = result
            self.game_b_log = self.current_log[:]
            # Tally results
            for r in [self.game_a_result, self.game_b_result]:
                if r == 1:
                    self.total_m1_wins += 1
                elif r == -1:
                    self.total_m2_wins += 1
                else:
                    self.total_draws += 1
            total = self.total_m1_wins + self.total_m2_wins + self.total_draws
            print(f"  Totals: M1={self.total_m1_wins}W  M2={self.total_m2_wins}W  D={self.total_draws}  ({total} games)", flush=True)
            self.phase = "results"

        self.thinking = False

    def reset_for_new_opening(self):
        """Reset board for a new opening setup."""
        self.game = ExtinctionChess()
        self.selected_square = None
        self.legal_moves = []
        self.last_move = None
        self.promotion_pending = False
        self.promotion_from = None
        self.promotion_to = None
        self.frozen_state = None
        self.pending_move = None
        self.thinking = False
        self.phase = "setup"

    def draw_sidebar(self):
        super().draw_sidebar()

        y = 400
        iter1 = self.meta1.get('iteration', '?')
        iter2 = self.meta2.get('iteration', '?')

        # Model info
        text = self.font.render(f"M1: iter {iter1}", True, (0, 0, 180))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 22
        text = self.font.render(f"M2: iter {iter2}", True, (180, 0, 0))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 22
        text = self.small_font.render(f"Sims: {self.sims}", True, (100, 100, 100))
        self.screen.blit(text, (SIDEBAR_X, y))
        y += 30

        if self.phase == "setup":
            text = self.font.render("Play opening moves", True, (0, 130, 0))
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 22
            text = self.font.render("Press SPACE to start", True, (0, 130, 0))
            self.screen.blit(text, (SIDEBAR_X, y))

        elif self.phase in ("game_a", "game_b"):
            text = self.font.render(self.current_game_label, True, (200, 100, 0))
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 22
            if self.thinking:
                is_m1 = (self.game.current_player == Color.WHITE) == self.m1_is_white
                who = "M1" if is_m1 else "M2"
                text = self.font.render(f"{who} thinking...", True, (200, 100, 0))
                self.screen.blit(text, (SIDEBAR_X, y))
            y += 22
            text = self.small_font.render(f"Move {self.move_count}", True, (100, 100, 100))
            self.screen.blit(text, (SIDEBAR_X, y))

        elif self.phase == "game_a_done":
            sym = "M1 wins" if self.game_a_result == 1 else ("M2 wins" if self.game_a_result == -1 else "Draw")
            text = self.font.render(f"Game A: {sym}", True, BLACK)
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 22
            text = self.font.render("Starting Game B...", True, (200, 100, 0))
            self.screen.blit(text, (SIDEBAR_X, y))

        elif self.phase == "results":
            def sym(r):
                return "M1 wins" if r == 1 else ("M2 wins" if r == -1 else "Draw")
            text = self.font.render(f"Game A (M1=W): {sym(self.game_a_result)}", True, BLACK)
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 22
            text = self.font.render(f"Game B (M1=B): {sym(self.game_b_result)}", True, BLACK)
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 30
            text = self.font.render("SPACE = next opening", True, (0, 130, 0))
            self.screen.blit(text, (SIDEBAR_X, y))

        # Running totals
        total = self.total_m1_wins + self.total_m2_wins + self.total_draws
        if total > 0:
            y = WINDOW_HEIGHT - 140
            text = self.font.render(f"Overall ({total} games):", True, BLACK)
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 22
            text = self.font.render(f"  M1: {self.total_m1_wins}W", True, (0, 0, 180))
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 20
            text = self.font.render(f"  M2: {self.total_m2_wins}W", True, (180, 0, 0))
            self.screen.blit(text, (SIDEBAR_X, y))
            y += 20
            text = self.font.render(f"  Draws: {self.total_draws}", True, (100, 100, 100))
            self.screen.blit(text, (SIDEBAR_X, y))

    def run(self):
        running = True
        clock = pygame.time.Clock()
        game_a_done_timer = 0  # brief pause before starting game B

        while running:
            # Apply move if background thread finished
            if self.thinking and self.pending_move is not None:
                self.apply_pending_move()

            # Auto-start game B after brief pause
            if self.phase == "game_a_done":
                game_a_done_timer += 1
                if game_a_done_timer > 90:  # ~1.5 seconds at 60fps
                    game_a_done_timer = 0
                    self._start_game("Game B", m1_is_white=False)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(pygame.mouse.get_pos())
                    elif event.button == 3 and self.phase == "setup":
                        self.selected_square = None
                        self.legal_moves = []

                elif event.type == pygame.KEYDOWN:
                    if self.phase == "setup" and self.promotion_pending:
                        self.handle_promotion_key(event.key)

                    elif event.key == pygame.K_SPACE:
                        if self.phase == "setup":
                            self.start_comparison()
                        elif self.phase == "results":
                            self.reset_for_new_opening()

                    elif event.key == pygame.K_r and self.phase == "setup":
                        self.reset_for_new_opening()

                    elif event.key == pygame.K_ESCAPE:
                        running = False

            self.screen.fill((255, 255, 255))
            self.draw_board()
            self.draw_highlights()
            self.draw_pieces()
            self.draw_sidebar()
            self.draw_promotion_dialog()
            pygame.display.flip()
            clock.tick(60)

        # Print final summary
        total = self.total_m1_wins + self.total_m2_wins + self.total_draws
        if total > 0:
            m1_score = (self.total_m1_wins + 0.5 * self.total_draws) / total
            m2_score = (self.total_m2_wins + 0.5 * self.total_draws) / total
            print(f"\nFinal Results ({total} games across {self.round_num} openings):")
            print(f"  M1 (iter {self.meta1.get('iteration', '?')}): {self.total_m1_wins}W {self.total_draws}D {self.total_m2_wins}L — {m1_score:.1%}")
            print(f"  M2 (iter {self.meta2.get('iteration', '?')}): {self.total_m2_wins}W {self.total_draws}D {self.total_m1_wins}L — {m2_score:.1%}")

        pygame.quit()
        sys.exit()


def main():
    parser = argparse.ArgumentParser(description='Compare two AlphaZero models')
    parser.add_argument('--model1', type=str, required=True, help='Path to first model')
    parser.add_argument('--model2', type=str, required=True, help='Path to second model')
    parser.add_argument('--sims', type=int, default=200, help='MCTS simulations per move (default: 200)')
    args = parser.parse_args()

    print(f"Loading model 1: {args.model1}")
    model1, meta1 = AlphaZeroNet.load_checkpoint(args.model1)
    eval1 = AlphaZeroEvaluator(model1, device="cpu")
    print(f"  Iteration: {meta1.get('iteration', '?')}, Win rate: {meta1.get('win_rate', '?')}")

    print(f"Loading model 2: {args.model2}")
    model2, meta2 = AlphaZeroNet.load_checkpoint(args.model2)
    eval2 = AlphaZeroEvaluator(model2, device="cpu")
    print(f"  Iteration: {meta2.get('iteration', '?')}, Win rate: {meta2.get('win_rate', '?')}")

    print(f"\nPlay opening moves, then press SPACE to watch the models play.")
    print(f"Each opening plays 2 games (M1 as white, M1 as black).")
    print(f"Press SPACE after results for a new opening, ESC to quit.\n")

    pygame.init()
    game = ExtinctionChess()
    gui = CompareGUI(game, eval1, eval2, args.sims, meta1, meta2)
    gui.run()


if __name__ == "__main__":
    main()
