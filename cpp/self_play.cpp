#include "self_play.h"
#include <algorithm>
#include <cstring>
#include <random>
#include <cmath>
#include <cassert>

namespace ext {

// ═══════════════════════════════════════════════════════════════════════════
// SelfPlayManager
// ═══════════════════════════════════════════════════════════════════════════

SelfPlayManager::SelfPlayManager(int num_parallel_games, int total_games,
                                 int num_simulations, float c_puct,
                                 float dirichlet_alpha, float noise_weight,
                                 bool tactical_shortcuts, int temp_threshold,
                                 int max_moves, int mcts_batch_size)
    : num_parallel_(num_parallel_games),
      total_games_(total_games),
      num_simulations_(num_simulations),
      c_puct_(c_puct),
      dirichlet_alpha_(dirichlet_alpha),
      noise_weight_(noise_weight),
      tactical_shortcuts_(tactical_shortcuts),
      temp_threshold_(temp_threshold),
      max_moves_(max_moves),
      mcts_batch_size_(mcts_batch_size),
      games_started_(0),
      games_completed_(0)
{
    // Start initial batch of games
    int initial = std::min(num_parallel_, total_games_);
    games_.resize(initial);
    for (int i = 0; i < initial; i++) {
        start_new_game(i);
    }
}

int SelfPlayManager::games_active() const {
    int count = 0;
    for (auto& g : games_) {
        if (g.phase != GamePhase::FINISHED)
            count++;
    }
    return count;
}

void SelfPlayManager::start_new_game(int slot) {
    auto& pg = games_[slot];
    pg.game = Game();  // fresh game
    pg.phase = GamePhase::NEED_ROOT_EVAL;
    delete pg.mcts;
    pg.mcts = nullptr;
    pg.move_count = 0;
    pg.boards.clear();
    pg.policies.clear();
    pg.players.clear();
    games_started_++;
}

void SelfPlayManager::finish_game(int slot) {
    auto& pg = games_[slot];

    // Determine outcome from white's perspective
    float outcome = 0.0f;
    if (pg.game.over) {
        if (pg.game.winner == WHITE) outcome = 1.0f;
        else if (pg.game.winner == BLACK) outcome = -1.0f;
        // else draw = 0.0f
    }
    // If max moves exceeded, treat as draw
    // outcome already 0.0f

    // Build GameRecord
    GameRecord rec;
    rec.boards = std::move(pg.boards);
    rec.policies = std::move(pg.policies);
    rec.players = std::move(pg.players);
    rec.outcome = outcome;
    completed_.push_back(std::move(rec));
    games_completed_++;

    // Clean up MCTS
    delete pg.mcts;
    pg.mcts = nullptr;
    pg.phase = GamePhase::FINISHED;

    // Start a new game in this slot if we haven't started enough
    if (games_started_ < total_games_) {
        start_new_game(slot);
    }
}

void SelfPlayManager::record_position(int slot) {
    auto& pg = games_[slot];

    // Encode current board
    std::vector<float> board_enc(14 * 64);
    pg.game.encode_board(board_enc.data());
    pg.boards.push_back(std::move(board_enc));

    // Player who is about to move
    pg.players.push_back(pg.game.side == WHITE ? 0 : 1);

    // Policy will be filled in when the move is made (make_move)
}

void SelfPlayManager::make_move(int slot) {
    auto& pg = games_[slot];
    assert(pg.mcts != nullptr);

    // Get MCTS visit counts
    auto results = pg.mcts->get_move_results();  // (from, to, promo, visits)

    // Build policy target from visit counts
    std::vector<float> policy(POLICY_SIZE, 0.0f);
    int total_visits = 0;
    for (auto& [from, to, promo, visits] : results) {
        total_visits += visits;
    }

    if (total_visits > 0) {
        for (auto& [from, to, promo, visits] : results) {
            Move m(from, to, promo);
            int idx = move_to_index(m);
            policy[idx] = static_cast<float>(visits) / static_cast<float>(total_visits);
        }
    }

    // Record policy for this position
    pg.policies.push_back(std::move(policy));

    // Select move based on temperature
    Move chosen_move;
    if (pg.move_count < temp_threshold_) {
        // Temperature = 1: sample proportional to visit counts
        std::mt19937 rng(std::random_device{}());
        std::vector<float> weights;
        weights.reserve(results.size());
        for (auto& [from, to, promo, visits] : results) {
            weights.push_back(static_cast<float>(visits));
        }
        std::discrete_distribution<int> dist(weights.begin(), weights.end());
        int pick = dist(rng);
        auto& [from, to, promo, visits] = results[pick];
        chosen_move = Move(from, to, promo);
    } else {
        // Temperature → 0: pick highest visit count
        int best_visits = -1;
        for (auto& [from, to, promo, visits] : results) {
            if (visits > best_visits) {
                best_visits = visits;
                chosen_move = Move(from, to, promo);
            }
        }
    }

    // We need to find the actual legal move that matches (for flags like castling/ep)
    auto legal = pg.game.legal_moves();
    Move actual_move = chosen_move;
    for (auto& lm : legal) {
        if (lm.from == chosen_move.from && lm.to == chosen_move.to &&
            lm.promo == chosen_move.promo) {
            actual_move = lm;
            break;
        }
    }

    // Make the move
    pg.game.make_move(actual_move);
    pg.move_count++;

    // Clean up MCTS tree
    delete pg.mcts;
    pg.mcts = nullptr;

    // Check if game is over or max moves reached
    if (pg.game.over || pg.move_count >= max_moves_) {
        finish_game(slot);
    } else {
        // Need root eval for next position
        pg.phase = GamePhase::NEED_ROOT_EVAL;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// collect_leaves: gather positions needing NN eval from all active games
// ═══════════════════════════════════════════════════════════════════════════

int SelfPlayManager::collect_leaves(float* out_boards, int max_batch) {
    leaf_map_.clear();
    int collected = 0;

    for (int gi = 0; gi < static_cast<int>(games_.size()); gi++) {
        if (collected >= max_batch) break;
        auto& pg = games_[gi];

        if (pg.phase == GamePhase::FINISHED) continue;

        if (pg.phase == GamePhase::NEED_ROOT_EVAL) {
            // Encode the root position for NN evaluation
            float* dst = out_boards + collected * 14 * 64;
            pg.game.encode_board(dst);

            LeafMapping lm;
            lm.game_idx = gi;
            lm.is_root_eval = true;
            leaf_map_.push_back(lm);
            collected++;
        }
        else if (pg.phase == GamePhase::SEARCHING) {
            // Collect leaves from this game's MCTS
            assert(pg.mcts != nullptr);

            if (pg.mcts->is_done()) {
                // MCTS finished, make a move
                pg.phase = GamePhase::MOVE_READY;
                make_move(gi);
                // After make_move, game might be FINISHED or NEED_ROOT_EVAL
                // If NEED_ROOT_EVAL, we can collect it on the next call
                continue;
            }

            // How many leaves can we collect from this game?
            int remaining = max_batch - collected;
            if (remaining < mcts_batch_size_) break;  // not enough room for a full MCTS batch

            // Collect leaves into the output buffer at the current offset
            float* dst = out_boards + collected * 14 * 64;
            int num_leaves = pg.mcts->select_leaves(dst);

            for (int li = 0; li < num_leaves; li++) {
                LeafMapping lm;
                lm.game_idx = gi;
                lm.is_root_eval = false;
                leaf_map_.push_back(lm);
            }
            collected += num_leaves;
        }
        else if (pg.phase == GamePhase::MOVE_READY) {
            // Shouldn't normally get here, but handle it
            make_move(gi);
        }
    }

    return collected;
}

// ═══════════════════════════════════════════════════════════════════════════
// process_results: distribute NN results back to the correct games
// ═══════════════════════════════════════════════════════════════════════════

void SelfPlayManager::process_results(const float* policies, const float* values, int n) {
    assert(n == static_cast<int>(leaf_map_.size()));

    // Group results by game for MCTS batch processing
    // We process in order, tracking how many leaves each game got

    int result_idx = 0;
    while (result_idx < n) {
        auto& lm = leaf_map_[result_idx];
        int gi = lm.game_idx;
        auto& pg = games_[gi];

        if (lm.is_root_eval) {
            // This was a root evaluation — create MCTS and expand root
            const float* policy = policies + result_idx * POLICY_SIZE;
            float value = values[result_idx];

            // Record position BEFORE creating MCTS (we want the board state)
            record_position(gi);

            // Create MCTS tree for this position
            pg.mcts = new MCTS(pg.game, num_simulations_, c_puct_,
                               dirichlet_alpha_, noise_weight_,
                               tactical_shortcuts_, mcts_batch_size_);
            pg.mcts->expand_root(policy);

            if (pg.mcts->is_done()) {
                // Tactical shortcut found an instant win — move immediately
                pg.phase = GamePhase::MOVE_READY;
                make_move(gi);
            } else {
                pg.phase = GamePhase::SEARCHING;
            }

            result_idx++;
        } else {
            // These are MCTS leaf results — collect consecutive leaves for same game
            int start = result_idx;
            int count = 0;
            while (result_idx < n &&
                   leaf_map_[result_idx].game_idx == gi &&
                   !leaf_map_[result_idx].is_root_eval) {
                count++;
                result_idx++;
            }

            // Feed results to this game's MCTS
            const float* p = policies + start * POLICY_SIZE;
            const float* v = values + start;
            pg.mcts->process_results(p, v, count);

            // Check if MCTS is now done
            if (pg.mcts->is_done()) {
                pg.phase = GamePhase::MOVE_READY;
                make_move(gi);
            }
        }
    }
}

} // namespace ext
