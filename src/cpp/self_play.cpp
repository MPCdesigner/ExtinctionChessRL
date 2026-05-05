#include "self_play.h"
#include <algorithm>
#include <cstring>
#include <random>
#include <cmath>
#include <cassert>
#include <cstdio>
#include <chrono>

namespace ext {

// ═══════════════════════════════════════════════════════════════════════════
// SelfPlayManager
// ═══════════════════════════════════════════════════════════════════════════

SelfPlayManager::SelfPlayManager(int num_parallel_games, int total_games,
                                 int num_simulations, float c_puct,
                                 float dirichlet_alpha, float noise_weight,
                                 bool tactical_shortcuts, int temp_threshold,
                                 int max_moves, int mcts_batch_size,
                                 int num_threads)
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
      num_threads_(std::max(1, num_threads)),
      games_started_(0),
      games_completed_(0)
{
    // Pre-allocate per-thread buffers
    // Each thread handles ceil(num_parallel/num_threads) games,
    // each producing up to mcts_batch_size leaves (or 1 root eval)
    int games_per_thread = (num_parallel_ + num_threads_ - 1) / num_threads_;
    int max_leaves_per_thread = games_per_thread * (mcts_batch_size_ + 1);
    thread_bufs_.resize(num_threads_);
    for (int t = 0; t < num_threads_; t++) {
        thread_bufs_[t].boards.resize(max_leaves_per_thread * Game::BOARD_ENCODING_SIZE);
        thread_bufs_[t].maps.reserve(max_leaves_per_thread);
    }

    printf("[SelfPlayManager] num_threads=%d, num_parallel=%d, mcts_batch=%d, "
           "buf_per_thread=%d leaves\n",
           num_threads_, num_parallel_, mcts_batch_size_, max_leaves_per_thread);
    fflush(stdout);

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
    std::vector<float> board_enc(Game::BOARD_ENCODING_SIZE);
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
    auto t_start = std::chrono::high_resolution_clock::now();
    leaf_map_.clear();
    collect_call_count_++;

    // ── Phase 1 (serial): handle finished MCTS and MOVE_READY games ──
    // These mutate shared state (completed_, games_started_, etc.)
    for (int gi = 0; gi < static_cast<int>(games_.size()); gi++) {
        auto& pg = games_[gi];
        if (pg.phase == GamePhase::FINISHED) continue;
        if (pg.phase == GamePhase::MOVE_READY) {
            make_move(gi);
        }
        if (pg.phase == GamePhase::SEARCHING && pg.mcts && pg.mcts->is_done()) {
            pg.phase = GamePhase::MOVE_READY;
            make_move(gi);
        }
    }

    // ── Phase 2: identify work items ──
    struct WorkItem {
        int game_idx;
        bool is_root;  // true = NEED_ROOT_EVAL, false = SEARCHING
    };
    std::vector<WorkItem> work;
    work.reserve(num_parallel_);
    for (int gi = 0; gi < static_cast<int>(games_.size()); gi++) {
        auto& pg = games_[gi];
        if (pg.phase == GamePhase::NEED_ROOT_EVAL) {
            work.push_back({gi, true});
        } else if (pg.phase == GamePhase::SEARCHING) {
            work.push_back({gi, false});
        }
    }

    if (work.empty()) return 0;

    // ── Phase 3 (parallel): run encode_board / select_leaves per thread ──
    int nt = std::min(num_threads_, static_cast<int>(work.size()));

    if (collect_call_count_ == 1) {
        printf("[collect_leaves] first call: work_size=%d, nt=%d\n",
               static_cast<int>(work.size()), nt);
        fflush(stdout);
    }

    // Distribute work items round-robin across threads
    std::vector<std::vector<int>> thread_work(nt);
    for (int i = 0; i < static_cast<int>(work.size()); i++) {
        thread_work[i % nt].push_back(i);
    }

    // Reset per-thread result counters
    for (int t = 0; t < nt; t++) {
        thread_bufs_[t].maps.clear();
        thread_bufs_[t].count = 0;
    }

    auto thread_fn = [&](int t) {
        auto& buf = thread_bufs_[t];
        float* boards = buf.boards.data();

        for (int wi : thread_work[t]) {
            auto& item = work[wi];
            auto& pg = games_[item.game_idx];

            if (item.is_root) {
                pg.game.encode_board(boards + buf.count * Game::BOARD_ENCODING_SIZE);
                buf.maps.push_back({item.game_idx, true});
                buf.count++;
            } else {
                float* dst = boards + buf.count * Game::BOARD_ENCODING_SIZE;
                int n = pg.mcts->select_leaves(dst);
                for (int j = 0; j < n; j++) {
                    buf.maps.push_back({item.game_idx, false});
                }
                buf.count += n;
            }
        }
    };

    if (nt <= 1) {
        // Single-threaded: run directly, no thread overhead
        thread_fn(0);
    } else {
        // Launch nt-1 worker threads, run one chunk on this thread
        std::vector<std::thread> threads;
        threads.reserve(nt - 1);
        for (int t = 1; t < nt; t++) {
            threads.emplace_back(thread_fn, t);
        }
        thread_fn(0);  // main thread does chunk 0
        for (auto& t : threads) t.join();
    }

    // ── Phase 4 (serial): combine thread results into output buffer ──
    int collected = 0;
    for (int t = 0; t < nt; t++) {
        auto& buf = thread_bufs_[t];
        if (buf.count == 0) continue;
        int to_copy = std::min(buf.count, max_batch - collected);
        std::memcpy(out_boards + collected * Game::BOARD_ENCODING_SIZE,
                     buf.boards.data(),
                     to_copy * Game::BOARD_ENCODING_SIZE * sizeof(float));
        for (int i = 0; i < to_copy; i++) {
            leaf_map_.push_back(buf.maps[i]);
        }
        collected += to_copy;
        if (collected >= max_batch) break;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    collect_time_us_ += std::chrono::duration<double, std::micro>(t_end - t_start).count();

    // Print timing summary every 5000 calls
    if (collect_call_count_ % 5000 == 0) {
        printf("[timing #%d] collect=%.1fs, process=%.1fs, avg_collect=%.0fus, avg_process=%.0fus\n",
               collect_call_count_,
               collect_time_us_ / 1e6, process_time_us_ / 1e6,
               collect_time_us_ / collect_call_count_,
               process_time_us_ / collect_call_count_);
        fflush(stdout);
    }

    return collected;
}

// ═══════════════════════════════════════════════════════════════════════════
// process_results: distribute NN results back to the correct games
// ═══════════════════════════════════════════════════════════════════════════

void SelfPlayManager::process_results(const float* policies, const float* values, int n) {
    auto t_start = std::chrono::high_resolution_clock::now();
    assert(n == static_cast<int>(leaf_map_.size()));

    // ── Phase 1: Group results by game ──
    struct GameWork {
        int game_idx;
        bool is_root;
        int start;   // index into policies/values
        int count;   // number of leaves
    };
    std::vector<GameWork> works;
    works.reserve(num_parallel_);

    int result_idx = 0;
    while (result_idx < n) {
        auto& lm = leaf_map_[result_idx];
        int gi = lm.game_idx;

        if (lm.is_root_eval) {
            works.push_back({gi, true, result_idx, 1});
            result_idx++;
        } else {
            int start = result_idx;
            int count = 0;
            while (result_idx < n &&
                   leaf_map_[result_idx].game_idx == gi &&
                   !leaf_map_[result_idx].is_root_eval) {
                count++;
                result_idx++;
            }
            works.push_back({gi, false, start, count});
        }
    }

    // ── Phase 2 (parallel): expand nodes / create MCTS trees ──
    // Each game's MCTS tree is independent — safe to parallelize.
    int nt = std::min(num_threads_, static_cast<int>(works.size()));
    std::vector<bool> became_done(works.size(), false);

    // Distribute work round-robin
    std::vector<std::vector<int>> thread_work(nt);
    for (int i = 0; i < static_cast<int>(works.size()); i++) {
        thread_work[i % nt].push_back(i);
    }

    auto work_fn = [&](int t) {
        for (int wi : thread_work[t]) {
            auto& gw = works[wi];
            auto& pg = games_[gw.game_idx];

            if (gw.is_root) {
                const float* policy = policies + gw.start * POLICY_SIZE;

                // Record position (per-game data, safe in parallel)
                record_position(gw.game_idx);

                // Create MCTS tree and expand root
                pg.mcts = new MCTS(pg.game, num_simulations_, c_puct_,
                                   dirichlet_alpha_, noise_weight_,
                                   tactical_shortcuts_, mcts_batch_size_);
                pg.mcts->expand_root(policy);

                if (pg.mcts->is_done()) {
                    became_done[wi] = true;
                } else {
                    pg.phase = GamePhase::SEARCHING;
                }
            } else {
                const float* p = policies + gw.start * POLICY_SIZE;
                const float* v = values + gw.start;
                pg.mcts->process_results(p, v, gw.count);

                if (pg.mcts->is_done()) {
                    became_done[wi] = true;
                }
            }
        }
    };

    if (nt <= 1) {
        work_fn(0);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(nt - 1);
        for (int t = 1; t < nt; t++) {
            threads.emplace_back(work_fn, t);
        }
        work_fn(0);
        for (auto& t : threads) t.join();
    }

    // ── Phase 3 (serial): handle games that finished MCTS ──
    // make_move modifies shared state (completed_, games_started_)
    for (int i = 0; i < static_cast<int>(works.size()); i++) {
        if (became_done[i]) {
            games_[works[i].game_idx].phase = GamePhase::MOVE_READY;
            make_move(works[i].game_idx);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    process_time_us_ += std::chrono::duration<double, std::micro>(t_end - t_start).count();
}

} // namespace ext
