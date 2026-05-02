#pragma once
#include "engine.h"
#include "mcts.h"
#include <vector>
#include <array>
#include <memory>

namespace ext {

// ── Training data from one completed game ─────────────────────────────────

struct GameRecord {
    std::vector<std::vector<float>> boards;    // each is BOARD_ENCODING_SIZE floats
    std::vector<std::vector<float>> policies;  // each is 4864 floats
    std::vector<int>                players;   // 0=WHITE, 1=BLACK
    float                           outcome;   // 1.0, -1.0, 0.0
};

// ── State of one parallel game ────────────────────────────────────────────

enum class GamePhase {
    NEED_ROOT_EVAL,  // waiting for NN eval of current position (root expansion)
    SEARCHING,       // MCTS in progress, collecting leaves
    MOVE_READY,      // MCTS finished, need to pick move and advance
    FINISHED         // game complete, training data recorded
};

struct ParallelGame {
    Game        game;
    GamePhase   phase;
    MCTS*       mcts;           // current MCTS tree (owned, nullable)
    int         move_count;

    // Training data accumulated during the game
    std::vector<std::vector<float>> boards;
    std::vector<std::vector<float>> policies;
    std::vector<int>                players;

    ParallelGame() : phase(GamePhase::NEED_ROOT_EVAL), mcts(nullptr), move_count(0) {}
    ~ParallelGame() { delete mcts; }

    // No copy, only move
    ParallelGame(const ParallelGame&) = delete;
    ParallelGame& operator=(const ParallelGame&) = delete;
    ParallelGame(ParallelGame&& o) noexcept
        : game(std::move(o.game)), phase(o.phase), mcts(o.mcts),
          move_count(o.move_count), boards(std::move(o.boards)),
          policies(std::move(o.policies)), players(std::move(o.players))
    { o.mcts = nullptr; }
    ParallelGame& operator=(ParallelGame&& o) noexcept {
        if (this != &o) {
            delete mcts;
            game = std::move(o.game);
            phase = o.phase;
            mcts = o.mcts; o.mcts = nullptr;
            move_count = o.move_count;
            boards = std::move(o.boards);
            policies = std::move(o.policies);
            players = std::move(o.players);
        }
        return *this;
    }
};

// ── Tracks which leaf in the batch belongs to which game ──────────────────

struct LeafMapping {
    int game_idx;
    bool is_root_eval;  // true = root expansion, false = MCTS leaf
};

// ── Batched self-play manager ─────────────────────────────────────────────

class SelfPlayManager {
public:
    SelfPlayManager(int num_parallel_games, int total_games,
                    int num_simulations, float c_puct,
                    float dirichlet_alpha, float noise_weight,
                    bool tactical_shortcuts, int temp_threshold,
                    int max_moves, int mcts_batch_size);

    // Collect positions needing NN evaluation from all active games.
    // Returns the number of positions. Boards are written to out_boards
    // (must have space for at least max_batch * BOARD_ENCODING_SIZE floats).
    int collect_leaves(float* out_boards, int max_batch);

    // Process NN results for the collected leaves.
    void process_results(const float* policies, const float* values, int n);

    // Is all self-play complete?
    bool is_done() const { return games_completed_ >= total_games_; }

    // Get all completed game records.
    const std::vector<GameRecord>& get_results() const { return completed_; }

    // Stats
    int games_completed() const { return games_completed_; }
    int games_active() const;

private:
    // Config
    int   num_parallel_;
    int   total_games_;
    int   num_simulations_;
    float c_puct_;
    float dirichlet_alpha_;
    float noise_weight_;
    bool  tactical_shortcuts_;
    int   temp_threshold_;
    int   max_moves_;
    int   mcts_batch_size_;

    // State
    std::vector<ParallelGame> games_;
    std::vector<LeafMapping>  leaf_map_;  // maps batch positions to games
    std::vector<GameRecord>   completed_;
    int games_started_;
    int games_completed_;

    // Helpers
    void start_new_game(int slot);
    void finish_game(int slot);
    void make_move(int slot);
    void record_position(int slot);
};

} // namespace ext
