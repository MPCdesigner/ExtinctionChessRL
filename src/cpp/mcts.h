#pragma once
#include "engine.h"
#include <vector>
#include <cmath>
#include <random>
#include <tuple>

namespace ext {

// ── Move encoding (Move ↔ policy index, matches Python alphazero.py) ──────
//
// 76 planes × 64 squares = 4864 total policy logits
//   Planes  0–55: Queen-type moves (8 directions × 7 distances)
//   Planes 56–63: Knight moves (8 offsets)
//   Planes 64–72: Underpromotions (knight/bishop/rook × 3 file deltas)
//   Planes 73–75: King promotions (3 file deltas)

static constexpr int POLICY_SIZE = 76 * 64;  // 4864

int move_to_index(const Move& m);

// ── MCTS Node (compact, stored in a pool) ─────────────────────────────────

struct MCTSNode {
    Game    game;            // game state at this node
    int     parent;          // index in pool (-1 = root)
    int     first_child;     // index of first child in pool (-1 = leaf)
    int     num_children;    // number of children
    Move    move;            // move that led here from parent
    float   prior;           // policy prior P(s,a)
    int     visit_count;
    float   value_sum;       // cumulative value from WHITE's perspective
    bool    is_expanded;
    int     virtual_loss;    // transient loss during batched selection

    MCTSNode()
        : parent(-1), first_child(-1), num_children(0),
          prior(1.0f), visit_count(0), value_sum(0.0f),
          is_expanded(false), virtual_loss(0) {}
};

// ── MCTS Search ───────────────────────────────────────────────────────────

class MCTS {
public:
    MCTS(const Game& root_game, int num_simulations, float c_puct,
         float dirichlet_alpha, float noise_weight,
         bool tactical_shortcuts, int batch_size);

    // Phase 1: Select up to batch_size leaves, encode their boards.
    // Returns the number of leaves needing NN evaluation.
    // out_boards must have space for batch_size * BOARD_ENCODING_SIZE floats.
    int select_leaves(float* out_boards);

    // Phase 2: Receive NN results and expand + backprop.
    // policies: (num_leaves, 4864), values: (num_leaves,)
    void process_results(const float* policies, const float* values, int num_leaves);

    // Is the search complete?
    bool is_done() const { return sims_done_ >= num_simulations_; }

    // Get results: (move_index, visit_count) for each child of root.
    std::vector<std::pair<int, int>> get_results() const;

    // Get results as (from, to, promo, visit_count) for mapping back to Move objects.
    std::vector<std::tuple<int, int, int, int>> get_move_results() const;

    // Expand root node with policy logits (called from Python after first NN eval)
    void expand_root(const float* policy_logits);

    // Promote the child matching `m` to become the new root, keeping its
    // subtree and discarding the rest of the pool. Called by SelfPlayManager
    // after a move is played, to reuse the search work already done for that
    // subtree.
    //
    // Returns true on success. On success:
    //   - nodes_ is rebuilt (compacted) to only contain the promoted subtree
    //   - root_ points at the new root (formerly the matching child)
    //   - sims_done_ is set to the new root's visit_count (top-up semantics)
    //   - virtual_loss is zeroed on every carried-over node (defensive; should
    //     already be 0 since promote() is only called after MCTS::is_done())
    //   - Fresh Dirichlet noise is applied to the new root's children priors
    //     (if dirichlet_alpha_ > 0 and noise_weight_ > 0)
    //
    // Returns false (and leaves the MCTS unchanged) if:
    //   - no child of root matches m (by from/to/promo), or
    //   - the matching child is not expanded, or
    //   - the matching child has visit_count == 0
    //
    // Caller should treat false as a signal to destroy this MCTS and start
    // a fresh one.
    bool promote(const Move& m);

private:
    // Node pool
    std::vector<MCTSNode> nodes_;
    int root_;

    // Config
    int   num_simulations_;
    float c_puct_;
    float dirichlet_alpha_;
    float noise_weight_;
    bool  tactical_shortcuts_;
    int   batch_size_;
    int   sims_done_;

    // Pending leaves from select_leaves (indices into nodes_)
    std::vector<int> pending_leaves_;
    // Pending terminals from select_leaves
    std::vector<std::pair<int, float>> pending_terminals_;

    // Helpers
    int  allocate_node();
    void expand_node(int node_idx, const float* policy_logits);
    int  select_child(int node_idx) const;
    void backpropagate(int node_idx, float white_value);
    float terminal_value(const Game& g) const;
    void add_virtual_loss(int node_idx);
    void remove_virtual_loss(int node_idx);
    void apply_dirichlet_noise_at_root();

    // Tactical shortcuts
    struct TacticalResult {
        bool instant_win;
        std::vector<Move> winning_moves;  // all moves that win immediately
        std::vector<Move> safe_moves;     // moves that don't lose in 1-2 ply
    };
    TacticalResult check_tactics(const Game& g) const;
};

} // namespace ext
