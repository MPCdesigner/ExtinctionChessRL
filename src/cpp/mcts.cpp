#include "mcts.h"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cassert>

namespace ext {

// ═══════════════════════════════════════════════════════════════════════════
// Move encoding — must match Python alphazero.py exactly
// ═══════════════════════════════════════════════════════════════════════════

// Queen-type directions: N, NE, E, SE, S, SW, W, NW
static const int QUEEN_DR[] = { 1, 1, 0,-1,-1,-1, 0, 1};
static const int QUEEN_DF[] = { 0, 1, 1, 1, 0,-1,-1,-1};

// Knight offsets (same order as Python)
static const int KNIGHT_DR[] = { 2, 2,-2,-2, 1, 1,-1,-1};
static const int KNIGHT_DF[] = { 1,-1, 1,-1, 2,-2, 2,-2};

int move_to_index(const Move& m) {
    int from_r = rank_of(m.from), from_f = file_of(m.from);
    int to_r   = rank_of(m.to),   to_f   = file_of(m.to);
    int from_sq = from_r * 8 + from_f;
    int dr = to_r - from_r;
    int df = to_f - from_f;

    // Underpromotions / king promotions
    if (m.promo != NONE_PT && m.promo != QUEEN) {
        int df_idx = df + 1;  // -1→0, 0→1, +1→2
        int plane;
        switch (m.promo) {
            case KNIGHT: plane = 64 + df_idx; break;
            case BISHOP: plane = 64 + 3 + df_idx; break;
            case ROOK:   plane = 64 + 6 + df_idx; break;
            case KING:   plane = 73 + df_idx; break;
            default:     plane = 0; break;
        }
        return plane * 64 + from_sq;
    }

    // Knight moves
    for (int i = 0; i < 8; i++) {
        if (dr == KNIGHT_DR[i] && df == KNIGHT_DF[i]) {
            int plane = 56 + i;
            return plane * 64 + from_sq;
        }
    }

    // Queen-type moves (includes queen promotions and castling king moves)
    if (dr != 0 || df != 0) {
        // Normalize to direction
        int abs_dr = std::abs(dr), abs_df = std::abs(df);
        int dist = std::max(abs_dr, abs_df);
        int norm_dr = (dr != 0) ? dr / abs_dr : 0;
        int norm_df = (df != 0) ? df / abs_df : 0;

        for (int d = 0; d < 8; d++) {
            if (norm_dr == QUEEN_DR[d] && norm_df == QUEEN_DF[d]) {
                int plane = d * 7 + (dist - 1);
                return plane * 64 + from_sq;
            }
        }
    }

    return 0;  // fallback
}


// ═══════════════════════════════════════════════════════════════════════════
// MCTS Implementation
// ═══════════════════════════════════════════════════════════════════════════

MCTS::MCTS(const Game& root_game, int num_simulations, float c_puct,
           float dirichlet_alpha, float noise_weight,
           bool tactical_shortcuts, int batch_size)
    : num_simulations_(num_simulations), c_puct_(c_puct),
      dirichlet_alpha_(dirichlet_alpha), noise_weight_(noise_weight),
      tactical_shortcuts_(tactical_shortcuts), batch_size_(batch_size),
      sims_done_(0)
{
    // Reserve space for nodes (rough estimate)
    nodes_.reserve(num_simulations * 4);

    // Create root node
    root_ = allocate_node();
    nodes_[root_].game = root_game;
}

int MCTS::allocate_node() {
    int idx = static_cast<int>(nodes_.size());
    nodes_.emplace_back();
    return idx;
}

float MCTS::terminal_value(const Game& g) const {
    if (g.over) {
        if (g.winner == WHITE) return 1.0f;
        if (g.winner == BLACK) return -1.0f;
        return 0.0f;
    }
    return 0.0f;
}

int MCTS::select_child(int node_idx) const {
    const MCTSNode& node = nodes_[node_idx];
    int best = -1;
    float best_score = -1e9f;

    float parent_vc = static_cast<float>(node.visit_count);
    float sqrt_parent = std::sqrt(parent_vc);

    for (int i = 0; i < node.num_children; i++) {
        int ci = node.first_child + i;
        const MCTSNode& child = nodes_[ci];

        float q;
        if (child.visit_count == 0) {
            q = 0.0f;
        } else {
            float wq = child.value_sum / static_cast<float>(child.visit_count);
            // q is from parent's perspective
            q = (node.game.side == WHITE) ? wq : -wq;
        }

        float u = c_puct_ * child.prior * sqrt_parent / (1.0f + child.visit_count);
        float score = q + u;

        if (score > best_score) {
            best_score = score;
            best = ci;
        }
    }
    return best;
}

void MCTS::backpropagate(int node_idx, float white_value) {
    int idx = node_idx;
    while (idx >= 0) {
        nodes_[idx].visit_count++;
        nodes_[idx].value_sum += white_value;
        idx = nodes_[idx].parent;
    }
}

void MCTS::expand_node(int node_idx, const float* policy_logits) {
    // Get legal moves and game state BEFORE any reallocation
    auto legal = nodes_[node_idx].game.legal_moves();

    if (legal.empty()) {
        nodes_[node_idx].is_expanded = true;
        return;
    }

    // Compute softmax priors for legal moves
    std::vector<float> logits(legal.size());
    float max_logit = -1e9f;

    for (size_t i = 0; i < legal.size(); i++) {
        int idx = move_to_index(legal[i]);
        logits[i] = policy_logits[idx];
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        logits[i] = std::exp(logits[i] - max_logit);
        sum_exp += logits[i];
    }
    for (size_t i = 0; i < logits.size(); i++) {
        logits[i] /= (sum_exp + 1e-8f);
    }

    // Copy parent game state before reallocation
    Game parent_game = nodes_[node_idx].game;

    // Pre-allocate to avoid reallocation during child creation
    nodes_.reserve(nodes_.size() + legal.size());

    // Set child info on parent (safe now since we reserved)
    nodes_[node_idx].first_child = static_cast<int>(nodes_.size());
    nodes_[node_idx].num_children = static_cast<int>(legal.size());

    for (size_t i = 0; i < legal.size(); i++) {
        int ci = allocate_node();
        nodes_[ci].parent = node_idx;
        nodes_[ci].move = legal[i];
        nodes_[ci].prior = logits[i];

        // Copy game and make move
        nodes_[ci].game = parent_game;
        nodes_[ci].game.make_move(legal[i]);
    }

    nodes_[node_idx].is_expanded = true;
}

void MCTS::expand_root(const float* policy_logits) {
    auto legal = nodes_[root_].game.legal_moves();

    if (legal.empty()) {
        nodes_[root_].is_expanded = true;
        return;
    }

    // Tactical shortcut: force instant wins (but allow blunders)
    std::vector<Move> filtered_legal = legal;
    if (tactical_shortcuts_) {
        auto tactics = check_tactics(nodes_[root_].game);
        if (tactics.instant_win) {
            // Copy root game before reallocation
            Game root_game = nodes_[root_].game;
            nodes_.reserve(nodes_.size() + legal.size());

            nodes_[root_].first_child = static_cast<int>(nodes_.size());
            nodes_[root_].num_children = static_cast<int>(legal.size());
            for (size_t i = 0; i < legal.size(); i++) {
                int ci = allocate_node();
                nodes_[ci].parent = root_;
                nodes_[ci].move = legal[i];
                nodes_[ci].prior = 0.0f;
                nodes_[ci].game = root_game;
                nodes_[ci].game.make_move(legal[i]);
                if (legal[i].from == tactics.winning_move.from &&
                    legal[i].to == tactics.winning_move.to &&
                    legal[i].promo == tactics.winning_move.promo) {
                    nodes_[ci].visit_count = num_simulations_;
                }
            }
            nodes_[root_].is_expanded = true;
            sims_done_ = num_simulations_;
            return;
        }
        // No loss avoidance filtering — let MCTS search all legal moves
    }

    // Compute softmax priors for legal moves
    std::vector<float> probs(filtered_legal.size());
    float max_logit = -1e9f;

    for (size_t i = 0; i < filtered_legal.size(); i++) {
        int idx = move_to_index(filtered_legal[i]);
        probs[i] = policy_logits[idx];
        if (probs[i] > max_logit) max_logit = probs[i];
    }

    float sum_exp = 0.0f;
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] = std::exp(probs[i] - max_logit);
        sum_exp += probs[i];
    }
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= (sum_exp + 1e-8f);
    }

    // Add Dirichlet noise at root
    if (dirichlet_alpha_ > 0 && noise_weight_ > 0) {
        std::mt19937 rng(std::random_device{}());
        std::gamma_distribution<float> gamma(dirichlet_alpha_, 1.0f);

        std::vector<float> noise(probs.size());
        float noise_sum = 0.0f;
        for (size_t i = 0; i < noise.size(); i++) {
            noise[i] = gamma(rng);
            noise_sum += noise[i];
        }
        for (size_t i = 0; i < noise.size(); i++) {
            noise[i] /= (noise_sum + 1e-8f);
            probs[i] = (1.0f - noise_weight_) * probs[i] + noise_weight_ * noise[i];
        }
    }

    // Copy root game before reallocation
    Game root_game = nodes_[root_].game;
    nodes_.reserve(nodes_.size() + filtered_legal.size());

    nodes_[root_].first_child = static_cast<int>(nodes_.size());
    nodes_[root_].num_children = static_cast<int>(filtered_legal.size());

    for (size_t i = 0; i < filtered_legal.size(); i++) {
        int ci = allocate_node();
        nodes_[ci].parent = root_;
        nodes_[ci].move = filtered_legal[i];
        nodes_[ci].prior = probs[i];

        nodes_[ci].game = root_game;
        nodes_[ci].game.make_move(filtered_legal[i]);
    }

    nodes_[root_].is_expanded = true;
}

int MCTS::select_leaves(float* out_boards) {
    pending_leaves_.clear();
    pending_terminals_.clear();

    int leaves_collected = 0;
    int batch = std::min(batch_size_, num_simulations_ - sims_done_);

    for (int b = 0; b < batch; b++) {
        // Selection: walk tree to a leaf
        int node_idx = root_;
        while (nodes_[node_idx].is_expanded && nodes_[node_idx].num_children > 0) {
            node_idx = select_child(node_idx);
        }

        const MCTSNode& node = nodes_[node_idx];

        if (node.game.over) {
            // Terminal node — handle immediately
            float tv = terminal_value(node.game);
            pending_terminals_.emplace_back(node_idx, tv);
        } else if (node.is_expanded) {
            // Expanded but no children (shouldn't happen with legal moves)
            float tv = terminal_value(node.game);
            pending_terminals_.emplace_back(node_idx, tv);
        } else {
            // Leaf needing NN evaluation
            pending_leaves_.push_back(node_idx);
            // Encode board into output buffer
            float* dst = out_boards + leaves_collected * Game::BOARD_ENCODING_SIZE;
            node.game.encode_board(dst);
            leaves_collected++;
        }
    }

    // Handle terminals immediately
    for (auto& [idx, val] : pending_terminals_) {
        backpropagate(idx, val);
        sims_done_++;
    }

    return leaves_collected;
}

void MCTS::process_results(const float* policies, const float* values, int num_leaves) {
    for (int i = 0; i < num_leaves; i++) {
        int node_idx = pending_leaves_[i];
        const float* policy = policies + i * POLICY_SIZE;
        float value = values[i];

        // Expand node with policy priors
        // Note: node_idx might have shifted if expand_node caused reallocation
        // We need to be careful here — expand_node appends to nodes_
        expand_node(node_idx, policy);

        // Value is from current player's perspective, convert to white's
        float white_value;
        if (nodes_[node_idx].game.side == WHITE) {
            white_value = value;
        } else {
            white_value = -value;
        }

        backpropagate(node_idx, white_value);
        sims_done_++;
    }
    pending_leaves_.clear();
}

std::vector<std::pair<int, int>> MCTS::get_results() const {
    const MCTSNode& root = nodes_[root_];
    std::vector<std::pair<int, int>> results;
    results.reserve(root.num_children);

    for (int i = 0; i < root.num_children; i++) {
        int ci = root.first_child + i;
        const MCTSNode& child = nodes_[ci];
        int idx = move_to_index(child.move);
        results.emplace_back(idx, child.visit_count);
    }
    return results;
}

std::vector<std::tuple<int, int, int, int>> MCTS::get_move_results() const {
    const MCTSNode& root = nodes_[root_];
    std::vector<std::tuple<int, int, int, int>> results;
    results.reserve(root.num_children);

    for (int i = 0; i < root.num_children; i++) {
        int ci = root.first_child + i;
        const MCTSNode& child = nodes_[ci];
        results.emplace_back(
            static_cast<int>(child.move.from),
            static_cast<int>(child.move.to),
            static_cast<int>(child.move.promo),
            child.visit_count
        );
    }
    return results;
}

// ── Tactical shortcuts ────────────────────────────────────────────────────

MCTS::TacticalResult MCTS::check_tactics(const Game& g) const {
    TacticalResult result;
    result.instant_win = false;

    Color current = g.side;
    auto legal = g.legal_moves();

    for (size_t i = 0; i < legal.size(); i++) {
        Game gc = g;  // copy
        if (!gc.make_move(legal[i])) continue;

        // 1-ply: this move immediately wins
        if (gc.over && gc.winner == current) {
            result.instant_win = true;
            result.winning_move = legal[i];
            return result;
        }
    }

    return result;
}

} // namespace ext
