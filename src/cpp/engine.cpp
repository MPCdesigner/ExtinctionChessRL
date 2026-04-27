#include "engine.h"
#include <algorithm>
#include <unordered_map>

namespace ext {

// ═══════════════════════════════════════════════════════════════════════════
// Pre-computed tables
// ═══════════════════════════════════════════════════════════════════════════

u64 knight_atk[64];
u64 king_atk[64];
u64 zobrist_pc[2][6][64];
u64 zobrist_side;
u64 zobrist_ep[64];
u64 zobrist_castle[16];

static bool s_initialized = false;

static u64 xorshift_state = 0x12345678ABCDEF01ULL;
static u64 xorshift_next() {
    xorshift_state ^= xorshift_state >> 12;
    xorshift_state ^= xorshift_state << 25;
    xorshift_state ^= xorshift_state >> 27;
    return xorshift_state * 0x2545F4914F6CDD1DULL;
}

void init() {
    if (s_initialized) return;
    s_initialized = true;

    // Knight attacks
    static const int kn_dr[] = {2, 2, -2, -2, 1, 1, -1, -1};
    static const int kn_df[] = {1, -1, 1, -1, 2, -2, 2, -2};
    for (int s = 0; s < 64; s++) {
        knight_atk[s] = 0;
        int r = rank_of(s), f = file_of(s);
        for (int i = 0; i < 8; i++) {
            int nr = r + kn_dr[i], nf = f + kn_df[i];
            if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
                knight_atk[s] |= bit(sq(nr, nf));
        }
    }

    // King attacks
    for (int s = 0; s < 64; s++) {
        king_atk[s] = 0;
        int r = rank_of(s), f = file_of(s);
        for (int dr = -1; dr <= 1; dr++)
            for (int df = -1; df <= 1; df++) {
                if (dr == 0 && df == 0) continue;
                int nr = r + dr, nf = f + df;
                if (nr >= 0 && nr < 8 && nf >= 0 && nf < 8)
                    king_atk[s] |= bit(sq(nr, nf));
            }
    }

    // Zobrist keys
    xorshift_state = 0x12345678ABCDEF01ULL;
    for (int c = 0; c < 2; c++)
        for (int pt = 0; pt < 6; pt++)
            for (int s = 0; s < 64; s++)
                zobrist_pc[c][pt][s] = xorshift_next();
    zobrist_side = xorshift_next();
    for (int s = 0; s < 64; s++)
        zobrist_ep[s] = xorshift_next();
    for (int i = 0; i < 16; i++)
        zobrist_castle[i] = xorshift_next();
}

// ── Sliding attacks (loop-based) ───────────────────────────────────────────

u64 bishop_attacks(int s, u64 occ) {
    u64 atk = 0;
    static const int dr[] = {1, 1, -1, -1};
    static const int df[] = {1, -1, 1, -1};
    for (int d = 0; d < 4; d++) {
        int r = rank_of(s), f = file_of(s);
        for (;;) {
            r += dr[d]; f += df[d];
            if (r < 0 || r > 7 || f < 0 || f > 7) break;
            int t = sq(r, f);
            atk |= bit(t);
            if (occ & bit(t)) break;
        }
    }
    return atk;
}

u64 rook_attacks(int s, u64 occ) {
    u64 atk = 0;
    static const int dr[] = {1, -1, 0, 0};
    static const int df[] = {0, 0, 1, -1};
    for (int d = 0; d < 4; d++) {
        int r = rank_of(s), f = file_of(s);
        for (;;) {
            r += dr[d]; f += df[d];
            if (r < 0 || r > 7 || f < 0 || f > 7) break;
            int t = sq(r, f);
            atk |= bit(t);
            if (occ & bit(t)) break;
        }
    }
    return atk;
}

// ═══════════════════════════════════════════════════════════════════════════
// Board
// ═══════════════════════════════════════════════════════════════════════════

Board::Board() { clear(); }

void Board::clear() {
    std::memset(bb, 0, sizeof(bb));
    std::memset(occ, 0, sizeof(occ));
    std::memset(pc, NONE_PT, sizeof(pc));
    std::memset(cl, -1, sizeof(cl));
    ep = -1;
    castling = 0;
    hmclock = 0;
    fmnum = 1;
    hash = 0;
    history.clear();
}

void Board::setup() {
    clear();
    static const PieceType back_rank[] = {
        ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK
    };
    for (int f = 0; f < 8; f++) {
        put(WHITE, back_rank[f], sq(0, f));
        put(WHITE, PAWN,         sq(1, f));
        put(BLACK, PAWN,         sq(6, f));
        put(BLACK, back_rank[f], sq(7, f));
    }
    castling = WK | WQ | BK | BQ;
    hash = compute_hash(WHITE);
}

void Board::put(Color c, PieceType pt, int s) {
    bb[c][pt] |= bit(s);
    occ[c]    |= bit(s);
    occ[2]    |= bit(s);
    pc[s] = pt;
    cl[s] = c;
}

void Board::del(int s) {
    if (pc[s] == NONE_PT) return;
    Color c = Color(cl[s]);
    PieceType pt = PieceType(pc[s]);
    bb[c][pt] &= ~bit(s);
    occ[c]    &= ~bit(s);
    occ[2]    &= ~bit(s);
    pc[s] = NONE_PT;
    cl[s] = -1;
}

std::array<int,6> Board::counts(Color c) const {
    std::array<int,6> r{};
    for (int pt = 0; pt < 6; pt++)
        r[pt] = popcnt(bb[c][pt]);
    return r;
}

u64 Board::compute_hash(Color side) const {
    u64 h = 0;
    for (int c = 0; c < 2; c++)
        for (int pt = 0; pt < 6; pt++) {
            u64 b = bb[c][pt];
            while (b) {
                int s = pop_lsb(b);
                h ^= zobrist_pc[c][pt][s];
            }
        }
    if (side == BLACK) h ^= zobrist_side;
    if (ep >= 0) h ^= zobrist_ep[ep];
    h ^= zobrist_castle[castling];
    return h;
}

// ═══════════════════════════════════════════════════════════════════════════
// Move string
// ═══════════════════════════════════════════════════════════════════════════

std::string Move::to_string() const {
    if (from < 0) return "(null)";
    char buf[16];
    int n = 0;
    buf[n++] = 'a' + file_of(from);
    buf[n++] = '1' + rank_of(from);
    buf[n++] = '-';
    buf[n++] = 'a' + file_of(to);
    buf[n++] = '1' + rank_of(to);
    if (promo != NONE_PT) {
        buf[n++] = '=';
        static const char pc[] = "PNBRQK";
        buf[n++] = pc[promo];
    }
    if (is_castle()) { buf[n++]=' '; buf[n++]='O'; }
    buf[n] = '\0';
    return std::string(buf);
}

// ═══════════════════════════════════════════════════════════════════════════
// Game
// ═══════════════════════════════════════════════════════════════════════════

Game::Game() : side(WHITE), over(false), winner(-1) {
    init();
    board.setup();
}

// ── Move generation ────────────────────────────────────────────────────────

void Game::gen_pawn(std::vector<Move>& mv) const {
    Color c = side;
    int dir = (c == WHITE) ? 1 : -1;
    int start_rank = (c == WHITE) ? 1 : 6;
    int promo_rank = (c == WHITE) ? 7 : 0;

    u64 pawns = board.bb[c][PAWN];
    while (pawns) {
        int s = pop_lsb(pawns);
        int r = rank_of(s), f = file_of(s);

        // Single push
        int tr = r + dir;
        if (tr < 0 || tr > 7) continue;
        int t = sq(tr, f);
        if (!board.has(t)) {
            if (tr == promo_rank) {
                for (int p : {QUEEN, ROOK, BISHOP, KNIGHT, KING})
                    mv.emplace_back(s, t, p);
            } else {
                mv.emplace_back(s, t);
                // Double push
                if (r == start_rank) {
                    int t2 = sq(r + 2 * dir, f);
                    if (!board.has(t2))
                        mv.emplace_back(s, t2);
                }
            }
        }

        // Captures
        for (int df : {-1, 1}) {
            int cf = f + df;
            if (cf < 0 || cf > 7) continue;
            int cs = sq(tr, cf);
            if (board.has(cs) && board.color_at(cs) != c) {
                if (tr == promo_rank) {
                    for (int p : {QUEEN, ROOK, BISHOP, KNIGHT, KING})
                        mv.emplace_back(s, cs, p);
                } else {
                    mv.emplace_back(s, cs);
                }
            }
            // En passant
            if (board.ep == cs) {
                mv.emplace_back(s, cs, NONE_PT, Move::F_EP);
            }
        }
    }
}

void Game::gen_knight(std::vector<Move>& mv) const {
    Color c = side;
    u64 pieces = board.bb[c][KNIGHT];
    while (pieces) {
        int s = pop_lsb(pieces);
        u64 targets = knight_atk[s] & ~board.occ[c];
        while (targets) {
            int t = pop_lsb(targets);
            mv.emplace_back(s, t);
        }
    }
}

void Game::gen_bishop(std::vector<Move>& mv) const {
    Color c = side;
    u64 pieces = board.bb[c][BISHOP];
    while (pieces) {
        int s = pop_lsb(pieces);
        u64 targets = bishop_attacks(s, board.occ[2]) & ~board.occ[c];
        while (targets) {
            int t = pop_lsb(targets);
            mv.emplace_back(s, t);
        }
    }
}

void Game::gen_rook(std::vector<Move>& mv) const {
    Color c = side;
    u64 pieces = board.bb[c][ROOK];
    while (pieces) {
        int s = pop_lsb(pieces);
        u64 targets = rook_attacks(s, board.occ[2]) & ~board.occ[c];
        while (targets) {
            int t = pop_lsb(targets);
            mv.emplace_back(s, t);
        }
    }
}

void Game::gen_queen(std::vector<Move>& mv) const {
    Color c = side;
    u64 pieces = board.bb[c][QUEEN];
    while (pieces) {
        int s = pop_lsb(pieces);
        u64 targets = queen_attacks(s, board.occ[2]) & ~board.occ[c];
        while (targets) {
            int t = pop_lsb(targets);
            mv.emplace_back(s, t);
        }
    }
}

void Game::gen_king(std::vector<Move>& mv) const {
    Color c = side;
    u64 pieces = board.bb[c][KING];
    while (pieces) {
        int s = pop_lsb(pieces);
        u64 targets = king_atk[s] & ~board.occ[c];
        while (targets) {
            int t = pop_lsb(targets);
            mv.emplace_back(s, t);
        }
    }

    // Castling — extinction chess allows castling through/into/out of check
    int rank = (c == WHITE) ? 0 : 7;
    int king_sq = sq(rank, 4);

    // Only castle if a king is actually on e1/e8
    if (!(board.bb[c][KING] & bit(king_sq))) return;

    // Kingside
    uint8_t ks_flag = (c == WHITE) ? Board::WK : Board::BK;
    if (board.castling & ks_flag) {
        int rook_sq = sq(rank, 7);
        if (board.has(rook_sq) && board.type_at(rook_sq) == ROOK &&
            board.color_at(rook_sq) == c) {
            if (!board.has(sq(rank, 5)) && !board.has(sq(rank, 6))) {
                mv.emplace_back(king_sq, sq(rank, 6), NONE_PT, Move::F_CASTLE,
                                rook_sq, sq(rank, 5));
            }
        }
    }

    // Queenside
    uint8_t qs_flag = (c == WHITE) ? Board::WQ : Board::BQ;
    if (board.castling & qs_flag) {
        int rook_sq = sq(rank, 0);
        if (board.has(rook_sq) && board.type_at(rook_sq) == ROOK &&
            board.color_at(rook_sq) == c) {
            if (!board.has(sq(rank, 1)) && !board.has(sq(rank, 2)) &&
                !board.has(sq(rank, 3))) {
                mv.emplace_back(king_sq, sq(rank, 2), NONE_PT, Move::F_CASTLE,
                                rook_sq, sq(rank, 3));
            }
        }
    }
}

std::vector<Move> Game::legal_moves() const {
    std::vector<Move> mv;
    mv.reserve(80);
    gen_pawn(mv);
    gen_knight(mv);
    gen_bishop(mv);
    gen_rook(mv);
    gen_queen(mv);
    gen_king(mv);
    return mv;
}

std::vector<Move> Game::legal_moves_from(int target_sq) const {
    auto all = legal_moves();
    std::vector<Move> result;
    for (auto& m : all)
        if (m.from == target_sq)
            result.push_back(m);
    return result;
}

// ── Make move ──────────────────────────────────────────────────────────────

bool Game::make_move(const Move& m) {
    if (over) return false;
    if (!board.has(m.from) || board.color_at(m.from) != side) return false;

    PieceType moving = board.type_at(m.from);

    // Record position hash for threefold repetition (before move, matching Python)
    board.history.push_back(board.compute_hash(side));

    // Halfmove clock
    bool is_capture = board.has(m.to);
    if (is_capture || moving == PAWN)
        board.hmclock = 0;
    else
        board.hmclock++;

    // Execute move
    if (m.is_castle()) {
        board.del(m.from);
        board.put(side, KING, m.to);
        board.del(m.castle_rf);
        board.put(side, ROOK, m.castle_rt);
    } else if (m.is_ep()) {
        int cap_sq = sq(rank_of(m.from), file_of(m.to));
        board.del(cap_sq);
        board.del(m.from);
        board.put(side, PAWN, m.to);
        board.hmclock = 0;
    } else {
        if (board.has(m.to)) board.del(m.to);
        board.del(m.from);
        if (m.promo != NONE_PT) {
            board.put(side, PieceType(m.promo), m.to);
            board.hmclock = 0;
        } else {
            board.put(side, moving, m.to);
        }
    }

    // Update en passant square
    board.ep = -1;
    if (moving == PAWN && std::abs(rank_of(m.to) - rank_of(m.from)) == 2) {
        board.ep = sq((rank_of(m.from) + rank_of(m.to)) / 2, file_of(m.from));
    }

    // Update castling rights
    if (moving == KING) {
        if (side == WHITE) board.castling &= ~(Board::WK | Board::WQ);
        else               board.castling &= ~(Board::BK | Board::BQ);
    }
    // Rook moved or captured from original square
    if (m.from == sq(0, 0) || m.to == sq(0, 0)) board.castling &= ~Board::WQ;
    if (m.from == sq(0, 7) || m.to == sq(0, 7)) board.castling &= ~Board::WK;
    if (m.from == sq(7, 0) || m.to == sq(7, 0)) board.castling &= ~Board::BQ;
    if (m.from == sq(7, 7) || m.to == sq(7, 7)) board.castling &= ~Board::BK;

    // Check extinction
    if (check_extinction(m)) return true;

    // Check draws
    if (check_draws()) return true;

    // Switch player
    side = flip(side);
    if (side == WHITE) board.fmnum++;

    return true;
}

// ── Extinction check ───────────────────────────────────────────────────────

bool Game::check_extinction(const Move& m) {
    auto wc = board.counts(WHITE);
    auto bc = board.counts(BLACK);

    bool white_extinct = false, black_extinct = false;
    for (int pt = 0; pt < 6; pt++) {
        if (wc[pt] == 0) white_extinct = true;
        if (bc[pt] == 0) black_extinct = true;
    }

    // Promotion can cause pawn extinction for the moving side
    if (m.promo != NONE_PT) {
        if (side == WHITE && wc[PAWN] == 0) white_extinct = true;
        if (side == BLACK && bc[PAWN] == 0) black_extinct = true;
    }

    // Both extinct: moving player wins
    if (white_extinct && black_extinct) {
        over = true;
        winner = side;
        return true;
    }
    if (white_extinct) {
        over = true;
        winner = BLACK;
        return true;
    }
    if (black_extinct) {
        over = true;
        winner = WHITE;
        return true;
    }
    return false;
}

// ── Draw check ─────────────────────────────────────────────────────────────

bool Game::check_draws() {
    // 50-move rule
    if (board.hmclock >= 100) {
        over = true;
        winner = -1;
        draw_reason = "50-move rule";
        return true;
    }

    // Threefold repetition (count in history)
    u64 current_hash = board.compute_hash(side);
    // Note: matching Python behavior — check ALL history entries, not just current
    std::unordered_map<u64, int> counts;
    for (u64 h : board.history)
        counts[h]++;
    for (auto& [h, cnt] : counts) {
        if (cnt >= 3) {
            over = true;
            winner = -1;
            draw_reason = "Threefold repetition";
            return true;
        }
    }

    // Stalemate — matching Python: checks current player (who just moved)
    if (legal_moves().empty()) {
        over = true;
        winner = -1;
        draw_reason = "Stalemate";
        return true;
    }

    return false;
}

// ── Piece analysis ─────────────────────────────────────────────────────────

std::vector<PieceType> Game::endangered(Color c) const {
    std::vector<PieceType> result;
    for (int pt = 0; pt < 6; pt++)
        if (board.count(c, PieceType(pt)) == 1)
            result.push_back(PieceType(pt));
    return result;
}

std::vector<PieceType> Game::extinct(Color c) const {
    std::vector<PieceType> result;
    for (int pt = 0; pt < 6; pt++)
        if (board.count(c, PieceType(pt)) == 0)
            result.push_back(PieceType(pt));
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Neural network encoding
// ═══════════════════════════════════════════════════════════════════════════

void Game::encode_board(float* out) const {
    std::memset(out, 0, 14 * 64 * sizeof(float));

    // Channels 0–11: piece planes  (channel = pt + color*6)
    for (int c = 0; c < 2; c++)
        for (int pt = 0; pt < 6; pt++) {
            int ch = pt + c * 6;
            u64 b = board.bb[c][pt];
            while (b) {
                int s = pop_lsb(b);
                out[ch * 64 + s] = 1.0f;
            }
        }

    // Channel 12: current player (all 1s if white's turn)
    if (side == WHITE)
        for (int i = 0; i < 64; i++)
            out[12 * 64 + i] = 1.0f;

    // Channel 13: endangered pieces (count == 1)
    u64 endangered_mask = 0;
    for (int c = 0; c < 2; c++)
        for (int pt = 0; pt < 6; pt++)
            if (popcnt(board.bb[c][pt]) == 1)
                endangered_mask |= board.bb[c][pt];
    {
        u64 e = endangered_mask;
        while (e) {
            int s = pop_lsb(e);
            out[13 * 64 + s] = 1.0f;
        }
    }
}

void Game::get_simple_features(float* out) const {
    int idx = 0;

    // 12 features: piece counts (white then black, 6 types each) / 8.0
    for (int c = 0; c < 2; c++)
        for (int pt = 0; pt < 6; pt++)
            out[idx++] = board.count(Color(c), PieceType(pt)) / 8.0f;

    // 12 features: endangered flags (for each type: white, black)
    bool w_end[6]{}, b_end[6]{};
    for (int pt = 0; pt < 6; pt++) {
        w_end[pt] = (board.count(WHITE, PieceType(pt)) == 1);
        b_end[pt] = (board.count(BLACK, PieceType(pt)) == 1);
    }
    for (int pt = 0; pt < 6; pt++) {
        out[idx++] = w_end[pt] ? 1.0f : 0.0f;
        out[idx++] = b_end[pt] ? 1.0f : 0.0f;
    }

    // 12 features: extinct flags (for each type: white, black)
    bool w_ext[6]{}, b_ext[6]{};
    for (int pt = 0; pt < 6; pt++) {
        w_ext[pt] = (board.count(WHITE, PieceType(pt)) == 0);
        b_ext[pt] = (board.count(BLACK, PieceType(pt)) == 0);
    }
    for (int pt = 0; pt < 6; pt++) {
        out[idx++] = w_ext[pt] ? 1.0f : 0.0f;
        out[idx++] = b_ext[pt] ? 1.0f : 0.0f;
    }

    // Current player
    out[idx++] = (side == WHITE) ? 1.0f : 0.0f;

    // Mobility (legal move count / 100)
    out[idx++] = static_cast<float>(legal_moves().size()) / 100.0f;

    // Game phase (total pieces / 32)
    int total = 0;
    for (int c = 0; c < 2; c++)
        for (int pt = 0; pt < 6; pt++)
            total += board.count(Color(c), PieceType(pt));
    out[idx++] = total / 32.0f;
}

} // namespace ext
