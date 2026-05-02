#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <string>

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(__popcnt64)
#pragma intrinsic(_BitScanForward64)
#endif

namespace ext {

// ── Types ──────────────────────────────────────────────────────────────────

using u64 = uint64_t;

enum PieceType : int8_t {
    PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5, NONE_PT = 6
};

enum Color : int8_t { WHITE = 0, BLACK = 1 };

inline Color flip(Color c) { return Color(c ^ 1); }

// ── Square helpers (rank-major: sq = rank*8 + file) ────────────────────────

inline int sq(int r, int f) { return (r << 3) | f; }
inline int rank_of(int s) { return s >> 3; }
inline int file_of(int s) { return s & 7; }
inline u64 bit(int s) { return 1ULL << s; }

inline int popcnt(u64 b) {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(b));
#else
    return __builtin_popcountll(b);
#endif
}

inline int lsb(u64 b) {
#ifdef _MSC_VER
    unsigned long idx;
    _BitScanForward64(&idx, b);
    return static_cast<int>(idx);
#else
    return __builtin_ctzll(b);
#endif
}

inline int pop_lsb(u64& b) { int s = lsb(b); b &= b - 1; return s; }

// ── Move ───────────────────────────────────────────────────────────────────

struct Move {
    int8_t from, to, promo, flags;
    int8_t castle_rf, castle_rt;          // rook from/to for castling

    static constexpr int8_t F_EP = 1, F_CASTLE = 2;

    Move() : from(-1), to(-1), promo(NONE_PT), flags(0),
             castle_rf(-1), castle_rt(-1) {}
    Move(int f, int t, int p = NONE_PT, int fl = 0, int crf = -1, int crt = -1)
        : from(static_cast<int8_t>(f)), to(static_cast<int8_t>(t)),
          promo(static_cast<int8_t>(p)), flags(static_cast<int8_t>(fl)),
          castle_rf(static_cast<int8_t>(crf)), castle_rt(static_cast<int8_t>(crt)) {}

    bool is_ep()     const { return flags & F_EP; }
    bool is_castle() const { return flags & F_CASTLE; }

    std::string to_string() const;
};

// ── Pre-computed attack tables ─────────────────────────────────────────────

extern u64 knight_atk[64];
extern u64 king_atk[64];
extern u64 zobrist_pc[2][6][64];
extern u64 zobrist_side;
extern u64 zobrist_ep[64];
extern u64 zobrist_castle[16];

void init();      // call once before using engine

u64 bishop_attacks(int s, u64 occ);
u64 rook_attacks(int s, u64 occ);
inline u64 queen_attacks(int s, u64 occ) {
    return bishop_attacks(s, occ) | rook_attacks(s, occ);
}

// ── Board ──────────────────────────────────────────────────────────────────

class Board {
public:
    u64 bb[2][6]{};         // pieces[color][piece_type]
    u64 occ[3]{};           // WHITE occupancy, BLACK occupancy, ALL

    int8_t pc[64];          // piece type on each square  (NONE_PT if empty)
    int8_t cl[64];          // color on each square       (-1 if empty)

    int8_t ep;              // en passant target square    (-1 if none)
    uint8_t castling;       // 4-bit castling rights
    int     hmclock;        // halfmove clock
    int     fmnum;          // fullmove number
    u64     hash;
    std::vector<u64> history;

    // History ring buffer for NN input (last 8 positions' piece bitboards)
    static constexpr int HIST_N = 8;
    u64  bb_hist[HIST_N][2][6]{};   // piece bitboards for past positions
    int  hist_len = 0;              // number of valid entries (0..HIST_N)
    int  hist_idx = 0;              // next write position in ring buffer

    void push_history();            // save current bb into ring buffer

    static constexpr uint8_t WK = 1, WQ = 2, BK = 4, BQ = 8;

    Board();
    void clear();
    void setup();
    Board copy() const { return *this; }

    void put(Color c, PieceType pt, int s);
    void del(int s);

    bool      has(int s)      const { return pc[s] != NONE_PT; }
    PieceType type_at(int s)  const { return PieceType(pc[s]); }
    Color     color_at(int s) const { return Color(cl[s]); }
    int       count(Color c, PieceType pt) const { return popcnt(bb[c][pt]); }

    std::array<int,6> counts(Color c) const;
    u64 compute_hash(Color side) const;
};

// ── Game ───────────────────────────────────────────────────────────────────

class Game {
public:
    Board board;
    Color side;             // current player
    bool  over;             // game_over flag
    int   winner;           // 0=WHITE, 1=BLACK, -1=none/draw

    std::string draw_reason;

    Game();

    std::vector<Move> legal_moves() const;
    std::vector<Move> legal_moves_from(int sq) const;
    bool make_move(const Move& m);

    bool check_extinction(const Move& m);
    bool check_draws();

    std::vector<PieceType> endangered(Color c) const;
    std::vector<PieceType> extinct(Color c) const;

    // Neural-network encoding
    // 115 channels × 8 × 8:
    //   0-11:    current position (12 piece planes)
    //   12-23:   T-1 position
    //   ...
    //   96-107:  T-8 position
    //   108:     current player
    //   109:     endangered pieces
    //   110-113: castling rights (WK, WQ, BK, BQ)
    //   114:     halfmove clock (normalized)
    static constexpr int NUM_INPUT_CHANNELS = 115;
    static constexpr int BOARD_ENCODING_SIZE = NUM_INPUT_CHANNELS * 64;

    void encode_board(float* out) const;          // 115×8×8 = 7360 floats
    void get_simple_features(float* out) const;   // 39 floats

private:
    void gen_pawn  (std::vector<Move>& mv) const;
    void gen_knight(std::vector<Move>& mv) const;
    void gen_bishop(std::vector<Move>& mv) const;
    void gen_rook  (std::vector<Move>& mv) const;
    void gen_queen (std::vector<Move>& mv) const;
    void gen_king  (std::vector<Move>& mv) const;
};

} // namespace ext
