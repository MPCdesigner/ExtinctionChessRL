#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "engine.h"
#include "mcts.h"
#include "self_play.h"

namespace py = pybind11;
using namespace ext;

static constexpr int NC = Game::NUM_INPUT_CHANNELS;  // 115
static constexpr int ENC_SIZE = Game::BOARD_ENCODING_SIZE;  // 115*64

// ── Helper: Python Position object ─────────────────────────────────────────

struct PyPosition {
    int rank, file;
    PyPosition() : rank(0), file(0) {}
    PyPosition(int r, int f) : rank(r), file(f) {}
    int to_sq() const { return sq(rank, file); }
    std::string to_algebraic() const {
        char buf[3] = { static_cast<char>('a' + file), static_cast<char>('1' + rank), '\0' };
        return std::string(buf);
    }
    bool eq(const PyPosition& o) const { return rank == o.rank && file == o.file; }
    size_t hash() const { return std::hash<int>()(rank * 8 + file); }
};

// ── Helper: Python PieceInfo wrapper ───────────────────────────────────────

struct PyPiece {
    PieceType piece_type;
    Color color;
    PyPosition position;
    bool has_moved;
    int rank, file;   // convenience — matches old Python Piece.position.rank/file
    PyPiece() : piece_type(PAWN), color(WHITE), position(0,0), has_moved(false), rank(0), file(0) {}
};

// ═══════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(_ext_chess, m) {
    m.doc() = "Fast C++ engine for Extinction Chess (bitboard-based)";

    // Call init on module load
    ext::init();

    // ── PieceType enum ─────────────────────────────────────────────────────

    py::enum_<PieceType>(m, "PieceType")
        .value("PAWN",   PAWN)
        .value("KNIGHT", KNIGHT)
        .value("BISHOP", BISHOP)
        .value("ROOK",   ROOK)
        .value("QUEEN",  QUEEN)
        .value("KING",   KING);

    // ── Color enum ─────────────────────────────────────────────────────────

    py::enum_<Color>(m, "Color")
        .value("WHITE", WHITE)
        .value("BLACK", BLACK);

    // ── Position ───────────────────────────────────────────────────────────

    py::class_<PyPosition>(m, "Position")
        .def(py::init<int, int>(), py::arg("rank"), py::arg("file"))
        .def_readwrite("rank", &PyPosition::rank)
        .def_readwrite("file", &PyPosition::file)
        .def("to_algebraic", &PyPosition::to_algebraic)
        .def("__eq__", &PyPosition::eq)
        .def("__hash__", &PyPosition::hash)
        .def("__repr__", [](const PyPosition& p) {
            return p.to_algebraic();
        });

    // ── Piece (returned by Board.get_piece) ────────────────────────────────

    py::class_<PyPiece>(m, "Piece")
        .def_readonly("piece_type", &PyPiece::piece_type)
        .def_readonly("color",      &PyPiece::color)
        .def_readonly("position",   &PyPiece::position)
        .def_readonly("has_moved",  &PyPiece::has_moved)
        .def("__repr__", [](const PyPiece& p) {
            const char* cc = (p.color == WHITE) ? "W" : "B";
            const char* names[] = {"P","N","B","R","Q","K"};
            return std::string(cc) + names[p.piece_type];
        });

    // ── Move ───────────────────────────────────────────────────────────────

    py::class_<Move>(m, "Move")
        .def(py::init([](PyPosition from, PyPosition to,
                         py::object promo, bool ep, bool castling,
                         py::object crook_from, py::object crook_to) {
            int p = NONE_PT;
            if (!promo.is_none())
                p = promo.cast<PieceType>();
            int fl = 0;
            if (ep) fl |= Move::F_EP;
            if (castling) fl |= Move::F_CASTLE;
            int crf = -1, crt = -1;
            if (!crook_from.is_none())
                crf = crook_from.cast<PyPosition>().to_sq();
            if (!crook_to.is_none())
                crt = crook_to.cast<PyPosition>().to_sq();
            return Move(from.to_sq(), to.to_sq(), p, fl, crf, crt);
        }),
            py::arg("from_pos"), py::arg("to_pos"),
            py::arg("promotion") = py::none(),
            py::arg("is_en_passant") = false,
            py::arg("is_castling") = false,
            py::arg("castling_rook_from") = py::none(),
            py::arg("castling_rook_to") = py::none())

        .def_property_readonly("from_pos", [](const Move& m) {
            return PyPosition(rank_of(m.from), file_of(m.from));
        })
        .def_property_readonly("to_pos", [](const Move& m) {
            return PyPosition(rank_of(m.to), file_of(m.to));
        })
        .def_property_readonly("promotion", [](const Move& m) -> py::object {
            if (m.promo == NONE_PT) return py::none();
            return py::cast(PieceType(m.promo));
        })
        .def_property_readonly("is_en_passant", &Move::is_ep)
        .def_property_readonly("is_castling",   &Move::is_castle)
        .def_property_readonly("castling_rook_from", [](const Move& m) -> py::object {
            if (m.castle_rf < 0) return py::none();
            return py::cast(PyPosition(rank_of(m.castle_rf), file_of(m.castle_rf)));
        })
        .def_property_readonly("castling_rook_to", [](const Move& m) -> py::object {
            if (m.castle_rt < 0) return py::none();
            return py::cast(PyPosition(rank_of(m.castle_rt), file_of(m.castle_rt)));
        })
        // captured_piece is always None from the C++ side (not tracked on Move)
        .def_property_readonly("captured_piece", [](const Move&) { return py::none(); })
        .def("__repr__", &Move::to_string);

    // ── Board ──────────────────────────────────────────────────────────────

    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def("copy", &Board::copy)
        .def("get_piece", [](const Board& b, PyPosition pos) -> py::object {
            int s = pos.to_sq();
            if (s < 0 || s > 63 || !b.has(s)) return py::none();
            PyPiece piece;
            piece.piece_type = b.type_at(s);
            piece.color = b.color_at(s);
            piece.position = pos;
            piece.rank = pos.rank;
            piece.file = pos.file;
            // Derive has_moved from castling rights for king/rook on starting squares
            piece.has_moved = true;
            int r = pos.rank, f = pos.file;
            if (piece.piece_type == KING) {
                if (piece.color == WHITE && r == 0 && f == 4 &&
                    (b.castling & (Board::WK | Board::WQ)))
                    piece.has_moved = false;
                if (piece.color == BLACK && r == 7 && f == 4 &&
                    (b.castling & (Board::BK | Board::BQ)))
                    piece.has_moved = false;
            }
            if (piece.piece_type == ROOK) {
                if (piece.color == WHITE && r == 0 && f == 0 && (b.castling & Board::WQ))
                    piece.has_moved = false;
                if (piece.color == WHITE && r == 0 && f == 7 && (b.castling & Board::WK))
                    piece.has_moved = false;
                if (piece.color == BLACK && r == 7 && f == 0 && (b.castling & Board::BQ))
                    piece.has_moved = false;
                if (piece.color == BLACK && r == 7 && f == 7 && (b.castling & Board::BK))
                    piece.has_moved = false;
            }
            return py::cast(piece);
        }, py::arg("pos"))

        .def("get_piece_count", [](const Board& b, Color c) {
            py::dict d;
            auto cts = b.counts(c);
            for (int pt = 0; pt < 6; pt++)
                d[py::cast(PieceType(pt))] = cts[pt];
            return d;
        }, py::arg("color"))

        .def("set_piece", [](Board& b, PyPosition pos, py::object piece_obj) {
            int s = pos.to_sq();
            if (piece_obj.is_none()) {
                b.del(s);
            } else {
                auto piece = piece_obj.cast<PyPiece>();
                if (b.has(s)) b.del(s);
                b.put(piece.color, piece.piece_type, s);
            }
        }, py::arg("pos"), py::arg("piece"))

        .def("is_valid_position", [](const Board&, PyPosition pos) {
            return pos.rank >= 0 && pos.rank < 8 && pos.file >= 0 && pos.file < 8;
        }, py::arg("pos"))

        .def("find_king", [](const Board& b, Color c) -> py::object {
            u64 kings = b.bb[c][KING];
            if (!kings) return py::none();
            int s = lsb(kings);
            PyPiece p;
            p.piece_type = KING;
            p.color = c;
            p.position = PyPosition(rank_of(s), file_of(s));
            p.rank = rank_of(s);
            p.file = file_of(s);
            p.has_moved = true;
            return py::cast(p);
        }, py::arg("color"))

        .def_property("en_passant_target",
            [](const Board& b) -> py::object {
                if (b.ep < 0) return py::none();
                return py::cast(PyPosition(rank_of(b.ep), file_of(b.ep)));
            },
            [](Board& b, py::object val) {
                if (val.is_none()) b.ep = -1;
                else b.ep = val.cast<PyPosition>().to_sq();
            })

        .def_readwrite("halfmove_clock",  &Board::hmclock)
        .def_readwrite("fullmove_number", &Board::fmnum)
        .def_property_readonly("position_history", [](const Board& b) {
            return b.history;
        });

    // ── ExtinctionChess (Game wrapper) ─────────────────────────────────────

    py::class_<Game>(m, "ExtinctionChess")
        .def(py::init<>())

        .def_readwrite("board", &Game::board)

        .def_property("current_player",
            [](const Game& g) { return g.side; },
            [](Game& g, Color c) { g.side = c; })

        .def_readwrite("game_over", &Game::over)

        .def_property("winner",
            [](const Game& g) -> py::object {
                if (g.winner < 0) return py::none();
                return py::cast(Color(g.winner));
            },
            [](Game& g, py::object val) {
                if (val.is_none()) g.winner = -1;
                else g.winner = val.cast<Color>();
            })

        .def_property("draw_reason",
            [](const Game& g) -> py::object {
                if (g.draw_reason.empty()) return py::none();
                return py::cast(g.draw_reason);
            },
            [](Game& g, py::object val) {
                if (val.is_none()) g.draw_reason.clear();
                else g.draw_reason = val.cast<std::string>();
            })

        .def("get_legal_moves", [](const Game& g, py::object from_pos) {
            if (from_pos.is_none())
                return g.legal_moves();
            auto pos = from_pos.cast<PyPosition>();
            return g.legal_moves_from(pos.to_sq());
        }, py::arg("from_pos") = py::none())

        .def("make_move", &Game::make_move, py::arg("move"))

        .def("get_endangered_pieces", &Game::endangered, py::arg("color"))
        .def("get_extinct_pieces",    &Game::extinct,    py::arg("color"))

        .def("get_game_state", [](const Game& g) {
            py::dict state;
            state["board"] = g.board;
            state["current_player"] = g.side;
            state["game_over"] = g.over;
            state["winner"] = (g.winner < 0) ? py::none().cast<py::object>()
                                              : py::cast(Color(g.winner));
            state["draw_reason"] = g.draw_reason.empty()
                ? py::none().cast<py::object>() : py::cast(g.draw_reason);
            // These are computed via Board methods
            py::dict wc, bc;
            auto wcts = g.board.counts(WHITE);
            auto bcts = g.board.counts(BLACK);
            for (int pt = 0; pt < 6; pt++) {
                wc[py::cast(PieceType(pt))] = wcts[pt];
                bc[py::cast(PieceType(pt))] = bcts[pt];
            }
            state["white_counts"] = wc;
            state["black_counts"] = bc;
            state["legal_moves"] = g.over ? py::list() : py::cast(g.legal_moves());
            if (g.board.ep >= 0)
                state["en_passant_target"] = PyPosition(rank_of(g.board.ep),
                                                        file_of(g.board.ep));
            else
                state["en_passant_target"] = py::none();
            state["halfmove_clock"] = g.board.hmclock;
            state["fullmove_number"] = g.board.fmnum;
            return state;
        })

        // ── Neural network encoding ────────────────────────────────────────

        .def("encode_board", [](const Game& g) {
            py::array_t<float> arr({NC, 8, 8});
            auto buf = arr.mutable_data();
            g.encode_board(buf);
            return arr;
        })

        .def("get_simple_features", [](const Game& g) {
            py::array_t<float> arr(39);
            auto buf = arr.mutable_data();
            g.get_simple_features(buf);
            return arr;
        })

        // reset_game (for GUI)
        .def("reset_game", [](Game& g) {
            g.board.setup();
            g.side = WHITE;
            g.over = false;
            g.winner = -1;
            g.draw_reason.clear();
        });

    // ── MCTS ──────────────────────────────────────────────────────────────

    m.def("move_to_index", [](const Move& m) {
        return ext::move_to_index(m);
    }, py::arg("move"), "Convert a Move to a policy index in [0, 4864)");

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<const Game&, int, float, float, float, bool, int>(),
             py::arg("game"),
             py::arg("num_simulations") = 800,
             py::arg("c_puct") = 2.5f,
             py::arg("dirichlet_alpha") = 0.3f,
             py::arg("noise_weight") = 0.25f,
             py::arg("tactical_shortcuts") = true,
             py::arg("batch_size") = 8)

        .def("select_leaves", [](MCTS& mcts, int max_leaves) {
            // Allocate output buffer
            py::array_t<float> boards({max_leaves, NC, 8, 8});
            auto buf = boards.mutable_data();
            int num_leaves = mcts.select_leaves(buf);
            // Return only the filled portion
            if (num_leaves == 0) {
                return py::array_t<float>({0, NC, 8, 8});
            }
            // Slice to actual number of leaves
            py::array_t<float> result({num_leaves, NC, 8, 8});
            std::memcpy(result.mutable_data(), buf,
                        num_leaves * ENC_SIZE * sizeof(float));
            return result;
        }, py::arg("max_leaves") = 8)

        .def("process_results", [](MCTS& mcts,
                                   py::array_t<float, py::array::c_style> policies,
                                   py::array_t<float, py::array::c_style> values) {
            auto p = policies.unchecked<2>();  // (N, 4864)
            auto v = values.unchecked<1>();    // (N,)
            int n = static_cast<int>(p.shape(0));
            mcts.process_results(policies.data(), values.data(), n);
        }, py::arg("policies"), py::arg("values"))

        .def("is_done", &MCTS::is_done)

        .def("get_results", &MCTS::get_results,
             "Returns list of (policy_index, visit_count) pairs")

        .def("get_move_results", &MCTS::get_move_results,
             "Returns list of (from_sq, to_sq, promo, visit_count) tuples")

        // Convenience: run root expansion separately (first NN call)
        .def("expand_root", [](MCTS& mcts,
                               py::array_t<float, py::array::c_style> policy_logits) {
            auto p = policy_logits.unchecked<1>();  // (4864,)
            mcts.expand_root(policy_logits.data());
        }, py::arg("policy_logits"))

        .def_property_readonly("sims_done", [](const MCTS& m) {
            // Access via is_done check
            return m.is_done();
        });

    // ── SelfPlayManager ──────────────────────────────────────────────────

    py::class_<SelfPlayManager>(m, "SelfPlayManager")
        .def(py::init<int, int, int, float, float, float, bool, int, int, int>(),
             py::arg("num_parallel_games"),
             py::arg("total_games"),
             py::arg("num_simulations") = 800,
             py::arg("c_puct") = 2.5f,
             py::arg("dirichlet_alpha") = 0.3f,
             py::arg("noise_weight") = 0.25f,
             py::arg("tactical_shortcuts") = true,
             py::arg("temp_threshold") = 15,
             py::arg("max_moves") = 200,
             py::arg("mcts_batch_size") = 8)

        .def("collect_leaves", [](SelfPlayManager& mgr, int max_batch) {
            // Allocate output buffer
            py::array_t<float> boards({max_batch, NC, 8, 8});
            auto buf = boards.mutable_data();
            int num_leaves = mgr.collect_leaves(buf, max_batch);
            if (num_leaves == 0) {
                return py::array_t<float>({0, NC, 8, 8});
            }
            // Return only the filled portion
            py::array_t<float> result({num_leaves, NC, 8, 8});
            std::memcpy(result.mutable_data(), buf,
                        num_leaves * ENC_SIZE * sizeof(float));
            return result;
        }, py::arg("max_batch") = 256)

        .def("process_results", [](SelfPlayManager& mgr,
                                   py::array_t<float, py::array::c_style> policies,
                                   py::array_t<float, py::array::c_style> values) {
            int n = static_cast<int>(policies.shape(0));
            mgr.process_results(policies.data(), values.data(), n);
        }, py::arg("policies"), py::arg("values"))

        .def("is_done", &SelfPlayManager::is_done)
        .def("games_completed", &SelfPlayManager::games_completed)
        .def("games_active", &SelfPlayManager::games_active)

        .def("get_results", [](const SelfPlayManager& mgr) {
            auto& records = mgr.get_results();
            py::list result;
            for (auto& rec : records) {
                py::dict d;
                // Convert boards: list of (BOARD_ENCODING_SIZE,) float vectors
                py::list boards_list;
                for (auto& b : rec.boards) {
                    py::array_t<float> arr(static_cast<py::ssize_t>(b.size()));
                    std::memcpy(arr.mutable_data(), b.data(), b.size() * sizeof(float));
                    boards_list.append(arr);
                }
                d["boards"] = boards_list;

                // Convert policies: list of (4864,) float vectors
                py::list policies_list;
                for (auto& p : rec.policies) {
                    py::array_t<float> arr(static_cast<py::ssize_t>(p.size()));
                    std::memcpy(arr.mutable_data(), p.data(), p.size() * sizeof(float));
                    policies_list.append(arr);
                }
                d["policies"] = policies_list;

                // Players
                py::list players_list;
                for (int pl : rec.players) {
                    players_list.append(pl);
                }
                d["players"] = players_list;

                d["outcome"] = rec.outcome;
                result.append(d);
            }
            return result;
        });
}
