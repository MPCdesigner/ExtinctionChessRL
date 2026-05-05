"""
Build the C++ extinction chess engine.

Usage:
    pip install pybind11    (if not already installed)
    python setup.py build_ext --inplace

This produces _ext_chess.*.so (Linux) or _ext_chess.*.pyd (Windows)
in the current directory, importable as:
    from _ext_chess import ExtinctionChess, Color, PieceType, Position, Move
"""

import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

extra_compile = []
extra_link = []
if sys.platform == "win32":
    extra_compile = ["/O2"]
else:
    extra_compile = ["-O3", "-ffast-math", "-pthread"]
    extra_link = ["-pthread"]

ext = Pybind11Extension(
    "_ext_chess",
    sources=["cpp/engine.cpp", "cpp/mcts.cpp", "cpp/self_play.cpp", "cpp/bindings.cpp"],
    include_dirs=["cpp"],
    cxx_std=17,
    extra_compile_args=extra_compile,
    extra_link_args=extra_link,
)

setup(
    name="extinction-chess-fast",
    version="1.0.0",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
)
