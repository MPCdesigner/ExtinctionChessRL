"""Positional evaluation tool for the AlphaZero Extinction Chess models.

Local pygame GUI for inspecting how models evaluate specific positions.
Supports two modes:
  - Game setup: play moves from the starting position (history planes populated)
  - Construction: manually place pieces (history planes empty; user sets turn)

Loads N models chosen at startup, evaluates the current position at chosen
sim counts (raw NN, 20, 50, 100, 200, 400, 800), and displays value + top-K
move distributions side-by-side per model.
"""
