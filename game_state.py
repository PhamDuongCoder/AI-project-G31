"""
Shared game state module.

This module contains global game state variables that are shared between
chess_game.py and best_move.py 
"""

# Game phase indicator:
# 0 = opening (early game)
# 1 = middlegame (mid game)
# 2 = endgame (late game)
game_phase = 0
