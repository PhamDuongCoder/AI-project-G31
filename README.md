# Interactive Chess Game

A fully interactive chess game with GUI built using Python and Pygame.

## Features

- **Interactive GUI**: Click on pieces to move them
- **Player Color Selection**: Choose to play as white or black
- **AI Opponent**: Play against a random-move AI
- **Visual Feedback**: 
  - Highlighted legal moves
  - Selected piece indication
  - Check indicator
  - Game over screen
- **Piece Images**: Uses custom piece images from the `img/` folder

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## How to Run

```bash
python chess_game.py
```

## How to Play

1. **Start the game**: Run `python chess_game.py`
2. **Choose your color**: Select whether you want to play as white or black
3. **Make moves**: Click on a piece to select it, then click on a legal destination square
4. **Game controls**:
   - Click "New Game" to restart
   - Click "Switch Color" to change your color and restart
5. **AI moves**: The AI will automatically make random legal moves on its turn

## Piece Images

The game uses piece images from the `img/` folder:
- `w` = white pieces, `b` = black pieces
- `p` = pawn, `r` = rook, `n` = knight, `b` = bishop, `q` = queen, `k` = king

## Game Rules

The game follows standard chess rules:
- All standard piece movements
- Castling, en passant, and pawn promotion
- Check and checkmate detection
- Draw conditions (stalemate, insufficient material, etc.)

Enjoy playing chess!
