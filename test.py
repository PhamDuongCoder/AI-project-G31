import chess
import random

def print_board(board):
    """Print the chess board in a readable format."""
    print("\n" + str(board) + "\n")
    print(f"Turn: {'White' if board.turn else 'Black'}")
    print("-" * 40)

def get_player_move(board):
    """Get a valid move from the player."""
    while True:
        print("\nLegal moves:", ", ".join([move.uci() for move in board.legal_moves]))
        move_input = input("\nEnter your move (e.g., 'e2e4') or 'quit' to exit: ").strip().lower()
        
        if move_input == 'quit':
            return None
        
        try:
            move = chess.Move.from_uci(move_input)
            if move in board.legal_moves:
                return move
            else:
                print("Illegal move! Try again.")
        except:
            print("Invalid format! Use format like 'e2e4' or 'e7e8q' for promotion.")

def get_random_move(board):
    """Get a random legal move for the AI."""
    legal_moves = list(board.legal_moves)
    return random.choice(legal_moves)

def main():
    """Main game loop."""
    board = chess.Board()
    
    print("=" * 40)
    print("   CHESS GAME - You vs Random AI")
    print("=" * 40)
    print("\nYou are White. Enter moves in UCI format (e.g., 'e2e4')")
    print("For pawn promotion, add the piece: 'e7e8q' for queen")
    
    while not board.is_game_over():
        print_board(board)
        
        if board.turn == chess.WHITE:
            # Player's turn (White)
            print("\n>>> YOUR TURN <<<")
            move = get_player_move(board)
            if move is None:
                print("Game quit by player.")
                break
            board.push(move)
            print(f"\nYou played: {move.uci()}")
        else:
            # AI's turn (Black)
            print("\n>>> AI THINKING <<<")
            move = get_random_move(board)
            board.push(move)
            print(f"AI played: {move.uci()}")
            input("\nPress Enter to continue...")
    
    # Game over
    print_board(board)
    print("\n" + "=" * 40)
    print("GAME OVER!")
    print("=" * 40)
    
    if board.is_checkmate():
        winner = "Black (AI)" if board.turn == chess.WHITE else "White (You)"
        print(f"Checkmate! {winner} wins!")
    elif board.is_stalemate():
        print("Stalemate! It's a draw.")
    elif board.is_insufficient_material():
        print("Draw due to insufficient material.")
    elif board.is_fifty_moves():
        print("Draw due to fifty-move rule.")
    elif board.is_repetition():
        print("Draw due to threefold repetition.")
    else:
        print("Draw!")

if __name__ == "__main__":
    main()