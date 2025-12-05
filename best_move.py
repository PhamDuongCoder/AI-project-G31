import chess
import random
from typing import List, Optional
import game_state

square_value_for_black = {}
square_value_for_white = {}

# Square point matrices for opening phase
# Rows are from rank 8 to rank 1 (top to bottom in the display)
# Columns are from file a to h (left to right)
SQUARE_POINT_BLACK_OPENING = [
    [4, 4, 4, 4, 4, 4, 4, 4],  # rank 8
    [3, 4, 4, 4, 4, 4, 4, 3],  # rank 7
    [2, 3, 4, 4, 4, 4, 3, 2],  # rank 6
    [2, 3, 4, 5, 5, 4, 3, 2],  # rank 5
    [2, 3, 4, 5, 5, 4, 3, 2],  # rank 4
    [1, 3, 3, 3, 3, 3, 3, 1],  # rank 3
    [1, 2, 2, 2, 2, 2, 2, 1],  # rank 2
    [0, 0, 0, 0, 0, 0, 0, 0],  # rank 1
]

SQUARE_POINT_WHITE_OPENING = [
    [0, 0, 0, 0, 0, 0, 0, 0],  # rank 8
    [1, 2, 2, 2, 2, 2, 2, 1],  # rank 7
    [1, 3, 3, 3, 3, 3, 3, 1],  # rank 6
    [2, 3, 4, 5, 5, 4, 3, 2],  # rank 5
    [2, 3, 4, 5, 5, 4, 3, 2],  # rank 4
    [2, 3, 4, 4, 4, 4, 3, 2],  # rank 3
    [3, 4, 4, 4, 4, 4, 4, 3],  # rank 2
    [4, 4, 4, 4, 4, 4, 4, 4],  # rank 1
]

def init():
    global square_value_for_black
    global square_value_for_white
    square_value_for_black = {square: 1 for square in chess.SQUARES}
    square_value_for_white = {square: 1 for square in chess.SQUARES}

def get_ai_move(board: chess.Board) -> Optional[chess.Move]:
    """Return an AI move (may return None if no legal moves)."""
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None
    return get_best_move_minimax(board, depth=3)

def get_best_move_minimax(board: chess.Board, depth: int = 3) -> Optional[chess.Move]:
    """Return best move found by minimax (alpha-beta) from the current board."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    root_color = board.turn
    best_move: Optional[chess.Move] = legal_moves[0]
    best_score = float('-inf')

    for move in legal_moves:
        board.push(move)
        score = minimax(board, depth - 1, False, float('-inf'), float('inf'), root_color)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move

def minimax(board: chess.Board, depth: int, maximizing: bool, alpha: float, beta: float, root_color: bool) -> float:

    legal_moves = list(board.legal_moves)

    if depth == 0 or board.is_game_over():
        return evaluate_position(board, root_color)

    if maximizing:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, False, alpha, beta, root_color)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
            if eval_score > alpha:
                alpha = eval_score
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, True, alpha, beta, root_color)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
            if eval_score < beta:
                beta = eval_score
            if beta <= alpha:
                break
        return min_eval
        
def evaluate_position(board: chess.Board, root_color: bool) -> float:
    """Evaluate the board from the point of view of root_color (True for WHITE, False for BLACK)."""
    # Terminal conditions
    if board.is_checkmate():
        # side to move is checkmated
        if board.turn == root_color:
            return float('-inf')
        else:
            return float('inf')

    # draw conditions
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return 0.0

    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

    score = 0.0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == root_color:
                score += value
            else:
                score -= value

    # mobility (small factor): compute legal moves count for each side on copies
    try:
        b1 = board.copy()
        b1.turn = root_color
        mobility_root = len(list(b1.legal_moves))
        b2 = board.copy()
        b2.turn = not root_color
        mobility_opp = len(list(b2.legal_moves))
    except Exception:
        mobility_root = mobility_opp = 0

    score += 0.05 * (mobility_root - mobility_opp)

    # small control and king safety heuristics (from root perspective)
    score += 0.01 * count_square_control(board, root_color) + 0.1 * king_safety(board, root_color)

    # phase-specific evaluation
    extra_point = 0.0
    if game_state.game_phase == 0:
        extra_point = evaluate_opening(board, root_color)
    elif game_state.game_phase == 1:
        extra_point = evaluate_middlegame(board, root_color)
    elif game_state.game_phase == 2:
        extra_point = evaluate_endgame(board, root_color)
    
    score += 0.1 * extra_point

    return score

"""count total square control"""
def count_square_control(board: chess.Board, root_color: bool) -> int:
    """Return a signed control value (positive if favorable to root_color, otherwise negative)."""
    controlled = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            attacks = board.attacks(square)
            if piece.color == root_color:
                controlled += len(attacks)
            else:
                controlled -= len(attacks)
    return controlled

def king_safety(board: chess.Board, root_color: bool) -> float:
    """Simple king safety heuristic: counts friendly pawns in front of king as shield.
    
    Positive values favor root_color.
    """
    total = 0.0
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is None:
            continue

        # Pawn shield in front of the king (existing heuristic)
        shield = 0.0
        direction = 1 if color == chess.WHITE else -1
        file = chess.square_file(king_sq)
        rank = chess.square_rank(king_sq)
        shield_rank = rank + direction

        if 0 <= shield_rank <= 7:
            for file_offset in (-1, 0, 1):
                f = file + file_offset
                if 0 <= f <= 7:
                    sq = chess.square(f, shield_rank)
                    piece = board.piece_at(sq)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        shield += 0.5

        # King neighborhood control (2-square radius)
        radius_score = 0.0
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)

        # Precompute attack sets for both colors to speed checks
        friendly_attacks = set()
        enemy_attacks = set()
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if not p:
                continue
            attacks = board.attacks(sq)
            if p.color == color:
                friendly_attacks.update(attacks)
            else:
                enemy_attacks.update(attacks)

        for df in range(-2, 3):
            for dr in range(-2, 3):
                if df == 0 and dr == 0:
                    continue
                f = king_file + df
                r = king_rank + dr
                if not (0 <= f <= 7 and 0 <= r <= 7):
                    continue
                sq = chess.square(f, r)

                occ = board.piece_at(sq)
                if occ:
                    # Occupied by friendly or enemy piece
                    if occ.color == color:
                        radius_score += 0.1
                    else:
                        radius_score -= 0.1
                else:
                    # Empty: check control
                    if sq in friendly_attacks:
                        radius_score += 0.1
                    if sq in enemy_attacks:
                        radius_score -= 0.1

        # Combine shield and radius for this king; add if root_color, subtract if opponent
        king_contrib = shield + radius_score
        if color == root_color:
            total += king_contrib
        else:
            total -= king_contrib

    return total


def evaluate_opening(board: chess.Board, root_color: bool) -> float:
    """Evaluate position in the opening phase using square point matrices.
    
    For each square, if root_color's piece/control is on it, add that square's point.
    Otherwise, subtract that square's point.
    
    Returns a float bonus/penalty based on piece placement and control.
    """
    point = 0.0
    
    for square in chess.SQUARES:
        # Get the rank and file of the square
        rank = chess.square_rank(square)  # 0-7 (1st rank to 8th rank)
        file = chess.square_file(square)  # 0-7 (a-file to h-file)
        
        # Get matrix index (convert rank to matrix row: rank 7 -> row 0, rank 0 -> row 7)
        matrix_row = 7 - rank
        matrix_col = file
        
        # Get the square point value based on piece color
        if root_color == chess.WHITE:
            square_point = SQUARE_POINT_WHITE_OPENING[matrix_row][matrix_col]
        else:
            square_point = SQUARE_POINT_BLACK_OPENING[matrix_row][matrix_col]
        
        # Check if root_color's piece is on this square or controls it
        piece = board.piece_at(square)
        
        # Check if root_color controls this square
        is_controlled = False
        if piece and piece.color == root_color:
            is_controlled = True
        else:
            # Check if any root_color piece attacks this square
            for attack_square in chess.SQUARES:
                attack_piece = board.piece_at(attack_square)
                if attack_piece and attack_piece.color == root_color:
                    if square in board.attacks(attack_square):
                        is_controlled = True
                        break
        
        if is_controlled:
            point += square_point
        else:
            point -= square_point
    
    return point


def evaluate_middlegame(board: chess.Board, root_color: bool) -> float:
    """Evaluate position in the middlegame phase.
    
    Returns a float bonus/penalty (defaults to 0 for now).
    """
    return 0.0


def evaluate_endgame(board: chess.Board, root_color: bool) -> float:
    """Evaluate position in the endgame phase.
    
    Returns a float bonus/penalty (defaults to 0 for now).
    """
    return 0.0
