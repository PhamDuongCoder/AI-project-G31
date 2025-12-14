"""best_move without killer move heuristic and move ordering"""
"""compared to best_move_4, this bot has quiescence search, which makes it stronger,
but much slower"""
import chess
import random
from typing import List, Optional
import game_state

INITIAL_DEPTH = 3

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
        score = minimax(board, depth - 1, False, float('-inf'), float('inf'), root_color, ply=1)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

    return best_move

MAX_PLY = INITIAL_DEPTH + 8

def minimax(board: chess.Board, depth: int, maximizing: bool, alpha: float, beta: float, root_color: bool, ply: int = 0) -> float:
    legal_moves = list(board.legal_moves)

    if depth == 0 or board.is_game_over():
        return quiescence_search(board, alpha, beta, root_color, ply)
    
    if maximizing:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, False, alpha, beta, root_color, ply=ply+1)
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
            eval_score = minimax(board, depth - 1, True, alpha, beta, root_color, ply=ply+1)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
            if eval_score < beta:
                beta = eval_score
            if beta <= alpha:
                break
        return min_eval
    
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}
    
def score_move(board: chess.Board, move: chess.Move) -> int:
    score = 0
    victim_piece = board.piece_at(move.to_square)
    aggressor_piece = board.piece_at(move.from_square)
    
    # 1. Kiểm tra Bắt quân (Capture)
    if victim_piece is not None:
        victim = victim_piece.piece_type
        aggressor = aggressor_piece.piece_type

        # Nếu victim_piece khác None, chắc chắn đây là nước bắt quân (capture)
        score += 100 * PIECE_VALUES.get(victim, 0) - PIECE_VALUES.get(aggressor, 0)
    
    # 2. Kiểm tra Phong cấp (Promotion)
    if move.promotion is not None:
        # Cộng điểm cao cho phong cấp lên Hậu
        score += 3000 + PIECE_VALUES.get(move.promotion, 0)

    return score

def quiescence_search(board: chess.Board, alpha: float, beta: float, root_color: bool, ply: int) -> float:
    # 1. Đánh giá tĩnh (Stand-pat)
    # Giả định evaluate_position trả về điểm tuyệt đối (Trắng +, Đen -)
    stand_pat = evaluate_position(board, root_color, ply)

    # 2. Xử lý Logic Cắt tỉa (Tách biệt Trắng/Đen)
    if board.turn == chess.WHITE:  # Maximizing Player (Trắng)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
    else:  # Minimizing Player (Đen)
        if stand_pat <= alpha:
            return alpha
        if stand_pat < beta:
            beta = stand_pat

    # 3. Tạo và Sắp xếp nước đi Động (Chỉ bắt quân)
    dynamic_moves = [move for move in board.legal_moves if board.is_capture(move)]
    
    scored_qmoves = []
    for move in dynamic_moves:
        # Dùng lại hàm score_move để lấy điểm MVV-LVA
        score = score_move(board, move) 
        scored_qmoves.append((score, move))
    
    ordered_qmoves = [move for score, move in scored_qmoves]

    # 4. Duyệt các nước đi
    if board.turn == chess.WHITE: # Trắng tìm Max
        for move in ordered_qmoves:
            board.push(move)
            # Gọi đệ quy Minimax-style (giữ nguyên alpha/beta, không đảo dấu)
            score = quiescence_search(board, alpha, beta, root_color, ply + 1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha
    
    else: # Đen tìm Min
        for move in ordered_qmoves:
            board.push(move)
            score = quiescence_search(board, alpha, beta, root_color, ply + 1)
            board.pop()

            if score <= alpha:
                return alpha
            if score < beta:
                beta = score
        return beta
        
def evaluate_position(board: chess.Board, root_color: bool, ply: int = 0) -> float:
    """Evaluate the board from the point of view of root_color (True for WHITE, False for BLACK)."""
    MATE_SCORE = 100000
    # Terminal conditions
    if board.is_checkmate():
        # side to move is checkmated
        if board.turn == root_color:
            return -MATE_SCORE + ply
        else:
            return MATE_SCORE - ply

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

    # Check piece count on board and update game phase if endgame threshold reached
    # Use the new rule: transition to endgame when there are fewer than 15 pieces on the board
    piece_count = len(board.piece_map())
    if piece_count < 15 and game_state.game_phase == 1:
        game_state.game_phase = 2

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

def evaluate_opening(board: chess.Board, root_color: bool) -> float:
    """Evaluate the board specifically for the opening phase."""
    # Simple opening evaluation: prioritize development and center control
    opening_score = 0.0

    center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
    development_squares = [
        chess.B1, chess.G1, chess.B8, chess.G8,  # Knights' starting squares
        chess.C1, chess.F1, chess.C8, chess.F8   # Bishops' starting squares
    ]

    for square in center_squares:
        piece = board.piece_at(square)
        if piece:
            if piece.color == root_color:
                opening_score += 0.5
            else:
                opening_score -= 0.5

    for square in development_squares:
        piece = board.piece_at(square)
        if piece:
            if piece.color == root_color:
                opening_score -= 0.3
            else:
                opening_score += 0.3

    # Castling bonus: +1 if the playing side's king is castled, -1 if opponent's king is castled
    def king_has_castled(bd: chess.Board, color: bool) -> bool:
        ks = bd.king(color)
        if ks is None:
            return False
        if color == chess.WHITE:
            return ks in (chess.G1, chess.C1)
        else:
            return ks in (chess.G8, chess.C8)

    if king_has_castled(board, root_color):
        opening_score += 1.0
    if king_has_castled(board, not root_color):
        opening_score -= 1.0

    return opening_score

def evaluate_middlegame(board: chess.Board, root_color: bool) -> float:
    """Evaluate the board specifically for the middlegame phase."""
    # Simple middlegame evaluation: prioritize piece activity and king safety
    middlegame_score = 0.0

    king_safety_score = king_safety(board, root_color)
    rook_score = rook_open_files(board, root_color)
    bishop_score_val = bishop_score(board, root_color)
    knight_score = knight_outposts(board, root_color)
    queen_score = queen_activity(board, root_color)
    
    middlegame_score = king_safety_score + rook_score + bishop_score_val + knight_score + queen_score
    
    return middlegame_score

def king_safety(board: chess.Board, root_color: bool) -> float:
    """Simple king safety heuristic: counts friendly pawns in front of king as shield.
    
    Positive values favor root_color.
    """
    total = 0.0
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is None:
            continue

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
        king_contrib = radius_score
        if color == root_color:
            total += king_contrib
        else:
            total -= king_contrib

    return total  

def rook_open_files(board: chess.Board, root_color: bool) -> float:
    """Evaluate rook activity on open and semi-open files.
    
    Positive values favor root_color.
    """
    total = 0.0
    for color in [chess.WHITE, chess.BLACK]:
        rooks = board.pieces(chess.ROOK, color)
        for rook_sq in rooks:
            file = chess.square_file(rook_sq)

            # Check if the file is open or semi-open
            has_friendly_pawn = False
            has_enemy_pawn = False
            for rank in range(8):
                sq = chess.square(file, rank)
                piece = board.piece_at(sq)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == color:
                        has_friendly_pawn = True
                    else:
                        has_enemy_pawn = True

            if not has_friendly_pawn and not has_enemy_pawn:
                # Open file
                if color == root_color:
                    total += 0.5
                else:
                    total -= 0.5
            elif not has_friendly_pawn and has_enemy_pawn:
                # Semi-open file
                if color == root_color:
                    total += 0.2
                else:
                    total -= 0.2

    return total

def bishop_score(board: chess.Board, root_color: bool) -> float:
    #for each bishop of the playing side, give 0.1 points for each forward square and 0.05 square for each backward square it controls
    #for each bishop of the opponent side, subtract 0.1 points for each forward square and 0.05 square for each backward square it controls
    total = 0.0
    for color in [chess.WHITE, chess.BLACK]:
        bishops = board.pieces(chess.BISHOP, color)
        for bishop_sq in bishops:
            bishop_file = chess.square_file(bishop_sq)
            bishop_rank = chess.square_rank(bishop_sq)

            # Determine forward and backward directions
            forward_dir = 1 if color == chess.WHITE else -1

            # Get all squares attacked by the bishop
            attacked_squares = board.attacks(bishop_sq)

            for sq in attacked_squares:
                sq_file = chess.square_file(sq)
                sq_rank = chess.square_rank(sq)

                if (sq_rank - bishop_rank) * forward_dir > 0:
                    # Forward square
                    if color == root_color:
                        total += 0.1
                    else:
                        total -= 0.1
                else:
                    # Backward square
                    if color == root_color:
                        total += 0.05
                    else:
                        total -= 0.05

    return total

def knight_outposts(board: chess.Board, root_color: bool) -> float:
    """Evaluate knight outposts.
    
    Positive values favor root_color.
    """
    total = 0.0
    for color in [chess.WHITE, chess.BLACK]:
        knights = board.pieces(chess.KNIGHT, color)
        for knight_sq in knights:
            knight_file = chess.square_file(knight_sq)
            knight_rank = chess.square_rank(knight_sq)

            # Determine forward direction
            forward_dir = 1 if color == chess.WHITE else -1

            # Check if knight is on an outpost (4th rank or beyond without enemy pawns controlling it)
            if (color == chess.WHITE and knight_rank >= 3) or (color == chess.BLACK and knight_rank <= 4):
                # Check if any enemy pawns control this square
                is_outpost = True
                for df in [-1, 1]:
                    f = knight_file + df
                    r = knight_rank - forward_dir
                    if 0 <= f <= 7 and 0 <= r <= 7:
                        sq = chess.square(f, r)
                        piece = board.piece_at(sq)
                        if piece and piece.piece_type == chess.PAWN and piece.color != color:
                            is_outpost = False
                            break

                if is_outpost:
                    if color == root_color:
                        total += 0.3
                    else:
                        total -= 0.3

    return total

def queen_activity(board: chess.Board, root_color: bool) -> float:
    #Evaluate queen activity based on the number of enemy pieces it attacks and friendly pieces it defends
    #for each enemy piece it attacks, add 10% the attacked piece's value points; for each friendly piece it defends, add 10% of the piece's value points
    total = 0.0
    
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    for color in [chess.WHITE, chess.BLACK]:
        queens = board.pieces(chess.QUEEN, color)
        
        for queen_sq in queens:
            attacked_squares = board.attacks(queen_sq)
            
            for sq in attacked_squares:
                piece = board.piece_at(sq)
                if not piece:
                    continue
                
                piece_value = piece_values.get(piece.piece_type, 0)
                
                if piece.color != color:
                    # Enemy piece: add 10% of its value
                    activity_bonus = 0.1 * piece_value
                    if color == root_color:
                        total += activity_bonus
                    else:
                        total -= activity_bonus
                else:
                    # Friendly piece: defend bonus (10% of its value)
                    defense_bonus = 0.1 * piece_value
                    if color == root_color:
                        total += defense_bonus
                    else:
                        total -= defense_bonus
    
    return total

def evaluate_endgame(board: chess.Board, root_color: bool) -> float:
    """
    Hàm đánh giá Cờ tàn tối ưu: 
    Kết hợp King Activity, Passed Pawn Score và King Proximity to Passed Pawns.
    """
    endgame_score = 0.0
    
    # 1. KING VS KING 
    # Khuyến khích Vua phe mình tiến về phía Vua đối phương (hoặc chiếm trung tâm)
    root_king = board.king(root_color)
    opp_king = board.king(not root_color)
    
    if root_king is not None and opp_king is not None:
        root_king_dist = abs(chess.square_file(root_king) - chess.square_file(opp_king)) + \
                         abs(chess.square_rank(root_king) - chess.square_rank(opp_king))
        opp_king_dist = abs(chess.square_file(opp_king) - chess.square_file(root_king)) + \
                        abs(chess.square_rank(opp_king) - chess.square_rank(root_king))
        
        # Nếu đang ưu thế, Vua cần tích cực.
        endgame_score += (opp_king_dist - root_king_dist) * 0.2

    # Biến lưu vị trí Tốt thông cao nhất để tính điểm King Activity sau này
    # (Rank, Square)
    most_advanced_passed_pawn = {chess.WHITE: (-1, None), chess.BLACK: (-1, None)}

    # 2. XỬ LÝ TỐT (PASSED PAWN & ADVANCEMENT)
    # Duyệt qua tất cả các quân Tốt một lần duy nhất
    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        
        # Xác định hướng tiến lên
        forward_dir = 1 if color == chess.WHITE else -1
        # Xác định hàng đích (rank 7 cho Trắng, rank 0 cho Đen)
        promotion_rank = 7 if color == chess.WHITE else 0
        
        for pawn_sq in pawns:
            file = chess.square_file(pawn_sq)
            rank = chess.square_rank(pawn_sq)
            
            # --- Logic kiểm tra Passed Pawn (Tốt thông) ---
            is_passed = True
            # Kiểm tra file hiện tại và 2 file bên cạnh xem có Tốt địch chặn đường không
            check_ranks = range(rank + 1, 8) if color == chess.WHITE else range(rank - 1, -1, -1)
            
            for df in [-1, 0, 1]:
                f = file + df
                if 0 <= f <= 7:
                    for r in check_ranks:
                        sq = chess.square(f, r)
                        piece = board.piece_at(sq)
                        # Nếu gặp Tốt địch chắn trước mặt hoặc ở cột bên cạnh -> Không phải Tốt thông
                        if piece and piece.piece_type == chess.PAWN and piece.color != color:
                            is_passed = False
                            break
                if not is_passed:
                    break
            
            # Tính điểm cho Tốt
            pawn_score = 0.0
            advancement = rank if color == chess.WHITE else (7 - rank)
            
            if is_passed:
                # A. Điểm cơ bản cho Tốt thông + Độ tiến xa
                passed_pawn_bonus = 0.2 + (advancement * 0.1) # Tăng trọng số khi tiến xa
                
                # B. Điểm kiểm soát ô phía trước (Control Bonus)
                # Kiểm tra các ô phía trước trên cùng cột
                friendly_control = False
                opp_control = False
                
                for r in check_ranks:
                    sq = chess.square(file, r)
                    if board.attackers(color, sq):
                        friendly_control = True
                    if board.attackers(not color, sq):
                        opp_control = True
                    if friendly_control and opp_control: # Chỉ cần biết có hay không
                        break
                
                if friendly_control:
                    passed_pawn_bonus += 0.15
                if opp_control:
                    passed_pawn_bonus -= 0.15
                
                pawn_score += passed_pawn_bonus
                
                # C. Cập nhật Tốt thông tiến xa nhất (cho phần King Activity)
                current_best_rank = most_advanced_passed_pawn[color][0]
                if advancement > current_best_rank: # advancement càng cao càng tốt cho cả 2 màu
                     most_advanced_passed_pawn[color] = (advancement, pawn_sq)
            else:
                # Nếu không phải Tốt thông, vẫn thưởng nhẹ nếu Tốt đã tiến sâu (để tránh thụ động)
                # Đây là phần thay thế cho logic cũ "endgame_score += advancement * 0.1"
                pawn_score += advancement * 0.05

            # Cộng điểm tổng
            if color == root_color:
                endgame_score += pawn_score
            else:
                endgame_score -= pawn_score

    # 3. KING ACTIVITY vs PASSED PAWN (Tích hợp logic từ hàm king_activity)
    # Logic: Vua phe ta nên ở gần Tốt thông phe ta (để hộ tống) VÀ gần Tốt thông phe địch (để chặn)
    
    if root_king is not None:
        king_file = chess.square_file(root_king)
        king_rank = chess.square_rank(root_king)
        
        # 3a. Hỗ trợ Tốt thông phe mình (Root Color Passed Pawn)
        best_adv, best_sq = most_advanced_passed_pawn[root_color]
        if best_sq is not None:
            pawn_file = chess.square_file(best_sq)
            pawn_rank = chess.square_rank(best_sq)
            distance = abs(king_file - pawn_file) + abs(king_rank - pawn_rank)
            # Càng gần càng tốt: Thưởng điểm
            endgame_score += (14 - distance) * 0.1

        # 3b. Chặn Tốt thông phe địch (Enemy Color Passed Pawn)
        best_adv_enemy, best_sq_enemy = most_advanced_passed_pawn[not root_color]
        if best_sq_enemy is not None:
            pawn_file = chess.square_file(best_sq_enemy)
            pawn_rank = chess.square_rank(best_sq_enemy)
            distance = abs(king_file - pawn_file) + abs(king_rank - pawn_rank)
            # Càng gần càng tốt: Thưởng điểm (để chạy tới chặn)
            endgame_score += (14 - distance) * 0.15 # Trọng số cao hơn chút vì chặn nguy hiểm quan trọng hơn

    # Tương tự tính cho Vua đối phương (để trừ điểm nếu nó làm tốt việc này)
    if opp_king is not None:
        king_file = chess.square_file(opp_king)
        king_rank = chess.square_rank(opp_king)
        
        # Opponent King supports Opponent Passed Pawn
        best_adv_enemy, best_sq_enemy = most_advanced_passed_pawn[not root_color]
        if best_sq_enemy is not None:
            pawn_file = chess.square_file(best_sq_enemy)
            pawn_rank = chess.square_rank(best_sq_enemy)
            distance = abs(king_file - pawn_file) + abs(king_rank - pawn_rank)
            endgame_score -= (14 - distance) * 0.1

        # Opponent King blocks Root Passed Pawn
        best_adv, best_sq = most_advanced_passed_pawn[root_color]
        if best_sq is not None:
            pawn_file = chess.square_file(best_sq)
            pawn_rank = chess.square_rank(best_sq)
            distance = abs(king_file - pawn_file) + abs(king_rank - pawn_rank)
            endgame_score -= (14 - distance) * 0.15

    return endgame_score