"""File best_move_4, không có killer move heuristic, sắp xếp nước đi và quiescence search"""

import chess
import random
from typing import List, Optional
import game_state

# giá trị quân cờ
EVAL_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

INITIAL_DEPTH = 3

def get_ai_move(board: chess.Board) -> Optional[chess.Move]:
    """Return an AI move (may return None if no legal moves)."""
    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return None
    return get_best_move_minimax(board, INITIAL_DEPTH)

def get_best_move_minimax(board: chess.Board, depth: int = 3) -> Optional[chess.Move]:
    """Return best move found by minimax (alpha-beta) from the current board."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    root_color = board.turn
    
    # Lưu tất cả nước đi và điểm vào list
    move_scores = []

    for move in legal_moves:
        board.push(move)
        will_repeat = board.is_repetition(count=2)
        score = minimax(board, depth - 1, False, root_color, ply=1)
        board.pop()
        
        move_scores.append((score, move, will_repeat))

    # Sắp xếp theo điểm giảm dần
    move_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Lấy nước tốt nhất
    best_score, best_move, best_will_repeat = move_scores[0]
    
    # Nếu nước tốt nhất gây repetition VÀ có nước thứ 2
    if best_will_repeat and len(move_scores) > 1:
        second_score, second_move, second_will_repeat = move_scores[1]
        
        # Chỉ chọn nước thứ 2 nếu không tệ hơn quá 100 điểm (1 quân Tốt)
        if best_score - second_score <= 100:
            return second_move
    
    return best_move

def minimax(board: chess.Board, depth: int, maximizing: bool, root_color: bool, ply: int = 0) -> float:
    legal_moves = list(board.legal_moves)

    if depth == 0 or board.is_game_over():
        return evaluate_position(board, root_color, ply)
    
    if maximizing:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, False, root_color, ply=ply+1)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, True, root_color, ply=ply+1)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
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
        
def evaluate_position(board: chess.Board, root_color: bool, ply: int = 0) -> float:
    """Evaluate the board from the point of view of root_color (True for WHITE, False for BLACK)."""
    MATE_SCORE = 100000
    # Điều kiện kết thúc ván cờ
    if board.is_checkmate():
        # Nếu đến lượt root_color mà bị chiếu hết, thì thua
        if board.turn == root_color:
            return -MATE_SCORE + ply
        else:
            return MATE_SCORE - ply

    # điều kiện hòa
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return 0.0

    piece_values = EVAL_PIECE_VALUES

    score = 0.0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == root_color:
                score += value
            else:
                score -= value

    # Sang tàn cuộc nếu còn dưới 15 quân
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
    # Đánh giá mở màn đơn giản: ưu tiên phát triển và kiểm soát trung tâm
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

    # Bonus nhập thành: +1 nếu vua của phe đang đi đã nhập thành, -1 nếu vua của đối thủ đã nhập thành
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
    # Đánh giá trung cuộc: ưu tiên hoạt động của quân cờ và an toàn của vua
    middlegame_score = 0.0

    king_safety_score = king_safety(board, root_color)
    rook_score = rook_open_files(board, root_color)
    bishop_score_val = bishop_score(board, root_color)
    knight_score = knight_outposts(board, root_color)
    queen_score = queen_activity(board, root_color)
    
    middlegame_score = king_safety_score + rook_score + bishop_score_val + knight_score + queen_score
    
    return middlegame_score

def king_safety(board: chess.Board, root_color: bool) -> float:
    """Simple king safety: bonus if castled, penalty if not castled.
    
    Positive values favor root_color.
    """
    total = 0.0
    
    for color in [chess.WHITE, chess.BLACK]:
        king_sq = board.king(color)
        if king_sq is None:
            continue
        
        # Kiểm tra nhập thành
        is_castled = False
        if color == chess.WHITE:
            if king_sq in [chess.G1, chess.C1]:
                is_castled = True
        else:  # BLACK
            if king_sq in [chess.G8, chess.C8]:
                is_castled = True
        
        # Assign bonus/penalty
        if is_castled:
            king_score = 0.5  # Điểm thưởng nhập thành
        else:
            king_score = -0.3  # Điểm phạt không nhập thành
        
        # Thêm điểm vào tổng
        if color == root_color:
            total += king_score
        else:
            total -= king_score
    
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
                # Cột mở
                if color == root_color:
                    total += 0.5
                else:
                    total -= 0.5
            elif not has_friendly_pawn and has_enemy_pawn:
                # Cột nửa mở
                if color == root_color:
                    total += 0.2
                else:
                    total -= 0.2

    return total

def bishop_score(board: chess.Board, root_color: bool) -> float:
    #Với mỗi quân tượng của phe mình, cộng 0.1 điểm cho mỗi ô tiến lên và 0.05 điểm cho mỗi ô lùi nó kiểm soát
    #Ngược lại với quân tượng của phe địch, trừ điểm tương ứng
    total = 0.0
    for color in [chess.WHITE, chess.BLACK]:
        bishops = board.pieces(chess.BISHOP, color)
        for bishop_sq in bishops:
            bishop_file = chess.square_file(bishop_sq)
            bishop_rank = chess.square_rank(bishop_sq)

            # Xác định hướng tiến lên của quân
            forward_dir = 1 if color == chess.WHITE else -1

            # Lấy các ô bị tấn công bởi con tượng
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

            # Xác định hướng tiến lên của quân
            forward_dir = 1 if color == chess.WHITE else -1

            # Kiểm tra "outpost" của quân mã
            if (color == chess.WHITE and knight_rank >= 3) or (color == chess.BLACK and knight_rank <= 4):
                # Kiểm tra nếu không có tốt địch kiểm soát các ô phía sau
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
    #Đánh giá hoạt động của quân hậu dựa trên số lượng quân địch nó tấn công và quân mình nó bảo vệ
    #cho mỗi quân địch nó tấn công, cộng 10% giá trị của quân đó; cho mỗi quân mình nó bảo vệ, cộng 10% giá trị của quân đó
    total = 0.0
    
    piece_values = EVAL_PIECE_VALUES
    
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
                    # Cộng điểm cho quân địch bị tấn công (10% của giá trị quân đó)
                    activity_bonus = 0.1 * piece_value
                    if color == root_color:
                        total += activity_bonus
                    else:
                        total -= activity_bonus
                else:
                    # Cộng điểm cho quân mình được bảo vệ (10% của giá trị quân đó)
                    defense_bonus = 0.1 * piece_value
                    if color == root_color:
                        total += defense_bonus
                    else:
                        total -= defense_bonus
    
    return total

def evaluate_endgame(board: chess.Board, root_color: bool) -> float:
    """
    Hàm đánh giá Cờ tàn tối ưu: 
    Kết hợp King Activity, Passed Pawn Score, King Proximity to Passed Pawns
    VÀ Logic dồn Vua đối phương vào góc + BAO VÂY để chiếu hết.
    """
    endgame_score = 0.0
    
    root_king = board.king(root_color)
    opp_king = board.king(not root_color)

    # Xây dựng tập hợp các ô bị kiểm soát bởi mỗi bên
    attacks_union = {chess.WHITE: set(), chess.BLACK: set()}
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if not p:
            continue
        attacks_union[p.color].update(board.attacks(sq))

    # --- 1. MOP-UP EVALUATION (DỒN VUA VÀO GÓC) ---
    piece_vals_small = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}
    my_material = 0
    opp_material = 0
    for sq, p in board.piece_map().items():
        if p.color == root_color:
            my_material += piece_vals_small.get(p.piece_type, 0)
        else:
            opp_material += piece_vals_small.get(p.piece_type, 0)
    material_advantage = my_material - opp_material

    if opp_king is not None and root_king is not None:
        # Khoảng cách từ vua đến trung tâm bàn cờ (e4,d4,e5,d5)
        root_center_dist = abs(chess.square_file(root_king) - 3.5) + abs(chess.square_rank(root_king) - 3.5)
        opp_center_dist = abs(chess.square_file(opp_king) - 3.5) + abs(chess.square_rank(opp_king) - 3.5)

        # Cộng điểm nếu vua mình gần trung tâm hơn vua đối phương
        endgame_score += (opp_center_dist - root_center_dist) * 0.1

        # Nếu chúng ta có lợi thế vật chất rõ rệt, tăng động lực đẩy đối thủ vào góc
        MOPUP_MATERIAL_THRESH = 3
        if material_advantage >= MOPUP_MATERIAL_THRESH:
            endgame_score += (opp_center_dist - root_center_dist) * 0.2

        # Th
        dist_between_kings = abs(chess.square_file(root_king) - chess.square_file(opp_king)) + \
                             abs(chess.square_rank(root_king) - chess.square_rank(opp_king))
        endgame_score += (14 - dist_between_kings) * 0.05

        # --- NEW: BAO VÂY VUA ĐỐI PHƯƠNG KHI ĐÃ DỒN VÀO GÓC ---
        # Chỉ áp dụng khi vua đối phương đã ở xa trung tâm (đã bị dồn vào góc)
        CORNER_THRESHOLD = 4.5  # Khoảng cách từ center, nếu > 4.5 là gần góc
        if opp_center_dist >= CORNER_THRESHOLD and material_advantage >= MOPUP_MATERIAL_THRESH:
            # Lấy tất cả ô adjacent với vua đối phương
            opp_king_file = chess.square_file(opp_king)
            opp_king_rank = chess.square_rank(opp_king)
            
            adjacent_squares = []
            for df in [-1, 0, 1]:
                for dr in [-1, 0, 1]:
                    if df == 0 and dr == 0:  # Skip vua itself
                        continue
                    f = opp_king_file + df
                    r = opp_king_rank + dr
                    if 0 <= f <= 7 and 0 <= r <= 7:
                        adjacent_squares.append(chess.square(f, r))
            
            # Đếm bao nhiêu ô adjacent được kiểm soát bởi quân của mình
            controlled_count = 0
            controlling_pieces = set()  # Xem những quân nào kiểm soát ô gần vua đối phương
            
            for adj_sq in adjacent_squares:
                if adj_sq in attacks_union[root_color]:
                    controlled_count += 1
                    
                    # Tìm quân nào đang kiểm soát ô này
                    for sq in chess.SQUARES:
                        p = board.piece_at(sq)
                        if p and p.color == root_color and adj_sq in board.attacks(sq):
                            controlling_pieces.add(sq)
            
            # Thưởng điểm cho việc kiểm soát nhiều ô xung quanh vua đối phương
            endgame_score += controlled_count * 0.3
            
            # Thưởng điểm ĐẶC BIỆT nếu có ít nhất 2 quân khác nhau tham gia bao vây
            if len(controlling_pieces) >= 2:
                endgame_score += 1.5  # Bonus lớn khi có coordination
            
            # Thưởng điểm thêm nếu kiểm soát hầu hết các ô (gần chiếu hết)
            if controlled_count >= 5:  # Vua có tối đa 8 ô adjacent
                endgame_score += 2.0  # Gần chiếu hết rồi!

    # --- 2. LOGIC CŨ: KING VS KING ---
    if root_king is not None and opp_king is not None:
        root_king_dist = abs(chess.square_file(root_king) - chess.square_file(opp_king)) + \
                         abs(chess.square_rank(root_king) - chess.square_rank(opp_king))
        opp_king_dist = abs(chess.square_file(opp_king) - chess.square_file(root_king)) + \
                        abs(chess.square_rank(opp_king) - chess.square_rank(root_king))
        
        endgame_score += (opp_king_dist - root_king_dist) * 0.1
        
    # --- 3. PASSED PAWN EVALUATION (TỐT THÔNG) ---
    most_advanced_passed_pawn = {chess.WHITE: (-1, None), chess.BLACK: (-1, None)}

    for color in [chess.WHITE, chess.BLACK]:
        pawns = board.pieces(chess.PAWN, color)
        
        for pawn_sq in pawns:
            file = chess.square_file(pawn_sq)
            rank = chess.square_rank(pawn_sq)
            
            is_passed = True
            check_ranks = range(rank + 1, 8) if color == chess.WHITE else range(rank - 1, -1, -1)
            
            for df in [-1, 0, 1]:
                f = file + df
                if 0 <= f <= 7:
                    for r in check_ranks:
                        sq = chess.square(f, r)
                        piece = board.piece_at(sq)
                        if piece and piece.piece_type == chess.PAWN and piece.color != color:
                            is_passed = False
                            break
                if not is_passed:
                    break
            
            pawn_score = 0.0
            advancement = rank if color == chess.WHITE else (7 - rank)
            
            if is_passed:
                passed_pawn_bonus = 0.2 + (advancement * 0.1)
                friendly_control = False
                opp_control = False
                for r in check_ranks:
                    if 0 <= r <= 7:
                        target_sq = chess.square(file, r)
                        if target_sq in attacks_union[color]:
                            friendly_control = True
                        if target_sq in attacks_union[not color]:
                            opp_control = True
                        if friendly_control and opp_control:
                            break
                
                if friendly_control: passed_pawn_bonus += 0.15
                if opp_control: passed_pawn_bonus -= 0.15
                
                pawn_score += passed_pawn_bonus
                
                current_best_rank = most_advanced_passed_pawn[color][0]
                if advancement > current_best_rank:
                     most_advanced_passed_pawn[color] = (advancement, pawn_sq)
            else:
                pawn_score += advancement * 0.05

            if color == root_color: endgame_score += pawn_score
            else: endgame_score -= pawn_score

    if root_king is not None:
        king_file = chess.square_file(root_king)
        king_rank = chess.square_rank(root_king)
        
        # 3a. Hỗ trợ Tốt thông phe mình
        _, best_sq = most_advanced_passed_pawn[root_color]
        if best_sq is not None:
            pawn_file = chess.square_file(best_sq)
            pawn_rank = chess.square_rank(best_sq)
            distance = abs(king_file - pawn_file) + abs(king_rank - pawn_rank)
            endgame_score += (14 - distance) * 0.1

        # 3b. Chặn Tốt thông phe địch
        _, best_sq_enemy = most_advanced_passed_pawn[not root_color]
        if best_sq_enemy is not None:
            pawn_file = chess.square_file(best_sq_enemy)
            pawn_rank = chess.square_rank(best_sq_enemy)
            distance = abs(king_file - pawn_file) + abs(king_rank - pawn_rank)
            endgame_score += (14 - distance) * 0.15

    if opp_king is not None:
        king_file = chess.square_file(opp_king)
        king_rank = chess.square_rank(opp_king)
        
        # Vua đối phương hỗ trợ tốt thông đối phương
        _, best_sq_enemy = most_advanced_passed_pawn[not root_color]
        if best_sq_enemy is not None:
            pawn_file = chess.square_file(best_sq_enemy)
            pawn_rank = chess.square_rank(best_sq_enemy)
            distance = abs(king_file - pawn_file) + abs(king_rank - pawn_rank)
            endgame_score -= (14 - distance) * 0.1

        # Vua đối phương chặn tốt thông của mình
        _, best_sq = most_advanced_passed_pawn[root_color]
        if best_sq is not None:
            pawn_file = chess.square_file(best_sq)
            pawn_rank = chess.square_rank(best_sq)
            distance = abs(king_file - pawn_file) + abs(king_rank - pawn_rank)
            endgame_score -= (14 - distance) * 0.15

    return endgame_score