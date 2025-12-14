import pygame
import chess
import os
from typing import Optional, Tuple, List
from best_move_4 import get_best_move_minimax
import game_state

# Initialize Pygame
pygame.init()

# Constants
BOARD_SIZE = 640
SQUARE_SIZE = BOARD_SIZE // 8
WINDOW_WIDTH = BOARD_SIZE + 200  # Extra space for UI
WINDOW_HEIGHT = BOARD_SIZE + 100

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (255, 255, 0)
SELECTED = (255, 0, 0)
LEGAL_MOVE = (0, 255, 0)

class ChessGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Interactive Chess Game")
        self.clock = pygame.time.Clock()
        
        # Game state
        self.board = chess.Board()
        self.selected_square = None
        self.player_color = chess.WHITE  # Default to white
        self.game_over = False
        self.winner = None
        self.move_count = 0  # Track number of moves
        self.player_just_moved = False  # Flag to delay AI move by one frame
        
        # Load piece images
        self.piece_images = self.load_piece_images()
        
        # Font for UI
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)
        
        # UI state
        self.show_color_selection = True
        self.show_game_over = False
        self.show_promotion = False
        self.promotion_square = None
        self.promotion_from_square = None
        self.promotion_pieces = ['Q', 'R', 'N', 'B']  # Queen, Rook, Knight, Bishop
        
    def load_piece_images(self) -> dict:
        """Load all piece images from the img folder."""
        piece_images = {}
        piece_mapping = {
            'wp': 'P', 'bp': 'p',
            'wr': 'R', 'br': 'r', 
            'wn': 'N', 'bn': 'n',
            'wb': 'B', 'bb': 'b',
            'wq': 'Q', 'bq': 'q',
            'wk': 'K', 'bk': 'k'
        }
        
        for filename, piece_char in piece_mapping.items():
            try:
                image_path = os.path.join('img', f'{filename}.png')
                if os.path.exists(image_path):
                    image = pygame.image.load(image_path)
                    # Scale image to fit square
                    piece_images[piece_char] = pygame.transform.scale(image, (SQUARE_SIZE - 10, SQUARE_SIZE - 10))
                else:
                    print(f"Warning: Could not find {image_path}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
        return piece_images
    
    def get_square_from_pos(self, pos: Tuple[int, int]) -> Optional[chess.Square]:
        """Convert screen position to chess square."""
        x, y = pos
        if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
            return None
        
        file = x // SQUARE_SIZE
        rank = 7 - (y // SQUARE_SIZE)  # Flip rank for chess notation
        
        if 0 <= file <= 7 and 0 <= rank <= 7:
            return chess.square(file, rank)
        return None
    
    def get_pos_from_square(self, square: chess.Square) -> Tuple[int, int]:
        """Convert chess square to screen position."""
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        x = file * SQUARE_SIZE
        y = (7 - rank) * SQUARE_SIZE  # Flip rank for display
        return (x, y)
    
    def draw_board(self):
        """Draw the chess board."""
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                x = file * SQUARE_SIZE
                y = (7 - rank) * SQUARE_SIZE  # Flip rank for display to match coordinate system
                
                # Alternate square colors
                color = LIGHT_SQUARE if (file + rank) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(self.screen, color, (x, y, SQUARE_SIZE, SQUARE_SIZE))
                
                # Highlight selected square
                if square == self.selected_square:
                    pygame.draw.rect(self.screen, SELECTED, (x, y, SQUARE_SIZE, SQUARE_SIZE), 3)
                
                # Highlight legal moves
                if self.selected_square is not None:
                    move = chess.Move(self.selected_square, square)
                    if move in self.board.legal_moves:
                        pygame.draw.circle(self.screen, LEGAL_MOVE, 
                                         (x + SQUARE_SIZE//2, y + SQUARE_SIZE//2), 10)
    
    def draw_pieces(self):
        """Draw all pieces on the board."""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_char = piece.symbol()
                if piece_char in self.piece_images:
                    x, y = self.get_pos_from_square(square)
                    image = self.piece_images[piece_char]
                    # Center the image in the square
                    img_x = x + (SQUARE_SIZE - image.get_width()) // 2
                    img_y = y + (SQUARE_SIZE - image.get_height()) // 2
                    self.screen.blit(image, (img_x, img_y))
    
    def draw_ui(self):
        """Draw the user interface."""
        # Draw turn indicator
        turn_text = "White's Turn" if self.board.turn == chess.WHITE else "Black's Turn"
        turn_color = WHITE if self.board.turn == chess.WHITE else BLACK
        turn_surface = self.font.render(turn_text, True, turn_color)
        self.screen.blit(turn_surface, (BOARD_SIZE + 10, 20))
        
        # Draw player color
        player_text = f"Playing as: {'White' if self.player_color == chess.WHITE else 'Black'}"
        player_surface = self.font.render(player_text, True, WHITE)
        self.screen.blit(player_surface, (BOARD_SIZE + 10, 50))
        
        # Draw game status
        if self.board.is_check():
            check_text = "CHECK!"
            check_surface = self.large_font.render(check_text, True, (255, 0, 0))
            self.screen.blit(check_surface, (BOARD_SIZE + 10, 80))
        
        # Draw restart button
        restart_rect = pygame.Rect(BOARD_SIZE + 10, 120, 150, 30)
        pygame.draw.rect(self.screen, (100, 100, 100), restart_rect)
        restart_text = self.font.render("New Game", True, WHITE)
        text_rect = restart_text.get_rect(center=restart_rect.center)
        self.screen.blit(restart_text, text_rect)
        
        # Draw color selection button
        color_rect = pygame.Rect(BOARD_SIZE + 10, 160, 150, 30)
        pygame.draw.rect(self.screen, (100, 100, 100), color_rect)
        color_text = self.font.render("Switch Color", True, WHITE)
        color_text_rect = color_text.get_rect(center=color_rect.center)
        self.screen.blit(color_text, color_text_rect)
    
    def draw_color_selection(self):
        """Draw the color selection screen."""
        self.screen.fill((50, 50, 50))
        
        title = self.large_font.render("Choose Your Color", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH//2, 200))
        self.screen.blit(title, title_rect)
        
        # White button
        white_rect = pygame.Rect(WINDOW_WIDTH//2 - 100, 300, 200, 50)
        pygame.draw.rect(self.screen, WHITE, white_rect)
        pygame.draw.rect(self.screen, BLACK, white_rect, 2)
        white_text = self.font.render("Play as White", True, BLACK)
        white_text_rect = white_text.get_rect(center=white_rect.center)
        self.screen.blit(white_text, white_text_rect)
        
        # Black button
        black_rect = pygame.Rect(WINDOW_WIDTH//2 - 100, 370, 200, 50)
        pygame.draw.rect(self.screen, BLACK, black_rect)
        pygame.draw.rect(self.screen, WHITE, black_rect, 2)
        black_text = self.font.render("Play as Black", True, WHITE)
        black_text_rect = black_text.get_rect(center=black_rect.center)
        self.screen.blit(black_text, black_text_rect)
    
    def draw_game_over(self):
        """Draw the game over screen."""
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        if self.winner:
            winner_text = f"{self.winner} Wins!"
        else:
            winner_text = "Draw!"
        
        text = self.large_font.render(winner_text, True, WHITE)
        text_rect = text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 - 50))
        self.screen.blit(text, text_rect)
        
        restart_text = self.font.render("Click anywhere to start a new game", True, WHITE)
        restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 20))
        self.screen.blit(restart_text, restart_rect)
    
    def draw_promotion_popup(self):
        """Draw the pawn promotion popup."""
        if not self.show_promotion or self.promotion_square is None:
            return
        
        # Get the position of the promotion square
        x, y = self.get_pos_from_square(self.promotion_square)
        
        # Determine if we're promoting white or black pieces
        is_white = self.board.turn == chess.WHITE
        
        # Create popup background
        popup_width = 200
        popup_height = 80
        popup_x = x - popup_width // 2
        popup_y = y - popup_height - 10  # Position above the square
        
        # Make sure popup stays within window bounds
        popup_x = max(10, min(popup_x, WINDOW_WIDTH - popup_width - 10))
        popup_y = max(10, min(popup_y, WINDOW_HEIGHT - popup_height - 10))
        
        # Draw popup background
        pygame.draw.rect(self.screen, (50, 50, 50), (popup_x, popup_y, popup_width, popup_height))
        pygame.draw.rect(self.screen, WHITE, (popup_x, popup_y, popup_width, popup_height), 2)
        
        # Draw title
        title_text = self.font.render("Choose Promotion:", True, WHITE)
        title_rect = title_text.get_rect(center=(popup_x + popup_width//2, popup_y + 15))
        self.screen.blit(title_text, title_rect)
        
        # Draw piece options
        piece_size = 40
        start_x = popup_x + 20
        start_y = popup_y + 30
        
        for i, piece_char in enumerate(self.promotion_pieces):
            piece_x = start_x + i * (piece_size + 10)
            piece_y = start_y
            
            # Create piece symbol (uppercase for white, lowercase for black)
            display_char = piece_char if is_white else piece_char.lower()
            
            # Draw piece background
            piece_rect = pygame.Rect(piece_x, piece_y, piece_size, piece_size)
            pygame.draw.rect(self.screen, LIGHT_SQUARE, piece_rect)
            pygame.draw.rect(self.screen, BLACK, piece_rect, 2)
            
            # Draw piece image if available
            if display_char in self.piece_images:
                piece_img = self.piece_images[display_char]
                # Scale to fit the popup
                scaled_img = pygame.transform.scale(piece_img, (piece_size - 8, piece_size - 8))
                img_x = piece_x + 4
                img_y = piece_y + 4
                self.screen.blit(scaled_img, (img_x, img_y))
            else:
                # Fallback: draw piece symbol
                piece_text = self.font.render(display_char, True, BLACK if is_white else WHITE)
                text_rect = piece_text.get_rect(center=(piece_x + piece_size//2, piece_y + piece_size//2))
                self.screen.blit(piece_text, text_rect)
    
    def handle_click(self, pos: Tuple[int, int]):
        """Handle mouse clicks."""
        print(f"Click detected at: {pos}, Color selection: {self.show_color_selection}, Promotion: {self.show_promotion}")
        if self.show_color_selection:
            self.handle_color_selection_click(pos)
        elif self.show_promotion:
            self.handle_promotion_click(pos)
        elif self.show_game_over:
            self.handle_game_over_click(pos)
        else:
            self.handle_game_click(pos)
    
    def handle_color_selection_click(self, pos: Tuple[int, int]):
        """Handle clicks on color selection screen."""
        x, y = pos
        # Calculate button positions based on window size
        center_x = WINDOW_WIDTH // 2
        white_y = 300
        black_y = 370
        button_width = 200
        button_height = 50
        
        # White button
        if (center_x - button_width//2 <= x <= center_x + button_width//2 and 
            white_y <= y <= white_y + button_height):
            self.player_color = chess.WHITE
            self.show_color_selection = False
            print("Selected White - Starting game...")
        # Black button  
        elif (center_x - button_width//2 <= x <= center_x + button_width//2 and 
              black_y <= y <= black_y + button_height):
            self.player_color = chess.BLACK
            self.show_color_selection = False
            print("Selected Black - Starting game...")
    
    def handle_promotion_click(self, pos: Tuple[int, int]):
        """Handle clicks on promotion popup."""
        if self.promotion_square is None:
            return
        
        x, y = pos
        is_white = self.board.turn == chess.WHITE
        
        # Get the position of the promotion square
        square_x, square_y = self.get_pos_from_square(self.promotion_square)
        
        # Calculate popup position
        popup_width = 200
        popup_height = 80
        popup_x = square_x - popup_width // 2
        popup_y = square_y - popup_height - 10
        
        # Make sure popup stays within window bounds
        popup_x = max(10, min(popup_x, WINDOW_WIDTH - popup_width - 10))
        popup_y = max(10, min(popup_y, WINDOW_HEIGHT - popup_height - 10))
        
        # Check if click is within popup area
        if (popup_x <= x <= popup_x + popup_width and 
            popup_y <= y <= popup_y + popup_height):
            
            # Check which piece was clicked
            piece_size = 40
            start_x = popup_x + 20
            start_y = popup_y + 30
            
            for i, piece_char in enumerate(self.promotion_pieces):
                piece_x = start_x + i * (piece_size + 10)
                piece_y = start_y
                
                if (piece_x <= x <= piece_x + piece_size and 
                    piece_y <= y <= piece_y + piece_size):
                    
                    # Promote the pawn
                    self.promote_pawn(piece_char)
                    break
    
    def promote_pawn(self, piece_char: str):
        """Promote the pawn to the selected piece."""
        if self.promotion_square is None or self.promotion_from_square is None:
            return
        
        try:
            # Convert piece character to piece type
            piece_type_map = {
                'Q': chess.QUEEN,
                'R': chess.ROOK,
                'N': chess.KNIGHT,
                'B': chess.BISHOP
            }
            
            piece_type = piece_type_map.get(piece_char.upper())
            if piece_type is None:
                print(f"Invalid piece type: {piece_char}")
                return
            
            # Create promotion move with the stored squares
            promotion_move = chess.Move(self.promotion_from_square, self.promotion_square, 
                                       promotion=piece_type)
            
            if promotion_move in self.board.legal_moves:
                self.board.push(promotion_move)
                self.move_count += 1
                if self.move_count == 10:
                    game_state.game_phase = 1
                self.player_just_moved = True  # Flag: draw player move before AI thinks
                print(f"Promoted pawn to {piece_char}")
            else:
                print(f"Invalid promotion move: {promotion_move}")
            
            # Close promotion popup
            self.show_promotion = False
            self.promotion_square = None
            self.promotion_from_square = None
            
            # Check for game over after promotion
            if self.board.is_game_over():
                self.game_over = True
                if self.board.is_checkmate():
                    self.winner = "Black" if self.board.turn == chess.WHITE else "White"
                else:
                    self.winner = None
                self.show_game_over = True
            
        except Exception as e:
            print(f"Error promoting pawn: {e}")
            self.show_promotion = False
            self.promotion_square = None
            self.promotion_from_square = None
    
    def handle_game_over_click(self, pos: Tuple[int, int]):
        """Handle clicks on game over screen."""
        self.reset_game()
    
    def handle_game_click(self, pos: Tuple[int, int]):
        """Handle clicks during the game."""
        # Check UI button clicks
        if BOARD_SIZE + 10 <= pos[0] <= BOARD_SIZE + 160 and 120 <= pos[1] <= 150:  # New Game
            self.reset_game()
            return
        elif BOARD_SIZE + 10 <= pos[0] <= BOARD_SIZE + 160 and 160 <= pos[1] <= 190:  # Switch Color
            self.player_color = chess.BLACK if self.player_color == chess.WHITE else chess.WHITE
            self.reset_game()
            return
        
        # Only allow moves for the current player
        if self.board.turn != self.player_color:
            return
        
        square = self.get_square_from_pos(pos)
        if square is None:
            return
        
        if self.selected_square is None:
            # Select a piece
            piece = self.board.piece_at(square)
            if piece and piece.color == self.player_color:
                self.selected_square = square
        else:
            # Check if this is a pawn promotion BEFORE making the move
            piece = self.board.piece_at(self.selected_square)
            if (piece and piece.piece_type == chess.PAWN and 
                ((piece.color == chess.WHITE and chess.square_rank(square) == 7) or
                 (piece.color == chess.BLACK and chess.square_rank(square) == 0))):
                
                # Show promotion popup
                self.show_promotion = True
                self.promotion_square = square
                self.promotion_from_square = self.selected_square
                self.selected_square = None
                print(f"Pawn promotion required! Piece: {piece}, From: {self.promotion_from_square}, To: {square}")
            else:
                # Try to make a regular move
                move = chess.Move(self.selected_square, square)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.move_count += 1
                    if self.move_count == 10:
                        game_state.game_phase = 1
                    self.selected_square = None
                    self.player_just_moved = True  # Flag: draw player move before AI thinks
                    
                    # Check for game over
                    if self.board.is_game_over():
                        self.game_over = True
                        if self.board.is_checkmate():
                            self.winner = "Black" if self.board.turn == chess.WHITE else "White"
                        else:
                            self.winner = None
                        self.show_game_over = True
                else:
                    # Try to select a different piece
                    piece = self.board.piece_at(square)
                    if piece and piece.color == self.player_color:
                        self.selected_square = square
                    else:
                        self.selected_square = None
    
    def get_best_move_minimax(self) -> chess.Move:
        """Get a move for the AI using the best_move module."""
        return get_best_move_minimax(self.board)
    
    def make_ai_move(self):
        """Make the AI's move."""
        # Skip if the player just moved (draw their move first before AI starts thinking)
        if self.player_just_moved:
            self.player_just_moved = False
            return
        
        if not self.game_over and self.board.turn != self.player_color:
            move = self.get_best_move_minimax()
            self.board.push(move)
            self.move_count += 1
            if self.move_count == 10:
                game_state.game_phase = 1
            
            # Check for game over after AI move
            if self.board.is_game_over():
                self.game_over = True
                if self.board.is_checkmate():
                    self.winner = "White" if self.board.turn == chess.BLACK else "Black"
                else:
                    self.winner = None
                self.show_game_over = True
    
    def reset_game(self):
        """Reset the game to initial state."""
        self.board = chess.Board()
        self.selected_square = None
        self.game_over = False
        self.winner = None
        self.show_color_selection = False
        self.show_game_over = False
        self.show_promotion = False
        self.player_just_moved = False
        self.promotion_square = None
        self.promotion_from_square = None
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
            
            # Make AI move if it's AI's turn
            if not self.show_color_selection and not self.show_game_over:
                self.make_ai_move()
            
            # Draw everything
            self.screen.fill((50, 50, 50))
            
            if self.show_color_selection:
                self.draw_color_selection()
            elif self.show_game_over:
                self.draw_board()
                self.draw_pieces()
                self.draw_ui()
                self.draw_game_over()
            else:
                self.draw_board()
                self.draw_pieces()
                self.draw_ui()
                if self.show_promotion:
                    self.draw_promotion_popup()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    game = ChessGame()
    game.run()
