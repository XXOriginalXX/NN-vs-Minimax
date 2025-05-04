import chess
import random
import numpy as np
import time
import os

class AdvancedNeuralNetwork:
    def __init__(self, input_size=768, hidden_size1=256, hidden_size2=128, output_size=1):
       
        np.random.seed(int(time.time()))
        
        self.W1 = np.random.randn(input_size, hidden_size1) * 0.01
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * 0.01
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))
        
        self._initialize_smart_weights()
    
    def _initialize_smart_weights(self):
        center_indices = self._get_center_square_indices()
        for idx in center_indices:
            self.W1[idx, :] *= 1.5
        
        queen_indices = self._get_piece_type_indices(chess.QUEEN)
        rook_indices = self._get_piece_type_indices(chess.ROOK)
        for idx in queen_indices:
            self.W1[idx, :] *= 2.0
        for idx in rook_indices:
            self.W1[idx, :] *= 1.8
    
    def _get_center_square_indices(self):
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        indices = []
        for piece_type in range(1, 7):
            for color in [True, False]:
                base_idx = ((piece_type - 1) * 2 + (0 if color else 1)) * 64
                for square in center_squares:
                    indices.append(base_idx + square)
        return indices
    
    def _get_piece_type_indices(self, piece_type):
        indices = []
        piece_offset = (piece_type - 1) * 128
        for color in [0, 1]:
            color_offset = color * 64
            for square in range(64):
                indices.append(piece_offset + color_offset + square)
        return indices
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = np.maximum(0, self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        return self.z3
    
    def evaluate_board(self, board):
        X = self._board_to_features(board)
        base_score = self.forward(X)[0][0]
        positional_score = self._positional_evaluation(board)
        tactical_score = self._tactical_evaluation(board)
        random_noise = np.random.normal(0, 5.0)
        return base_score + positional_score + tactical_score + random_noise
    
    def _board_to_features(self, board):
        features = np.zeros((1, 768))
        feature_idx = 0
        
        for piece_type in range(1, 7):
            for color in [True, False]:
                for square in chess.SQUARES:
                    piece = board.piece_at(square)
                    if piece and piece.piece_type == piece_type and piece.color == color:
                        features[0, feature_idx] = 1
                    feature_idx += 1
        
        return features
    
    def _positional_evaluation(self, board):
        score = 0
        
        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                if piece.color == chess.BLACK:
                    score += 50
                else:
                    score -= 50
        
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        
        for square in black_pawns:
            rank = chess.square_rank(square)
            score += (7 - rank) * 10
        
        for square in white_pawns:
            rank = chess.square_rank(square)
            score -= rank * 10
            
        black_king_square = board.king(chess.BLACK)
        white_king_square = board.king(chess.WHITE)
        
        if black_king_square:
            black_king_rank = chess.square_rank(black_king_square)
            if board.fullmove_number < 15:
                score += max(0, black_king_rank - 5) * 30
        
        if white_king_square:
            white_king_rank = chess.square_rank(white_king_square)
            if board.fullmove_number < 15:
                score -= max(0, 2 - white_king_rank) * 30
                
        return score
    
    def _tactical_evaluation(self, board):
        score = 0
        
        if board.is_check():
            if board.turn == chess.BLACK:
                score += 100
            else:
                score -= 100
        
        current_turn = board.turn
        num_moves = len(list(board.legal_moves))
        
        board.turn = not board.turn
        opponent_moves = len(list(board.legal_moves))
        board.turn = current_turn
        
        if current_turn == chess.WHITE:
            score -= (num_moves - opponent_moves) * 5
        else:
            score += (opponent_moves - num_moves) * 5
            
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue
                
            if piece.color == chess.BLACK:
                if piece.piece_type == chess.KNIGHT:
                    score += len(list(board.attacks(square))) * 5
                elif piece.piece_type == chess.BISHOP:
                    score += len(list(board.attacks(square))) * 5
                elif piece.piece_type == chess.ROOK:
                    score += len(list(board.attacks(square))) * 4
                elif piece.piece_type == chess.QUEEN:
                    score += len(list(board.attacks(square))) * 2
            else:
                if piece.piece_type == chess.KNIGHT:
                    score -= len(list(board.attacks(square))) * 5
                elif piece.piece_type == chess.BISHOP:
                    score -= len(list(board.attacks(square))) * 5
                elif piece.piece_type == chess.ROOK:
                    score -= len(list(board.attacks(square))) * 4
                elif piece.piece_type == chess.QUEEN:
                    score -= len(list(board.attacks(square))) * 2
        
        return score

class MinimaxPlayer:
    def __init__(self, depth=3):
        self.depth = depth
        self.nodes_evaluated = 0
        self.team_name = "Team Minimax"
        
    def evaluate_board(self, board):
        if board.is_checkmate():
            return -10000 if board.turn else 10000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        material = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                material += value
            else:
                material -= value
                
        board.push(chess.Move.null())
        black_mobility = len(list(board.legal_moves))
        board.pop()
        white_mobility = len(list(board.legal_moves))
        mobility = white_mobility - black_mobility
        random_noise = random.uniform(-10, 10)
        
        evaluation = material + mobility * 10 + random_noise
        
        return evaluation
    
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        self.nodes_evaluated += 1
        
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def get_best_move(self, board):
        self.nodes_evaluated = 0
        best_moves = []
        best_value = float('-inf') if board.turn else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            value = self.minimax(board, self.depth - 1, alpha, beta, not board.turn)
            board.pop()
            value_with_noise = value + random.uniform(-5, 5)
            
            if board.turn:
                if value_with_noise > best_value:
                    best_value = value_with_noise
                    best_moves = [move]
                    alpha = max(alpha, value)
                elif abs(value_with_noise - best_value) < 10: 
                    best_moves.append(move)
            else:
                if value_with_noise < best_value:
                    best_value = value_with_noise
                    best_moves = [move]
                    beta = min(beta, value)
                elif abs(value_with_noise - best_value) < 10: 
                    best_moves.append(move)
        
        if best_moves:
            chosen_move = random.choice(best_moves)
        else:
            chosen_move = random.choice(list(board.legal_moves))
        
        return chosen_move, best_value

class NeuralNetworkPlayer:
    def __init__(self, depth=2):
        self.model = AdvancedNeuralNetwork()
        self.depth = depth
        self.team_name = "Team Neural Network"
        
    def evaluate_position(self, board):
        return self.model.evaluate_board(board)
    
    def minimax_search(self, board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_game_over():
            return self.evaluate_position(board)
        
        if maximizing_player:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax_search(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax_search(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
        
    def get_best_move(self, board):
        best_moves = []
        maximizing = not board.turn
        best_value = float('-inf') if maximizing else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        
        legal_moves = list(board.legal_moves)
        random.shuffle(legal_moves)
        
        for move in legal_moves:
            board.push(move)
            
            if self.depth > 0:
                value = self.minimax_search(board, self.depth - 1, alpha, beta, not maximizing)
            else:
                value = self.evaluate_position(board)
                
            board.pop()
            value_with_noise = value + random.uniform(-5, 5)
            
            if maximizing:
                if value_with_noise > best_value:
                    best_value = value_with_noise
                    best_moves = [move]
                    alpha = max(alpha, value)
                elif abs(value_with_noise - best_value) < 10:  
                    best_moves.append(move)
            else:
                if value_with_noise < best_value:
                    best_value = value_with_noise
                    best_moves = [move]
                    beta = min(beta, value)
                elif abs(value_with_noise - best_value) < 10:  
                    best_moves.append(move)
        if best_moves:
            chosen_move = random.choice(best_moves)
        else:
            chosen_move = random.choice(list(board.legal_moves))
                
        return chosen_move, best_value

def display_board(board):
    unicode_pieces = {
        'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
        'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
        '.': '·'
    }
    
    board_str = str(board)
    
    for piece in unicode_pieces:
        board_str = board_str.replace(piece, unicode_pieces[piece])
    
    col_labels = '  a b c d e f g h  '
    
    rows = board_str.split('\n')
    bordered_board = col_labels + '\n'
    for i, row in enumerate(rows):
        bordered_board += f"{8-i} {row} {8-i}\n"
    bordered_board += col_labels
    
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("\n" + "=" * 40)
    print(f"{'TEAM MINIMAX (WHITE)':^40}")
    print("=" * 40)
    
    print('\n' + bordered_board + '\n')
    
    print("=" * 40)
    print(f"{'TEAM NEURAL NETWORK (BLACK)':^40}")
    print("=" * 40 + "\n")

def play_game(minimax_depth=3, nn_depth=2, max_moves=100):
    board = chess.Board()
    random.seed(int(time.time()))
    
    white_player = MinimaxPlayer(depth=minimax_depth)
    black_player = NeuralNetworkPlayer(depth=nn_depth)
    
    move_count = 0
    white_eval_history = []
    black_eval_history = []
    
    print("\n===== CHESS AI BATTLE: TEAM MINIMAX VS TEAM NEURAL NETWORK =====\n")
    print(f"White: {white_player.team_name} (depth={minimax_depth})")
    print(f"Black: {black_player.team_name} (depth={nn_depth})")
    display_board(board)
    
    while not board.is_game_over() and move_count < max_moves:
        start_time = time.time()
        
        if board.turn:
            move, eval_score = white_player.get_best_move(board)
            player_name = white_player.team_name
            white_eval_history.append(eval_score)
        else:
            move, eval_score = black_player.get_best_move(board)
            player_name = black_player.team_name
            black_eval_history.append(eval_score)
        
        board.push(move)
        move_count += 1
        
        elapsed_time = time.time() - start_time
        
        print(f"\nMove {move_count}: {player_name} played {move} in {elapsed_time:.2f} seconds")
        display_board(board)
        
        minimax_eval = white_player.evaluate_board(board)
        nn_eval = black_player.evaluate_position(board)
        print(f"{white_player.team_name} evaluation: {minimax_eval}")
        print(f"{black_player.team_name} evaluation: {nn_eval}")
        
        if minimax_eval > 500:
            prediction = f"{white_player.team_name} is likely to win"
        elif minimax_eval < -500:
            prediction = f"{black_player.team_name} is likely to win"
        else:
            prediction = "Game is currently balanced"
            
        print(f"Prediction: {prediction}")
        
        if move_count > 1:
            print("\nMove history:")
            history_start = max(0, move_count - 5)
            for i in range(history_start, move_count):
                move_index = i % 2
                move_number = (i // 2) + 1
                move_player = white_player.team_name if move_index == 0 else black_player.team_name
                move_str = board.move_stack[i].uci()
                if move_index == 0:
                    print(f"{move_number}. {move_str}", end=" ")
                else:
                    print(f"{move_str}")
            if move_count % 2 == 1:
                print()
        
        time.sleep(1)
    
    print("\n===== GAME OVER =====")
    display_board(board)
    
    if board.is_checkmate():
        winner = black_player.team_name if board.turn else white_player.team_name
        print(f"{winner} wins by checkmate!")
    elif board.is_stalemate():
        print("Game ended in stalemate!")
    elif board.is_insufficient_material():
        print("Game ended due to insufficient material!")
    elif board.is_fifty_moves():
        print("Game ended due to fifty-move rule!")
    elif board.is_repetition():
        print("Game ended due to threefold repetition!")
    else:
        print("Game ended due to move limit!")
    
    print("\nFinal Stats:")
    print(f"Total moves: {move_count}")
    print(f"Final evaluation ({white_player.team_name}): {minimax_eval}")
    print(f"Final evaluation ({black_player.team_name}): {nn_eval}")
    
    print("\nComplete move history:")
    for i in range(move_count):
        move_index = i % 2
        move_number = (i // 2) + 1
        move_player = white_player.team_name if move_index == 0 else black_player.team_name
        move_str = board.move_stack[i].uci()
        print(f"Move {i+1}: {move_player} played {move_str}")

if __name__ == "__main__":
    # Use time-based random seed for initial randomization
    random.seed(int(time.time()))
    np.random.seed(int(time.time()))
    
    try:
        minimax_depth = int(input("Enter depth for Team Minimax (I would recommend 1-4): ") or "3")
        if minimax_depth < 1 or minimax_depth > 10:
            print("Invalid depth. Using default depth of 3.")
            minimax_depth = 3
    except ValueError:
        print("Invalid input. Using default depth of 3.")
        minimax_depth = 3
    
    try:
        nn_depth = int(input("Enter depth for Team Neural Network (I would recommend 1-3,): ") or "2")
        if nn_depth < 0 or nn_depth > 3:
            print("Invalid depth. Using default depth of 2.")
            nn_depth = 2
    except ValueError:
        print("Invalid input. Using default depth of 2.")
        nn_depth = 2
    
    try:
        max_moves = int(input("Enter maximum number of moves (default is 50): ") or "50")
        if max_moves < 1:
            print("Invalid number of moves. Using default of 50.")
            max_moves = 50
    except ValueError:
        print("Invalid input. Using default of 50 moves.")
        max_moves = 50
    
    play_game(minimax_depth=minimax_depth, nn_depth=nn_depth, max_moves=max_moves)