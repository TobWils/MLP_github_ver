import numpy as np
from MLP import MLP
import time as t

class bot():
    def __init__(self):
        self.brain = MLP(64,1,"n",5)
        self.BOARD_SIZE = 8
        self.is_player = 1 # 1 coresponds to being white player, change to -1 for black
    
    def read_board(self,board):
        board = np.char.replace(board, " ", "1")
        board = np.char.replace(board, "X", "0")
        board = np.char.replace(board, "O", "2")
        return np.int8(board) - np.ones((self.BOARD_SIZE,self.BOARD_SIZE), dtype = np.int8)

    def evaluate_moves(self,board,player):
        """Return all valid moves for the player."""
        opponent = -player
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        moves = np.array([[0,0]],dtype=np.ndarray)

        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if board[r][c] != 0:
                    continue
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    found_opponent = False
                    while 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE and board[nr][nc] == opponent:
                        nr += dr
                        nc += dc
                        found_opponent = True
                    if found_opponent and 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE and board[nr][nc] == player:
                        moves = np.concat((moves,np.array([[r,c]])),axis=0)
                        break
        moves = moves[1:]
        
        scores = np.array(moves,dtype=np.ndarray)
        for i in range(len(moves)):
            temp_board = np.array(board)
            temp_board[moves[i][0]][moves[i][1]] = player
            scores[i] = self.brain.propigate_withought_softmax(temp_board.flatten())
        
        return moves[np.argmax(scores.transpose() * player)] # the *player bit is to acount for the negative values coresponding to black
    
    def play_training_game(self):
        board = self.create_board()
        current_player = 'X'
        player_idx = -1
        game = np.array([self.read_board(board)], dtype = np.ndarray)

        while True:
            # check if moves can be made
            moves = self.valid_moves(board, current_player)

            if not moves:
                if not self.valid_moves(board, 'O' if current_player == 'X' else 'X'): # stop game if no moves can be made by anyone
                    break
                current_player = 'O' if current_player == 'X' else 'X'
                player_idx = 1 if current_player == 'O' else -1
                continue

            # make moves
            if np.random.randint(0,3) == 0: # the partialy random action is to alow the bot to see more game posibilities in training
                move = moves[np.random.randint(0,len(moves))]
            else:
                move = self.evaluate_moves(self.read_board(board),player_idx)
            self.make_move(board, current_player,move[0],move[1])

            # change player
            current_player = 'O' if current_player == 'X' else 'X'
            player_idx = 1 if current_player == 'O' else -1

            game = np.concat((game, [self.read_board(board)]), axis = 0)
        
        game = game[1:]
        totalx, totalo = self.score(board)

        winner = totalo - totalx # signed diference is a good predictor of who is winning in the end
        
        for turn in game:
            self.brain.back_propigate_once_root_mean_squared_Adam(np.array(turn).flatten(),[winner])

    # Othello (Reversi) - Console Version
    # Two-player version (Black = X, White = O) Black = -1 White = 1, black gets index 0 white gets index 1

    def create_board(self):
        board: np.ndarray = np.array([[' 'for _ in range(self.BOARD_SIZE)]for _ in range(self.BOARD_SIZE)])
        mid = self.BOARD_SIZE // 2
        board[mid - 1][mid - 1] = "O"
        board[mid][mid] = "O"
        board[mid - 1][mid] = "X"
        board[mid][mid - 1] = "X"
        return board

    def print_board(self,board):
        """Display the board."""
        print("  " + " ".join(str(i) for i in range(self.BOARD_SIZE)))
        for i, row in enumerate(board):
            print(i, " ".join(str(row)))
        print()

    def valid_moves(self,board, player):
        """Return all valid moves for the player."""
        opponent = 'O' if player == 'X' else 'X'
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        moves = []

        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if board[r][c] != ' ':
                    continue
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    found_opponent = False
                    while 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE and board[nr][nc] == opponent:
                        nr += dr
                        nc += dc
                        found_opponent = True
                    if found_opponent and 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE and board[nr][nc] == player:
                        moves.append((r, c))
                        break
        return moves

    def make_move(self,board, player, row, col):
        """Place a piece and flip opponent pieces."""
        opponent = 'O' if player == 'X' else 'X'
        directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
        board[row][col] = player

        for dr, dc in directions:
            tiles_to_flip = []
            nr, nc = row + dr, col + dc
            while 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE and board[nr][nc] == opponent:
                tiles_to_flip.append((nr, nc))
                nr += dr
                nc += dc
            if 0 <= nr < self.BOARD_SIZE and 0 <= nc < self.BOARD_SIZE and board[nr][nc] == player:
                for rr, cc in tiles_to_flip:
                    board[rr][cc] = player

    def score(self,board):
        """Return the count of X and O."""
        x = np.sum(np.strings.count(board,"X"))
        o = np.sum(np.strings.count(board,"O"))
        return x, o

    def main(self):
        board = self.create_board()
        current_player = 'X'
        player_idx = -1

        while True:
            moves = self.valid_moves(board, current_player)

            if not moves:
                if not self.valid_moves(board, 'O' if current_player == 'X' else 'X'):
                        break
                print("No valid moves, skipping turn.\n")
                current_player = 'O' if current_player == 'X' else 'X'
                player_idx = 1 if current_player == 'O' else -1
                continue

            if player_idx != self.is_player:
                self.print_board(board)
                x_count, o_count = self.score(board)
                print(f"Score → X: {x_count}, O: {o_count}")
                print(f"{current_player}'s turn")


                print("Valid moves:", moves)
                try:
                    row, col = map(int, input("Enter row and column (e.g., 2 3): ").split())
                    if (row, col) not in moves:
                        print("Invalid move. Try again.\n")
                        continue
                except:
                    print("Invalid input. Use two numbers like '2 3'.\n")
                    continue

                self.make_move(board, current_player, row, col)

            elif player_idx == self.is_player: # plays both, aparent problems with white are just the bot playing as intended
                move = self.evaluate_moves(self.read_board(board),player_idx)
                self.make_move(board,current_player,move[0],move[1])

            current_player = 'O' if current_player == 'X' else 'X'
            player_idx = 1 if current_player == 'O' else -1

        self.print_board(board)
        x_count, o_count = self.score(board)
        print(f"Final Score → X: {x_count}, O: {o_count}")
        if x_count > o_count:
            print("X wins!")
        elif o_count > x_count:
            print("O wins!")
        else:
            print("It's a tie!")


test = bot()
start = t.time()
for i in range(500): # trains bot, first games seem to indicate that its pretty good or your realy bad. either way it seems to actualy play the game
    test.play_training_game()
end = t.time()
print(end - start)
print()
test.main()

