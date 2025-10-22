import numpy as np
from MLP import MLP

class bot():
    def __init__(self):
        self.board: np.ndarray = np.zeros((8,8),dtype=np.int8)
        self.board[3][3] = 1
        self.board[3][4] = -1
        self.board[4][3] = -1
        self.board[4][4] = 1

        self.player = -1
        self.peices_left = 30

    def player_peice_locations(self, val):
        return np.array(np.where(self.board == val)).transpose()
    
    def is_pos_inline(self, x, y, pval):
        peices = np.array(np.where(self.board == pval)).transpose()
        for peice in peices:
            if x == peice[0] or y == peice[0]:
                return True
            if x-peice[0] == y-peice[1] or x-peice[0] == -y+peice[1]:
                return True
        return False
    
    def is_line(self, x1, y1, x2, y2, board, player):
        m = (y2-y1)/(x2-x1)
        b = m*x1 - y1
        if x1 > x2:
            temp = x1
            x1 = x2
            x2 = temp
            temp = y1
            y1 = y2
            y2 = temp
        for x in range(x1+1,x2):
            if board[x,int(m*x+b)] != -player:
                return False
        return True


    def find_moves(self):
        moves = np.array([])
        for x in range(8):
            for y in range(8):
                move = np.array(self.board)
                if self.board[x][y] == 0 and ((self.board[x+1][y+1] == -self.player or self.board[x+1][y] == -self.player or self.board[x+1][y-1] == -self.player) or (self.board[x][y+1] == -self.player or self.board[x][y-1] == -self.player) or (self.board[x-1][y+1] == -self.player or self.board[x-1][y] == -self.player or self.board[x-1][y-1] == -self.player)):
                    if self.is_pos_inline(x,y,-self.player):
                        for pos in self.player_peice_locations(-self.player):
                            if self.is_line(x,y,pos[0],pos[1],self.board,-self.player):
                                pass