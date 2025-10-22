import numpy as np
from MLP import MLP

class bot():
    def __init__(self):
        self.board: np.ndarray = np.zeros((9,9))
        