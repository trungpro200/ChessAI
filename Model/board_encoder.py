import torch
import bulletchess as chess

SHIFTS = torch.arange(64, dtype=torch.int64)
PIECES = [chess.Piece(chess.WHITE, x) for x in chess.PIECE_TYPES] + [chess.Piece(chess.BLACK, x) for x in chess.PIECE_TYPES] 

def to_signed_64(x):
    return (x + (1 << 63)) % (1 << 64) - (1 << 63)

def bitboards_to_tensor(bitboards):
    pass

def encode_board(board: chess.Board):
    pass