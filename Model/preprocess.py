import bulletchess as chess
import torch
import numpy as np

SHIFTS = torch.arange(64, dtype=torch.int64)
PIECES = [chess.Piece(chess.WHITE, x) for x in chess.PIECE_TYPES] + [chess.Piece(chess.BLACK, x) for x in chess.PIECE_TYPES] 

PLANES = 73

DIRECTIONS = [
    (0,1),(1,1),(1,0),(1,-1),
    (0,-1),(-1,-1),(-1,0),(-1,1)
]

KNIGHT_DIRS = [
    (1,2),(2,1),(2,-1),(1,-2),
    (-1,-2),(-2,-1),(-2,1),(-1,2)
]

def to_signed_64(x):
    return (x + (1 << 63)) % (1 << 64) - (1 << 63)

def square_to_xy(sq):
    return sq % 8, sq // 8


def xy_to_square(x,y) ->int | None:
    if x < 0 or x > 7 or y < 0 or y > 7:
        return None
    return y*8 + x

MOVE_TABLE:list = [[None]*PLANES for _ in range(64)]

for from_sq in range(64):

    fx, fy = square_to_xy(from_sq)

    # sliding moves
    for d,(dx,dy) in enumerate(DIRECTIONS):
        for dist in range(1,8):

            plane = d*7 + dist

            tx = fx + dx*dist
            ty = fy + dy*dist

            to_sq = xy_to_square(tx,ty)

            if to_sq is not None:
                MOVE_TABLE[from_sq][plane] = (to_sq, None)

    # knight moves
    for i,(dx,dy) in enumerate(KNIGHT_DIRS):

        plane = 56+i

        tx = fx + dx
        ty = fy + dy

        to_sq = xy_to_square(tx,ty)

        if to_sq is not None:
            MOVE_TABLE[from_sq][plane] = (to_sq, None)

    # promotions
    for direction,dx in enumerate([-1,0,1]):
        for promo_i,promo in enumerate([
            chess.KNIGHT,
            chess.BISHOP,
            chess.ROOK
        ]):

            plane = 64 + direction*3 + promo_i

            tx = fx + dx
            ty = fy + 1

            to_sq = xy_to_square(tx,ty)

            if to_sq is not None:
                MOVE_TABLE[from_sq][plane] = (to_sq, promo)
        plane = 64 + direction*3 + 3

ENCODE_TABLE = {}

def build_encode_table():
    for from_sq in range(64):
        for plane in range(73):

            entry = MOVE_TABLE[from_sq][plane]

            if entry is None:
                continue

            to_sq, promo = entry

            key = (from_sq, to_sq, promo)

            action = from_sq*73 + plane

            ENCODE_TABLE[key] = action

build_encode_table()

def decode_move_fast(board: chess.Board, action: int):

    from_sq = action // 73
    plane = action % 73

    entry = MOVE_TABLE[from_sq][plane]

    if entry is None:
        return None

    to_sq, promo = entry
    
    f_square = chess.SQUARES[from_sq]
    t_square = chess.SQUARES[to_sq]
    
    piece = board[f_square].piece_type  # type: ignore

    if promo == None and piece is chess.PAWN and t_square in (chess.RANK_8 | chess.RANK_1):
        promo = chess.QUEEN
    
    return chess.Move(chess.SQUARES[from_sq], chess.SQUARES[to_sq], promo) # type: ignore

def encode_move_fast(move: chess.Move | None):
    key = (move.origin.index(), move.destination.index(), None if move.promotion is chess.QUEEN else move.promotion) # type: ignore

    return ENCODE_TABLE[key]

def bitboards_to_tensor(bitboards):
    bb = torch.tensor(bitboards, dtype=torch.int64).unsqueeze(1)

    bits = ((bb >> SHIFTS) & 1).bool()

    return bits.view(-1, 8, 8)

def encode_board(board: chess.Board):
    planes = bitboards_to_tensor([to_signed_64(int(board[x])) for x in PIECES])
    return planes

def test_roundtrip(board, uci):

    move = chess.Move.from_uci(uci)

    action = encode_move_fast(move)

    decoded = decode_move_fast(board, action)

    print("move:", move)
    print("action:", action)
    print("decoded:", decoded)
    print("===========")
    assert move == decoded

def roundtrip_cases():
    board = chess.Board()
    board.apply(None)

    test_roundtrip(board, "e2e4")
    test_roundtrip(board, "d2d4")

    test_roundtrip(board, "g1f3")
    test_roundtrip(board, "b1c3")

    board.apply(chess.Move.from_uci("e2e4"))
    board.apply(chess.Move.from_uci("e7e5"))

    test_roundtrip(board, "f1c4")

    board = chess.Board.from_fen("8/8/8/8/8/8/4K3/R6k w - - 0 1")

    test_roundtrip(board, "a1a8")

    board = chess.Board.from_fen("8/8/8/8/3Q4/8/8/4k3 w - - 0 1")

    test_roundtrip(board, "d4h8")

    board = chess.Board()

    test_roundtrip(board, "e1g1")
    test_roundtrip(board, "e1c1")

    board = chess.Board.from_fen("8/P7/8/8/8/8/8/k6K w - - 0 1")

    test_roundtrip(board, "a7a8n")
    test_roundtrip(board, "a7a8b")
    test_roundtrip(board, "a7a8r")

    board = chess.Board.from_fen("8/P7/8/8/8/8/8/k6K w - - 0 1")

    test_roundtrip(board, "a7a8q")

if __name__ == "__main__":
    board = chess.Board()
    
    for move in board.legal_moves():
        test_roundtrip(board, move.uci())