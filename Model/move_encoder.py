import bulletchess as chess
import torch

PLANES = 73

DIRECTIONS = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
]

KNIGHT_DIRS = [
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
]


def square_to_xy(sq: int) -> tuple[int, int]:
    return sq % 8, sq // 8


def xy_to_square(x: int, y: int) -> int | None:
    if x < 0 or x > 7 or y < 0 or y > 7:
        return None
    return y * 8 + x


def flip_square(sq: int) -> int:
    return sq ^ 56


def flip_move(move: chess.Move) -> chess.Move:
    return chess.Move(
        chess.SQUARES_FLIPPED[move.origin.index()],
        chess.SQUARES_FLIPPED[move.destination.index()],
        move.promotion,
    )


# ── Build MOVE_TABLE  [64][73] → (to_sq, promo) | None ────────────
MOVE_TABLE: list[list] = [[None] * PLANES for _ in range(64)]

for _from_sq in range(64):
    _fx, _fy = square_to_xy(_from_sq)

    for _d, (_dx, _dy) in enumerate(DIRECTIONS):
        for _dist in range(1, 8):
            _tx = _fx + _dx * _dist
            _ty = _fy + _dy * _dist
            _to = xy_to_square(_tx, _ty)
            if _to is not None:
                MOVE_TABLE[_from_sq][_d * 7 + _dist] = (_to, None)

    for _i, (_dx, _dy) in enumerate(KNIGHT_DIRS):
        _to = xy_to_square(_fx + _dx, _fy + _dy)
        if _to is not None:
            MOVE_TABLE[_from_sq][56 + _i] = (_to, None)

    for _direction, _dx in enumerate([-1, 0, 1]):
        for _promo_i, _promo in enumerate([chess.KNIGHT, chess.BISHOP, chess.ROOK]):
            _to = xy_to_square(_fx + _dx, _fy + 1)
            if _to is not None:
                MOVE_TABLE[_from_sq][64 + _direction * 3 + _promo_i] = (_to, _promo)

# ── Build ENCODE_TABLE  (from_sq, to_sq, promo) → (from_sq, plane) ─
ENCODE_TABLE: dict[tuple, tuple] = {}

for _from_sq in range(64):
    for _plane in range(PLANES):
        entry = MOVE_TABLE[_from_sq][_plane]
        if entry is None:
            continue
        _to_sq, _promo = entry
        ENCODE_TABLE[(_from_sq, _to_sq, _promo)] = (_from_sq, _plane)

# ── Fast encode cache: (is_black, from_sq, to_sq, promo) → action ──
# Eliminates both flip_move() calls (chess.Move allocations) in encode_move.
_FAST_ENCODE: dict[tuple, tuple] = {}

for (_from_sq, _to_sq, _promo), _action in ENCODE_TABLE.items():
    # White move: key as-is
    _FAST_ENCODE[(False, _from_sq, _to_sq, _promo)] = _action
    # Black move: the flip of this white-encoding maps to a black move
    _flip_from = flip_square(_from_sq)
    _flip_to   = flip_square(_to_sq)
    _FAST_ENCODE[(True, _flip_from, _flip_to, _promo)] = _action

# ── Fast decode cache: (is_black, from_sq, plane) → chess.Move ─────
# decode_move is called on every tree-traversal step; caching it avoids
# repeated chess.Move construction and conditional flipping.
_FAST_DECODE: dict[tuple, chess.Move | None] = {}

for _from_sq in range(64):
    for _plane in range(PLANES):
        entry = MOVE_TABLE[_from_sq][_plane]
        if entry is None:
            _FAST_DECODE[(False, _from_sq, _plane)] = None
            _FAST_DECODE[(True,  _from_sq, _plane)] = None
            continue

        _to_sq, _promo = entry

        # White move: only valid if, for promotions, destination is a back rank.
        if _promo is not None and (_to_sq // 8) not in (0, 7):
            _FAST_DECODE[(False, _from_sq, _plane)] = None
        else:
            _move_w = chess.Move(chess.SQUARES[_from_sq], chess.SQUARES[_to_sq], _promo)
            _FAST_DECODE[(False, _from_sq, _plane)] = _move_w

        # Black move (apply flip to both squares)
        # Only create a promotion move if the flipped destination is a back rank.
        # Promotion planes exist for all from-squares in MOVE_TABLE, but a black
        # pawn can only promote from rank 6 (to rank 7 after flip → rank 0).
        # For any other from-rank the flipped destination is not a back rank, so
        # the black entry is None (the plane is simply unreachable for black).
        _flip_from = _from_sq ^ 56
        _flip_to   = _to_sq ^ 56
        if _promo is not None and (_flip_to // 8) not in (0, 7):
            _FAST_DECODE[(True, _from_sq, _plane)] = None
        else:
            _move_b = chess.Move(
                chess.SQUARES_FLIPPED[_from_sq],
                chess.SQUARES_FLIPPED[_to_sq],
                _promo,
            )
            _FAST_DECODE[(True, _from_sq, _plane)] = _move_b


# ── Public API ─────────────────────────────────────────────────────

def encode_move(move: chess.Move | None, board: chess.Board) -> tuple:
    """
    Encode a chess.Move into an action tuple (from_sq, plane).
    Uses a pre-built lookup table — no chess.Move allocations at call time.
    """
    is_black = board.turn == chess.BLACK
    origin   = move.origin.index()       # type: ignore
    dest     = move.destination.index()  # type: ignore
    # Queen promotion is encoded as None (implicit auto-queen)
    promo    = None if move.promotion is chess.QUEEN else move.promotion  # type: ignore

    return _FAST_ENCODE[(is_black, origin, dest, promo)]


def decode_move(board: chess.Board, action: tuple) -> chess.Move | None:
    """
    Decode an action tuple (from_sq, plane) into a chess.Move.
    Uses a pre-built lookup table — no chess.Move allocations at call time.

    Handles the auto-queen promotion rule: a pawn reaching the back rank
    with no explicit promotion piece is promoted to queen.
    """
    from_sq, plane = action
    is_black = board.turn == chess.BLACK

    move = _FAST_DECODE.get((is_black, from_sq, plane))
    if move is None:
        return None

    # Auto-queen: pawn to back rank with no explicit promotion
    if move.promotion is None:
        piece = board[move.origin]
        if (piece is not None
                and piece.piece_type is chess.PAWN
                and move.destination in (chess.RANK_8 | chess.RANK_1)):
            # Create a new Move only in this rare case
            move = chess.Move(move.origin, move.destination, chess.QUEEN)

    return move


# ── Tests ───────────────────────────────────────────────────────────

def test_roundtrip(board: chess.Board, uci: str) -> None:
    move    = chess.Move.from_uci(uci)
    action  = encode_move(move, board)
    decoded = decode_move(board, action)
    print(f"move: {move}  action: {action}  decoded: {decoded}")
    assert move == decoded, f"Roundtrip failed: {move} != {decoded}"
    print("===========")


def roundtrip_cases() -> None:
    board = chess.Board()
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
    test_roundtrip(board, "a7a8q")

    board = chess.Board.from_fen("1nb1kbnr/pppppp1p/8/8/BP6/KN6/PPQ1PPpP/R7 b k - 0 1")
    test_roundtrip(board, "g2f1r")
    test_roundtrip(board, "g2g1r")
    test_roundtrip(board, "g2h1r")

    board = chess.Board()
    board.apply(chess.Move.from_uci("e2e4"))
    test_roundtrip(board, "e7e5")

    board = chess.Board()
    board.apply(chess.Move.from_uci("d2d4"))
    test_roundtrip(board, "d7d5")

    board = chess.Board()
    board.apply(chess.Move.from_uci("g1f3"))
    test_roundtrip(board, "g8f6")

    board = chess.Board()
    board.apply(chess.Move.from_uci("b1c3"))
    test_roundtrip(board, "b8c6")

    board = chess.Board.from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    test_roundtrip(board, "c7c5")
    test_roundtrip(board, "e7e6")
    test_roundtrip(board, "g8f6")


if __name__ == "__main__":
    roundtrip_cases()