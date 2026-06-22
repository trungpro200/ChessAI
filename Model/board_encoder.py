import torch
import bulletchess as chess
from .move_encoder import encode_move, decode_move
from collections import deque

SHIFTS = torch.arange(64, dtype=torch.int64)
PIECES = (
    [chess.Piece(chess.WHITE, x) for x in chess.PIECE_TYPES] +
    [chess.Piece(chess.BLACK, x) for x in chess.PIECE_TYPES]
)
PIECES_INT_DICT: dict[chess.Piece, int] = {p: i for i, p in enumerate(PIECES)}  # type: ignore

HISTORY_LEN = 8   # must match MCTS/src/encoder.rs


class State:
    def __init__(self, board: chess.Board, init_board: bool = True) -> None:
        self.board          = board
        self.history_planes = torch.zeros(64, 12, dtype=torch.float32)  # read-only zero fallback
        self.meta_planes    = torch.zeros(64, 7,  dtype=torch.float32)
        self.pos_cache: dict[int, torch.Tensor] = {}  # hash → [64, 12]
        self.move_stack: deque[int] = deque()

        if init_board:
            self.encode_board_init(board)

    # ── Utilities ──────────────────────────────────────────────────
    @staticmethod
    def to_signed_64(x: int) -> int:
        return (x + (1 << 63)) % (1 << 64) - (1 << 63)

    def bitboards_to_tensor(self, bitboards: list[int]) -> torch.Tensor:
        """Convert 12 piece bitboards → [64, 12] bool tensor."""
        bb   = torch.tensor(bitboards, dtype=torch.int64)   # [12]
        bits = ((bb[:, None] >> SHIFTS) & 1).to(torch.bool) # [12, 64]
        return bits.transpose(0, 1).to(torch.float32)        # [64, 12]

    # ── Metadata ───────────────────────────────────────────────────
    def update_metadata(self) -> None:
        """Write current board's meta features in-place into self.meta_planes."""
        board = self.board
        meta  = self.meta_planes

        meta[:, 0] = 1.0 if board.turn == chess.WHITE else 0.0

        cr = board.castling_rights
        meta[:, 1] = float(cr.kingside(chess.WHITE))
        meta[:, 2] = float(cr.queenside(chess.WHITE))
        meta[:, 3] = float(cr.kingside(chess.BLACK))
        meta[:, 4] = float(cr.queenside(chess.BLACK))

        meta[:, 5] = 0.0
        ep = board.en_passant_square
        if ep is not None:
            meta[ep.index(), 5] = 1.0  # type: ignore

        meta[:, 6] = board.halfmove_clock * 0.01

    # ── Initialisation ─────────────────────────────────────────────
    def encode_board_init(self, board: chess.Board) -> None:
        """Cache the starting position's piece tensor."""
        bitboards = [self.to_signed_64(int(board[p])) for p in PIECES]
        self.pos_cache[board.__hash__()] = self.bitboards_to_tensor(bitboards)

    # ── Move application ───────────────────────────────────────────
    def make_move(self, action: tuple) -> torch.Tensor | None:
        board  = self.board
        move: chess.Move = decode_move(board, action)  # type: ignore

        origin = move.origin.index()
        dest   = move.destination.index()
        piece: chess.Piece = board[move.origin]  # type: ignore
        ep     = board.en_passant_square
        turn   = board.turn

        prev_hash = board.__hash__()
        prev      = self.pos_cache[prev_hash]   # [64, 12]

        board.apply(move)
        curr_hash = board.__hash__()
        self.move_stack.append(curr_hash)

        if curr_hash in self.pos_cache:     # already encoded (transposition / revisit)
            return None

        curr = prev.clone()

        # En-passant capture
        if ep is not None and piece.piece_type == chess.PAWN and dest == ep.index():
            captured_sq = dest - 8 if turn == chess.WHITE else dest + 8
            curr[captured_sq] = 0.0

        # Move piece (overwrites captured piece naturally)
        curr[dest]   = prev[origin]
        curr[origin] = 0.0

        if move.promotion:
            curr[dest] = 0.0
            idx = PIECES_INT_DICT[chess.Piece(board.turn, move.promotion)]
            curr[dest, idx] = 1.0
        elif move.is_castling(board):
            if dest > origin:   # kingside
                rook_from, rook_to = origin + 3, origin + 1
            else:               # queenside
                rook_from, rook_to = origin - 4, origin - 1
            curr[rook_to]   = prev[rook_from]
            curr[rook_from] = 0.0

        self.pos_cache[curr_hash] = curr
        return curr

    def unmake_move(self) -> None:
        self.board.undo()
        self.move_stack.pop()

    # ── Token tensor ───────────────────────────────────────────────
    @property
    def tokens(self) -> torch.Tensor:
        """
        Return  [1, 64, 103]  — 8 history frames (newest first, 12 planes each)
        followed by 7 meta planes per square.
        """
        self.update_metadata()

        frames: list[torch.Tensor] = []
        stack = self.move_stack
        curr_hash = self.board.__hash__()
        cache = self.pos_cache

        for k in range(HISTORY_LEN):
            if k == 0:
                h = curr_hash
            elif len(stack) >= k + 1:
                h = stack[-(k + 1)]
            else:
                h = None

            frames.append(cache[h] if (h is not None and h in cache) else self.history_planes)

        return torch.cat((*frames, self.meta_planes), dim=1).unsqueeze(0)