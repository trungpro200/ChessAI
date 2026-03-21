import torch
import bulletchess as chess
from .move_encoder import encode_move, decode_move

SHIFTS = torch.arange(64, dtype=torch.int64)
PIECES = [chess.Piece(chess.WHITE, x) for x in chess.PIECE_TYPES] + [chess.Piece(chess.BLACK, x) for x in chess.PIECE_TYPES] 
PIECES_INT_DICT: dict[chess.Piece] = dict([reversed(x) for x in enumerate(PIECES)])  # type: ignore


class Tokens:
    def __init__(self, board: chess.Board) -> None:
        self.board = board
        self.head = 0
        
        self.history_planes = torch.zeros(8, 64, 12, dtype=torch.float32)
        self.meta_planes = torch.zeros(64,7)
        
        self.encode_board_init(board)
    
    @staticmethod
    def to_signed_64(x):
        return (x + (1 << 63)) % (1 << 64) - (1 << 63)

    def bitboards_to_tensor(self, bitboards: list[int]):
        bb = torch.tensor(bitboards, dtype=torch.int64)  # [12]
        bits = ((bb[:, None] >> SHIFTS) & 1).to(torch.float32)  # [12,64]
        return bits.transpose(0, 1)  # [64,12]

    def update_metadata(self):
        board = self.board
        meta = self.meta_planes
        
        #Turn plane - 1 layer
        meta[:, 0] = 1 if board.turn == chess.WHITE else 0
        
        # Castling planes - 4 layers
        sides = "KQkq"
        
        meta[:, 1] = int(board.castling_rights.kingside(chess.WHITE))
        meta[:, 2] = int(board.castling_rights.queenside(chess.WHITE))
        meta[:, 3] = int(board.castling_rights.kingside(chess.BLACK))
        meta[:, 4] = int(board.castling_rights.queenside(chess.BLACK))
        
        # En passant - 1 layer
        meta[:, 5] = 0
        if board.en_passant_square is not None: # The square is not an int it's chess.Square
            meta[board.en_passant_square.index(), 5] = 1 # type: ignore
        
        # Repetition - 1 layer
        meta[:, 6] = board.halfmove_clock*0.01 # This is good enough

    def encode_board_init(self, board: chess.Board): # -> [64, 103]
        """## Tokenization
        
        Encode board into 64 tokens [64,103]
        
        ### Planes:
            0-95: positions
            96: turn
            97-100: castling rights
            101: en-passant
            102: repetition
        """
        self.history_planes[0] = self.bitboards_to_tensor([self.to_signed_64(int(board[x])) for x in PIECES])
        self.update_metadata()
        

    def encode_board_propagate(self, action):
        h_plane = self.history_planes
        board = self.board

        move: chess.Move = decode_move(board, action) #type: ignore

        origin = move.origin.index()
        dest = move.destination.index()

        piece: chess.Piece = board[move.origin] #type:ignore
        ep = board.en_passant_square
        turn = board.turn

        # Rotate head 
        new_head = (self.head - 1) % 8
        prev_head = self.head

        curr = h_plane[new_head]
        prev = h_plane[prev_head]

        # Copy previous board
        curr.copy_(prev)

        # Handle en-passant capture 
        if ep and piece.piece_type == chess.PAWN and dest == ep.index():
            captured_sq = dest - 8 if turn == chess.WHITE else dest + 8
            curr[captured_sq] = 0
               
        # Move piece 
        curr[dest] = prev[origin] # This overwritten captured piece by nature
        curr[origin] = 0
        
        if move.promotion:
            curr[dest] = 0
            idx = PIECES_INT_DICT[chess.Piece(board.turn, move.promotion)]
            curr[dest, idx] = 1
        elif piece.piece_type == chess.KING and abs(dest - origin) == 2:
            if dest > origin:  # kingside
                rook_from = origin + 3
                rook_to = origin + 1
            else:  # queenside
                rook_from = origin - 4
                rook_to = origin - 1

            curr[rook_to] = prev[rook_from]
            curr[rook_from] = 0

        # Apply move to board 
        board.apply(move)

        # Update head 
        self.head = new_head

        self.update_metadata()

        return h_plane
    
    @property
    def tokens(self):
        pos = self.history_planes.roll(-self.head, dims=0)
        pos = pos.reshape(64, -1)
        return torch.cat((pos, self.meta_planes), dim=1)