import torch
import bulletchess as chess
from .move_encoder import encode_move, decode_move

SHIFTS = torch.arange(64, dtype=torch.int64)
PIECES = [chess.Piece(chess.WHITE, x) for x in chess.PIECE_TYPES] + [chess.Piece(chess.BLACK, x) for x in chess.PIECE_TYPES] 
PIECES_INT_DICT = dict([reversed(x) for x in enumerate(PIECES)])  # type: ignore


class Tokens:
    def __init__(self, board: chess.Board) -> None:
        self.board = board
        self.tokens = self.encode_board_init(board)
        self.head = 0
    
    @staticmethod
    def to_signed_64(x):
        return (x + (1 << 63)) % (1 << 64) - (1 << 63)

    def bitboards_to_tensor(self, bitboards: list[int]):
        
        bb = torch.tensor(bitboards, dtype=torch.int64).unsqueeze(1)

        bits = ((bb >> SHIFTS) & 1).bool()

        return bits.view(-1, 64).transpose(0,1)

    def build_metadata(self) -> torch.Tensor:
        board = self.board
        
        #Turn plane - 1 layer
        turn = torch.ones(64,1) if board.turn == chess.WHITE else torch.zeros(64,1)
        
        # Castling planes - 4 layers
        sides = "KQkq"
        castling = torch.zeros(64,4)
        
        for i, side in enumerate(sides):
            if side in board.castling_rights.fen():
                castling[:, i] = 1
        
        # En passant - 1 layer
        enps = torch.zeros(64,1)
        if board.en_passant_square:
            enps[board.en_passant_square.index(), 0] = 1 # type: ignore
        
        # Repetition - 1 layer
        rep_50 = torch.full([64,1], board.halfmove_clock/100) 
        
        meta = torch.concat([
            turn, castling, enps, rep_50
        ], dim=1)
        
        return meta

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
        current_board = self.bitboards_to_tensor([self.to_signed_64(int(board[x])) for x in PIECES])
        history = torch.zeros(64,84)
        
        # Metadata
        meta = self.build_metadata()
        
        return torch.concat([current_board, history, meta], 1)

    def encode_board_propagate(self, action):
        tokens = self.tokens
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

        curr = tokens[:, 12*new_head : 12*(new_head+1)]
        prev = tokens[:, 12*prev_head : 12*(prev_head+1)]

        # Copy previous board
        curr.copy_(prev)

        # Handle en-passant capture 
        if ep and piece.piece_type == chess.PAWN and dest == ep.index():
            captured_sq = dest - 8 if turn else dest + 8
            curr[captured_sq] = 0

        # Move piece 
        curr[dest] = prev[origin]
        curr[origin] = 0

        # Apply move to board 
        board.apply(move)

        # Update head 
        self.head = new_head

        # Turn plane 
        tokens[:, 96] = int(turn == chess.BLACK)

        # En-passant plane 
        tokens[:, 101] = 0
        ep = board.en_passant_square
        if ep:
            tokens[ep.index(), 101] = 1

        # Repetition 
        tokens[:, 102] = board.halfmove_clock * 0.01

        return tokens