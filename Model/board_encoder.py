import torch
import bulletchess as chess
from .move_encoder import encode_move, decode_move
from collections import deque

SHIFTS = torch.arange(64, dtype=torch.int64)
PIECES = [chess.Piece(chess.WHITE, x) for x in chess.PIECE_TYPES] + [chess.Piece(chess.BLACK, x) for x in chess.PIECE_TYPES] 
PIECES_INT_DICT: dict[chess.Piece] = dict([reversed(x) for x in enumerate(PIECES)])  # type: ignore


class State:
    def __init__(self, board: chess.Board, init_board = True) -> None:
        self.board = board
        
        self.history_planes = torch.zeros(64, 96, dtype=torch.float32)
        self.meta_planes = torch.zeros(64,7)
        self.pos_cache:dict[int, torch.Tensor] = {} # hash -> [12,64]
        
        self.move_stack: deque[int] = deque()
        
        if init_board:
            self.encode_board_init(board)
    
    @staticmethod
    def to_signed_64(x):
        return (x + (1 << 63)) % (1 << 64) - (1 << 63)
    
    def bitboards_to_tensor(self, bitboards: list[int]):
        bb = torch.tensor(bitboards, dtype=torch.int64)  # [12]
        bits = ((bb[:, None] >> SHIFTS) & 1).to(torch.bool)  # [12,64]
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

    def encode_board_init(self, board: chess.Board): # Create the root board state and put into histories
        """## Tokenization
        
        Encode board into 64 tokens [64,103]
        
        ### Planes:
            0-95: positions
            96: turn
            97-100: castling rights
            101: en-passant
            102: repetition
        """
        self.pos_cache[board.__hash__()] = self.bitboards_to_tensor([self.to_signed_64(int(board[x])) for x in PIECES])
        

    def make_move(self, action):
        board = self.board

        move: chess.Move = decode_move(board, action) #type: ignore
        
        # assert move in board.legal_moves(), "Ilegal move"

        origin = move.origin.index()
        dest = move.destination.index()

        piece: chess.Piece = board[move.origin] #type:ignore
        ep = board.en_passant_square
        turn = board.turn

        prev = self.pos_cache[board.__hash__()]
        
        # Apply move to board 
        board.apply(move)
        curr_hash = board.__hash__()
        
        self.move_stack.append(curr_hash)
        
        if self.pos_cache.get(curr_hash) is not None: # Already generated
            return
        
        # Copy previous board
        curr = prev.clone()

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

        self.pos_cache[board.__hash__()] = curr

        return curr
    
    def unmake_move(self):
        self.board.undo()
        self.move_stack.pop()
    
    @property
    def tokens(self): # Reorder to feed into NN
        self.update_metadata()
        
        size = min(len(self.move_stack), 8)
        
        if size < 8:
            self.history_planes.zero_()
        
        for i in range(min(len(self.move_stack), 8)):
            pos = self.pos_cache[self.move_stack[i]]
            self.history_planes[:, i*12:(i+1)*12] = pos
        return torch.cat((self.history_planes, self.meta_planes), dim=1).unsqueeze(0)