import torch
import bulletchess as chess
from .move_encoder import encode_move, decode_move

SHIFTS = torch.arange(64, dtype=torch.int64)
PIECES = [chess.Piece(chess.WHITE, x) for x in chess.PIECE_TYPES] + [chess.Piece(chess.BLACK, x) for x in chess.PIECE_TYPES] 

def to_signed_64(x):
    return (x + (1 << 63)) % (1 << 64) - (1 << 63)

def bitboards_to_tensor(bitboards: list[int]):
    
    bb = torch.tensor(bitboards, dtype=torch.int64).unsqueeze(1)

    bits = ((bb >> SHIFTS) & 1).bool()

    return bits.view(-1, 64).transpose(0,1)

def build_metadata(board: chess.Board) -> torch.Tensor:
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

def encode_board_init(board: chess.Board): # -> [64, 103]
    """## Tokenization
    
    Encode board into 64 tokens [64,103]
    
    ### Planes:
        0-95: positions
        96: turn
        97-100: castling rights
        101: en-passant
        102: repetition
    """
    current_board = bitboards_to_tensor([to_signed_64(int(board[x])) for x in PIECES])
    history = torch.zeros(64,84)
    
    # Metadata
    meta = build_metadata(board)
    
    return torch.concat([current_board, history, meta], 1)

def encode_board_propagate(tokens: torch.Tensor, board: chess.Board, action: tuple[int, int]):
    move = decode_move(board, action)
    
    # assert move in board.legal_moves(), "Illegal Move"
    
    # Shift history down
    for i in range(7):
        start_l = 84 - (i*12)
        end_l = start_l + 12
        
        start_h, end_h = start_l - 12, end_l - 12
        
        tokens[: ,start_l:end_l] = tokens[:, start_h:end_h]
    
    # Flip the turn plane
    if board.turn == chess.WHITE: # Flip to black
        tokens[:, 96] = 0
    else:
        tokens[:, 96] = 1
    
    #Reset enpassant plane
    tokens[:,101] = 0
    
    
    
    return tokens