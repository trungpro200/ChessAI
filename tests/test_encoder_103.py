import bulletchess as chess

from Model.board_encoder import HISTORY_LEN, State
from Model.move_encoder import encode_move


def test_tokens_shape_start_position():
    board = chess.Board()
    state = State(board)
    tokens = state.tokens
    assert tokens.shape == (1, 64, 8 * 12 + 7)
    assert tokens.shape == (1, 64, 103)


def test_tokens_after_replay():
    board = chess.Board()
    state = State(board)
    for _ in range(20):
        move = next(iter(board.legal_moves()))
        action = encode_move(move, board)
        state.make_move(action)

    tokens = state.tokens
    assert tokens.shape == (1, 64, 103)
    assert tokens.sum() > 0


def test_history_len_constant():
    assert HISTORY_LEN == 8
