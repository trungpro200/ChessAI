import math
from dataclasses import dataclass

import bulletchess as bc
import chess as pychess
import numpy as np

from Model.board_encoder import State
from Model.move_encoder import encode_move
from training.pgn_parser import eval_to_z, parse_eval, parse_result, z_from_outcome


@dataclass
class EncodedSample:
    features: np.ndarray  # [64, 103] float32
    from_sq: int
    plane: int
    z: float
    z_eval: float
    has_eval: bool


def _pychess_board_to_bullet(board: pychess.Board) -> bc.Board:
    return bc.Board.from_fen(board.fen())


def _pychess_move_to_bullet(move: pychess.Move) -> bc.Move:
    return bc.Move.from_uci(move.uci())


def encode_game(
    game: pychess.pgn.Game,
    eval_cp_scale: float = 4.0,
    min_plies: int = 8,
) -> list[EncodedSample]:
    """Encode main-line positions from a python-chess Game."""
    outcome_white = parse_result(game.headers.get("Result"))
    if outcome_white is None:
        return []

    node = game
    board_py = game.board()
    state = State(_pychess_board_to_bullet(board_py))
    eval_cp = parse_eval(node.comment)

    samples: list[EncodedSample] = []

    while node.variations:
        child = node.variation(0)
        move_py = child.move
        if move_py is None:
            break

        white_to_move = board_py.turn == pychess.WHITE
        z = z_from_outcome(outcome_white, white_to_move)

        has_eval = eval_cp is not None
        z_eval = (
            eval_to_z(eval_cp, white_to_move, eval_cp_scale) if has_eval else math.nan
        )

        tokens = state.tokens.squeeze(0).numpy()
        move_bc = _pychess_move_to_bullet(move_py)
        from_sq, plane = encode_move(move_bc, state.board)

        samples.append(
            EncodedSample(
                features=tokens.astype(np.float32),
                from_sq=from_sq,
                plane=plane,
                z=z,
                z_eval=z_eval,
                has_eval=has_eval,
            )
        )

        state.make_move((from_sq, plane))
        board_py.push(move_py)
        node = child
        eval_cp = parse_eval(node.comment)

    if len(samples) < min_plies:
        return []

    return samples
