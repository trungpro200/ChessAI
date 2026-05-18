import hashlib
import math
import re
from typing import Iterator, TextIO

import chess.pgn

EVAL_RE = re.compile(r"\[%eval\s+([-\d.]+)\]")


def parse_eval(comment: str | None) -> float | None:
    if not comment:
        return None
    m = EVAL_RE.search(comment)
    if not m:
        return None
    return float(m.group(1))


def parse_result(result: str | None) -> float | None:
    """White-centric game outcome: +1 white win, -1 black win, 0 draw."""
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    if result == "1/2-1/2":
        return 0.0
    return None


def z_from_outcome(outcome_white: float, white_to_move: bool) -> float:
    return outcome_white if white_to_move else -outcome_white


def eval_to_z(cp_white: float, white_to_move: bool, scale: float = 4.0) -> float:
    cp_stm = cp_white if white_to_move else -cp_white
    return math.tanh(cp_stm / scale)


def game_split_key(headers: chess.pgn.Headers, game_index: int) -> str:
    site = headers.get("Site", "")
    return hashlib.sha256(f"{site}:{game_index}".encode()).hexdigest()


def is_val_split(key: str, val_fraction: float) -> bool:
    return int(key[:8], 16) % 1000 < int(val_fraction * 1000)


def stream_games(handle: TextIO) -> Iterator[tuple[chess.pgn.Game, int]]:
    index = 0
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            break
        yield game, index
        index += 1
