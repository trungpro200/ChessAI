import io

import chess.pgn

from training.encode_game import encode_game
from training.pgn_parser import parse_eval, parse_result, z_from_outcome


MINI_PGN = """[Event "Test"]
[Site "https://lichess.org/test"]
[Result "1-0"]

1. e4 { [%eval 0.20] [%clk 0:03:00] } 1... e5 { [%eval -0.10] } 2. Nf3 Nc6 3. Bb5 a6 1-0
"""


def test_parse_result():
    assert parse_result("1-0") == 1.0
    assert parse_result("0-1") == -1.0
    assert parse_result("1/2-1/2") == 0.0
    assert parse_result("*") is None


def test_parse_eval():
    assert parse_eval("{ [%eval 0.17] [%clk 0:03:00] }") == 0.17
    assert parse_eval("{ [%clk 0:03:00] }") is None


def test_encode_mini_game():
    game = chess.pgn.read_game(io.StringIO(MINI_PGN))
    assert game is not None
    samples = encode_game(game, min_plies=4)
    assert len(samples) >= 4
    assert samples[0].features.shape == (64, 103)
    assert 0 <= samples[0].from_sq < 64
    assert 0 <= samples[0].plane < 73
    assert samples[0].z in (-1.0, 0.0, 1.0)
    assert any(s.has_eval for s in samples)


def test_z_side_to_move():
    assert z_from_outcome(1.0, True) == 1.0
    assert z_from_outcome(1.0, False) == -1.0
