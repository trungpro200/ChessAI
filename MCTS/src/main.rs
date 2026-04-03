use cozy_chess::Board;

// mod mcts;
mod encoder;


fn main() {
    let _board = Board::default();

    let pp = encoder::encode_board_build(&_board);
}
