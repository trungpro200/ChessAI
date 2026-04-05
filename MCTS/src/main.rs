#[allow(unused)]

// mod mcts;
// mod encoder;
mod api;
mod encoder;
use cozy_chess::*;

use crate::{api::ZmqClient, encoder::BatchBuffer};

fn main() {
    let board: Board = Board::from_fen("rnbqkbnr/p1ppp1pp/1p6/4Pp2/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 3", false).unwrap();
    board.null_move();
    board.null_move();
    board.null_move();
    board.null_move();

    let mut h = encoder::History::default();
    let zmq = ZmqClient::new();
    h.update(&board);

    let mut batch = BatchBuffer::new(1);
    batch.fill_slot(0, &h, &board);

    zmq.send(&batch);
}
