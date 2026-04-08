#[allow(unused)]

// mod mcts;
// mod encoder;
mod api;
mod encoder;
use cozy_chess::*;

use crate::{api::ZmqClient, encoder::BatchBuffer};

fn main() {
    let mut h = encoder::History::default();
    let mut board: Board = Board::default();

    h.update(&board);

    board.play("g1f3".parse().unwrap());
    h.update(&board);
    board.play("g8f6".parse().unwrap());
    h.update(&board);

    board.play("f3g1".parse().unwrap());
    h.update(&board);
    board.play("f6g8".parse().unwrap());
    h.update(&board);

    let zmq = ZmqClient::new();
    

    let mut batch = BatchBuffer::new(4);
    batch.fill_slot(0, &h, &board);
    batch.fill_slot(1, &h, &board);
    batch.fill_slot(2, &h, &board);
    batch.fill_slot(3, &h, &board);

    zmq.send(&batch);
}
