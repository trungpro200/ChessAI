#[allow(unused)]

// mod mcts;
// mod encoder;
mod api;
mod encoder;
use cozy_chess::*;

fn main() {
    let board: Board = Board::default();
    let mut h = encoder::History::default();
    h.update(&board);

    let mut batch: [[f32;6592]; 8] = [[0.0f32; 6592]; 8]; // 8 Batchs
    let slice = &mut batch[0];

    h.get_ordered_data(slice, &board);

    println!("{:?}", batch[0]);
}
