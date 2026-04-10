#![allow(unused)]

// mod mcts;
// mod encoder;
mod api;
mod encoder;

use cozy_chess::*;
use encoder::*;
use api::*;

fn main() {
    let encoder = MoveEncoder::new();
    let mv = Move { from: Square::A1, to: Square::A8, promotion: Some(Piece::Bishop) };
    
    let encoded = encoder.encode(mv);
    println!("From: {}, Plane: {}", encoded[0], encoded[1]); // [0, 6] (Direction: North, Dist: 7)
    let decoded = encoder.decode(encoded[0], encoded[1]);
}
