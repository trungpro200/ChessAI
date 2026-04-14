#![allow(unused)]

// mod mcts;
// mod encoder;
mod api;
mod encoder;
mod mcts;
mod tester;

use cozy_chess::*;
use encoder::*;
use api::*;

fn main() {
    println!("Starting tests");
    tester::zmq_stress_test();
}
