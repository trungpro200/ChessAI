use rand;
use std::thread;
use std::time::Instant;
use std::sync::{mpsc, Arc};
use std::sync::atomic::Ordering;
use cozy_chess::*;
use crate::mcts::*;
use zmq::*;

pub fn test_mcts_parallel() {
    let board = Board::default();
    let tree = Arc::new(Tree::new());
    let (tx, rx) = mpsc::channel();

    // Spawn 4 workers to simulate your i3-10105F environment
    for i in 0..4 {
        let t_tree = Arc::clone(&tree);
        let t_board = board.clone();
        let t_tx = tx.clone();

        thread::spawn(move || {
            // Simulate selection phase
            let (path, leaf) = t_tree.select_leaf(&t_board);
            
            // In a real scenario, this is where the Batcher/GPU would act.
            // For this test, we simulate a value of 0.5 (slight advantage).
            t_tree.backprop(path, 0.5);
            
            t_tx.send(format!("Worker {} finished expansion", i)).unwrap();
        });
    }

    for _ in 0..4 {
        println!("{}", rx.recv().unwrap());
    }

    // Verify root node was visited 4 times
    let root_node = tree.tt.get(&board.hash()).expect("Root should exist");
    let total_visits: u32 = root_node.moves.iter().map(|m| m.n.load(Ordering::SeqCst)).sum();
    
    assert_eq!(total_visits, 4);
    println!("MCTS Thread Safety Test: PASSED (Visits: {})", total_visits);
}

pub fn test_repetition_logic() {
    let mut board = Board::default();
    let mut history = Vec::new(); // Use a Vec to track the order
    history.push(board.hash());

    let moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"];

    for mv_str in moves {
        board.play(mv_str.parse().unwrap());
        let h = board.hash();
        
        // Count how many times this hash has appeared
        let count = history.iter().filter(|&&x| x == h).count();
        history.push(h);

        if count >= 2 {
            println!("3-fold Repetition reached! (Hash appeared {} times before)", count);
            // In your real MCTS, you would return 0.0 (Draw) here
        }
    }

    // Correcting the Stalemate Test
    // This FEN is a classic stalemate: White to move, but no moves, not in check.
    let stalemate_fen = "k7/8/K7/8/8/8/8/1Q6 b - - 0 1"; 
    let s_board = Board::from_fen(stalemate_fen, false).unwrap();
    
    println!("Status: {:?}", s_board.status());
    // cozy_chess returns Drawn for stalemate only if generate_moves is empty 
    // AND the king is not in check.
    assert_eq!(s_board.status(), GameStatus::Drawn);
}

pub fn zmq_stress_test(){
    let ctx = Context::new();
    let socket = ctx.socket(zmq::REQ).unwrap();
    socket.connect("tcp://localhost:3636").unwrap();

    // Simulate 256 batch, 64 tokens, 103 features (f32)
    // Roughly 6.75 MB per batch
    let data_size = 256 * 64 * 103;
    let junk_data: Vec<f32> = (0..data_size).map(|_| rand::random::<f32>()).collect();
    let bytes: &[u8] = bytemuck::cast_slice(&junk_data);

    println!("Starting stress test: Sending ~6.75MB per message...");
    let start = Instant::now();
    let iterations = 1000;

    for i in 0..iterations {
        // Send as multipart to simulate real protocol
        socket.send_multipart(vec![
            "batch_256".as_bytes(), 
            bytes
        ], 0).unwrap();

        // Wait for dummy response from Python
        let _ = socket.recv_bytes(0).unwrap();
        
        if i % 100 == 0 {
            println!("Sent {} batches...", i);
        }
    }

    let duration = start.elapsed();
    let total_mb = (iterations as f64 * 6.75);
    println!("--- Results ---");
    println!("Time: {:?}", duration);
    println!("Throughput: {:.2} MB/s", total_mb / duration.as_secs_f64());
    println!("Batches/sec: {:.2}", iterations as f64 / duration.as_secs_f64());
}