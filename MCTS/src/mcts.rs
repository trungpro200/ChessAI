use std::collections::HashMap;
use cozy_chess::PieceMoves;
pub use cozy_chess::{Board, Move, Color};

const C_PUCT: f64 = 1.4;

#[derive(Default)]
pub struct Node {
    pub n: HashMap<Move, i32>,   // visit count
    pub w: HashMap<Move, f64>,   // total value
    pub q: HashMap<Move, f64>,   // mean value
    pub p: HashMap<Move, f64>,   // prior (uniform for now)
}

pub struct Tree {
    pub tt: HashMap<u64, Node>, // transposition table
    pub root: u64,
}

impl Tree {
    pub fn new(board: &Board) -> Self {
        let mut tt = HashMap::new();
        tt.insert(board.hash(), Node::default());

        Self {
            tt,
            root: board.hash(),
        }
    }

    /// Run one MCTS simulation
    pub fn simulate(&mut self, board: &Board) {
        let mut path: Vec<(u64, Move)> = Vec::new();
        let mut current: Board = board.clone();

        // === 1. SELECTION ===
        loop {
            let hash: u64 = current.hash();

            let node: &mut Node = self.tt.entry(hash).or_default();
            let mut moves: Vec<Move> = Vec::new();

            

            // If leaf (not expanded)
            if node.n.is_empty() {
                current.generate_moves(|mv |{
                    moves.extend(mv);
                    false
                });
            }

            // Select best move via PUCT
            let total_n: f64 = node.n.values().map(|&v| v as f64).sum();

            let mut best_move = None;
            let mut best_score = f64::NEG_INFINITY;

            for mv in moves {
                let n = *node.n.get(mv).unwrap_or(&0) as f64;
                let q = *node.q.get(mv).unwrap_or(&0.0);
                let p = *node.p.get(mv).unwrap_or(&0.0);

                let u = C_PUCT * p * (total_n.sqrt() / (1.0 + n));
                let score = q + u;

                if score > best_score {
                    best_score = score;
                    best_move = Some(*mv);
                }
            }

            let mv = best_move.unwrap();
            path.push((hash, mv));
            current.play(mv);
        }

        // === 2. SIMULATION ===
        let value = self.rollout(&current);

        // === 3. BACKPROP ===
        self.backprop(path, value, board.side_to_move());
    }

    fn expand(&mut self, node: &mut Node, moves: &[Move]) {
        let prior = 1.0 / moves.len() as f64;

        for &mv in moves {
            node.n.insert(mv, 0);
            node.w.insert(mv, 0.0);
            node.q.insert(mv, 0.0);
            node.p.insert(mv, prior); // uniform prior
        }
    }

    fn rollout(&self, board: &Board) -> f64 {
        // VERY basic random rollout
        let mut b = board.clone();

        for _ in 0..100 {
            if let Some(result) = b.status() {
                return match result {
                    cozy_chess::GameStatus::Won => 1.0,
                    cozy_chess::GameStatus::Drawn => 0.0,
                    cozy_chess::GameStatus::Ongoing => continue,
                };
            }

            let moves: Vec<Move> = b.generate_moves().collect();
            if moves.is_empty() {
                return 0.0;
            }

            let mv = moves[rand::random::<usize>() % moves.len()];
            b.play(mv);
        }

        0.0
    }

    fn backprop(
        &mut self,
        path: Vec<(u64, Move)>,
        mut value: f64,
        root_player: Color,
    ) {
        for (hash, mv) in path.into_iter().rev() {
            let node = self.tt.get_mut(&hash).unwrap();

            let n = node.n.entry(mv).or_insert(0);
            let w = node.w.entry(mv).or_insert(0.0);
            let q = node.q.entry(mv).or_insert(0.0);

            *n += 1;
            *w += value;
            *q = *w / *n as f64;

            // Flip perspective each step
            value = -value;
        }
    }

    /// Get best move after simulations
    pub fn best_move(&self, board: &Board) -> Option<Move> {
        let node = self.tt.get(&board.hash())?;

        node.n
            .iter()
            .max_by_key(|(_, &n)| n)
            .map(|(&mv, _)| mv)
    }
}