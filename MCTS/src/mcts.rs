use dashmap::DashMap;
pub use cozy_chess::*;
use std::sync::atomic::{AtomicI64, AtomicU32, Ordering};
use std::sync::Arc;

const C_PUCT: f32 = 1.4;
const V_LOSS: f32 = 1.0;
const F_SCALE: f32 = 1000.0; // Scale for atomic f32 storage

pub struct Node {
    // We use a single map for all move data to keep it cache-local
    pub moves: DashMap<Move, MoveStats>,
}

pub struct MoveStats {
    pub n: AtomicU32,
    pub w: AtomicI64,      // Total value scaled by F_SCALE
    pub p: f32,            // Prior
    pub v_loss: AtomicU32, // Virtual loss count
}

pub struct Tree {
    pub tt: DashMap<u64, Arc<Node>>,
}

impl Tree {
    pub fn new() -> Self {
        Self { tt: DashMap::new() }
    }

    /// Primary search function for a worker thread
    pub fn select_leaf(&self, board: &Board) -> (Vec<(u64, Move)>, Board) {
        let mut path = Vec::new();
        let mut current = board.clone();

        loop {
            let hash = current.hash();
            
            // 1. Expansion: If node doesn't exist, it's a leaf
            let node: dashmap::mapref::one::RefMut<'_, u64, Arc<Node>> = self.tt.entry(hash).or_insert_with(|| {
                let mut moves: DashMap<Move, MoveStats> = DashMap::new();
                current.generate_moves(|mvs| {
                        for mv in mvs {
                            moves.insert(mv, MoveStats {
                            n: AtomicU32::new(0),
                            w: AtomicI64::new(0),
                            p: 0.0, // Will be updated by Policy Head
                            v_loss: AtomicU32::new(0),
                        });
                    }
                    false
                });
                Arc::new(Node { moves })
            });

            // 2. Selection
            let total_n: u32 = node.moves.iter().map(|m| m.n.load(Ordering::Relaxed)).sum();
            let total_vloss: u32 = node.moves.iter().map(|m| m.v_loss.load(Ordering::Relaxed)).sum();
            let sqrt_n = ((total_n + total_vloss) as f32).sqrt();

            let best_move = node.moves.iter().max_by(|a, b| {
                let score_a = self.calculate_puct(a.value(), sqrt_n);
                let score_b = self.calculate_puct(b.value(), sqrt_n);
                score_a.partial_cmp(&score_b).unwrap()
            }).map(|m| *m.key());

            match best_move {
                Some(mv) => {
                    // Apply Virtual Loss to discourage other workers
                    let stats = node.moves.get(&mv).unwrap();
                    stats.v_loss.fetch_add(1, Ordering::SeqCst);
                    
                    path.push((hash, mv));
                    current.play(mv);

                    // If this move leads to a node we haven't seen, it's our expansion target
                    if !self.tt.contains_key(&current.hash()) {
                        return (path, current);
                    }
                }
                None => return (path, current), // Terminal state
            }
        }
    }

    fn calculate_puct(&self, stats: &MoveStats, sqrt_n: f32) -> f32 {
        let n = stats.n.load(Ordering::Relaxed) as f32;
        let v_loss = stats.v_loss.load(Ordering::Relaxed) as f32;
        
        // Q = (W - VLOSS) / (N + VLOSS)
        let w = (stats.w.load(Ordering::Relaxed) as f32) / F_SCALE;
        let q = if n + v_loss > 0.0 {
            (w - (v_loss * V_LOSS)) / (n + v_loss)
        } else {
            0.0
        };

        let u = C_PUCT * stats.p * (sqrt_n / (1.0 + n + v_loss));
        q + u
    }

    pub fn backprop(&self, path: Vec<(u64, Move)>, mut value: f32) {
        // value is from the perspective of the player who just moved
        for (hash, mv) in path.into_iter().rev() {
            if let Some(node) = self.tt.get(&hash) {
                if let Some(stats) = node.moves.get(&mv) {
                    // Update stats
                    stats.n.fetch_add(1, Ordering::SeqCst);
                    let scaled_v = (value * F_SCALE) as i64;
                    stats.w.fetch_add(scaled_v, Ordering::SeqCst);
                    
                    // Remove Virtual Loss
                    stats.v_loss.fetch_sub(1, Ordering::SeqCst);
                }
            }
            // Flip value for previous player
            value = -value;
        }
    }
}