use cozy_chess::{Board, Color, Piece, Square};

const HISTORY_LEN: usize = 8;
const HISTORY_ARR: usize = HISTORY_LEN*12;

pub type Batch = Vec<History>;
pub struct History {
    data: [[f32; 768]; 8],
    head: usize,
}

impl Default for History {
    fn default() -> Self {
        Self {
            data: [[0.0f32; 768]; 8],
            head: 0,
        }
    }
}

impl History {
    pub fn update(&mut self, board: &Board) {
        // Move the head forward (wrapping around 0-7)
        self.head = (self.head + 1) % HISTORY_LEN;
        
        // Zero out the old data in this slot
        self.data[self.head] = [0.0; 768];
        
        // Encode the new board directly into the new head position
        self.encode_at_head(board);
    }

    fn encode_at_head(&mut self, board: &Board) {
        let current_frame = &mut self.data[self.head];
        for &color in &Color::ALL {
            for &piece in &Piece::ALL {
                let bitboard = board.pieces(piece) & board.colors(color);
                let offset = (piece as usize + (color as usize * 6)) * 64;
                for sq in bitboard {
                    current_frame[offset + sq as usize] = 1.0;
                }
            }
        }
    }

    // Take a slice (&mut [f32]) instead of a fixed array reference
    pub fn get_ordered_data(&self, out_data: &mut [f32]) {
        // Safety check: ensure the provided slice is large enough
        assert!(out_data.len() >= 768 * 8, "Output slice is too small!");

        for i in 0..8 {
            // i=0 is current (head), i=1 is head-1, etc.
            let source_idx = (self.head + 8 - i) % 8;
            
            let start = i * 768;
            let end = start + 768;
            
            // Copy from our internal ring buffer to the provided slice
            out_data[start..end].copy_from_slice(&self.data[source_idx]);
        }
    }
}