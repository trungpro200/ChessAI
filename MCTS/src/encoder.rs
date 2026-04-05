use cozy_chess::{Board, Color, Piece, Square};

const HISTORY_LEN: usize = 8;
const HISTORY_ARR: usize = HISTORY_LEN*12;



//768 = 64*12 (64 Squares and 12 Piece types)
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
    // Then get metadata from the board and write to the output
    pub fn get_ordered_data(&self, out_data: &mut [f32], board: &Board) {
        // 64 squares * ( (8 frames * 12 pieces) + 7 meta planes ) = 6592 floats
        assert!(out_data.len() >= 64 * (8 * 12 + 7), "Output slice is too small!");

        // 1. Fill History Tokens (Interleaved)
        // We want: [Square 0: Frame 0-7], [Square 1: Frame 0-7]...
        for sq in 0..64 {
            let token_offset = sq * (8 * 12 + 7);
            
            for frame_idx in 0..8 {
                let source_idx = (self.head + 8 - frame_idx) % 8;
                let frame_data = &self.data[source_idx];
                
                for piece_layer in 0..12 {
                    // Map the flat history into the specific square's token slot
                    let piece_val = frame_data[piece_layer * 64 + sq];
                    
                    // Index: Square-Start + (Frame * 12) + Piece
                    let out_idx = token_offset + (frame_idx * 12) + piece_layer;
                    out_data[out_idx] = piece_val;
                }
            }

            // 2. Fill Metadata into each token
            let meta_start = token_offset + (8 * 12);
            
            // 0: Turn (Global)
            out_data[meta_start + 0] = if board.side_to_move() == Color::White { 1.0 } else { 0.0 };

            // 1-4: Castling (Global)
            let white_rights = board.castle_rights(Color::White);
            let black_rights = board.castle_rights(Color::Black);
            out_data[meta_start + 1] = if white_rights.short.is_some() { 1.0 } else { 0.0 };
            out_data[meta_start + 2] = if white_rights.long.is_some() { 1.0 } else { 0.0 };
            out_data[meta_start + 3] = if black_rights.short.is_some() { 1.0 } else { 0.0 };
            out_data[meta_start + 4] = if black_rights.long.is_some() { 1.0 } else { 0.0 };

            // 5: En Passant (Square-specific)
            // Only the specific EP square gets a 1.0
            out_data[meta_start + 5] = if board.en_passant().map_or(false, |ep| ep as usize == sq) { 
                1.0 
            } else { 
                0.0 
            };

            // 6: Halfmove Clock (Global)
            out_data[meta_start + 6] = board.halfmove_clock() as f32 * 0.01;
        }
    }
}

pub struct BatchBuffer {
    // If batch size is 64: 64 * 8 * 768 = 393,216 floats (~1.5 MB)
    pub buffer: Vec<f32>, 
    pub batch_size: usize,
}

impl BatchBuffer {
    pub fn new(batch_size: usize) -> Self {
        Self {
            buffer: vec![0.0f32; batch_size * 8 * 768],
            batch_size,
        }
    }

    pub fn fill_slot(&mut self, slot_idx: usize, history: &History, board: &Board) {
        let start = slot_idx * 8 * 768;
        let end = start + 8 * 768;
        // Re-order the ring buffer into this specific slice of the flat buffer
        history.get_ordered_data(&mut self.buffer[start..end], board);
    }
}