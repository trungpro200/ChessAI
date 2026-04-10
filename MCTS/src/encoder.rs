use cozy_chess::*;

const HISTORY_LEN: usize = 8;



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
            out_data[meta_start + 5] = if let Some(ep_file) = board.en_passant() {
                // depending on whose turn it is.
                let ep_rank = match board.side_to_move() {
                    Color::White => Rank::Third, // White captures to 3rd rank
                    Color::Black => Rank::Sixth, // Black captures to 6th rank
                };
                
                // Construct the actual square and compare its 0-63 index to the current loop 'sq'
                let target_sq = Square::new(ep_file, ep_rank);
                if target_sq as usize == sq { 1.0 } else { 0.0 }
            } else {
                0.0
            };

            // 6: Halfmove Clock (Global)
            out_data[meta_start + 6] = board.halfmove_clock() as f32 * 0.01;
        }
    }
}

pub struct BatchBuffer {
    pub buffer: Vec<f32>, 
    pub batch_size: usize,
}

impl BatchBuffer {
    pub fn new(batch_size: usize) -> Self {
        Self {
            buffer: vec![0.0f32; batch_size * 64 * 103],
            batch_size,
        }
    }

    pub fn fill_slot(&mut self, slot_idx: usize, history: &History, board: &Board) {
        let start = slot_idx * 64 * 103;
        let end = start + 64 * 103;
        // Re-order the ring buffer into this specific slice of the flat buffer
        history.get_ordered_data(&mut self.buffer[start..end], board);
    }
}

use cozy_chess::{Move, Square, Piece, File, Rank};

pub const MOVE_CHANNELS: usize = 73;

// Directional offsets for Queen-like moves
const DIRECTIONS: [(i8, i8); 8] = [
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1)
];

pub struct MoveEncoder {
    // lookup[from_sq][to_sq] -> plane_index
    knight_map: [[Option<usize>; 64]; 64],
}

impl MoveEncoder {
    pub fn new() -> Self {
        let mut knight_map = [[None; 64]; 64];
        let knight_offsets = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2)
        ];

        for sq in 0..64 {
            let f = (sq % 8) as i8;
            let r = (sq / 8) as i8;
            for (idx, (df, dr)) in knight_offsets.iter().enumerate() {
                let nf = f + df;
                let nr = r + dr;
                if nf >= 0 && nf < 8 && nr >= 0 && nr < 8 {
                    knight_map[sq][(nr * 8 + nf) as usize] = Some(56 + idx);
                }
            }
        }
        Self { knight_map }
    }

    pub fn encode(&self, mv: Move) -> [usize; 2] {
        let from = mv.from as usize;
        let to = mv.to as usize;
        
        let from_f = (from % 8) as i8;
        let from_r = (from / 8) as i8;
        let to_f = (to % 8) as i8;
        let to_r = (to / 8) as i8;

        let df = to_f - from_f;
        let dr = to_r - from_r;

        let plane = if let Some(p) = mv.promotion {
            // Under-promotions (Knight=0, Bishop=1, Rook=2)
            // Queen promotions are handled as normal queen moves
            if p != Piece::Queen {
                let promo_idx = match p {
                    Piece::Knight => 0,
                    Piece::Bishop => 1,
                    Piece::Rook => 2,
                    _ => 0,
                };
                let dir_idx = (df + 1) as usize; // -1 -> 0, 0 -> 1, 1 -> 2
                64 + (promo_idx * 3) + dir_idx
            } else {
                self.get_queen_plane(df, dr)
            }
        } else if let Some(p) = self.knight_map[from][to] {
            p
        } else {
            self.get_queen_plane(df, dr)
        };

        [from, plane]
    }

    fn get_queen_plane(&self, df: i8, dr: i8) -> usize {
        let dist = df.abs().max(dr.abs());
        let (nf, nr) = (df / dist, dr / dist);
        
        let dir_idx = DIRECTIONS.iter().position(|&d| d == (nf, nr)).unwrap();
        (dir_idx * 7) + (dist as usize - 1)
    }

    pub fn decode(&self, from: usize, plane: usize) -> (Square, Square, Option<Piece>) {
        let from_sq = Square::index(from);
        let from_f = from % 8;
        let from_r = from / 8;

        if plane < 56 {
            // Queen moves
            let dir_idx = plane / 7;
            let dist = (plane % 7) + 1;
            let (df, dr) = DIRECTIONS[dir_idx];
            let to_sq = Square::new(
                File::index((from_f as i8 + df * dist as i8) as usize),
                Rank::index((from_r as i8 + dr * dist as i8) as usize)
            );
            (from_sq, to_sq, None)
        } else if plane < 64 {
            // Knight moves - Reverse lookup
            let knight_idx = plane - 56;
            let knight_offsets = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];
            let (df, dr) = knight_offsets[knight_idx];
            let to_sq = Square::new(
                File::index((from_f as i8 + df) as usize),
                Rank::index((from_r as i8 + dr) as usize)
            );
            (from_sq, to_sq, None)
        } else {
            // Under-promotions
            let p_type = (plane - 64) / 3;
            let dir_idx = (plane - 64) % 3;
            let df = dir_idx as i8 - 1;
            let dr = 1; // Always forward for promotion
            let piece = match p_type {
                0 => Piece::Knight,
                1 => Piece::Bishop,
                2 => Piece::Rook,
                _ => Piece::Queen
            };
            let to_sq = Square::new(
                File::index((from_f as i8 + df) as usize),
                Rank::index((from_r as i8 + dr) as usize)
            );
            (from_sq, to_sq, Some(piece))
        }
    }
}