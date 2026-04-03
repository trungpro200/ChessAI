use cozy_chess::{Board, Color, Piece, Square};
use std::collections::VecDeque;

const HISTORY_LEN: i32 = 8;

pub type PosPlanes = VecDeque<[f32; 768]>;

pub fn encode_board_build(board: &Board) -> PosPlanes {
    //768 = 12*64
    let mut plane: [f32; 768] = [0.0f32; 768];
    let mut pos_planes: PosPlanes = VecDeque::new();

    // square → index
    let sq_idx = |sq: Square| -> usize {
        let file: usize = sq.file() as usize;
        let rank: usize = sq.rank() as usize;
        rank * 8 + file
    };

    for sq in Square::ALL {
        if let Some(piece) = board.piece_on(sq) {
            let color = board.color_on(sq).unwrap();

            let piece_idx = match piece {
                Piece::Pawn => 0,
                Piece::Knight => 1,
                Piece::Bishop => 2,
                Piece::Rook => 3,
                Piece::Queen => 4,
                Piece::King => 5,
            };

            let layer = match color {
                Color::White => piece_idx,
                Color::Black => piece_idx + 6,
            };

            plane[layer * 64 + sq_idx(sq)] = 1.0;
        }
    }

    for _ in 0..HISTORY_LEN-1 {
        pos_planes.push_back([0.0f32; 768]);
    }

    pos_planes.push_back(plane);
    pos_planes
}
