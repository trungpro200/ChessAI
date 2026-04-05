use crate::encoder::*;
use zmq;
use bytemuck;

pub struct ZmqClient {
    socket: zmq::Socket,
}

impl ZmqClient {
    pub fn new() -> Self {
        let ctx = zmq::Context::new();
        let socket = ctx.socket(zmq::REQ).unwrap();
        socket.connect("tcp://127.0.0.1:3636").unwrap();

        Self { socket }
    }

    pub fn send(&self, data: &BatchBuffer) {
        let bytes: &[u8] = bytemuck::cast_slice(&data.buffer);
        let batch_size_bytes: [u8; _] = data.batch_size.to_ne_bytes();

        let res = self.socket.send_multipart(vec![
            batch_size_bytes.as_slice(),
            bytes
        ], 0);
    }
}
