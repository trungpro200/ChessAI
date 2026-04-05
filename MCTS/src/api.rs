use crate::encoder::*;
use zmq;

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
        
    }
}
