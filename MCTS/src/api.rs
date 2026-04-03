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

    pub fn send(&self) {
        let batch: u32 = 4;
        let arr = [0.0f32; 16];

        let data_arr: &[u8] = unsafe {
            std::slice::from_raw_parts(
            arr.as_ptr() as *const u8,
            arr.len() * 4,
            )
        };

        let _res: Result<(), zmq::Error> = self.socket.send_multipart(&[
            &batch.to_le_bytes(),
            data_arr 
        ], 0);
    }
}
