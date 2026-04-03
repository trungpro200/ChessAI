// mod mcts;
// mod encoder;
mod api;


fn main() {
    let zmq = api::ZmqClient::new();

    zmq.send();
}
