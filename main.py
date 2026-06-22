from Model import SelfPlay, ChessModel, State
import torch
import bulletchess

ckpt = torch.load("checkpoints/best.pt")

model = ChessModel()
model.load_state_dict(ckpt["model"])
splay = SelfPlay(model, num_simulations=100)

state = State(bulletchess.Board())
splay.play_game(state)