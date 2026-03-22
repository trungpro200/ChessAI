import math
import numpy as np
import bulletchess as chess
from .board_encoder import State
from .move_encoder import encode_move, decode_move
from copy import deepcopy
from .chess_model import ChessModel
from collections import deque
import torch, random, gc


CPUCT = 1.5
GLOBAL_BUFFER = deque()
TERMINAL_STATES = [chess.THREEFOLD_REPETITION, chess.FIFTY_MOVE_TIMEOUT, chess.STALEMATE, chess.CHECKMATE]
def is_terminal(board: chess.Board):
    for state in TERMINAL_STATES:
        if board in state:
            if state == chess.CHECKMATE:
                return int(board.turn != chess.WHITE)*2 - 1
    
    return None



class Node:
    def __init__(self, policy: dict[tuple, float]):
        self.P = policy        # prior probabilities (dict: move -> prob)
        self.N: dict[tuple, int] = {}            # visit count per move
        self.W: dict[tuple, float] = {}            # total value per move
        self.Q: dict[tuple, float] = {}            # mean value per move

        self.expanded = False

class SelfPlay:
    def __init__(self, model: ChessModel, num_simulations=200, temperature=1.0):
        self.model = model
        self.num_simulations = 50
        self.temperature = temperature

        self.TT: dict[int, Node] = {}   # zobrist -> Node
        
        model.eval()

    # =========================
    # PUBLIC ENTRY POINT
    # =========================
    def play_game(self, state: State):
        game_data = []

        while is_terminal(state.board) is None:
            zhash = state.board.__hash__()

            # run MCTS
            self.run_mcts(state)

            # get improved policy
            pi = self.get_policy(zhash)

            # store training sample
            game_data.append((state.tokens, pi))

            # sample move
            move = self.sample_move(pi)

            # apply move
            print(decode_move(state.board, move), sum(self.TT[zhash].N.values()))
            state.make_move(move)

        # game finished → assign values
        outcome = is_terminal(state.board)  # +1 / 0 / -1

        return self.assign_values(game_data, outcome, GLOBAL_BUFFER)

    # =========================
    # MCTS
    # =========================
    def run_mcts(self, root_state: State):
        root_hash = root_state.board.__hash__()

        for _ in range(self.num_simulations):
            state = root_state.clone()
            

            path: deque[tuple] = deque()
            value = self.simulate(state, root_hash, path)

            self.backpropagate(path, value, state.board)

    def simulate(self, state: State, zhash, path: deque[tuple]):
        while True:
            if zhash not in self.TT:
                # expand
                with torch.no_grad():
                    policy, value = self.model(state.tokens)
                    masked_p: dict[tuple, float] = dict()
                
                policy: torch.Tensor
                value: torch.Tensor
                
                for lmove in state.board.legal_moves():
                    e = encode_move(lmove)
                    masked_p[e] = policy[0, *e].item()

                node = Node(masked_p)
                self.TT[zhash] = node
                
                # for obj in gc.get_objects():
                #     try:
                #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                #             if obj.device == 'cuda':
                #                 print(type(obj), obj.size())
                #     except:
                #         pass
                return value

            node = self.TT[zhash]

            move = self.select_move(node)

            path.append((zhash, move))

            state.make_move(move)
            zhash = state.board.__hash__()

            terminal_state = is_terminal(state.board)
            if terminal_state is not None:
                return terminal_state

    def select_move(self, node: Node):
        total_N = sum(node.N.values())

        best_score = -1e9
        best_move = None

        for move in node.P:
            Q = node.Q.get(move, 0)
            U = CPUCT * node.P.get(move, 0) * math.sqrt(total_N + 1) / (1 + node.N.get(move, 0)) #type: ignore
            score = Q + U #type: ignore

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def backpropagate(self, path: deque[tuple], value, board: chess.Board):
        for hash, move in reversed(path):
            node = self.TT[hash]
            node: Node
            move: tuple
            node.N[move] = node.N.get(move, 0) + 1
            node.W[move] = node.W.get(move, 0) + value
            node.Q[move] = node.W[move] / node.N[move]
            
            board.undo()
            value = -value  # switch player

    # =========================
    # POLICY + SAMPLING
    # =========================
    def get_policy(self, zhash):
        node = self.TT[zhash]

        visits = np.array(list(node.N.values()), dtype=np.float32)
        moves = list(node.N.keys())

        if self.temperature == 0:
            best = np.argmax(visits)
            pi = np.zeros_like(visits)
            pi[best] = 1.0
        else:
            visits = visits ** (1 / self.temperature)
            pi = visits / visits.sum()
        
        return dict(zip(moves, pi))

    def sample_move(self, pi: dict[tuple, float]) -> tuple:
        moves = list(pi.keys())
        probs = list(pi.values())
        return random.choices(moves, weights=probs, k=1)[0]

    # =========================
    # 🎓 TRAINING TARGETS
    # =========================
    def assign_values(self, game_data, outcome, buffer: deque | None = None):
        if buffer is not None:
            results = buffer
        else:
            results = deque()

        for state_enc, pi, player in game_data:
            value = outcome if player == 1 else -outcome
            results.append((state_enc, pi, value))

        return results