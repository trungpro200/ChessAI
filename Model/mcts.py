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
VIRTUAL_LOSS = 1

def is_terminal(board: chess.Board):
    if board.halfmove_clock >= 24:
        return 0
    
    for state in TERMINAL_STATES:
        if board in state:
            if state == chess.CHECKMATE:
                return int(board.turn != chess.WHITE)*2 - 1
    return None



class Node:
    def __init__(self, policy: dict[tuple, float]):
        self.P = policy        # prior probabilities (dict: move -> prob)
        
        # dicts: move -> value
        self.N: dict[tuple, int] = {}            # visit count per move
        self.W: dict[tuple, float] = {}            # total value per move
        self.Q: dict[tuple, float] = {}            # mean value per move
        
        self.is_terminal:dict[tuple, bool] = {} # move -> terminal state

class SelfPlay:
    def __init__(self, model: ChessModel, num_simulations=200, temperature=1.0, batch_size=32):
        self.model = model
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.batch_size = batch_size

        self.TT: dict[int, Node] = {}   # zobrist -> Node
        
        self.model.eval()

    # =========================
    # PUBLIC ENTRY POINT
    # =========================
    def play_game(self, state: State):
        game_data = []
        step = 0

        while is_terminal(state.board) is None:
            if step >= 30:
                self.temperature = 0.0
            else:
                self.temperature = 1.0
            
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
            # print(decode_move(state.board, move).san(state.board), state.board.halfmove_clock)
            state.make_move(move)
            # print(move, self.TT[zhash].N[move]) 
            print(state.board.pretty()) # Debug line
            
            step += 1

        # game finished → assign values
        outcome = is_terminal(state.board)  # +1 / 0 / -1 from white perspective as always
        # print(outcome)

        return self.assign_values(game_data, outcome, GLOBAL_BUFFER)

    # =========================
    # MCTS
    # =========================
    def run_mcts(self, root_state: State):
        root_hash = root_state.board.__hash__()

        for _ in range(self.num_simulations):

            paths: deque[deque[tuple]] = deque() # hash, move
            with torch.no_grad():
                batch = self.simulate(root_state, root_hash, paths)
                p, v = self.model(batch)
            p: torch.Tensor 
            v: torch.Tensor
            # print(paths)
            for i, path in enumerate(paths): # Apply policies, values
                turn_value = path.pop()
                leaf_hash, leaf_move = path[-1]
                leaf_node = self.TT[leaf_hash] # hash, move
                policies = p[i]
                value = v[i].item()
                
                # Apply policies
                for move in leaf_node.P:
                    leaf_node.P[move] = policies[move].item()
                    
                # Apply values/ Remove virtual loss
                # print(VIRTUAL_LOSS+(value*turn_value))
                self.backpropagate(path, value, increase_visit=False, undo_move=False, demand_flip=turn_value, v_loss=-VIRTUAL_LOSS, state=root_state) # type: ignore
            

    def simulate(self, root_state: State, zhash, paths: deque[deque[tuple]]):
        batch = []
        path: deque[tuple] = deque()
        
        root_hash = root_state.board.__hash__()
        
        for i in range(self.batch_size):
            path.clear()
            
            while True: # Travelling
                if zhash not in self.TT: # Unexpanded node
                    # expand
                    # with torch.no_grad():
                    #     policy, value = self.model(state.tokens)
                    #     masked_p: dict[tuple, float] = dict()
                    
                    policies = {}
                    node = Node(policies)
                    for lmove in root_state.board.legal_moves():
                        e = encode_move(lmove, root_state.board)
                        policies[e] = 0
                        node.N[e] = 1
                        node.W[e] = -1

                    self.TT[zhash] = node
                    
                    # Append to the batch for GPU
                    batch.append(root_state.tokens)
                    
                    # adding an int that represent current turn for back-prop when the batch is full (-1 for black, 1 for white)
                    turn_value = int(not root_state.board.turn==chess.WHITE)*2-1
                    
                    self.backpropagate(path, value=0, v_loss=VIRTUAL_LOSS, state=root_state) # Apply virtual loss
                    
                    path.append(turn_value) #type: ignore // do this after back prop to not break it
                    
                    if zhash != root_hash: # No path needed if we're expanding root node
                        paths.append(path.copy())
                    
                    zhash = root_hash # travel from the root again
                    break
                
                # Travelling down
                node = self.TT[zhash]

                move = self.select_move(node)

                path.append((zhash, move))
                root_state.make_move(move)

                # If the node is a terminal 
                terminal_state = is_terminal(root_state.board)
                if terminal_state is not None:
                    self.backpropagate(path, value = terminal_state, state=root_state)
                    self.TT[zhash].is_terminal[move] = True
                    zhash = root_hash
                    break
                
                zhash = root_state.board.__hash__()
                
        
        # print(*paths, sep='\n')
        batch = torch.concat(batch)
        return batch

    def select_move(self, node: Node)-> tuple:
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
        return best_move # type: ignore


    def backpropagate(self, path: deque[tuple], value, state: State, increase_visit = True, undo_move = True, v_loss = 0, demand_flip = None):
        board = state.board
        
        # Revert based on demand or on turn
        if not demand_flip:
            if board.turn == chess.WHITE:
                value = -value
        else:
            value *= demand_flip

        
        iv = int(increase_visit)
        
        for hash, move in reversed(path):
            node = self.TT[hash]
            node: Node
            move: tuple
            node.N[move] = node.N.get(move, 0) + v_loss + iv
            node.W[move] = node.W.get(move, 0) + value - v_loss
            node.Q[move] = node.W[move] / node.N[move]
            value = -value  # switch player
            
            if undo_move and isinstance(board, chess.Board):
                state.unmake_move()

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

        for state_enc, pi in game_data:
            results.append((state_enc, pi, outcome))

        return results