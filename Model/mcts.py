import math
import numpy as np
import bulletchess as chess
from .board_encoder import State
from .move_encoder import encode_move, decode_move
from copy import deepcopy
from .chess_model import ChessModel
from collections import deque
import torch, random, gc


CPUCT = 1.41
GLOBAL_BUFFER = deque()
TERMINAL_STATES = [chess.THREEFOLD_REPETITION, chess.FIFTY_MOVE_TIMEOUT, chess.STALEMATE, chess.CHECKMATE]
VIRTUAL_LOSS = 1

PIECE_VALUES = {
    chess.PAWN: 100,   # pawn
    chess.KNIGHT: 320,   # knight
    chess.BISHOP: 330,   # bishop
    chess.ROOK: 500,   # rook
    chess.QUEEN: 900,   # queen
    chess.KING: 2000  # king (để cho đủ)
}

def mvv_lva_score(board: chess.Board, move: chess.Move):
    # No capture → lowest priority
    victim_piece: chess.Piece = board[move.destination] #type: ignore
    attacker_piece: chess.Piece = board[move.origin] #type: ignore
    
    score = 0
    
    if is_enpassant(board, move): # is_capture() won't detect enpassant so we do this :/
        victim_value = PIECE_VALUES[chess.PAWN]
        attacker_value = PIECE_VALUES[attacker_piece.piece_type]
        score = victim_value * 10 - attacker_value
    elif move.is_capture(board):
        victim_value = PIECE_VALUES[victim_piece.piece_type]
        attacker_value = PIECE_VALUES[attacker_piece.piece_type]
        score = victim_value * 10 - attacker_value

    # --- checks ---
    board.apply(move)
    if board in chess.CHECK:
        score += 50
    board.undo()
    # --- castling ---
    if move.is_castling(board):
        score += 30

    # MVV-LVA: maximize victim, minimize attacker
    return score

def is_enpassant(board: chess.Board, move: chess.Move):
    piece = board[move.origin]
    if piece is None:
        return False
    if piece.piece_type != chess.PAWN:
        return False
    if move.destination == board.en_passant_square:
        return True
    
    return False

def debug_path(path: deque[tuple], board: chess.Board):
    a = []
    for hash, item in reversed(path):
        board.undo()
        a.append(decode_move(board, item))
    
    print(list(reversed(a)))

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
        self.total_visit = 0
        
        self.matdiff: dict[tuple, float] = {}

class SelfPlay:
    def __init__(self, model: ChessModel, num_simulations=50, temperature=1.0, batch_size=64, late_mul=2, latethresh=25):
        self.model = model
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.batch_size = batch_size
        
        self.latethresh = latethresh
        self.late_mul = late_mul

        self.TT: dict[int, Node] = {}   # zobrist -> Node
        self.step = 0
        self.model.eval()

    def get_num_sim(self):
        if self.step <= self.latethresh:
            num_sim = self.num_simulations
        else:
            num_sim = self.num_simulations*self.late_mul*self.late_mul # Increase search depth
        return num_sim
    
    def get_batchsize(self):
        if self.step <= self.latethresh:
            batch_size = self.batch_size
        else:
            batch_size = int(self.batch_size/self.late_mul) # Lower search width
        
        return batch_size
    
    def get_cpuct(self):
        if self.step <= self.latethresh:
            c_puct = 2.5
        else:
            c_puct = CPUCT # Do more accurate moves
        
        return c_puct
    
    # =========================
    # PUBLIC ENTRY POINT
    # =========================
    def play_game(self, state: State):
        game_data = []
        self.step = 0

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
            san = decode_move(state.board, move).san(state.board) #type: ignore
            state.make_move(move)
            # print(move, self.TT[zhash].N[move]) 
            print(san, self.TT[zhash].matdiff.get(move, 0))
            print(state.board.pretty()) # Debug line
            
            self.step += 1

        # game finished → assign values
        outcome = is_terminal(state.board)  # +1 / 0 / -1 from white perspective as always
        # print(outcome)

        return self.assign_values(game_data, outcome, GLOBAL_BUFFER)

    # =========================
    # MCTS
    # =========================
    def run_mcts(self, root_state: State):
        root_hash = root_state.board.__hash__()
        
        if self.TT.get(root_hash) is None: # Expand root first
            policies = {}
            node = Node(policies)
            for lmove in sorted(root_state.board.legal_moves(), key=lambda x: mvv_lva_score(root_state.board, x), reverse=True):
                e = encode_move(lmove, root_state.board)
                policies[e] = 0
                node.N[e] = 0
                node.W[e] = 0

            self.TT[root_hash] = node
        
        
        root_node = self.TT[root_hash]
        
        num_sim = self.get_num_sim()
        bs = self.get_batchsize()

        while root_node.total_visit < num_sim*bs:

            paths: deque[deque[tuple]] = deque() # hash, move
            batch = self.simulate(root_state, paths)
            if batch != []:
                with torch.no_grad():
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
            

    def simulate(self, root_state: State, paths: deque[deque[tuple]]):
        batch = []
        path: deque[tuple] = deque()
        
        root_hash = root_state.board.__hash__()
        root_node = self.TT[root_hash]
        zhash = root_hash
        pzhash = None
        
        batch_size = self.get_batchsize()
        
        for i in range(batch_size):
            path.clear()
            curr_diff = root_node.matdiff
            
            while True: # Travelling
                if zhash not in self.TT: # Unexpanded node
                    # expand
                    # with torch.no_grad():
                    #     policy, value = self.model(state.tokens)
                    #     masked_p: dict[tuple, float] = dict()
                    
                    policies = {}
                    node = Node(policies)
                    node.matdiff = curr_diff
                    
                    for lmove in sorted(root_state.board.legal_moves(), key=lambda x: mvv_lva_score(root_state.board, x), reverse=True):
                        e = encode_move(lmove, root_state.board)
                        policies[e] = 0
                        node.N[e] = 0
                        node.W[e] = 0
                        
                        turn_value = 2*int(root_state.board.turn == chess.WHITE) - 1
                        
                        if is_enpassant(root_state.board, lmove):
                            node.matdiff[e] = 1 * turn_value
                        if lmove.is_capture(root_state.board):
                            captured: chess.Piece = root_state.board[lmove.destination] #type: ignore
                            # print(root_state.board.pretty(), lmove.san(root_state.board), end='\n==================\n')
                            node.matdiff[e] = turn_value * PIECE_VALUES[captured.piece_type]/100
                            
                            print(lmove.san(root_state.board))
                            print(root_state.board.pretty())

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
                    pzhash = None
                    break
                
                # Travelling down
                node = self.TT[zhash]

                move = self.select_move(node, root_state.board)

                path.append((zhash, move))
                root_state.make_move(move)
                
                curr_diff = self.TT[zhash].matdiff

                # If the node is a terminal 
                terminal_state = is_terminal(root_state.board)
                if terminal_state is not None:
                    # print(terminal_state)
                    # print(root_state.board.pretty())
                    # debug_path(path, root_state.board.copy())
                    self.backpropagate(path, value = terminal_state, state=root_state)
                    self.TT[zhash].is_terminal[move] = True
                    zhash = root_hash 
                    pzhash = None
                    break
                pzhash = zhash
                zhash = root_state.board.__hash__() # Travel down
                
        
        # print(*paths, sep='\n')
        if batch:
            batch = torch.concat(batch)
        return batch

    def select_move(self, node: Node, board: chess.Board)-> tuple:
        total_N = sum(node.N.values())

        best_score = -1e9
        best_move = None
        
        c_puct = self.get_cpuct()
        

        for move in node.P:
            Q = node.Q.get(move, 0)
            U = c_puct * node.P.get(move, 0) * math.sqrt(total_N + 1) / (1 + node.N.get(move, 0)) #type: ignore
            
            diff = node.matdiff.get(move, 0)*0.01
            if board.turn == chess.BLACK:
                diff = -diff
            
            score = Q + U + diff #type: ignore

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
            
            node.total_visit += v_loss + iv
            
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
        
        if self.step <= self.latethresh:
            temperature = 1
        else:
            temperature = 0

        if temperature == 0:
            best = np.argmax(visits)
            pi = np.zeros_like(visits)
            pi[best] = 1.0
        else:
            visits = visits ** (1 / temperature)
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