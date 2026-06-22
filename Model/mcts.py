import math
import numpy as np
import bulletchess as chess
from .board_encoder import State
from .move_encoder import encode_move, decode_move
from .chess_model import ChessModel
from collections import deque
import torch
import random
import threading
import queue
from copy import deepcopy
from .device import device


# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
CPUCT        = 1.41
GLOBAL_BUFFER: deque = deque()
TERMINAL_STATES = [
    chess.THREEFOLD_REPETITION,
    chess.FIFTY_MOVE_TIMEOUT,
    chess.STALEMATE,
    chess.CHECKMATE,
]
VIRTUAL_LOSS = 1

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:  2000,
}

# Sentinel pushed by each worker into the leaf_queue when it finishes
_WORKER_DONE = object()


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────
def is_enpassant(board: chess.Board, move: chess.Move) -> bool:
    piece = board[move.origin]
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    return move.destination == board.en_passant_square


def mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    score = 0
    attacker: chess.Piece = board[move.origin]  # type: ignore

    if is_enpassant(board, move):
        score = PIECE_VALUES[chess.PAWN] * 10 - PIECE_VALUES[attacker.piece_type]
    elif move.is_capture(board):
        victim: chess.Piece = board[move.destination]  # type: ignore
        score = PIECE_VALUES[victim.piece_type] * 10 - PIECE_VALUES[attacker.piece_type]

    board.apply(move)
    if board in chess.CHECK:
        score += 50
    board.undo()

    if move.is_castling(board):
        score += 30

    return score


def is_terminal(board: chess.Board):
    """
    Returns:
        None  → game still in progress
        0     → draw
        +1    → White wins  (from White's perspective)
        -1    → Black wins
    """
    if board.halfmove_clock >= 24:
        return 0

    for state in TERMINAL_STATES:
        if board in state:
            if state == chess.CHECKMATE:
                return int(board.turn != chess.WHITE) * 2 - 1
            return 0

    return None


# ──────────────────────────────────────────────────────────────────
# Transposition-table node
# ──────────────────────────────────────────────────────────────────
class Node:
    __slots__ = ("P", "N", "W", "Q", "is_terminal", "total_visit", "matdiff")

    def __init__(self, policy: dict):
        self.P: dict[tuple, float]      = policy
        self.N: dict[tuple, int]        = {}
        self.W: dict[tuple, float]      = {}
        self.Q: dict[tuple, float]      = {}
        self.is_terminal: dict[tuple, bool] = {}
        self.total_visit: int           = 0
        self.matdiff: dict[tuple, float] = {}


# ──────────────────────────────────────────────────────────────────
# SelfPlay  —  Multi-threaded MCTS with batched GPU inference
# ──────────────────────────────────────────────────────────────────
#
# Architecture
# ────────────
# • N worker threads each hold their own Board + move_stack + meta_planes copy
#   and walk the tree independently.  pos_cache is intentionally SHARED so that
#   incremental board encodings computed by one worker are visible to all others.
#
# • When a worker reaches an un-expanded leaf it:
#     1. Expands the node (under _tt_lock, double-checked).
#     2. Applies virtual loss for every node along the path (under _tt_lock).
#     3. Pushes  (tokens_tensor, path_snapshot, turn_value)  onto leaf_queue.
#     4. Unwinds back to root and starts the next simulation.
#
# • The main thread drains leaf_queue in chunks of `batch_size`, runs a single
#   GPU forward pass, then backpropagates all results (undoing virtual loss).
#
# • The TT (transposition table) is protected by a single RLock.
#   select_move reads N/Q/P without holding the lock (stale reads are harmless
#   in UCB — they never corrupt state).  Only _expand and backprop hold the lock.
#
class SelfPlay:
    def __init__(
        self,
        model: ChessModel,
        num_simulations: int = 50,
        temperature: float   = 1.0,
        batch_size: int      = 64,
        late_mul: int        = 2,
        latethresh: int      = 25,
        num_workers: int     = 4,
    ):
        self.model           = model
        self.num_simulations = num_simulations
        self.temperature     = temperature
        self.batch_size      = batch_size
        self.late_mul        = late_mul
        self.latethresh      = latethresh
        self.num_workers     = num_workers

        self.TT: dict[int, Node] = {}
        self._tt_lock = threading.RLock()   # RLock: re-entrant so _expand can nest

        self.step = 0
        self.model.eval()

    # ── Adaptive hyper-parameters ──────────────────────────────────
    def _is_late(self) -> bool:
        return self.step > self.latethresh

    def get_num_sim(self) -> int:
        mul = self.late_mul * self.late_mul if self._is_late() else 1
        return self.num_simulations * mul

    def get_batchsize(self) -> int:
        div = self.late_mul if self._is_late() else 1
        return max(1, self.batch_size // div)

    def get_cpuct(self) -> float:
        return CPUCT if self._is_late() else 2.5

    # ── Public entry point ─────────────────────────────────────────
    def play_game(self, state: State):
        game_data = []
        self.step  = 0

        while is_terminal(state.board) is None:
            zhash = state.board.__hash__()

            self.run_mcts(state)

            pi   = self.get_policy(zhash)
            game_data.append((state.tokens, pi))

            move = self.sample_move(pi)
            san  = decode_move(state.board, move).san(state.board)  # type: ignore
            state.make_move(move)

            print(san, self.TT[zhash].matdiff.get(move, 0))
            print(state.board.pretty())

            self.step += 1

        outcome = is_terminal(state.board)
        return self.assign_values(game_data, outcome, GLOBAL_BUFFER)

    # ── MCTS orchestrator ──────────────────────────────────────────
    def run_mcts(self, root_state: State):
        root_hash = root_state.board.__hash__()
        self._ensure_root_expanded(root_state, root_hash)

        target_visits = self.get_num_sim() * self.get_batchsize()
        leaf_queue: queue.Queue = queue.Queue()

        # Spawn workers — each gets its own board copy but shares pos_cache
        workers = []
        for _ in range(self.num_workers):
            worker_state = self._make_worker_state(root_state)
            t = threading.Thread(
                target=self._worker_loop,
                args=(worker_state, root_hash, target_visits, leaf_queue),
                daemon=True,
            )
            t.start()
            workers.append(t)

        # Main thread: drain queue → GPU inference → backprop
        batch_size  = self.get_batchsize()
        done_count  = 0
        pending: list[tuple] = []

        while done_count < self.num_workers:
            try:
                item = leaf_queue.get(timeout=0.05)
            except queue.Empty:
                if pending:
                    self._infer_and_backprop(pending, root_state)
                    pending.clear()
                continue

            if item is _WORKER_DONE:
                done_count += 1
                if pending:
                    self._infer_and_backprop(pending, root_state)
                    pending.clear()
                continue

            pending.append(item)  # type: ignore
            if len(pending) >= batch_size:
                self._infer_and_backprop(pending, root_state)
                pending.clear()

        if pending:
            self._infer_and_backprop(pending, root_state)

        for t in workers:
            t.join()

    # ── Worker state construction ──────────────────────────────────
    def _make_worker_state(self, root_state: State) -> State:
        """
        Create a worker-local State that is safe to mutate independently.

        Sharing rules:
          board          — deepcopy (workers traverse different branches)
          move_stack     — fresh empty deque (each worker tracks its own path)
          meta_planes    — clone (update_metadata writes in-place; must be isolated)
          history_planes — shared read-only (zero tensor, never mutated after init)
          pos_cache      — SHARED intentionally (incremental encodings benefit all
                           workers; concurrent writes are idempotent under the GIL)
        """
        ws = State.__new__(State)
        ws.board          = root_state.board.copy()
        ws.move_stack     = deque()
        ws.meta_planes    = root_state.meta_planes.clone()
        ws.history_planes = root_state.history_planes   # read-only zero tensor
        ws.pos_cache      = root_state.pos_cache        # shared, see docstring
        return ws

    # ── Worker thread body ─────────────────────────────────────────
    def _worker_loop(
        self,
        state: State,
        root_hash: int,
        target_visits: int,
        out: queue.Queue,
    ):
        """
        Repeatedly simulate from root until the root node has enough visits.
        Leaf states are pushed onto `out` for the main thread's GPU batch.
        """
        try:
            root_node = self.TT[root_hash]

            while root_node.total_visit < target_visits:
                path: deque[tuple] = deque()
                zhash = root_hash

                while True:
                    # ── Unexpanded leaf? ───────────────────────────
                    with self._tt_lock:
                        already_in_tt = zhash in self.TT

                    if not already_in_tt:
                        self._expand(state, zhash)

                        # Apply virtual loss along path before other workers see it
                        with self._tt_lock:
                            self._backpropagate_locked(
                                path, value=0,
                                board=state.board,
                                increase_visit=False,
                                v_loss=VIRTUAL_LOSS,
                            )

                        turn_value = int(not state.board.turn == chess.WHITE) * 2 - 1

                        # Push leaf for GPU inference (only non-root leaves)
                        if zhash != root_hash:
                            out.put((state.tokens.clone(), list(path), turn_value))

                        # Unwind back to root
                        for _ in path:
                            state.unmake_move()

                        break

                    # ── Selection ──────────────────────────────────
                    node = self.TT[zhash]
                    move = self._select_move(node, state.board)

                    path.append((zhash, move))
                    state.make_move(move)

                    # ── Terminal check ─────────────────────────────
                    terminal = is_terminal(state.board)
                    if terminal is not None:
                        with self._tt_lock:
                            node.is_terminal[move] = True
                            self._backpropagate_locked(
                                path, value=terminal,
                                board=state.board,
                                increase_visit=True,
                            )
                        for _ in path:
                            state.unmake_move()
                        break

                    zhash = state.board.__hash__()

        finally:
            out.put(_WORKER_DONE)

    # ── GPU inference + backpropagation ───────────────────────────
    def _infer_and_backprop(self, pending: list, root_state: State):
        """
        Batched forward pass → write policy priors → undo virtual loss.
        `pending` is a list of  (tokens_tensor, path_list, turn_value).
        """
        batch = torch.cat([item[0] for item in pending], dim=0).to(device)

        with torch.no_grad():
            p_batch, v_batch = self.model(batch)

        with self._tt_lock:
            for i, (_, path_list, turn_value) in enumerate(pending):
                if not path_list:
                    continue

                leaf_hash, _ = path_list[-1]
                if leaf_hash not in self.TT:
                    continue

                leaf_node = self.TT[leaf_hash]
                policies  = p_batch[i]
                value     = v_batch[i].item()

                # Write network policy priors into the leaf node
                for move in leaf_node.P:
                    leaf_node.P[move] = policies[move].item()

                # Undo virtual loss and credit real value in one pass
                self._backpropagate_locked(
                    deque(path_list),
                    value=value,
                    board=root_state.board,
                    increase_visit=True,
                    v_loss=-VIRTUAL_LOSS,
                    demand_flip=turn_value,
                )

    # ── Node expansion ─────────────────────────────────────────────
    def _expand(self, state: State, zhash: int) -> Node:
        """
        Create a new Node for `zhash` and register it in TT.
        Double-checked locking: only the first worker to arrive writes the node.
        """
        with self._tt_lock:
            if zhash in self.TT:        # another worker expanded it first
                return self.TT[zhash]

            board      = state.board
            is_white   = board.turn == chess.WHITE
            turn_value = 2 * int(is_white) - 1

            policies: dict[tuple, float] = {}
            node = Node(policies)

            for lmove in sorted(
                board.legal_moves(),
                key=lambda m: mvv_lva_score(board, m),
                reverse=True,
            ):
                e = encode_move(lmove, board)
                policies[e] = 0.0
                node.N[e]   = 0
                node.W[e]   = 0.0

                if is_enpassant(board, lmove):
                    node.matdiff[e] = turn_value * 1.0
                elif lmove.is_capture(board):
                    captured: chess.Piece = board[lmove.destination]  # type: ignore
                    node.matdiff[e] = turn_value * PIECE_VALUES[captured.piece_type] / 100

            self.TT[zhash] = node
            return node

    def _ensure_root_expanded(self, state: State, root_hash: int):
        with self._tt_lock:
            if root_hash not in self.TT:
                self._expand(state, root_hash)

    # ── Move selection (UCB + material heuristic) ──────────────────
    def _select_move(self, node: Node, board: chess.Board) -> tuple:
        c_puct   = self.get_cpuct()
        total_N  = sum(node.N.values())
        sqrt_N   = math.sqrt(total_N + 1)
        is_black = board.turn == chess.BLACK

        best_score = -1e9
        best_move  = None

        for move in node.P:
            Q    = node.Q.get(move, 0.0)
            U    = c_puct * node.P.get(move, 0.0) * sqrt_N / (1 + node.N.get(move, 0))
            diff = node.matdiff.get(move, 0.0) * 0.01
            if is_black:
                diff = -diff
            score = Q + U + diff
            if score > best_score:
                best_score = score
                best_move  = move

        return best_move  # type: ignore

    # Public alias for external callers
    def select_move(self, node: Node, board: chess.Board) -> tuple:
        return self._select_move(node, board)

    # ── Backpropagation  (caller MUST hold _tt_lock) ───────────────
    def _backpropagate_locked(
        self,
        path,
        value: float,
        board: chess.Board,
        increase_visit: bool = True,
        v_loss: int          = 0,
        demand_flip          = None,
    ):
        """
        Walk path in reverse and update N / W / Q.

        demand_flip: ±1 captured at the leaf's turn; used when the board position
                     no longer reflects the path (batched backprop).
                     Pass None to infer flip from board.turn.
        v_loss:      +VIRTUAL_LOSS to penalise, -VIRTUAL_LOSS to undo.
        """
        if demand_flip is None:
            # Value is from the perspective of the side that just moved; negate
            # so it becomes from the perspective of the node's parent.
            if board.turn == chess.WHITE:
                value = -value
        else:
            value *= demand_flip

        iv = int(increase_visit)

        for hash_, move in reversed(path):
            node = self.TT.get(hash_)
            if node is None:
                value = -value
                continue

            node.N[move]      = node.N.get(move, 0) + v_loss + iv
            node.W[move]      = node.W.get(move, 0.0) + value - v_loss
            n                 = node.N[move]
            node.Q[move]      = node.W[move] / n if n else 0.0
            node.total_visit += v_loss + iv

            value = -value

    # Public alias for external callers (acquires lock)
    def backpropagate(self, path, value, state: State,
                      increase_visit=True, v_loss=0, demand_flip=None):
        with self._tt_lock:
            self._backpropagate_locked(
                path, value, state.board,
                increase_visit=increase_visit,
                v_loss=v_loss,
                demand_flip=demand_flip,
            )

    # ── Policy + sampling ──────────────────────────────────────────
    def get_policy(self, zhash: int) -> dict:
        node   = self.TT[zhash]
        moves  = list(node.N.keys())
        visits = np.array(list(node.N.values()), dtype=np.float32)

        if self.step <= self.latethresh:
            total = visits.sum()
            pi    = visits / total if total > 0 else np.ones_like(visits) / max(len(visits), 1)
        else:
            pi        = np.zeros_like(visits)
            pi[int(np.argmax(visits))] = 1.0

        return dict(zip(moves, pi))

    def sample_move(self, pi: dict) -> tuple:
        moves = list(pi.keys())
        probs = list(pi.values())
        return random.choices(moves, weights=probs, k=1)[0]

    # ── Training targets ───────────────────────────────────────────
    def assign_values(self, game_data, outcome, buffer: deque | None = None):
        results = buffer if buffer is not None else deque()
        for state_enc, pi in game_data:
            results.append((state_enc, pi, outcome))
        return results