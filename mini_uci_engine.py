#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn
INF = 10_000
LAST_BESTMOVE_UCI = None

# ---------- MODEL LOADING ----------

MODEL_PATH = Path(__file__).resolve().parent / "models" / "value_model.pt"
INPUT_DIM = 64 * 6  # must match training
print(f"Loaded model from: {MODEL_PATH}")

class SearchTimeout(Exception):
    pass

class ValueNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),  # value in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


def fen_to_vector(fen: str) -> np.ndarray:
    """Same encoding we used during training."""
    board = chess.Board(fen)
    x = np.zeros(INPUT_DIM, dtype=np.float32)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        pt_index = piece.piece_type - 1  # piece_type is 1..6
        idx = pt_index * 64 + sq
        x[idx] = 1.0 if piece.color == chess.WHITE else -1.0

    return x


def load_value_model():
    model = ValueNet(input_dim=INPUT_DIM)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model



value_model = load_value_model()


# ---------- EVALUATION USING THE MODEL ----------


@torch.no_grad()
def evaluate(board: chess.Board) -> float:
    if board.is_checkmate():
        return -10000.0

    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves():
        return 0.0

    vec = fen_to_vector(board.fen())
    x = torch.from_numpy(vec).unsqueeze(0)
    value = value_model(x).item() * 1000.0

    # ---- MATERIAL PROGRESS BONUS ----
    material = sum(
        len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK))
        for pt in chess.PIECE_TYPES
    )
    value += material * 5

    # discourage repetition
    if board.can_claim_threefold_repetition():
        value -= 50

    return value if board.turn == chess.WHITE else -value




# ---------- SEARCH: MINIMAX + ALPHA-BETA ----------

INF = 1e9


def alphabeta(board: chess.Board, depth: int, alpha: float, beta: float, deadline: float) -> float:
    if time.time() >= deadline:
        raise SearchTimeout()
    # Soft penalty to avoid repetition lines deep in the tree
    if board.is_repetition(2) or board.can_claim_threefold_repetition():
        return evaluate(board) - 150.0

    if depth == 0 or board.is_game_over():
        return evaluate(board)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return evaluate(board)

    maximizing = board.turn
    if maximizing:
        value = -INF
        for move in legal_moves:
            board.push(move)
            value = max(value, alphabeta(board, depth - 1, alpha, beta, deadline))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = INF
        for move in legal_moves:
            board.push(move)
            value = min(value, alphabeta(board, depth - 1, alpha, beta, deadline))
            board.pop()
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value



def ordered_moves(board: chess.Board):
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: board.is_capture(m), reverse=True)
    return moves


def choose_best_move_iterative(board: chess.Board, time_limit_ms: int, max_depth_cap: int = 6):
    global LAST_BESTMOVE_UCI

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None, 0, 0

    start = time.time()
    deadline = start + time_limit_ms / 1000.0

    best_move = legal_moves[0]
    best_score = -INF
    reached_depth = 0

    for depth in range(1, max_depth_cap + 1):
        if time.time() >= deadline:
            break

        current_best_move = best_move
        current_best_score = -INF

        for move in ordered_moves(board):
            if time.time() >= deadline:
                break

            # --- anti ping-pong: don't immediately repeat our last bestmove ---
            if LAST_BESTMOVE_UCI is not None and move.uci() == LAST_BESTMOVE_UCI:
                continue

            board.push(move)
            try:
                score = -alphabeta(board, depth - 1, -INF, INF, deadline)

                # discourage repetition / shuffling at the root
                if board.is_repetition(2) or board.can_claim_threefold_repetition():
                    score -= 150.0

            except SearchTimeout:
                break
            finally:
                board.pop()

            if score > current_best_score:
                current_best_score = score
                current_best_move = move

        if current_best_score > -INF:
            best_move = current_best_move
            best_score = current_best_score
            reached_depth = depth
        else:
            break

    # remember what we played so we avoid undoing it next turn
    if best_move is not None:
        LAST_BESTMOVE_UCI = best_move.uci()

    return best_move, best_score, reached_depth


def depth_from_movetime(ms: int) -> int:
    if ms < 300:
        return 2
    if ms < 800:
        return 3
    if ms < 1500:
        return 4
    return 5





# ---------- UCI LOOP ----------

def uci_loop():
    board = chess.Board()
    movetime_ms = 1000
    max_depth = 3

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        cmd = line.strip()

        if cmd == "uci":
            print("id name MiniCaliprideEngine")
            print("id author Calipride")
            print("uciok")
            sys.stdout.flush()


        elif cmd == "isready":
            print("readyok")
            sys.stdout.flush()

        elif cmd.startswith("setoption"):
            # optional: handle engine options here if you want later
            pass

        elif cmd == "ucinewgame":
            board = chess.Board()

        elif cmd.startswith("position"):
            parts = cmd.split()
            if "startpos" in parts:
                board = chess.Board()
                if "moves" in parts:
                    idx = parts.index("moves") + 1
                    for mv in parts[idx:]:
                        board.push_uci(mv)
            elif "fen" in parts:
                fi = parts.index("fen") + 1
                fen = " ".join(parts[fi : fi + 6])
                board = chess.Board(fen)
                if "moves" in parts:
                    idx = parts.index("moves") + 1
                    for mv in parts[idx:]:
                        board.push_uci(mv)

                elif cmd.startswith("go"):
                    parts = cmd.split()
            movetime_ms = 1000

            if "movetime" in parts:
                movetime_ms = int(parts[parts.index("movetime") + 1])
            elif "wtime" in parts and "btime" in parts:
                try:
                    wtime = int(parts[parts.index("wtime") + 1])
                    btime = int(parts[parts.index("btime") + 1])
                    remain = wtime if board.turn == chess.WHITE else btime
                    movetime_ms = max(200, remain // 25)
                except Exception:
                    movetime_ms = 1000

            movetime_ms = min(movetime_ms, 3000)

            start = time.time()
            try:
                move, score_cp, depth = choose_best_move_iterative(
                    board,
                    movetime_ms,
                    max_depth_cap=6,
                )
            except SearchTimeout:
                legal = list(board.legal_moves)
                move = legal[0] if legal else None
                score_cp = 0
                depth = 1

            elapsed = int((time.time() - start) * 1000)

            if move is None:
                print("bestmove 0000")
            else:
                # IMPORTANT: lichess-bot expects score cp to be an INTEGER
                score_cp_int = int(round(score_cp))
                print(f"info depth {depth} time {elapsed} score cp {score_cp_int}")
                print(f"bestmove {move.uci()}")

            sys.stdout.flush()

        elif cmd == "quit":
            break


if __name__ == "__main__":
    uci_loop()
