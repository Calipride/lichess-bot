#!/usr/bin/env python3
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn


# ---------- MODEL LOADING ----------

MODEL_PATH = Path(__file__).resolve().parent / "models" / "value_model.pt"
INPUT_DIM = 64 * 6  # must match training


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


def load_value_model() -> ValueNet:
    data = torch.load(MODEL_PATH, map_location="cpu")
    model = ValueNet(input_dim=data["input_dim"])
    model.load_state_dict(data["model_state_dict"])
    model.eval()
    return model


value_model = load_value_model()


# ---------- EVALUATION USING THE MODEL ----------

@torch.no_grad()
def evaluate(board: chess.Board) -> float:
    """Return a score from the side-to-move perspective (bigger = better)."""
    # Hard terminal cases first
    if board.is_checkmate():
        # side to move is checkmated -> losing
        return -10000.0
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_threefold_repetition()
        or board.can_claim_fifty_moves()
    ):
        return 0.0

    vec = fen_to_vector(board.fen())
    x = torch.from_numpy(vec).unsqueeze(0)  # shape (1, 384)
    value = value_model(x).item()  # in [-1, 1]
    return value * 1000.0  # scale up for search


# ---------- SEARCH: MINIMAX + ALPHA-BETA ----------

INF = 1e9


def alphabeta(board: chess.Board, depth: int, alpha: float, beta: float) -> float:
    """Classic alpha–beta search from side-to-move perspective."""
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return evaluate(board)

    maximizing = board.turn  # True if white to move, False if black
    if maximizing:
        value = -INF
        for move in legal_moves:
            board.push(move)
            value = max(value, alphabeta(board, depth - 1, alpha, beta))
            board.pop()
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = INF
        for move in legal_moves:
            board.push(move)
            value = min(value, alphabeta(board, depth - 1, alpha, beta))
            board.pop()
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value


def choose_best_move(board: chess.Board, max_depth: int) -> chess.Move | None:
    """Pick the best move for the current side-to-move."""
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    best_move = None
    best_value = -INF if board.turn == chess.WHITE else INF

    for move in legal_moves:
        board.push(move)
        value = alphabeta(board, max_depth - 1, -INF, INF)
        board.pop()

        if board.turn == chess.WHITE:
            # we just popped, so board.turn is original side-to-move
            # but value is from side-to-move perspective at deeper node
            if value > best_value:
                best_value = value
                best_move = move
        else:
            if value < best_value:
                best_value = value
                best_move = move

    return best_move


def depth_from_movetime(ms: int) -> int:
    """Very rough mapping: more time -> deeper search."""
    if ms < 400:
        return 2
    if ms < 1200:
        return 3
    if ms < 3000:
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
                    pass

            max_depth = depth_from_movetime(movetime_ms)
            start = time.time()
            move = choose_best_move(board, max_depth=max_depth)
            elapsed = int((time.time() - start) * 1000)

            if move is None:
                print("bestmove 0000")
            else:
                board.push(move)
                print(f"info depth {max_depth} time {elapsed} score cp 0")
                print(f"bestmove {move.uci()}")
            sys.stdout.flush()

        elif cmd == "quit":
            break


if __name__ == "__main__":
    uci_loop()
