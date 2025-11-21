#!/usr/bin/env python3
import sys, time
import chess
import chess.polyglot

# ---------- Evaluation: material + piece-square + simple features ----------

PIECE_VALUE = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
}

PAWN = [
     0,  5,  5, -5, -5,  5,  5,  0,
     0, 10, -5,  0,  0, -5, 10,  0,
     0, 10, 10, 20, 20, 10, 10,  0,
     5, 15, 15, 25, 25, 15, 15,  5,
    10, 20, 20, 30, 30, 20, 20, 10,
    20, 25, 25, 35, 35, 25, 25, 20,
    50, 50, 50, 60, 60, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
]
KNIGHT = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]
BISHOP = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]
ROOK = [
     0,  0,  5, 10, 10,  5,  0,  0,
    -5,  0,  0,  5,  5,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
QUEEN = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -10,  5,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  0,  5,  5,  5,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]
KING_MID = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]

PSQT = {
    chess.PAWN: PAWN, chess.KNIGHT: KNIGHT, chess.BISHOP: BISHOP,
    chess.ROOK: ROOK, chess.QUEEN: QUEEN, chess.KING: KING_MID
}

def psqt_score(board: chess.Board) -> int:
    s = 0
    for ptype, table in PSQT.items():
        for sq in board.pieces(ptype, chess.WHITE):
            s += table[chess.square_mirror(sq)]
        for sq in board.pieces(ptype, chess.BLACK):
            s -= table[chess.square_mirror(chess.square_mirror(sq))]
    return s

def material(board: chess.Board) -> int:
    s = 0
    for p in PIECE_VALUE:
        s += PIECE_VALUE[p] * (len(board.pieces(p, chess.WHITE)) - len(board.pieces(p, chess.BLACK)))
    return s

def mobility(board: chess.Board) -> int:
    m = board.legal_moves.count()
    board.push(chess.Move.null())
    m_opp = board.legal_moves.count()
    board.pop()
    return 2 * m - m_opp

def pawn_structure(board: chess.Board) -> int:
    s = 0
    for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
        files = [0]*8
        pawns = board.pieces(chess.PAWN, color)
        for sq in pawns:
            files[chess.square_file(sq)] += 1
        s += sign * sum(-12 for f in files if f >= 2)
        for f, cnt in enumerate(files):
            if cnt > 0:
                left = files[f-1] if f-1 >= 0 else 0
                right = files[f+1] if f+1 <= 7 else 0
                if left == 0 and right == 0:
                    s += sign * -10
    return s

def evaluate(board: chess.Board) -> int:
    if board.is_checkmate():
        return -999999 if board.turn else 999999
    if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_threefold_repetition():
        return 0
    return material(board) + psqt_score(board) + mobility(board) + pawn_structure(board)

# ---------- Search: alpha-beta + simple q-search, TT, move ordering ----------

INF = 10**9
TT = {}

def zobrist(board: chess.Board) -> int:
    return chess.polyglot.zobrist_hash(board)

def is_capture_or_promo(board: chess.Board, move: chess.Move) -> bool:
    return board.is_capture(move) or move.promotion is not None

def order_moves(board: chess.Board, moves):
    caps, quiets = [], []
    for mv in moves:
        (caps if is_capture_or_promo(board, mv) else quiets).append(mv)
    caps.sort(
        key=lambda m: PIECE_VALUE.get(board.piece_type_at(m.to_square), 0)
                      - PIECE_VALUE.get(board.piece_type_at(m.from_square), 0),
        reverse=True,
    )
    return caps + quiets

def quiescence(board: chess.Board, alpha: int, beta: int, start_time: float, time_ms: int) -> int:
    stand_pat = evaluate(board)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    for move in order_moves(board, [m for m in board.legal_moves if board.is_capture(m)]):
        if (time.time() - start_time) * 1000 > time_ms:
            break
        board.push(move)
        score = -quiescence(board, -beta, -alpha, start_time, time_ms)
        board.pop()
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha

def alphabeta(board: chess.Board, depth: int, alpha: int, beta: int, start_time: float, time_ms: int) -> int:
    if (time.time() - start_time) * 1000 > time_ms:
        return evaluate(board)
    key = (zobrist(board), depth, board.turn)
    if key in TT:
        d, val, flag, _ = TT[key]
        if d >= depth:
            if flag == 0:
                return val
            elif flag == -1 and val <= alpha:
                return val
            elif flag == 1 and val >= beta:
                return val

    if depth == 0:
        return quiescence(board, alpha, beta, start_time, time_ms)

    legal = list(board.legal_moves)
    if not legal:
        return evaluate(board)

    best_val = -INF
    any_move = None
    for move in order_moves(board, legal):
        if (time.time() - start_time) * 1000 > time_ms:
            break
        board.push(move)
        val = -alphabeta(board, depth - 1, -beta, -alpha, start_time, time_ms)
        board.pop()
        if val > best_val:
            best_val = val
            any_move = move
        if val > alpha:
            alpha = val
        if alpha >= beta:
            break

    flag = 0
    if best_val <= alpha:
        flag = -1
    if best_val >= beta:
        flag = 1
    TT[key] = (depth, best_val, flag, any_move)
    return best_val

def choose_move(board: chess.Board, time_ms: int = 1500, max_depth: int = 5) -> chess.Move:
    start = time.time()
    best_move = None
    for depth in range(1, max_depth + 1):
        alpha, beta = -INF, INF
        local_best = None
        legal = list(board.legal_moves)
        if not legal:
            return None
        for move in order_moves(board, legal):
            if (time.time() - start) * 1000 > time_ms:
                break
            board.push(move)
            val = -alphabeta(board, depth - 1, -beta, -alpha, start, time_ms)
            board.pop()
            if local_best is None or val > alpha:
                alpha = val
                local_best = move
        if local_best:
            best_move = local_best
        if (time.time() - start) * 1000 > time_ms * 0.95:
            break
    return best_move or next(iter(board.legal_moves))

# ---------- UCI loop ----------

def uci_loop():
    board = chess.Board()
    movetime_ms = 1500
    max_depth = 5

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        cmd = line.strip()

        if cmd == "uci":
            print("id name CaliprideEngine")
            print("id author Calipride")
            print("uciok")
            sys.stdout.flush()

        elif cmd == "isready":
            print("readyok")
            sys.stdout.flush()

        elif cmd.startswith("setoption"):
            parts = cmd.split()
            if "Depth" in parts:
                try:
                    max_depth = int(parts[parts.index("value") + 1])
                except Exception:
                    pass
            if "MoveTime" in parts:
                try:
                    movetime_ms = int(parts[parts.index("value") + 1])
                except Exception:
                    pass

        elif cmd == "ucinewgame":
            board = chess.Board()
            TT.clear()

        elif cmd.startswith("position"):
            parts = cmd.split()
            if "startpos" in parts:
                board = chess.Board()
                if "moves" in parts:
                    idx = parts.index("moves") + 1
                    for uci in parts[idx:]:
                        board.push_uci(uci)
            elif "fen" in parts:
                fi = parts.index("fen") + 1
                fen = " ".join(parts[fi:fi+6])
                board = chess.Board(fen)
                if "moves" in parts:
                    idx = parts.index("moves") + 1
                    for uci in parts[idx:]:
                        board.push_uci(uci)

        elif cmd.startswith("go"):
            parts = cmd.split()
            if "movetime" in parts:
                movetime_ms = int(parts[parts.index("movetime")+1])
            elif "wtime" in parts and "btime" in parts:
                try:
                    wtime = int(parts[parts.index("wtime")+1])
                    btime = int(parts[parts.index("btime")+1])
                    remain = wtime if board.turn == chess.WHITE else btime
                    movetime_ms = max(200, remain // 30)
                except Exception:
                    pass
            move = choose_move(board, time_ms=movetime_ms, max_depth=max_depth)
            if move is None:
                print("bestmove 0000")
            else:
                print(f"bestmove {move.uci()}")
            sys.stdout.flush()

        elif cmd == "quit":
            break

if __name__ == "__main__":
    uci_loop()
