import argparse
import csv
import math
from pathlib import Path

import chess
import chess.engine


def cp_to_value(cp: float) -> float:
    return math.tanh(cp / 400.0)


def score_to_value(score: chess.engine.PovScore) -> float:
    if score.is_mate():
        mate_in = score.mate()
        return 1.0 if mate_in and mate_in > 0 else -1.0

    cp = score.score(mate_score=100000)
    if cp is None:
        return 0.0
    return cp_to_value(float(cp))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--in_csv", default="data/training_positions.csv")
    parser.add_argument("--out_csv", default="data/training_positions_stockfish.csv")
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--time", type=float, default=0.03)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    in_csv = (root / args.in_csv).resolve()
    out_csv = (root / args.out_csv).resolve()

    print(f"Reading from {in_csv}")
    print(f"Writing to {out_csv}")

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)

    try:
        with open(in_csv, newline="", encoding="utf-8") as fin, open(
            out_csv, "w", newline="", encoding="utf-8"
        ) as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=["fen", "value"])
            writer.writeheader()

            count = 0
            for row in reader:
                if count >= args.limit:
                    break

                fen = row["fen"]
                board = chess.Board(fen)

                info = engine.analyse(board, chess.engine.Limit(time=args.time))
                value = score_to_value(info["score"].pov(board.turn))

                writer.writerow({"fen": fen, "value": value})
                count += 1

                if count % 1000 == 0:
                    print(f"Labeled {count} positions")

        print(f"Done. Labeled {count} positions.")

    finally:
        engine.quit()


if __name__ == "__main__":
    main()
