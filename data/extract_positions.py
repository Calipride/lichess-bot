import io
import csv
from pathlib import Path

import zstandard as zstd
import chess
import chess.pgn

# Paths to your two PGN dumps
DATA_DIR = Path(__file__).resolve().parent
PGN_FILES = [
    DATA_DIR / "lichess_db_standard_rated_2017-06.pgn.zst",
    DATA_DIR / "lichess_db_standard_rated_2017-07.pgn.zst",
]

OUTPUT_CSV = DATA_DIR / "training_positions.csv"

# How much to sample (tune these if needed)
MAX_GAMES_PER_FILE = 5000      # e.g. first 5000 games from each file
SAMPLE_EVERY_N_PLIES = 3       # take every 3rd half-move to reduce size


def result_to_value(result_str: str) -> float:
    """
    Map PGN 'Result' to a value from White's perspective.
    1-0 -> +1, 0-1 -> -1, 1/2-1/2 -> 0, else -> None (skip).
    """
    if result_str == "1-0":
        return 1.0
    if result_str == "0-1":
        return -1.0
    if result_str == "1/2-1/2":
        return 0.0
    return None


def process_pgn_zst(zst_path: Path, writer: csv.writer,
                    max_games: int, sample_every: int) -> int:
    """
    Stream a .pgn.zst file, extract (fen, value) pairs and write to CSV.

    Returns number of games processed.
    """
    print(f"Processing {zst_path.name} ...")
    dctx = zstd.ZstdDecompressor(max_window_size=2**31)

    games_done = 0

    with zst_path.open("rb") as fh:
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")

            while games_done < max_games:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    # end of file
                    break

                games_done += 1
                if games_done % 500 == 0:
                    print(f"  {games_done} games processed...")

                result_str = game.headers.get("Result", "")
                res_white = result_to_value(result_str)
                if res_white is None:
                    continue  # skip unfinished / weird games

                board = game.board()
                ply = 0

                # Walk through main line moves
                for move in game.mainline_moves():
                    # Sample every Nth half-move to keep dataset smaller
                    if ply % sample_every == 0:
                        fen = board.fen()

                        # Convert result from side-to-move perspective
                        if board.turn == chess.WHITE:
                            value = res_white          # +1 if white eventually wins, etc.
                        else:
                            value = -res_white         # black's perspective

                        writer.writerow([fen, value])

                    board.push(move)
                    ply += 1

    print(f"Finished {zst_path.name}: {games_done} games used.")
    return games_done


def main():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["fen", "value"])  # header row

        total_games = 0
        for pgn_file in PGN_FILES:
            if not pgn_file.exists():
                print(f"WARNING: {pgn_file} not found, skipping.")
                continue
            total_games += process_pgn_zst(
                pgn_file,
                writer,
                max_games=MAX_GAMES_PER_FILE,
                sample_every=SAMPLE_EVERY_N_PLIES,
            )

    print(f"Done. Total games processed: {total_games}")
    print(f"Training data written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
