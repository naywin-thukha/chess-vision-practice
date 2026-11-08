"""Evening block: python-chess + Stockfish (UCI).

Run from the repo root:
    python3 engine/engine_test.py            # engine answers 1.e4, then a small play-along
    python3 engine/engine_test.py --play     # you type moves, Stockfish replies
Stockfish path: found automatically, or pass --engine /path/to/stockfish
"""
import argparse
import shutil
import sys

import chess
import chess.engine

CANDIDATES = ["/opt/homebrew/bin/stockfish",   # Mac, Apple Silicon (brew)
              "/usr/local/bin/stockfish",      # Mac, Intel (brew)
              "/usr/games/stockfish"]          # Linux apt


def find_stockfish(explicit=None):
    path = explicit or shutil.which("stockfish")
    if not path:
        for p in CANDIDATES:
            if shutil.which(p):
                path = p
                break
    if not path:
        sys.exit("Stockfish not found. On Mac run: brew install stockfish")
    return path


def best_move(engine, board, seconds=0.2):
    """Ask Stockfish for its move. Returns a chess.Move (not played yet)."""
    result = engine.play(board, chess.engine.Limit(time=seconds))
    return result.move


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine")
    ap.add_argument("--play", action="store_true")
    args = ap.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci(find_stockfish(args.engine))
    try:
        # --- Part 1: python-chess basics ---
        board = chess.Board()                       # starting position
        print(board)                                # ASCII board
        print("legal moves:", board.legal_moves.count())          # 20
        print("is e2e4 legal?", chess.Move.from_uci("e2e4") in board.legal_moves)

        # --- Part 2: engine answers 1.e4 ---
        board.push_uci("e2e4")                      # play 1.e4 for White
        move = best_move(engine, board)
        print("Stockfish replies:", move.uci(), "| SAN:", board.san(move))
        assert move in board.legal_moves
        print("OK: engine returned a legal move")

        # --- Part 3: load any position from a FEN ---
        fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
        board = chess.Board(fen)
        print("from FEN, White to move:", board.turn == chess.WHITE,
              "| engine says:", best_move(engine, board).uci())

        # --- Part 4 (optional): play against it in the terminal ---
        if args.play:
            board = chess.Board()
            while not board.is_game_over():
                print(board, "\n")
                text = input("Your move (e.g. e2e4, q to quit): ").strip()
                if text == "q":
                    break
                try:
                    board.push_uci(text)
                except ValueError:
                    print("Illegal or badly typed move, try again.")
                    continue
                if board.is_game_over():
                    break
                reply = best_move(engine, board)
                print("Stockfish plays:", reply.uci())
                board.push(reply)
            print("Result:", board.result())
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
