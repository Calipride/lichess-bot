import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import chess

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_CSV = ROOT / "data" / "training_positions.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "value_model.pt"

# ---------- Feature encoding ----------
INPUT_DIM = 64 * 6  # 6 piece types × 64 squares


def fen_to_vector(fen: str) -> np.ndarray:
    """
    Convert a FEN string into a 384-dim vector (64 squares × 6 piece types).
    For each square & piece type:
      +1 = white piece
      -1 = black piece
       0 = no such piece
    """
    board = chess.Board(fen)
    x = np.zeros(INPUT_DIM, dtype=np.float32)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        pt_index = piece.piece_type - 1  # piece_type is 1..6
        idx = pt_index * 64 + sq        # each piece type gets a block of 64
        x[idx] = 1.0 if piece.color == chess.WHITE else -1.0

    return x


def load_dataset(csv_path: Path):
    fens = []
    values = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fens.append(row["fen"])
            values.append(float(row["value"]))

    X = np.stack([fen_to_vector(fen) for fen in fens])
    y = np.array(values, dtype=np.float32)
    return X, y


# ---------- Model ----------
class ValueNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),  # outputs in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


def train():
    print(f"Loading data from {DATA_CSV} ...")
    X, y = load_dataset(DATA_CSV)
    print(f"Dataset size: {len(X)} positions")

    # shuffle + train/val split (80/20)
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)

    split = int(0.8 * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx]).unsqueeze(1)

    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx]).unsqueeze(1)

    # DataLoaders
    batch_size = 256
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ValueNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5  # you can increase to 10+ later

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_loader.dataset)

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
        },
        MODEL_PATH,
    )

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
