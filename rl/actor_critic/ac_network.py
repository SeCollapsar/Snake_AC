import numpy as np
import os
from datetime import datetime

from config import Config


class ACNetwork:
    """
    Actor + Critic 一体网络（统一保存）
    """

    def __init__(self, input_dim=300, hidden=128, output=4):

        # ---------- Actor ----------
        self.w1_a = np.random.randn(input_dim, hidden) * 0.01
        self.b1_a = np.zeros(hidden)

        self.w2_a = np.random.randn(hidden, output) * 0.01
        self.b2_a = np.zeros(output)

        # ---------- Critic ----------
        self.w1_c = np.random.randn(input_dim, hidden) * 0.01
        self.b1_c = np.zeros(hidden)

        self.w2_c = np.random.randn(hidden, 1) * 0.01
        self.b2_c = np.zeros(1)

    # =========================
    # forward
    # =========================
    def forward(self, x):

        # Actor
        h_a = np.dot(x, self.w1_a) + self.b1_a
        h_a = np.tanh(h_a)

        logits = np.dot(h_a, self.w2_a) + self.b2_a
        probs = self.softmax(logits)

        # Critic
        h_c = np.dot(x, self.w1_c) + self.b1_c
        h_c = np.tanh(h_c)

        value = np.dot(h_c, self.w2_c) + self.b2_c

        return probs, h_a, value[0], h_c

    def softmax(self, x, temperature=Config.TEMPERATURE):

        x = x / temperature
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    # =========================
    # 保存主模型（共享路径）
    # =========================
    def save(self):

        grid = Config.GRID_SIZE

        os.makedirs(Config.MODEL_DIR, exist_ok=True)

        path = os.path.join(
            Config.MODEL_DIR,
            f"policy_{grid}_ac.npy"
        )

        np.save(path, {
            "w1_a": self.w1_a,
            "b1_a": self.b1_a,
            "w2_a": self.w2_a,
            "b2_a": self.b2_a,
            "w1_c": self.w1_c,
            "b1_c": self.b1_c,
            "w2_c": self.w2_c,
            "b2_c": self.b2_c
        })

    # =========================
    # 加载模型
    # =========================
    def load(self):

        grid = Config.GRID_SIZE

        path = os.path.join(
            Config.MODEL_DIR,
            f"policy_{grid}_ac.npy"
        )

        if not os.path.exists(path):
            print("[INFO] No AC model found, training from scratch.")
            return

        data = np.load(path, allow_pickle=True).item()

        self.w1_a = data["w1_a"]
        self.b1_a = data["b1_a"]
        self.w2_a = data["w2_a"]
        self.b2_a = data["b2_a"]

        self.w1_c = data["w1_c"]
        self.b1_c = data["b1_c"]
        self.w2_c = data["w2_c"]
        self.b2_c = data["b2_c"]

        print(f"[INFO] Loaded AC model: {path}")

    # =========================
    # Top-K备份（AC专用）
    # =========================
    def save_backup(self, score):

        grid = Config.GRID_SIZE

        backup_dir = os.path.join(
            Config.MODEL_DIR,
            f"policy_{grid}_backup_ac"
        )

        os.makedirs(backup_dir, exist_ok=True)

        existing = []

        for f in os.listdir(backup_dir):
            if f.endswith(".npy"):
                try:
                    parts = f.replace(".npy", "").split("_")
                    reward = float(parts[2])
                    existing.append((reward, f))
                except:
                    continue

        # ---------- Top-K ----------
        if len(existing) >= Config.MAX_BACKUP_MODELS:

            existing.sort(key=lambda x: x[0])
            worst_reward, worst_file = existing[0]

            if score <= worst_reward:
                return

            os.remove(os.path.join(backup_dir, worst_file))
            print(f"[AC REMOVE WORST] {worst_file}")

        # ---------- 保存 ----------
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"policy_{grid}_{score:.2f}_{time_str}.npy"

        path = os.path.join(backup_dir, filename)

        np.save(path, {
            "w1_a": self.w1_a,
            "b1_a": self.b1_a,
            "w2_a": self.w2_a,
            "b2_a": self.b2_a,
            "w1_c": self.w1_c,
            "b1_c": self.b1_c,
            "w2_c": self.w2_c,
            "b2_c": self.b2_c
        })

        print(f"[AC TOP-K SAVE] {filename}")