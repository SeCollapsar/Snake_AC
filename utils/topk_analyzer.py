import os
import matplotlib.pyplot as plt
from config import Config


def analyze_topk():

    grid = Config.GRID_SIZE

    backup_dir = os.path.join(
        Config.MODEL_DIR,
        f"policy_{grid}_backup_r"
    )

    rewards = []

    for f in os.listdir(backup_dir):
        if f.endswith(".npy"):
            try:
                parts = f.replace(".npy", "").split("_")
                reward = float(parts[2])
                rewards.append(reward)
            except:
                continue

    rewards.sort(reverse=True)

    plt.figure()
    plt.bar(range(len(rewards)), rewards)
    plt.title("Top-K Models Reward")
    plt.xlabel("Rank")
    plt.ylabel("Reward")

    plt.savefig("logs/topk.png")
    plt.close()