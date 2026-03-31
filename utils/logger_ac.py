import os
import matplotlib.pyplot as plt


class ACLogger:

    def __init__(self):

        self.rewards = []
        self.values = []
        self.td_errors = []

        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, reward, value, td_error):

        self.rewards.append(reward)
        self.values.append(value)
        self.td_errors.append(td_error)

    def save(self):

        # ---------- Reward ----------
        plt.figure()
        plt.plot(self.rewards)
        plt.title("AC Reward Curve")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(self.log_dir, "ac_reward_curve.png"))
        plt.close()

        # ---------- Value ----------
        plt.figure()
        plt.plot(self.values)
        plt.title("Value Estimate Curve")
        plt.xlabel("Step")
        plt.ylabel("V(s)")
        plt.savefig(os.path.join(self.log_dir, "ac_value_curve.png"))
        plt.close()

        # ---------- TD Error ----------
        plt.figure()
        plt.plot(self.td_errors)
        plt.title("TD Error Curve")
        plt.xlabel("Step")
        plt.ylabel("TD Error")
        plt.savefig(os.path.join(self.log_dir, "ac_td_error_curve.png"))
        plt.close()