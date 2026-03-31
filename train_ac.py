from env.snake_env import SnakeEnv
from rl.actor_critic.ac_network import ACNetwork
from rl.actor_critic.actor_critic import ActorCriticAgent
from config import Config
from utils.logger_ac import ACLogger

logger = ACLogger()
env = SnakeEnv()

net = ACNetwork()
net.load()

agent = ActorCriticAgent(net)

logger = ACLogger()

episodes = 5000
best_reward = -1e9


for ep in range(episodes):

    state = env.reset()
    total_reward = 0

    while True:

        action, probs, h_a, value, h_c = agent.sample_action(state)

        next_state, reward, done = env.step(action)

        td_error = agent.update(
            state, action, reward, next_state, done,
            probs, h_a, value, h_c
        )

        # ---------- 记录 ----------
        logger.log(total_reward, value, td_error)

        state = next_state
        total_reward += reward

        if done:
            break

    # ---------- 保存 ----------
    net.save()

    if total_reward > best_reward:
        best_reward = total_reward
        net.save_backup(total_reward)

    if ep % 100 == 0:
        print(f"[AC] Episode {ep}, Reward: {total_reward}")
        logger.save()