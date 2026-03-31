import numpy as np


class ActorCriticAgent:

    def __init__(self, net, lr_actor=0.001, lr_critic=0.001, gamma=0.99):

        self.net = net
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma

    def sample_action(self, state):

        probs, h_a, value, h_c = self.net.forward(state)

        action = np.random.choice(len(probs), p=probs)

        return action, probs, h_a, value, h_c

    def update(self, state, action, reward, next_state, done,
               probs, h_a, value, h_c):

        _, _, next_value, _ = self.net.forward(next_state)

        assert np.isscalar(value), "value must be scalar"
        assert np.isscalar(next_value), "next_value must be scalar"

        target = reward + (0 if done else self.gamma * next_value)

        td_error = target - value

        # ---------- Critic ----------
        dvalue = td_error

        dw2 = np.outer(h_c, [dvalue])
        db2 = np.array([dvalue])

        dh = self.net.w2_c.flatten() * dvalue
        dh = (1 - h_c ** 2) * dh

        dw1 = np.outer(state, dh)
        db1 = dh

        self.net.w2_c += self.lr_critic * dw2
        self.net.b2_c += self.lr_critic * db2
        self.net.w1_c += self.lr_critic * dw1
        self.net.b1_c += self.lr_critic * db1

        # ---------- Actor ----------
        dlog = -probs
        dlog[action] += 1

        dlog *= td_error

        dw2 = np.outer(h_a, dlog)
        db2 = dlog

        dh = np.dot(self.net.w2_a, dlog)
        dh = (1 - h_a ** 2) * dh

        dw1 = np.outer(state, dh)
        db1 = dh

        self.net.w2_a += self.lr_actor * dw2
        self.net.b2_a += self.lr_actor * db2
        self.net.w1_a += self.lr_actor * dw1
        self.net.b1_a += self.lr_actor * db1

        return td_error