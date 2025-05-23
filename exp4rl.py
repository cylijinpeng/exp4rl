import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gym
from env import register_env
from EXP4RL.cloud import select_topk
# ------------------------------
# 1. Q-Network Definition
# ------------------------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------
# 2. EXP4-RL Meta-Algorithm (Multi-Arm Selection)
# ------------------------------
class EXP4RL:
    def __init__(self, experts, action_dim, gamma=0.99, eta=0.1, eps=1e-8):
        self.experts = experts       # list of expert agents
        self.K = action_dim          # number of arms
        self.gamma = gamma           # exploration parameter
        self.eta = eta               # learning rate for weights
        self.eps = eps
        self.weights = np.ones(len(experts))

    def select_action(self, state, n):
        # each expert outputs a probability vector over arms
        expert_probs = [expert.act(state) for expert in self.experts]  # list of shape (K,)
        W = np.sum(self.weights)
        # aggregate probabilities
        agg = np.zeros(self.K)
        for i, prob in enumerate(expert_probs):
            agg += (self.weights[i] / W) * prob
        # add small uniform exploration and renormalize
        agg = (1 - self.gamma) * agg + self.gamma / self.K
        agg = agg / agg.sum()
        # select top-n arms by aggregated probability
        topn = np.argsort(agg)[-n:][::-1]
        # build selection distribution p: equal weight for chosen arms
        p = np.zeros(self.K)
        p[topn] = 1.0 / n
        return topn, expert_probs, p

    def update_weights(self, expert_probs, p, rewards):
        # rewards: array shape (K,)
        for i, prob in enumerate(expert_probs):
            # compute importance-weighted payoff for expert i
            payoff = 0.0
            for j, r in enumerate(rewards):
                if p[j] > 0:
                    payoff += r / (p[j] + self.eps)
            self.weights[i] *= np.exp(self.eta * payoff / self.K)
        self.weights /= np.sum(self.weights)

    def step(self, state, selected_arms, expert_probs, p, rewards, next_state, done):
        # update each expert's internal model with experiences
        for expert in self.experts:
            for arm in selected_arms:
                expert.observe(state, arm, rewards[arm], next_state, done)
        # update EXP4 weights
        self.update_weights(expert_probs, p, rewards)

# ------------------------------
# 3. Expert Agents (output probabilities)
# ------------------------------
import torch.nn.functional as F

class DQNExpert:
    def __init__(self, state_dim, action_dim, lr=1e-3, buffer_size=10000, batch_size=64, target_update=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.target_update = target_update
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_q = QNetwork(state_dim, action_dim)
        self.target_q.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)
        self.learn_steps = 0
        self.epsilon = 1.0
        self.eps_min = 0.1
        self.eps_decay = 1e-5

    def act(self, state):
        # get Q-values
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            q_vals = self.q_net(s).squeeze(0)
        # softmax to probability
        probs = F.softmax(q_vals, dim=0).cpu().numpy()
        return probs

    def observe(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.learn()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        with torch.no_grad():
            q_next = self.target_q(next_states).max(1, keepdim=True)[0]
            target = rewards + (1 - dones) * 0.99 * q_next
        q_values = self.q_net(states).gather(1, actions)
        loss = nn.MSELoss()(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.target_q.load_state_dict(self.q_net.state_dict())
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_decay)

class RNDDQNExpert(DQNExpert):
    def __init__(self, state_dim, action_dim, beta=0.1, **kwargs):
        super().__init__(state_dim, action_dim, **kwargs)
        self.rnd_target = QNetwork(state_dim, state_dim)
        self.rnd_predictor = QNetwork(state_dim, state_dim)
        self.rnd_optimizer = optim.Adam(self.rnd_predictor.parameters(), lr=kwargs.get('lr', 1e-3))
        self.beta = beta

    def observe(self, state, action, reward, next_state, done):
        s = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            target_feat = self.rnd_target(s)
        pred_feat = self.rnd_predictor(s)
        int_reward = (target_feat - pred_feat).pow(2).mean().item()
        loss_rnd = nn.MSELoss()(pred_feat, target_feat)
        self.rnd_optimizer.zero_grad()
        loss_rnd.backward()
        self.rnd_optimizer.step()
        total_reward = reward + self.beta * int_reward
        super().observe(state, action, total_reward, next_state, done)

class AIExpert:
    def __init__(self, action_dim):
        self.action_dim = action_dim


    def act(self, state):
        DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
        JSON_FILE_PATH = "record.json"
        topk = select_topk(DEEPSEEK_API_KEY, JSON_FILE_PATH, self.action_dim)
        return topk

    def observe(self, state, action, reward, next_state, done):
        pass

# ------------------------------
# 4. Training Loop Example
# ------------------------------
def train_exp4(env, num_steps=100000, n_cache=5):
    state_dim = env.state_dim
    action_dim = env.action_dim
    eg_expert = DQNExpert(state_dim, action_dim)
    rnd_expert = RNDDQNExpert(state_dim, action_dim)
    ai_expert = AIExpert()
    meta = EXP4RL([eg_expert, rnd_expert, ai_expert], action_dim)
    state = env.reset()[0]
    for t in range(num_steps):
        selected_arms, expert_probs, p = meta.select_action(state, n_cache)
        # env.step should return a reward vector of length K
        next_state, rewards_vector, terminated, truncated, _ = env.step(selected_arms)
        done = terminated
        meta.step(state, selected_arms, expert_probs, p, rewards_vector, next_state, done)
        state = next_state if not done else env.reset()[0]
    return meta

def main():
    register_env()
    env = gym.make('MultiActionZeroEnv-v0',40,50,50000)  # 参数：状态维度、总臂数、回合数
    meta = train_exp4(env, num_steps=50000, n_cache=50) # n_cached是一回合选多少臂
    # print("训练完成，专家权重：", meta.weights)
    env.close()

if __name__ == "__main__":
    main()
