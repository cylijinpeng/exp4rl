import gym
from gym import spaces
import numpy as np

class MultiActionZeroEnv(gym.Env):
    """
    A simple Gym environment where each step accepts an array of n action indices
    and returns a zero reward vector of length n.
    Observation is a zero vector of length n.
    """

    metadata = {"render.modes": []}

    def __init__(self, state_dim=4, action_dim=5, max_steps=100):
        super(MultiActionZeroEnv, self).__init__()
        self.state_dim=state_dim  # 状态维度
        self.action_dim = action_dim # 动作总数/臂总数
        self.max_steps = max_steps
        # Define action space: n actions, each with discrete choices
        self.action_space = spaces.MultiDiscrete([action_dim] * self.state_dim)

        # Define observation space: state_dim length, each dimension is continuous
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # 重置环境信息，返回初始状态
        self.current_step = 0
        obs = np.zeros(self.state_dim, dtype=np.float32)  # 初始状态为零向量
        info = {}  # 额外的信息（这里为空）
        return obs, info

    def step(self, select_arms):
        # 输入当前选择的臂，返回执行完动作后的状态、奖励（数组，每个臂一个奖励）、是否结束
        next_state = np.ones(self.state_dim)
        rewards = np.random.rand(self.action_dim)
        done = False
        truncated = False

        # 每次执行时，增加一步
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True  # 达到最大步骤时结束

        info = {}  # 额外的信息（这里为空）
        return next_state, rewards, done, truncated, info

    def render(self, mode="human"):
        pass


# Optional: register this environment with Gym
from gym.envs.registration import register

def register_env(env_id="MultiActionZeroEnv-v0"):
    try:
        register(
            id=env_id,
            entry_point=__name__ + ":MultiActionZeroEnv",
            max_episode_steps=1000
        )
    except Exception:
        # already registered
        pass

# If run as a script, demonstrate one step
if __name__ == "__main__":
    register_env()
    env = gym.make("MultiActionZeroEnv-v0")
    obs, _ = env.reset()
    print("Initial obs:", obs)
    acts = [0] * env.action_dim
    obs, rewards, done, truncated, info = env.step(acts)
    print("Actions:", acts)
    print("Rewards:", rewards)
    print("Done:", done)
    env.close()
