# EXP4-RL: Multi-Arm Bandit Reinforcement Learning with Expert Aggregation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/cylijinpeng/exp4rl.svg)](https://github.com/cylijinpeng/exp4rl/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/cylijinpeng/exp4rl.svg)](https://github.com/cylijinpeng/exp4rl/network)
[![Repository Size](https://img.shields.io/github/repo-size/cylijinpeng/exp4rl.svg)](https://github.com/cylijinpeng/exp4rl)

EXP4-RL is a Python implementation of the EXP4 (Exponential-weight algorithm for Exploration and Exploitation with Experts) algorithm adapted for Reinforcement Learning environments. This project combines multiple expert agents using a meta-learning approach to solve multi-arm bandit problems efficiently.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Expert Agents](#expert-agents)
- [Environment Setup](#environment-setup)
- [API Integration](#api-integration)
- [Contributing](#contributing)
- [License](#license)

## Overview

EXP4-RL implements a sophisticated multi-expert system for reinforcement learning problems, particularly suited for multi-arm bandit scenarios. The system aggregates predictions from multiple expert agents (DQN, RND-DQN, and AI-based experts) using the EXP4 algorithm to make optimal action selections.

### Key Components:
- **EXP4 Meta-Algorithm**: Combines expert predictions with exponential weighting
- **Multiple Expert Types**: DQN, RND-DQN, and AI-based experts
- **Custom Environment**: Multi-action Gym environment for testing
- **Cloud Integration**: AI expert using DeepSeek API for intelligent recommendations

## Features

- **Multi-Expert Aggregation**: Combines predictions from diverse expert agents
- **EXP4 Algorithm**: Theoretical guarantees for regret minimization
- **Reinforcement Learning**: Compatible with OpenAI Gym environments
- **AI Integration**: Cloud-based expert using DeepSeek API
- **Flexible Configuration**: Customizable hyperparameters and expert combinations
- **Testing Environment**: Built-in multi-action bandit environment

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/cylijinpeng/exp4rl.git
cd exp4rl
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is not available, install the required packages manually:

```bash
pip install torch gym numpy requests
```

## Quick Start

1. **Set up your environment**:

```python
from env import register_env
import gym

# Register the custom environment
register_env()

# Create the environment
env = gym.make('MultiActionZeroEnv-v0', state_dim=40, action_dim=50, max_steps=50000)
```

2. **Run the training loop**:

```python
from exp4rl import train_exp4

# Train the EXP4-RL system
meta = train_exp4(env, num_steps=50000, n_cache=50)
print("Training completed. Expert weights:", meta.weights)
```

## Project Structure

```
exp4rl/
├── exp4rl.py          # Main EXP4-RL implementation
├── env.py             # Custom Gym environment
├── cloud.py           # AI expert with DeepSeek API integration
├── record.json        # Sample access history data
└── __pycache__/       # Python cache directory
```

## Usage

### Basic Usage

```python
import gym
from env import register_env
from exp4rl import train_exp4

# Register and create environment
register_env()
env = gym.make('MultiActionZeroEnv-v0', state_dim=40, action_dim=50, max_steps=50000)

# Train the system
meta = train_exp4(env, num_steps=50000, n_cache=50)

# Check expert weights
print("Final expert weights:", meta.weights)
```

### Custom Expert Configuration

You can create custom expert configurations by modifying the expert list:

```python
from exp4rl import EXP4RL, DQNExpert, RNDDQNExpert, AIExpert

# Create custom experts
experts = [
    DQNExpert(state_dim=40, action_dim=50),
    RNDDQNExpert(state_dim=40, action_dim=50),
    AIExpert(action_dim=50)
]

# Initialize EXP4-RL with custom experts
meta = EXP4RL(experts, action_dim=50, gamma=0.99, eta=0.1)
```

## Expert Agents

### DQNExpert
- Deep Q-Network based expert
- Uses neural networks for value function approximation
- Experience replay for stable learning

### RNDDQNExpert
- Random Network Distillation DQN expert
- Intrinsic motivation for exploration
- Combines external and internal rewards

### AIExpert
- Cloud-based expert using DeepSeek API
- Analyzes access history for intelligent recommendations
- Requires API key configuration

## Environment Setup

The project includes a custom Gym environment `MultiActionZeroEnv`:

- **State Space**: Continuous vector of specified dimension
- **Action Space**: Multi-discrete actions (multiple arms selection)
- **Rewards**: Random reward vector for each arm
- **Termination**: After specified maximum steps

### Environment Parameters

```python
env = gym.make('MultiActionZeroEnv-v0', 
               state_dim=40,      # State dimension
               action_dim=50,     # Number of arms/actions
               max_steps=50000)   # Maximum steps per episode
```

## API Integration

The AI expert uses DeepSeek API for intelligent recommendations. To use this feature:

1. **Get API Key**: Obtain a DeepSeek API key
2. **Configure**: Set your API key in the AIExpert initialization
3. **Data**: Provide access history in `record.json` format

### Example Configuration

```python
# In cloud.py, set your API key
DEEPSEEK_API_KEY = "your-api-key-here"
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- EXP4 algorithm based on theoretical work by Auer et al.
- OpenAI Gym for the reinforcement learning environment interface
- PyTorch for deep learning capabilities
- DeepSeek for AI API integration

## Contact

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This project is under active development. Features and APIs may change in future versions.
