import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym
from deepQN_cartpole_torch import plot_durations
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def add(self, state, action, reward, done, log_prob_action):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_terminals.append(done)
        self.logprobs.append(log_prob_action)

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return Categorical(logits=x)


class PPO:
    def __init__(self, state_dim, action_dim, lr, betas, gamma, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.policy = Policy(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

    def update(self, memory):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalize the rewards
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.stack(memory.states).detach().to(device)
        old_actions = torch.stack(memory.actions).detach().to(device)
        old_logprobs = torch.stack(memory.logprobs).detach().to(device)

        # Optimize policy for K epochs
        for _ in range(4):
            # Evaluating old actions and values
            dist = self.policy(old_states)
            logprobs = dist.log_prob(old_actions)
            prob_ratio = torch.exp(logprobs - old_logprobs)

            # Finding the surrogate loss
            surrogate1 = rewards * prob_ratio
            surrogate2 = torch.clamp(prob_ratio, 1 - self.eps_clip, 1 + self.eps_clip) * rewards
            loss = -torch.min(surrogate1, surrogate2)

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def main():
    ############## Hyperparameters ##############
    env_name = "CartPole-v1"
    state_dim = 4
    action_dim = 2
    lr = 0.002
    betas = (0.9, 0.9)
    gamma = 0.99
    eps_clip = 0.02
    max_episodes = 5000
    max_timesteps = 500
    ############################################
    episode_durations = []

    # Create environment
    env = gym.make(env_name)
    # env = gym.make("CartPole-v1", render_mode="human")

    # Initialize PPO and memory
    ppo = PPO(state_dim, action_dim, lr, betas, gamma, eps_clip)
    memory = Memory()

    # Main loop
    for i_episode in range(1, max_episodes + 1):
        state = env.reset()[0]
        # env.render()
        for t in range(max_timesteps):
            state = torch.FloatTensor(state).to(device)
            dist = ppo.policy(state)
            action = dist.sample()

            next_state, reward, done, _, _ = env.step(action.item())
            memory.add(state, action, reward, done, dist.log_prob(action))
            # env.render()
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break

            state = next_state

        ppo.update(memory)
        memory.clear_memory()

        if i_episode % 100 == 0:
            print('Episode {}/{}'.format(i_episode, max_episodes))

    print('Complete')
    plot_durations(episode_durations, show_result=True)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    main()
