import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

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
        self.policy = Policy(state_dim, action_dim)
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
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.stack(memory.states).detach()
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()

        # Optimize policy for K epochs
        for _ in range(4):
            # Evaluating old actions and values
            dist = self.policy(old_states)
            logprobs = dist.log_prob(old_actions)
            prob_ratio = torch.exp(logprobs - old_logprobs.detach())

            # Finding the surrogate loss
            surrogate1 = rewards * prob_ratio
            surrogate2 = torch.clamp(prob_ratio, 1-self.eps_clip, 1+self.eps_clip) * rewards
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
    betas = (0.9, 0.999)
    gamma = 0.99
    eps_clip = 0.2
    max_episodes = 5000
    max_timesteps = 500
    ############################################

    # Create environment
    env = gym.make(env_name)

    # Initialize PPO and memory
    ppo = PPO(state_dim, action_dim, lr, betas, gamma, eps_clip)
    memory = Memory()

    # Main loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()[0]
        for t in range(max_timesteps):
            state = torch.FloatTensor(state)
            dist = ppo.policy(state)
            action = dist.sample()

            next_state, reward, done, _ = env.step(action.item())
            memory.states.append(state)
            memory.actions.append(action)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            memory.logprobs.append(dist.log_prob(action))

            if done:
                break

            state = next_state

        ppo.update(memory)
        memory.clear_memory()

        if i_episode % 100 == 0:
            print('Episode {}/{}'.format(i_episode, max_episodes))

if __name__ == '__main__':
    main()
