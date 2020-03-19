import gym, os
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v0").unwrapped

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.0001

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step] # mask를 통해, 마지막 step(끝)이면 rewards가 그대로 next_value가 된다.
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        for i in count():
            env.render()
            state = torch.FloatTensor(state).to(device)
            dist, value = actor(state), critic(state)

            action = dist.sample()
            next_state, reward, done, _ = env.step(action.cpu().numpy())

            log_prob = dist.log_prob(action).unsqueeze(0) # log(policy)
            entropy += dist.entropy().mean() # policy * log(policy)

            log_probs.append(log_prob) # sigma (log(policy))를 구하기 위한 list
            values.append(value) # 각 critic의 value function을 구하기 위한 list
            rewards.append(torch.tensor([reward], dtype=torch.float, device=device)) # 각 단계의 reward를 리스트에 추가, TD target을 계산하는데 필요.
            masks.append(torch.tensor([1-done], dtype=torch.float, device=device)) # 1 minus 0, 즉 끝났는지 확인하기 위한 리스트

            state = next_state

            if done:
                print('Iteration: {}, Score: {}'.format(iter, i))
                break


        next_state = torch.FloatTensor(next_state).to(device)
        next_value = critic(next_state) # V(s_t+1)을 구하기 위함
        returns = compute_returns(next_value, rewards, masks) # TD target을 구함. returns는 list이며 각 단계마다의 TD target이 표기됨.

        log_probs = torch.cat(log_probs)  # 한 iteration에서 구한 log_probs을 붙임
        returns = torch.cat(returns).detach()   # 한 iteration에서 구한 TD target을 붙임
        values = torch.cat(values)        # 한 iteration에서 구한 value을 붙임

        advantage = returns - values    # TD Error

        actor_loss = -(log_probs * advantage.detach()).mean()   # actor는 policy 이므로, log(TD error)를 loss로 계산. (앞에 -1을 통해 ascent)
        critic_loss = advantage.pow(2).mean()    # critic은 TD error를 MSE 으로 loss로 계산.

        optimizerA.zero_grad()
        optimizerC.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    torch.save(actor, 'model/actor.pkl')
    torch.save(critic, 'model/critic.pkl')
    env.close()


if __name__ == '__main__':
    if os.path.exists('model/actor.pkl'):
        actor = torch.load('model/actor.pkl')
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size).to(device)
    if os.path.exists('model/critic.pkl'):
        critic = torch.load('model/critic.pkl')
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size).to(device)
    trainIters(actor, critic, n_iters=100)
