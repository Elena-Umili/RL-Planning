from copy import deepcopy
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from ddqn_ae_esterno.experience_replay_buf import experienceReplayBuffer
from ddqn_ae_esterno.AutoEncoder import AutoEncoder
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

def normalize_vec(vec):
    min_vec = [-0.9948723, -0.25610653, -4.6786265, -1.7428349, -1.9317414, -2.0567605, 0.0, 0.0]
    max_vec = [0.99916536, 1.5253401, 3.9662757, 0.50701684, 2.3456542, 2.0852647, 1.0, 1.0]
    for i in range(len(vec)):
        vec[i] = (vec[i] - min_vec[i])/(max_vec[i] - min_vec[i])

        vec[i] = min(vec[i], 1)
        vec[i] = max(vec[i], 0)

    return vec

class DDQNAgent:

    def __init__(self, env, network, buffer, epsilon=0.05, batch_size=32):

        self.ae = AutoEncoder(100)
        self.ae.load_state_dict(torch.load('lunar_models/code100.pt'))
        self.env = env
        self.network = network
        self.target_network = deepcopy(network)
        self.buffer = buffer
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.window = 100
        self.reward_threshold = 140  # Avg reward before CartPole is "solved"
        self.initialize()
        self.f = open("res/ddqn_ae_rw.txt", "w+")

    def take_step(self, mode='train'):

        #norm_s0 = normalize_vec(self.s_0)

        ae_out, ae_code = self.ae(Variable(torch.from_numpy(self.s_0).to('cuda')), 100, 100, 50)
        new_s = ae_code.detach().to('cpu').numpy()
        #new_s = ae_out.detach().to('cpu').numpy()

        if mode == 'explore':
            action = self.env.action_space.sample()
        else:
            action = self.network.get_action(new_s, epsilon=self.epsilon)
            self.step_count += 1
        s_1, r, done, _ = self.env.step(action)
        self.rewards += r
        #norm_s1 = normalize_vec(s_1)
        ae_out, ae_code = self.ae(Variable(torch.from_numpy(s_1).to('cuda')), 100, 100, 50)
        new_s1 = ae_code.detach().to('cpu').numpy()
        #new_s1 = ae_out.detach().to('cpu').numpy()

        self.buffer.append(new_s, action, r, done, new_s1)
        #self.buffer.append(norm_s0, action, r, done, norm_s1)

        self.s_0 = s_1.copy()
        if done:
            self.s_0 = self.env.reset()
        return done

    # Implement DQN training algorithm
    def train(self, gamma=0.99, max_episodes=10000,
              batch_size=32,
              network_update_frequency=4,
              network_sync_frequency=2000):
        self.gamma = gamma
        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            self.take_step(mode='explore')

        ep = 0
        training = True
        while training:
            self.s_0 = self.env.reset()
            self.rewards = 0
            done = False
            while done == False:
                '''
                if((ep % 50) == 0 ):
                    self.env.render()
                '''
                done = self.take_step(mode='train')
                # Update network
                if self.step_count % network_update_frequency == 0:
                    self.update()
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.network.state_dict())
                    self.sync_eps.append(ep)

                if done:
                    ep += 1
                    self.training_rewards.append(self.rewards)
                    self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(
                        self.training_rewards[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)
                    print("\rEpisode {:d} Mean Rewards {:.2f}\t\t".format(
                        ep, mean_rewards), end="")
                    self.f.write(str(mean_rewards)+"\n")
                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            ep))
                        break

    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards).to(device=self.network.device).reshape(-1, 1)
        actions_t = torch.LongTensor(np.array(actions)).reshape(-1, 1).to(
            device=self.network.device)
        dones_t = torch.ByteTensor(dones).to(device=self.network.device)

        qvals = torch.gather(self.network.get_qvals(states), 1, actions_t)

        #################################################################
        # DDQN Update
        next_actions = torch.max(self.network.get_qvals(next_states), dim=-1)[1]
        next_actions_t = torch.LongTensor(next_actions).reshape(-1, 1).to(
            device=self.network.device)
        target_qvals = self.target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_t).detach()
        #################################################################
        qvals_next[dones_t] = 0  # Zero-out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t
        loss = nn.MSELoss()(qvals, expected_qvals)
        return loss

    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.network.optimizer.step()
        if self.network.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

    def initialize(self):
        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.rewards = 0
        self.step_count = 0
        self.s_0 = self.env.reset()