from copy import deepcopy
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from ddqn_ae_interno.experience_replay_buf import experienceReplayBuffer
import random
import warnings
from ddqn_ae_interno.AutoEncoder import AutoEncoder,Encoder,Decoder
from ddqn_ae_interno.Gumbel_AE import DecoderGumbel,EncoderGumbel
from ddqn_ae_interno.transition_model import Transition,TransitionDelta
from ddqn_ae_interno.QNetwork import QNetwork
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

def normalize_vec(vec):
    min_vec = [-0.9948723, -0.25610653, -4.6786265, -1.7428349, -1.9317414, -2.0567605, 0.0, 0.0]
    max_vec = [0.99916536, 1.5253401, 3.9662757, 0.50701684, 2.3456542, 2.0852647, 1.0, 1.0]
    for i in range(len(vec)):
        vec[i] = (vec[i] - min_vec[i])/(max_vec[i] - min_vec[i])
        vec[i] = min(vec[i], 1)
        vec[i] = max(vec[i], 0)
    return vec

class plan_node():
    def __init__(self, code_state, value):
        self.code_state = code_state
        self.value = value
        self.action_vec = []

    def add_action(self, a):
        self.action_vec.extend(a)


class DDQNAgent:

    def __init__(self, env, buffer, epsilon=0.05, batch_size=32):
        #ROBA DI GUMBEL
        self.temp_max = 1
        self.temp_min = 0.5
        self.temperature = self.temp_max
        self.batch_epochs = 10
        self.ANNEAL_RATE = 0.07
        self.categorical_size = 10
        self.latent_size = 20
        self.encoder = EncoderGumbel(8,categorical_size=self.categorical_size,latent_size=self.latent_size)
        self.decoder = DecoderGumbel(input_size=8, code_size= self.categorical_size * self.latent_size)

        self.trans_delta = TransitionDelta(self.categorical_size * self.latent_size,4)
        self.transition= Transition(self.encoder,self.decoder,self.trans_delta)

        #self.ae = VAE_gumbel(1.0)
        #self.ae.load_state_dict(torch.load('lunar_models/code_gumbel_mod1_no_norm.pt'))
        self.env = env
        self.network = QNetwork(env=env, n_hidden_nodes=256, encoder=self.encoder)
        self.f = open("res/planner_enc_DDQN.txt", "a+")
        self.target_network = deepcopy(self.network)
        self.buffer = buffer
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.window = 100
        self.reward_threshold = 100  # Avg reward before LL is "solved"
        self.initialize()
        self.action = 0
        self.temp_s1 = 0
        self.step_count = 0
        self.cum_rew = 0

    def planner_action(self, depth=1):
        if np.random.random() < 0.1:
            return np.random.choice(4)

        origin_code = self.encoder(torch.from_numpy(self.s_0),50,50)
        origin_value = self.network.get_enc_value(origin_code)
        origin_node = plan_node(origin_code, origin_value)
        origin_node.action_vec = [0]
        old_vec = [origin_node]
        new_vec = []
        #print("Origin: " + str(origin_code))
        #print("s_a1 = "+ str(self.trans_delta(origin_code, torch.from_numpy(to_categorical(0,4) ))))
        #print("s_a2 = " + str(self.trans_delta(origin_code, torch.from_numpy(to_categorical(1, 4)))))
        #print("s_a3 = " + str(self.trans_delta(origin_code, torch.from_numpy(to_categorical(2, 4)))))
        #print("s_a4 = " + str(self.trans_delta(origin_code, torch.from_numpy(to_categorical(3, 4)))))
        for i in range(depth):
            for n in old_vec:
                for a in range(4):
                    a_ = torch.FloatTensor(to_categorical(a,4)).to('cuda')
                    pred_state = torch.cat((n.code_state, a_), 0)
                    new_code = self.predictor(pred_state)
                    new_value = n.value + self.network.get_enc_value(new_code)
                    new_node = plan_node(new_code,new_value)
                    new_node.action_vec.append(a)
                    for act in n.action_vec:
                        new_node.action_vec.append(act)
                    #new_node.add_action(n.action_vec)
                    new_vec.append(new_node)
            old_vec = new_vec
            new_vec = []
        v_max = -1000
        max_ind = 0
        random.shuffle(old_vec)
        for n in range(len(old_vec)):

            if(old_vec[n].value.cpu().detach().numpy() >= v_max):
                v_max = old_vec[n].value.cpu().detach().numpy()
                max_ind = n
            #print(old_vec[n].value)
        #print(max_ind)
        return old_vec[max_ind].action_vec[-2]

    def is_diff(self, s1, s0):
        for i in range(len(s0)):
            if(s0[0][i] != s1[0][i]):
                return True
        return False

    def take_step(self, mode='train'):

        s_1, r, done, _ = self.env.step(self.action)
        self.cum_rew += r
        enc_s1, _ = self.encoder(torch.from_numpy(np.asarray([s_1])).to('cuda'), self.temperature, False)
        enc_s0, _ = self.encoder(torch.from_numpy(np.asarray([self.s_0])).to('cuda'), self.temperature,False)
        if(self.is_diff(enc_s0, enc_s1)):
            self.buffer.append(self.s_0, self.action, r, done, s_1)
            self.cum_rew = 0

            if mode == 'explore':
                self.action = self.env.action_space.sample()
            else:
                self.action = self.network.get_action(self.s_0, temperature=self.temperature)
                #self.action = self.planner_action()


            self.rewards += r

            self.s_0 = s_1.copy()

        self.step_count += 1
        if done:
            self.s_0 = self.env.reset()
        return done


    # Implement DQN training algorithm
    def train(self, gamma=0.99, max_episodes=1000,
              batch_size=32,
              network_update_frequency=4,
              network_sync_frequency=200):
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
                if((ep % 50) == 0 ):
                    self.env.render()

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
                    self.f.write(str(mean_rewards)+ "\n")

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

        ###############
        # DDQN Update #
        ###############
        self.temperature = self.temp_max
        for ep in range(self.batch_epochs):
            qvals = torch.gather(self.network.get_qvals(states, self.temperature).to('cpu'), 1, actions_t)
            next_actions = torch.max(self.network.get_qvals(next_states, self.temperature).to('cpu'), dim=-1)[1]
            next_actions_t = torch.LongTensor(next_actions).reshape(-1, 1).to(
                device=self.network.device)
            target_qvals = self.target_network.get_qvals(next_states, self.temperature).to('cpu')
            qvals_next = torch.gather(target_qvals, 1, next_actions_t).detach()
            ###############
            qvals_next[dones_t] = 0  # Zero-out terminal states
            expected_qvals = self.gamma * qvals_next + rewards_t
            loss = nn.MSELoss()(qvals, expected_qvals)
            loss.backward()
            self.network.optimizer.step()
            self.temperature = np.maximum(self.temperature * np.exp(-self.ANNEAL_RATE * ep), self.temp_min)

        return loss

    def pred_update(self, batch):
        loss_function = nn.MSELoss()
        states, actions, rewards, dones, next_states = [i for i in batch]
        cat_actions = []

        #modifica struttura actions
        for act in actions:
            cat_actions.append(np.asarray(to_categorical(act,4)))
        cat_actions = np.asarray(cat_actions)
        a_t = torch.FloatTensor(cat_actions).to('cuda')

        #Modifiche struttura states
        if type(states) is tuple:
            states = np.array([np.ravel(s) for s in states])
        states = torch.FloatTensor(states).to('cuda')

        # Modifiche struttura states
        if type(next_states) is tuple:
            next_states = np.array([np.ravel(s) for s in next_states])
        next_states = torch.FloatTensor(next_states).to('cuda')

        self.temperature = self.temp_max
        for ep in range(self.batch_epochs):
            error_z, x_prime_hat, qy, qy_prime = self.transition(states, a_t, next_states, self.temperature, False)
            L = self.transition.loss_function_transition(error_z,next_states,x_prime_hat,qy)
            L.backward()
            self.transition.optimizer.step()
            self.temperature = np.maximum(self.temperature * np.exp(-self.ANNEAL_RATE * ep), self.temp_min)

        return

    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)

        batch2 = self.buffer.sample_batch(batch_size=self.batch_size)
        self.pred_update(batch2)

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