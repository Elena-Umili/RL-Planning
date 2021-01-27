from copy import deepcopy
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from planner_system.experience_replay_buf import experienceReplayBuffer
import random
import warnings
from planner_system.AutoEncoder import AutoEncoder,Encoder,Decoder
from planner_system.Gumbel_AE import DecoderGumbel,EncoderGumbel
from planner_system.transition_model import Transition,TransitionDelta
from planner_system.QNetwork import QNetwork
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
        self.temp_min = 0.2
        self.temperature = self.temp_max
        self.ANNEAL_RATE = 0.015
        self.categorical_size = 10
        self.latent_size = 50
        self.action_size = 3
        '''
        self.encoder = Encoder(200)
        self.decoder = Decoder(200)
        self.trans_delta = TransitionDelta(200, self.action_size)
        self.transition = Transition(self.encoder,self.decoder,self.trans_delta)
        '''
        self.encoder = Encoder(200)
        self.decoder = Decoder(200)
        self.trans_delta = TransitionDelta(200, self.action_size)
        self.transition = Transition(self.encoder, self.decoder, self.trans_delta)

        #self.ae = VAE_gumbel(1.0)
        #self.ae.load_state_dict(torch.load('lunar_models/code_gumbel_mod1_no_norm.pt'))
        self.env = env
        self.network = QNetwork(env=env, n_hidden_nodes=64, encoder=self.encoder)
        self.f = open("res/planner_enc_DDQN.txt", "a+")
        self.target_network = deepcopy(self.network)
        self.buffer = buffer
        self.epsilon = epsilon
        self.batch_size = 64
        self.window = 20
        self.reward_threshold = -120  # Avg reward before LL is "solved"
        self.initialize()
        self.action = 0
        self.temp_s1 = 0
        self.step_count = 0
        self.cum_rew = 0
        self.timestamp = 0
        self.episode = 0
        self.difference = 0

    def planner_action(self, depth=1):
        if np.random.random() < 0.05:
            return np.random.choice(self.action_size)

        origin_code = self.encoder(torch.from_numpy(self.s_0).type(torch.FloatTensor),50,50)
        origin_value = self.network.get_enc_value(origin_code)
        origin_node = plan_node(origin_code, origin_value)
        origin_node.action_vec = [0]
        action = torch.argmax(origin_value).to('cpu').detach().numpy()

        a0 = to_categorical(0,self.action_size)
        a1 = to_categorical(1,self.action_size)
        a2 = to_categorical(2,self.action_size)
        #a3 = to_categorical(3,self.action_size)
        #a4 = to_categorical(3, self.action_size)
        #a5 = to_categorical(3, self.action_size)


        _, ns0 = self.trans_delta(origin_code, torch.from_numpy(a0).type(torch.FloatTensor).to('cuda'))
        _, ns1 = self.trans_delta(origin_code, torch.from_numpy(a1).type(torch.FloatTensor).to('cuda'))
        _, ns2 = self.trans_delta(origin_code, torch.from_numpy(a2).type(torch.FloatTensor).to('cuda'))
        #_, ns3 = self.trans_delta(origin_code, torch.from_numpy(a3).type(torch.FloatTensor).to('cuda'))
        #_, ns4 = self.trans_delta(origin_code, torch.from_numpy(a4).type(torch.FloatTensor).to('cuda'))
        #_, ns5 = self.trans_delta(origin_code, torch.from_numpy(a5).type(torch.FloatTensor).to('cuda'))

        v0 = self.network.get_enc_value(ns0)
        v1 = self.network.get_enc_value(ns1)
        v2 = self.network.get_enc_value(ns2)
        #v3 = self.network.get_enc_value(ns3)
        #v4 = self.network.get_enc_value(ns4)
        #v5 = self.network.get_enc_value(ns5)

        max0 = torch.max(v0).to('cpu').detach().numpy()
        arg_max0 = torch.argmax(v0).to('cpu').detach().numpy()

        max1 = torch.max(v1).to('cpu').detach().numpy()
        arg_max1 = torch.argmax(v1).to('cpu').detach().numpy()


        max2 = torch.max(v2).to('cpu').detach().numpy()
        arg_max2 = torch.argmax(v2).to('cpu').detach().numpy()

        '''
        max3 = torch.max(v3).to('cpu').detach().numpy()
        arg_max3 = torch.argmax(v3).to('cpu').detach().numpy()

        
        max4 = torch.max(v4).to('cpu').detach().numpy()
        arg_max4 = torch.argmax(v4).to('cpu').detach().numpy()

        max5 = torch.max(v5).to('cpu').detach().numpy()
        arg_max5 = torch.argmax(v5).to('cpu').detach().numpy()
        '''

        l_max = [max0, max1, max2]
        l_amax = [arg_max0, arg_max1, arg_max2]

        if(action != l_amax[np.argmax(l_max)]):
            print("DIVERSO!")

        return l_amax[np.argmax(l_max)]

    def is_diff(self, s1, s0):
        for i in range(len(s0)):
            if(s0[i] != s1[i]):
                return True
        return False

    def take_step(self, mode='train'):
        '''
        s_1, r, done, _ = self.env.step(self.action)


        self.buffer.append(self.s_0, self.action, r, done, s_1)
        if mode == 'explore':
            self.action = self.env.action_space.sample()
        else:
            self.action = self.network.get_action(self.s_0, temperature=1)
            # self.action = self.planner_action()

        self.rewards += r

        self.s_0 = s_1.copy()

        self.step_count += 1
        if done:
            self.s_0 = self.env.reset()
        return done
        '''
######### da rivedere

        s_1, r, done, _ = self.env.step(self.action)
        #print(self.env.action_space)
        enc_s1 = self.encoder(torch.from_numpy(np.asarray(s_1)).type(torch.FloatTensor), 50, 50)
        enc_s0 = self.encoder(torch.from_numpy(np.asarray(self.s_0)).type(torch.FloatTensor).to('cuda'), 50,50)
        #print("Reward = ", r)
        if(self.is_diff(enc_s0,enc_s1)):
        #if(True):
            #print("step passati = ", self.step_count - self.timestamp)
            self.timestamp = self.step_count

            self.buffer.append(self.s_0, self.action, r, done, s_1)
            self.cum_rew = 0

            if mode == 'explore':
                self.action = self.env.action_space.sample()

            else:
                self.action = self.network.get_action(self.s_0, temperature=self.temperature)
                #self.action = self.planner_action()


            #self.rewards += r

            self.s_0 = s_1.copy()

        self.rewards += r
        self.step_count += 1
        if done:

            self.s_0 = self.env.reset()
        return done

    # Implement DQN training algorithm
    def train(self, gamma=0.99, max_episodes=300,
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
                    self. episode = ep
                    self.training_rewards.append(self.rewards)
                    self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    mean_rewards = np.mean(
                        self.training_rewards[-self.window:])
                    self.mean_training_rewards.append(mean_rewards)
                    print("\rEpisode {:d} Mean Rewards {:.2f}  Episode reward = {:.2f}\t\t".format(
                        ep, mean_rewards, self.rewards), end="")
                    self.f.write(str(mean_rewards)+ "\n")


                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            ep))
                        '''
                        torch.save(self.encoder, "models/encode_model_ng")
                        torch.save(self.transition, "models/transition_model_ng")
                        torch.save(self.decoder, "models/decoder_model_ng")
                        torch.save(self.trans_delta, "models/trans_delta_model_ng")
                        '''
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

        qvals, qy = self.network.get_qvals(states, self.temperature)
        #qy = qy.to('cpu')
        qvals = torch.gather(qvals.to('cpu'), 1, actions_t)

        next_vals, _ = self.network.get_qvals(next_states, self.temperature)
        next_actions = torch.max(next_vals.to('cpu'), dim=-1)[1]
        next_actions_t = torch.LongTensor(next_actions).reshape(-1, 1).to(
            device=self.network.device)
        target_qvals, _ = self.target_network.get_qvals(next_states, self.temperature)
        qvals_next = torch.gather(target_qvals.to('cpu'), 1, next_actions_t).detach()
        ###############
        qvals_next[dones_t] = 0  # Zero-out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t
        #log_ratio = torch.log(qy * self.categorical_size + 1e-20)
        #KLD = torch.sum(qy * log_ratio, dim=-1).mean()
        #print("KLD = ", KLD)
        loss = (nn.MSELoss()(qvals, expected_qvals))

        #print("loss = ", loss)
        loss.backward()
        self.network.optimizer.step()
        self.temperature = np.maximum(self.temperature * np.exp(-self.ANNEAL_RATE * self.episode), self.temp_min)

        return loss

    def pred_update(self, batch):
        loss_function = nn.MSELoss()
        states, actions, rewards, dones, next_states = [i for i in batch]
        cat_actions = []

        #modifica struttura actions
        for act in actions:
            cat_actions.append(np.asarray(to_categorical(act,self.action_size)))
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

        error_z, x_prime_hat = self.transition(states, a_t, next_states, self.temperature, False)
        L = self.transition.loss_function_transition(error_z,next_states,x_prime_hat)
        L = L * 0.01
        L.backward()
        #print("pred_loss = ", L)
        self.transition.optimizer.step()
        self.temperature = np.maximum(self.temperature * np.exp(-self.ANNEAL_RATE * self.episode), self.temp_min)

        return

    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)

        loss = self.calculate_loss(batch)
        #print("q loss = ", loss)

        self.transition.optimizer.zero_grad()
        batch2 = self.buffer.sample_batch(batch_size=self.batch_size)
        self.pred_update(batch2)

        batch_cons = self.buffer.consecutive_sample(batch_size=64)
        #print(batch_cons)

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
