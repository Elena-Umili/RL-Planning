from copy import deepcopy
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from old.experience_replay_buf import experienceReplayBuffer
import random
import warnings
from old.AutoEncoder import AutoEncoder,Encoder,Decoder
from old.Gumbel_AE import VAE_gumbel
from old.transition_model import Transition,TransitionDelta, Predictor
from old.QNetwork import QNetwork
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

        self.encoder = Encoder(200)
        self.decoder = Decoder(200)
        self.ae = AutoEncoder(self.encoder,self.decoder, 200)
        self.trans_delta = TransitionDelta(200,2)
        self.transition= Transition(self.encoder,self.decoder,self.trans_delta)
        self.predictor = Predictor(self.trans_delta)

        #self.ae = VAE_gumbel(1.0)
        #self.ae.load_state_dict(torch.load('lunar_models/code_gumbel_mod1_no_norm.pt'))
        self.env = env
        self.network = QNetwork(env=env, n_hidden_nodes=8, encoder=self.encoder)
        self.f = open("res/planner_enc_DDQN.txt", "a+")
        self.target_network = deepcopy(self.network)
        self.buffer = buffer
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.window = 100
        self.reward_threshold = 199  # Avg reward before LL is "solved"
        self.initialize()
        self.action = 0
        self.temp_s1 = 0

    def planner_action(self, depth=2):
        if np.random.random() < 0.3:
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
                    a_ = torch.from_numpy(to_categorical(a,4))
                    new_code = self.trans_delta(n.code_state, a_)
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




    def take_step(self, mode='train'):

        s_1, r, done, _ = self.env.step(self.action)

        if(self.step_count % 1 == 0):
            '''
            print("#########################")
            print("s0 = " + str(self.s_0))
            print("s1 = " + str(s_1))
            print("_________________________")
            '''
            self.buffer.append(self.s_0, self.action, r, done, s_1)
            if mode == 'explore':
                self.action = self.env.action_space.sample()
            else:
                self.action = self.network.get_action(self.s_0, epsilon=self.epsilon)
                #self.action = self.planner_action()


            self.rewards += r

            self.s_0 = s_1.copy()

        self.step_count += 1
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

        loss_function = nn.MSELoss()
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

        #return loss, pred_loss
        return loss
    def calculate_pred_loss(self, batch):
        loss_function = nn.MSELoss()
        states, actions, rewards, dones, next_states = [i for i in batch]
        pred_loss = 0
        rec_loss = 0
        for i in range(16):
            state = torch.from_numpy(np.asarray(states[i]))
            _action = torch.from_numpy(to_categorical(actions[i],4))
            next_state = torch.from_numpy(np.asarray(next_states[i]))
            enc_state = self.encoder(state, 50, 50)
            enc_nxt_state = self.encoder(next_state, 50, 50)
            #print("state = " + str(enc_state))
            #print("next_state = " + str(enc_nxt_state))
            out = self.trans_delta(enc_state, _action)
            #diff, x_prime_hat, z_prime_hat = self.transition(state,_action,next_state, 50,50)
            #target_z = torch.zeros(100).to('cuda')
            #pred_loss = pred_loss + loss_function(diff, target_z) + 2* loss_function(x_prime_hat, next_state.to('cuda'))
            pred_loss = pred_loss + loss_function(out, enc_nxt_state)
            rec, code = self.ae(state,10,50)

            rec_loss = rec_loss + loss_function(rec.to('cpu'),state)
            #print("rec = " + str(rec))
            #print("state = " + str(state))


        return pred_loss, 0

    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.network.optimizer.step()

        batch2 = self.buffer.sample_batch(batch_size=self.batch_size)
        #pred_loss,rec_loss = self.calculate_pred_loss(batch2)
        #print("pred loss = " + str(pred_loss))
        #print("rec loss = " + str(rec_loss))
        self.trans_delta.optimizer.zero_grad()
        self.ae.optimizer.zero_grad()
        #pred_loss.backward()
        #rec_loss.backward()
        self.trans_delta.optimizer.step()
        self.ae.optimizer.step()

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