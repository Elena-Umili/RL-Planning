import gym
import cv2
import numpy as np
#from filtering import findPlayer, lato
#from rewardMachine import RewardMachine
import scipy.misc
import pickle

class RestrainingBoltRewardWrapper(gym.Wrapper):
    def __init__(self, env, seqFileName):
        super().__init__(env)
        self.env = env
        self.rewMach = RewardMachine(seqFileName)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # modify reward
        reward += self.restrainingBoltReward(observation)
        return observation, reward, done, info

    #fa fare uno step alla reward machine e calcola la reward del restraining bolt
    def restrainingBoltReward(self, observation):

      oldState = self.rewMach.currentState()
      self.rewMach.step(observation)
      currState = self.rewMach.currentState()

      #se la reward machine ha fatto reset => reward = -100
      #if oldState != 0 and currState == 0:
      #  cautious
      #  return -100
      #  brave
      #  return 0

      if currState == 0:
        return 0

      #se la reward machine ha fatto un passo avanti => reward = +100
      ### nota: con l'automa sequenza qualunque step è uno step in avanti
      if oldState != currState:
        return 100

      #se la reward machine non ha cambiato stato => reward = 0
      return 0

#test
#env = RestrainingBoltRewardWrapper(gym.make('MontezumaRevenge-v0'), "desiredSequence.txt")
#obs = cv2.imread('globalObservation.png')

#print(env.restrainingBoltReward(obs))

class TakeDatasetMZimagesWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.oldObservation = np.zeros(shape = (lato, lato), dtype = int)
        self.fileDataset = open("transitions.txt", "a")
        self.photoIndex = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        ############### Montezuma
        #localObservation, position = findPlayer(observation)
        scipy.misc.imsave("datasetLL/images/"+str(self.photoIndex)+".png", localObservation)
        self.fileDataset.write("{}\t{}\t{}\n".format(self.photoIndex - 1, action, self.photoIndex))
        self.photoIndex += 1
        return observation, reward, done, info

class TakeDatasetLLvectorsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.fileDataset = open("transitions.txt", "a")
        self.stateData =  np.zeros(shape = (1,8), dtype = float)
        self.actionData = []

    def step(self, action, timestep):
        observation, reward, done, info = self.env.step(action)
        #print("action: ", action)
        ############### Lunar Lander
        if timestep % 3 == 0:
          self.actionData.append(action)
        if timestep != 0 and timestep % 2 == 0:
          observationd = observation.reshape((1,8))
          self.stateData = np.append(self.stateData, observationd, axis = 0)
        #print(observationd.shape)
        return observation, reward, done, info

    def saveData(self):
        
        self.stateData = self.stateData[1:]

        for i in range(self.stateData.shape[1]):
           print("component ", i)
           print("min ", min(self.stateData[:,i]))
           print("max ", max(self.stateData[:,i]))
        print("stateData.shape: ", self.stateData.shape)
        print("len(actionData): ", len(self.stateData))
        with open('datasetLLstates.pickle', 'wb') as f:
           pickle.dump(self.stateData, f)
        with open('datasetLLactions.pickle', 'wb') as f:
           pickle.dump(self.actionData, f)


class StateDiscretizerWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.n = [5,5,5,5,5,5,1,1]
        self.min = [-1, -0.5, -2, -2, -3, -6, 0, 0]
        self.max = [1, 2, 2, 2, 3, 6, 1, 1]

    def step(self, action, verbose = False):
        observation, reward, done, info = self.env.step(action)
        if verbose:
          print("NORMAL")
          print(observation)
        for i in range(len(observation)):
          
          if observation[i] <= self.min[i]:
            observation[i] = self.min[i]
          else:
            if observation[i] >= self.max[i]:
              observation[i] = self.max[i]
            else:
              step = (self.max[i] - self.min[i]) / self.n[i]
              for j in range(self.n[i]):
                a = self.min[i] + step*j
                b = self.min[i] + step*(j+1)
                if observation[i] >= a and observation[i] <= b:
                  if observation[i]-a > b - observation[i]:
                    observation[i] = b
                  else:
                    observation[i] = a
        if verbose:
          print("DISCRETIZED")
          print(observation)
        return observation, reward, done, info


