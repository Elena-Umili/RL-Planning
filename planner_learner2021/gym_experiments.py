import gym
from planner_system.DDQN import DDQNAgent
from planner_system.RewardWrapper import RewardWrapperEncFisso
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from planner_system.experience_replay_buf import experienceReplayBuffer
from gym_duckietown.envs import DuckietownEnv
from planner_system.project_utils import PositionObservation, DtRewardWrapper, DiscreteActionWrapperTrain, NoiseWrapper

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)



er_buf = experienceReplayBuffer(burn_in=1000)
env = gym.make("CartPole-v0")
'''
env = DuckietownEnv(
        seed=123,  # random seed
        map_name="small_loop_cw",
        max_steps=500001,  # we don't want the gym to reset itself
        domain_rand=0,
        accept_start_angle_deg=1,  # start close to straight
        full_transparency=True,

    )
    # discrete actions, 4 value observation and modified reward
env = NoiseWrapper(env)
env = DiscreteActionWrapperTrain(env)
env = PositionObservation(env)
env = DtRewardWrapper(env)
'''
ddqn = DDQNAgent(env=env, buffer=er_buf, batch_size=128 )
ddqn.train()

