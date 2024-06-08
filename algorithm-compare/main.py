import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tianshou.env import DummyVectorEnv
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import time
import json
import math
from tqdm import tqdm
from env import SDN_Env
from network import conv_mlp_net
import torch
import numpy as np
import torch.nn as nn


import gym
import numpy as np
import matplotlib.pyplot as plt
from egreedy import run_egreedy
from softmax import run_softmax
from ucb import run_ucb1
from ppo import run_ppo

INPUT_CH = 67
FEATURE_CH = 512
MLP_CH = 1024

config = 'multi-edge'
cloud_num = 1
edge_num = 1
is_gpu_default = torch.cuda.is_available()
class sdn_net(nn.Module):
    def __init__(self, mode='actor', is_gpu=is_gpu_default):
        super().__init__()
        self.is_gpu = is_gpu
        self.mode = mode

        if self.mode == 'actor':
            self.network = conv_mlp_net(conv_in=INPUT_CH, conv_ch=FEATURE_CH, mlp_in=(edge_num+cloud_num)*FEATURE_CH,\
                                    mlp_ch=MLP_CH, out_ch=edge_num+cloud_num, block_num=3)
        else:
            self.network = conv_mlp_net(conv_in=INPUT_CH, conv_ch=FEATURE_CH, mlp_in=(edge_num+cloud_num)*FEATURE_CH,\
                                    mlp_ch=MLP_CH, out_ch=edge_num, block_num=3)
        
    def load_model(self, filename):
        map_location=lambda storage, loc:storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        # print('save model!')

    def forward(self, obs, state=None, info={}):
        state = obs#['servers']
        state = torch.tensor(state).float()
        if self.is_gpu:
            state = state.cuda()

        logits = self.network(state)
        
        return logits, state

class Actor(nn.Module):
    def __init__(self, is_gpu=is_gpu_default, dist_fn=None):
        super().__init__()
        self.is_gpu = is_gpu

        self.net = sdn_net(mode='actor')

    def load_model(self, filename):
        map_location=lambda storage, loc:storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        # print('save model!')

    def forward(self, obs, state=None, info={}):
            
        logits,_ = self.net(obs)
        # Adjust output size according to the new action space
        logits = F.sigmoid(logits)

        return logits, state

class Critic(nn.Module):
    def __init__(self, is_gpu=is_gpu_default):
        # print(f"Batch keys: {self.__dict__.keys()}")
        super().__init__()

        self.is_gpu = is_gpu

        self.net = sdn_net(mode='critic')

    def load_model(self, filename):
        map_location=lambda storage, loc:storage
        self.load_state_dict(torch.load(filename, map_location=map_location))
        print('load model!')
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)
        # print('save model!')

    def forward(self, obs, state=None, info={}):
            
        v,_ = self.net(obs)

        return v

def main():
    # Initialize the SDN environment
    env = SDN_Env(conf_file='../env/config1.json', conf_name=config, w=1.0, fc=4e9, fe=2e9, edge_num=edge_num, cloud_num=cloud_num)
    # Set hyperparameters
    num_episodes = 1000
    
    observation_space = env.get_obs()
    # Load PPO model for the last epoch
    actor = Actor(is_gpu=is_gpu_default)
    critic = Critic(is_gpu=is_gpu_default)
    actor.load_model('/home/ad/mec_morl_multipolicy/env/save/pth-e1/cloud1/exp1/w100/ep10-actor.pth')  # Load the last epoch
    critic.load_model('/home/ad/mec_morl_multipolicy/env/save/pth-e1/cloud1/exp1/w100/ep10-critic.pth')  # Load the last epoch

    # Run PPO algorithm with loaded models
    ppo_delays_avg, ppo_link_utilisations_avg = run_ppo(env, num_episodes=num_episodes, actor=actor, critic=critic)

    # Run ε-Greedy algorithm
    egreedy_delays_avg, egreedy_link_utilisations_avg = run_egreedy(env, observation_space, num_episodes=num_episodes)

    # Run Softmax algorithm
    softmax_delays_avg, softmax_link_utilisations_avg = run_softmax(env, observation_space, num_episodes=num_episodes)

    # Run UCB1 algorithm
    ucb_delays_avg, ucb_link_utilisations_avg = run_ucb1(env, observation_space, num_episodes=num_episodes)

    print(np.sum(egreedy_delays_avg)/num_episodes, np.sum(softmax_delays_avg)/num_episodes, np.sum(ucb_delays_avg)/num_episodes, np.sum(ppo_delays_avg)/num_episodes)
    print(np.sum(egreedy_link_utilisations_avg)/num_episodes, np.sum(softmax_link_utilisations_avg)/num_episodes, np.sum(ucb_link_utilisations_avg)/num_episodes, np.sum(ppo_link_utilisations_avg)/num_episodes)
    # Plotting the results
    plt.figure(figsize=(15, 7))

    # Tạo subplot cho delay
    plt.subplot(2, 1, 1)
    plt.plot(egreedy_delays_avg, label='ε-Greedy')
    plt.plot(softmax_delays_avg, label='Softmax')
    plt.plot(ucb_delays_avg, label='UCB1')
    plt.plot(ppo_delays_avg, label='MORL')
    plt.xlabel('Episode')
    plt.ylabel('Delay (s)')
    plt.title('Comparison of Delay for ε-Greedy, Softmax, UCB1, and PPO Algorithms')
    plt.legend()
    plt.grid(True)

    # Tạo subplot cho link utilisation
    plt.subplot(2, 1, 2)
    plt.plot(egreedy_link_utilisations_avg, label='ε-Greedy')
    plt.plot(softmax_link_utilisations_avg, label='Softmax')
    plt.plot(ucb_link_utilisations_avg, label='UCB1')
    plt.plot(ppo_link_utilisations_avg, label='MORL')
    plt.xlabel('Episode')
    plt.ylabel('Link Utilisation (Mbps)')
    plt.title('Comparison of Link Utilisation for ε-Greedy, Softmax, UCB1, and PPO Algorithms')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # Đảm bảo không có trùng lặp trong các label và title
    plt.show()
if __name__ == "__main__":
    main()
