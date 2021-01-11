import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from tcp_base import TcpEventBased
import numpy as np
import gym
#Hyperparameters

learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 30
T_horizon     = 20

BATCH_SIZE = 30
LR =0.01
EPSILON= 0.7
GAMMA = 0.9                 # reward discount
N_ACTIONS = 3
N_STATES = 14



class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()

        self.data = []
        self.memory_counter=0
        self.fc1   = nn.Linear(14,256)
        self.fc_pi = nn.Linear(256,2)
        self.fc_v  = nn.Linear(256,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):

        self.data.append(transition)
        self.memory_counter=self.memory_counter+1        

    def make_batch(self):

        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []

        for transition in self.data:

            s, a, r, s_prime, prob_a, done = transition

            

            s_lst.append(s)

            a_lst.append([a])

            r_lst.append([r])

            s_prime_lst.append(s_prime)

            prob_a_lst.append([prob_a])

            done_mask = 0 if done else 1

            done_lst.append([done_mask])

            

        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst),torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float),torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)

        self.data = []

        return s, a, r, s_prime, done_mask, prob_a

    def get_action(self, obs, reward, done, info):
        socketUuid=obs[0]
        envType=obs[1]
        ssThresh = obs[2]
        cWnd = obs[3]
        segmentSize=obs[4]
        #segmentsAcked =obs[5]
        bytesInFlight = obs[5]
        rtt=obs[6]

        x = torch.unsqueeze(torch.FloatTensor(obs), 0)
        
        
        # input only one sample

        #self.prob = self.pi(x)
        #m = Categorical(self.prob)
        #self.action_cwnd = m.sample().item()

        prob = self.pi(x)
        action = torch.max(prob, 1)[1].data.numpy()
        self.action_cwnd = action[0]
        self.prob_a=torch.max(prob)
        

        new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))
        new_cWnd = cWnd
        if self.action_cwnd==0:
            new_cWnd += 0
        elif self.action_cwnd==1:
            new_cWnd +=segmentSize*3
        else:
            new_cWnd += segmentSize

        return [new_ssThresh, new_cWnd]

    def learn(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)
            pi = self.pi(s, softmax_dim=1)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
