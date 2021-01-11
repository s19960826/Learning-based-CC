'''

torch = 0.41

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time

#####################  hyper parameters  ####################
MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 100
BATCH_SIZE = 30
TAU = 0.01
RENDER = False

N_ACTIONS = 3
N_STATES = 14

###############################  DDPG  ####################################

class ANet(nn.Module):   # ae(s)=a
    def __init__(self):
        super(ANet,self).__init__()
        self.fc1 = nn.Linear(N_STATES,30)
        self.fc1.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(30,N_ACTIONS)
        self.out.weight.data.normal_(0,0.1) # initialization

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.tanh(x)
        actions_value = x*2
        return actions_value

class CNet(nn.Module):   # ae(s)=a
    def __init__(self):
        super(CNet,self).__init__()
        self.fcs = nn.Linear(N_STATES,30)
        self.fcs.weight.data.normal_(0,0.1) # initialization
        self.fca = nn.Linear(N_ACTIONS,30)
        self.fca.weight.data.normal_(0,0.1) # initialization
        self.out = nn.Linear(30,1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self,s,a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x+y)
        actions_value = self.out(net)
        return actions_value

class DDPG(object):
    def __init__(self):
        self.N_ACTIONS, self.N_STATES = N_ACTIONS, N_STATES
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2), dtype=np.float32)
        self.pointer = 0
        #self.sess = tf.Session()

        self.Actor_eval = ANet()
        self.Actor_target = ANet()
        self.Critic_eval = CNet()
        self.Critic_target = CNet()
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(),lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(),lr=LR_A)
        self.loss_td = nn.MSELoss()

    #def get_action(self, s,reward, done, info):
     #   s = torch.unsqueeze(torch.FloatTensor(s), 0)
      #  return self.Actor_eval(s)[0].detach() # ae（s）

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

        self.prob = self.Actor_eval(x)
        action = torch.max(self.prob, 1)[1].data.numpy()
        print("the action is: ",self.prob)
        self.action_cwnd = action[0]
        

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
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # soft target replacement
        #self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.N_STATES])
        ba = torch.FloatTensor(bt[:, self.N_STATES: self.N_STATES + self.N_ACTIONS])
        br = torch.FloatTensor(bt[:, -self.N_STATES - 1: -self.N_STATES])
        bs_ = torch.FloatTensor(bt[:, -self.N_STATES:])

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs,a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_

        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q) 

        #print(q)

        #print(loss_a)

        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_,a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = br+GAMMA*q_  # q_target = 负的

        #print(q_target)

        q_v = self.Critic_eval(bs,ba)

        #print(q_v)

        td_error = self.loss_td(q_target,q_v)

        # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确

        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()



    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

