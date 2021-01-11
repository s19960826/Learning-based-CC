from tcp_base import TcpEventBased
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

BATCH_SIZE = 30
LR =0.01
EPSILON= 0.9
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 100
N_ACTIONS = 3
N_STATES = 14

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization
        self.action_cwnd=0

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value



class TcpDeepRl(TcpEventBased):
    """docstring for TcpNewReno"""
    def __init__(self):
        super(TcpDeepRl, self).__init__()
        self.eval_net,self.target_net=Net(),Net()
        self.learn_step_counter = 0                    # for target updating
        self.memory_counter = 0                        # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def get_action(self, obs, reward, done, info):
        socketUuid=obs[0]
        envType=obs[1]
        ssThresh = obs[2]
        cWnd = obs[3]
        segmentSize=obs[4]
        #segmentsAcked =obs[5]
        bytesInFlight = obs[5]
        rtt=obs[6]
        #congestionState= obs[8]
        #congestionEvent=obs[9]
        #ecnState=obs[10]
        #avgSend=obs[11]
        #avgRecv=obs[12]
        #avgDelay = obs[7]

        x = torch.unsqueeze(torch.FloatTensor(obs), 0)
        '''        
        if cWnd<340*60:
            new_cWnd = cWnd + segmentSize
            self.action_cwnd=2
        '''
        if(1):
            if np.random.uniform() < EPSILON:   # greedy
                actions_value = self.eval_net.forward(x)
                action = torch.max(actions_value, 1)[1].data.numpy()
                self.action_cwnd = action[0]
            else:   # random
                action = np.random.randint(0, N_ACTIONS)  # 0 or 1
                self.action_cwnd = action
            new_cWnd = cWnd

            if self.action_cwnd==0:
                new_cWnd += 0
            elif self.action_cwnd==1:
                new_cWnd += segmentSize*3
            else:
                new_cWnd += segmentSize 
        
        new_ssThresh = int(max(2 * segmentSize, bytesInFlight / 2))
    

        return [new_ssThresh, new_cWnd]


    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
