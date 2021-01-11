#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from ns3gym import ns3env
from tcp_base import TcpTimeBased
from tcp_newreno import TcpNewReno
from tcp_deeprl import TcpDeepRl
from tcp_ppo import PPO
from tcp_ddpg import DDPG
import numpy as np
import torch
import datetime
import csv
import json
from os.path import join, exists
import time

__author__ = "Piotr Gawlowicz"
__copyright__ = "Copyright (c) 2018, Technische Universität Berlin"
__version__ = "0.1.0"
__email__ = "gawlowicz@tkn.tu-berlin.de"


parser = argparse.ArgumentParser(description='Start simulation script on/off')
parser.add_argument('--start',
                    type=int,
                    default=1,
                    help='Start ns-3 simulation script 0/1, Default: 1')
parser.add_argument('--iterations',
                    type=int,
                    default=1,
                    help='Number of iterations, Default: 1')
parser.add_argument('--rounds',
                    type=int,
                    default=1,
                    help='Number of rounds, Default: 1')

args = parser.parse_args()
startSim = bool(args.start)
iterationNum = int(args.iterations)

port = 5555
simTime = 10 # seconds
stepTime = 0.05  # seconds
seed = 12
simArgs = {"--duration": simTime,}
debug = False
#bottleneck_bandwidth=5.0*(10**6)



MEMORY_CAPACITY = 100

env = ns3env.Ns3Env(port=port, stepTime=stepTime, startSim=startSim, simSeed=seed, simArgs=simArgs, debug=debug)
# simpler:
#env = ns3env.Ns3Env()
env.reset()

ob_space = env.observation_space
ac_space = env.action_space
#print("Observation space: ", ob_space,  ob_space.dtype)
#print("Action space: ", ac_space, ac_space.dtype)

stepIdx = 0
currIt = 0

#save the training data
m_overall=[]

def utilities(rtt,min_rtt,throughput,lostrate,bottleneck_bandwidth):

    if int(bottleneck_bandwidth/1000000)<1:
        bandwidth=10
    else:
        bandwidth=int(bottleneck_bandwidth/1000000)*10+10
    #print(bandwidth)
    return np.log(throughput/(1000000.0*bandwidth))-np.log((rtt-min_rtt)/float(10**6))/4.0+np.log(1-lostrate/float(10**6))

def get_agent(obs):
    socketUuid = obs[0]
    tcpEnvType = obs[1]
    tcpAgent = get_agent.tcpAgents.get(socketUuid, None)
    
    if tcpAgent is None:
        if tcpEnvType == 0:
            # event-based = 0
            #tcpAgent = TcpDeepRl()
            tcpAgent=TcpNewReno()
            #tcpAgent = PPO()
            #tcpAgent=DDPG()
        else:
            # time-based = 1
            tcpAgent = TcpTimeBased()
        #tcpAgent.set_spaces(get_agent.ob_space, get_agent.ac_space)
        #get_agent.tcpAgents[socketUuid] = tcpAgent

    return tcpAgent

# initialize variable
get_agent.tcpAgents = {}
get_agent.ob_space = ob_space
get_agent.ac_space = ac_space

try:
    while True:
        #print("Start iteration: ", currIt)
        init_time=time.time()
        obs = env.reset()
        reward = 0
        done = False
        info = None
        #print("Step: ", stepIdx)
        #print("---obs: ", obs)

        # get existing agent of create new TCP agent if needed
        tcpAgent = get_agent(obs)
        #load trained model
        #(tcpAgent.eval_net).load_state_dict(torch.load("iter_"+str(args.rounds)+'/eval_iter-1.0.model'))
        #(tcpAgent.target_net).load_state_dict(torch.load("iter_"+str(args.rounds)+'/target_iter-1.0.model'))
        while True:
            stepIdx += 1
            begin=datetime.datetime.now()
            action = tcpAgent.get_action(obs, reward, done, info)
            #end=datetime.datetime.now()
            #k=end-begin
            #print("the time",k.total_seconds()) 
            #print("............................")
            #print("---action: ", action)

            print("Step: ", stepIdx)
            next_obs, reward, done, info = env.step(action)
            #print("---next_obs, reward, done, info: ", next_obs, reward, done, info)
            reward=1000*(utilities(next_obs[6],next_obs[7],next_obs[11],next_obs[12],next_obs[13])-utilities(obs[6],obs[7],obs[11],obs[12],obs[13]))
            """         
            #print("ratios: ",ratio)
            if ratio>30:
                reward=10
            if ratio>=0 and ratio<30:
                reward=2
            if ratio<0 and ratio>=-30:
                reward=-2
            if ratio<-30:
                reward=-10
            """
            #tcpAgent.store_transition(obs,tcpAgent.action_cwnd,reward,next_obs)
            #tcpAgent.put_data([obs,tcpAgent.action_cwnd,reward,next_obs,tcpAgent.prob_a,done])
            #if tcpAgent.pointer>MEMORY_CAPACITY:
            #if tcpAgent.memory_counter>MEMORY_CAPACITY:
             #   tcpAgent.learn()
              #  print("learning")
            print("---obs,action,reward: ", obs,action,reward)
            m_overall.append([obs[13]/1000000.0,obs[0],obs[1],obs[2],obs[3],obs[4],obs[5],obs[6]/1000000.0,obs[7]/1000000.0,obs[8],obs[9],obs[10],obs[11]/1000000.0,obs[12]/1000000.0,int(obs[13]/1000000),action[0],action[1],reward])

            # get existing agent of create new TCP agent if needed
            #tcpAgent = get_agent(obs)
            obs=next_obs

            #if done:
             #   stepIdx = 0
              #  if currIt + 1 < iterationNum:
               #     env.reset()
                #break
            
            """
            if int(obs[13]/1000000)==5:   #almost 10mins
                
                eval_netfilename="iter_"+str(args.rounds+1)+"/eval_iter-"+str(1.0)+".model"
                torch.save((tcpAgent.eval_net).state_dict(),eval_netfilename)

                target_netfilename="iter_"+str(args.rounds+1)+"/target_iter-"+str(1.0)+".model"
                torch.save((tcpAgent.target_net).state_dict(),target_netfilename)
                
                m_overall=np.array(m_overall)

                writer1=csv.writer(open(join("iter_"+str(args.rounds+1),"rltcp_iter-" + str(1.0) + ".csv" ), 'w'))
                writer1.writerows(m_overall)
                m_overall=[]

                break;
            """        
                               

        
        currIt += 1
        if currIt == iterationNum:
            break

except KeyboardInterrupt:
    print("Ctrl-C -> Exit")
finally:
    env.close()
    print("Done")
