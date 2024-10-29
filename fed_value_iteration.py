import numpy as np
import matplotlib.pyplot as plt
import value_iteration as vi

import argparse
import os
#import json


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)



class FedValueIteration:
    def __init__(self, envs,  common_kernel, common_reward, ep = 0.0, er = 0.0, **kwargs):
        self.kwargs = kwargs
        # get hyperparameters
        self.T = kwargs.get('T')
        self.epsilon_r = er
        self.epsilon_p = ep
        
        self.common_kernel = common_kernel
        self.common_reward = common_reward

        # get time horizon
        self.H = self.common_kernel.shape[0]
        self.environnement = kwargs.get('environment')
        
        # set number of agents
        self.N = len(envs)
        self.M = len(envs)
        self.delta = kwargs.get('confidence')

        self.verbose = kwargs.get('verbose')

        # set number of threads
        self.num_threads = kwargs.get('num_threads')
        
        # choose aggregation method
        self.standard_aggregation = kwargs.get('standard_aggregation')
        
        # get the environment
        self.envs = envs
        # initialize episodes counter
        self.t = 0

        # init r
        self.r = 0
        # initialize federation
        self.vi_objs = [vi.ValueIteration(env, self.t, self.common_kernel, self.common_reward, ep = self.epsilon_p, er= self.epsilon_r , M=self.M,**self.kwargs) for env in self.envs]
        # get states and actions
        if self.environnement == 0:
            self.states = list(range(self.envs[0].observation_dim()))
            self.actions = list(range(self.envs[0].action_dim()))
        elif self.environnement == 1:
            self.states = list(range(self.envs[0].observation_space.n))
            self.actions = list(range(self.envs[0].action_space.n))
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)   
        self.R_max = self.num_states * self.num_actions * np.log(self.T)
        self.L     = np.log(5 * self.H * self.N * self.num_actions * self.num_states * self.R_max / self.delta)
        # initialize policy for (one policy for every h in H)
        self.policy = self.init_policy()

        # initialize Q_hat and V_hat
        self.Q_hat, self.V_hat = self.init_Q_and_V()
        self.Q_hat_old = self.Q_hat
        
         # init big_N tracker
        self.global_trackers = self.init_global_trackers()
        
        self.beta_star = np.log(12 * self.num_states * self.num_actions * self.H / self.delta)
    def init_global_trackers(self):
        global_trackers = np.zeros((self.num_states,self.num_actions, self.H))
        return global_trackers
  
    def init_policy(self):    
        policy_ = []
        for _ in range(self.H):
            policy = np.zeros((self.num_states, self.num_actions))
            for state in range(self.num_states):
                action = np.random.choice(self.num_actions)
                policy[state, action] = 1
            policy_.append(policy)
        return np.array(policy_)

    def init_Q_and_V(self):
        Q_hat = np.array([np.full((self.num_states, self.num_actions), self.H - h, dtype=float) for h in range(self.H)])
        V_hat = np.array([np.full((self.num_states, 1), self.H - h, dtype=float) for h in range(self.H+1)])
        return Q_hat, V_hat

    def beta_c(self, n):
        return np.log(6 * self.num_states * self.num_actions * self.H / self.delta) + np.log(6 * np.exp(1) * (2 * n + 1))
    
    def compute_variance(self, state, action, h, big_N, p, sp, n):
        first_term = sum(n[i][state, action,h] * sp[i][h][(state, action)] for i in range(self.N))
        second_term = sum(n[i][state, action,h] * p[i][h][(state, action)] for i in range(self.N))
        return first_term / big_N - (second_term ** 2) / (big_N ** 2)
        
    def aggregate(self, Qs, p, sp, n , h):
        bonuses = np.zeros((self.num_states, self.num_actions))
        for state in self.states:
            for action in self.actions:
                big_N = self.sum_n(n, state, action, h)
                self.global_trackers[state, action,h] = big_N
                bonuses[state, action] = self.get_bonus(state, action, h, p, sp, n, big_N)
                if big_N > 0:
                    n_Q = self.sum_n_Q(Qs, n, state, action, h)
                    self.Q_hat[h][state, action] = min(n_Q / big_N + bonuses[state, action], self.H)
                else: self.Q_hat[h][state, action] = self.H
        return
    
    def aggregate_avg(self, Qs, p, sp, n, h):
        bonuses = np.zeros((self.num_states, self.num_actions))
        for state in self.states:
            for action in self.actions:
                big_N = self.sum_n(n, state, action, h)

                self.global_trackers[state, action,h] = big_N
                #V = self.compute_variance(state, action, h, big_N, p, sp, n)
                bonuses[state, action] = self.get_bonus(state, action, h, p, sp, n, big_N)
                if big_N > 0:
                    #n_Q = self.sum_n_Q(Qs, n, state, action, h)
                    Qsum = self.sum_Q(Qs, state, action , h)
                    self.Q_hat[h][state, action] = min(Qsum / self.N + bonuses[state, action], self.H)
                else: self.Q_hat[h][state, action] = self.H
        return

    def sum_n(self, n, state, action, h):
        s = 0    
        for i in range(len(n)):
            visit_tracker_pair = n[i]
            s += visit_tracker_pair[state, action,h]
        return s
        
    def sum_n_Q(self, Qs, n, state, action, h):
        s = 0
        for i in range(self.N):
            visit_tracker_pair = n[i]
            Q_hat = Qs[i][h]
            s += visit_tracker_pair[state, action,h] * Q_hat[state, action]
        return s
    
    def sum_Q(self, Qs, state, action, h):
        s = 0
        for i in range(self.N):
            Q_hat = Qs[i][h]
            s += Q_hat[state, action]
        return s
            
    def get_bonus(self, state, action, h, p, sp, n, big_N):
        if big_N >= 2:
            V = self.compute_variance(state, action, h, big_N, p, sp, n)
            first_term = (self.H) / big_N
            second_term = np.sqrt(max(V, 0) / big_N)
            return  first_term + second_term
        elif big_N <= 1:
            return self.H
    
    def broadcast_policy(self):
        for h in range(self.H):
            [obj.receive_policy(self.policy[h], h) for obj in self.vi_objs]
        return
    
    def broadcast_global_trackers(self):
        for h in range(self.H):
            [obj.receive_global_trackers(self.global_trackers[h], h) for obj in self.vi_objs]
        return
            
    def broadcast_V_hat(self, h):
        [obj.receive_V_hat(self.V_hat[h], h) for obj in self.vi_objs]
        return
    
    def update(self, h):
        for state in range(self.num_states):
            self.V_hat[h][state, 0] = np.max(self.Q_hat[h][state, :])
            best_action = np.argmax(self.Q_hat[h][state, :])
            self.policy[h][state, :] = 0
            self.policy[h][state, best_action] = 1
        return 

    def train(self):
        average_rewards = []
        regrets = []
        aggregations_t = []
        average_old_regret = 0
        while self.t <= self.T:
            self.broadcast_policy()
            self.broadcast_global_trackers()
            not_abortion = True
            while not_abortion and self.t <= self.T:
                rewards = []
                actual_regrets = []
                for id_client, obj in enumerate(self.vi_objs):
                    reward, regret, sync = obj.rollout()
                    rewards.append(reward)
                    actual_regrets.append(regret)
                    if sync== True:
                        not_abortion = False
                average_reward = sum(rewards) / self.N
                average_rewards.append(average_reward)
                # append the regrets
                average_actual_regret = sum(actual_regrets) / self.N
                Reg = average_actual_regret + average_old_regret
                regrets.append(Reg)
                average_old_regret = Reg
                if self.verbose:
                    print("t {} --> average_reward = {}".format(self.t, average_reward))
                self.t = self.t + 1 

            # update of the transition kernels
            for id_client in range(self.N):
                self.vi_objs[id_client].update_transitions()
            
            
            for h in reversed(range(self.H)):
                self.broadcast_V_hat(h + 1)
                # list that contains the Q hats
                Qs = []
                # list that contains all the products PV(s,a)
                p  = []
                # list that contains all the products with squared V, PV(s,a)^2
                sp = []
                # list that contains visit trackers
                n  = []
                for id_client in range(self.N):
                    QQ, pp, spp, nn = self.vi_objs[id_client].update_Q_hat(h)
                    Qs.append(QQ)
                    p.append(pp)
                    sp.append(spp)
                    n.append(nn)
                    self.vi_objs[id_client].update_actual_value_func()

                # perform aggregation
                if self.standard_aggregation:
                    self.aggregate_avg(Qs, p, sp, n, h)
                else: 
                    self.aggregate(Qs, p, sp, n , h)
                self.update(h)
            aggregations_t.append(self.t)
            #self.t += l
            self.r += 1

        avg_r_hat = np.zeros((self.num_states, self.num_actions), dtype=float)
        for id_client in range(self.N):
            avg_r_hat += self.vi_objs[id_client].get_r_hat(self.H - 1)
        return average_rewards, regrets, self.r, aggregations_t

            

                
                
                
                        

