
import numpy as np
#from Env import FiniteStateFiniteActionMDP

import matplotlib.pyplot as plt

class FedQlearning_gen:
    def __init__(self, envs, c, total_episodes, num_agents, 
                 is_fed = False, is_bern = False,
                 cb = 2.0, using_bern_min = 100, using_bern_samp = 100, H = 5,
                 common_kernel = None, common_reward = None,environment = 0):
        #self.mdp = mdp
        self.envs = envs
        
        self.environnement = environment
        self.common_kernel = common_kernel
        self.common_reward = common_reward
        self.H = self.common_kernel.shape[0]

        # this comes from the modified rlberry code
        self.transitions = [env.get_P() for env in self.envs]
        self.average_P = np.mean(self.transitions, axis=0)
        

        self.R = [env.get_r() for env in self.envs]
        self.average_R = np.mean(self.R, axis=0)


        # get states and actions
        if self.environnement == 0:
            self.states = list(range(self.envs[0].observation_dim()))
            self.actions = list(range(self.envs[0].action_dim()))
        elif self.environnement == 1:
            self.states = list(range(self.envs[0].observation_space.n))
            self.actions = list(range(self.envs[0].action_space.n))
        
        self.S = len(self.states)
        self.A = len(self.actions)

        self.c = c
        self.cb = cb
        self.total_episodes = total_episodes
        self.num_agents = num_agents
        if not is_fed:
            self.total_episodes = total_episodes * num_agents
            self.num_agents = 1
        self.V_func = np.zeros((self.H, self.S),dtype = np.float32)
        self.trigger_times = 0
        self.comm_episode_collection = []
        self.using_bern_min = using_bern_min
        self.using_bern_samp = using_bern_samp
        self.V_sum_all = np.zeros((self.H, self.S, self.A),dtype = np.float32)
        self.V2_sum_all = np.zeros((self.H, self.S, self.A),dtype = np.float32)
        self.count_variance = np.zeros((self.H, self.S, self.A),dtype = np.float32)
        self.theta2_sum = np.zeros((self.H, self.S, self.A),dtype = np.float32)
        self.bonous_all = np.zeros((self.H, self.S, self.A),dtype = np.float32)
        self.is_fed = is_fed
        self.is_bern = is_bern

        self.global_Q = np.full((self.H, self.S, self.A), self.H, dtype=np.float32)
        for i in range(self.H):
            self.global_Q[i,:,:] = self.H - i


        self.global_N = np.zeros((self.H, self.S, self.A), dtype=np.int32)

        self.agent_N = np.zeros((num_agents, self.H, self.S, self.A), dtype=np.int32)
        
        self.agent_V_sum = np.zeros((num_agents, self.H, self.S, self.A), dtype=np.float32)
        self.agent_V_sum2 = np.zeros((num_agents, self.H, self.S, self.A), dtype=np.float32)

        self.regret = []
        self.raw_gap = []
        self.regret_federation = []

    def run_episode(self, agent_id):
        # Get the policy (actions for all states and steps)
        #V_func[h,s]
        event_triggered = False
        actions_policy = self.choose_action()
        if self.environnement == 0:
            state =self.envs[agent_id].reset()
        elif self.environnement == 1:
            state, info = self.envs[agent_id].reset()
        state_init = state
        rewards = np.zeros((self.H, self.S, self.A))  # To store rewards for each state-step pair
        cumulative_reward = 0
        for step in range(self.H):
            # Select the action based on the agent's policy
            action = np.argmax(actions_policy[step, state])
            if self.environnement == 0:
                next_state, reward = self.envs[agent_id].step(action)
            elif self.environnement == 1:
                next_state, reward, terminated, truncated, info = self.envs[agent_id].step(action)

            cumulative_reward += reward
            # Increment visit count for the current state-action pair
            self.agent_N[agent_id, step, state, action] += 1
            
            if step < self.H - 1:
                self.agent_V_sum[agent_id, step, state, action] += self.V_func[step, next_state]
                self.agent_V_sum2[agent_id, step, state, action] += ((self.V_func[step, next_state])*(self.V_func[step, next_state]))

            # Store the received reward
            rewards[step, state, action] = reward
            # Check if the event-triggered condition is met

            flag = self.check_event_triggered(agent_id, step, state, action, self.is_fed)
            if flag:
                event_triggered = True
            state = next_state
        return rewards, event_triggered, state_init, cumulative_reward

    def choose_action(self):
        actions = np.zeros([self.H, self.S, self.A])

        for step in range(self.H):
            for state in range(self.S):
                best_action = np.argmax(self.global_Q[step, state])
                actions[step, state, best_action] = 1

        return actions


    def check_event_triggered(self, agent_id, step, state, action, is_fed):
        # Calculate the threshold for triggering the event
        tilde_C = 1.0 / (self.H * (self.H + 1))
        if not is_fed:
            tilde_C = tilde_C/(10**9)
        global_visits = self.global_N[step, state, action]
        threshold = max(1, int(np.floor((tilde_C / self.num_agents) * global_visits)))

        # Check if the visit count exceeds the threshold
        return self.agent_N[agent_id, step, state, action] >= threshold

    def aggregate_data(self, policy_k, rewards, is_bern, is_fed):
        H, M = self.H, self.num_agents
        i_0 = 2 * M * H * (H + 1)
        if not is_fed:
            i_0 = i_0 * (10**9)
        for h in range(H):
            for s in range(self.S):
                for a in range(self.A):
                    if a != np.argmax(policy_k[h, s]) or self.agent_N[:, h, s, a].sum() == 0:
                        # No update required, retain previous Q-values
                        continue
                    else:
                        # Calculate aggregated values
                        N_h_k = self.global_N[h, s, a]
                        n_h_k = self.agent_N[:, h, s, a].sum()

                        if N_h_k < i_0:
                            # Case 1: Update rule for small N_h_k (update Q sequentially)
                            t00 = N_h_k
                            for ag_id in range(self.num_agents):
                                if self.agent_N[ag_id, h, s, a] > 0:
                                    t00 = t00 + 1
                                    step_size = (H + 1) / (H + t00)
                                    self.theta2_sum[h,s,a] = ((1 - step_size)**2) * self.theta2_sum[h,s,a] + step_size**2
                                    self.global_Q[h, s, a] = (1 - step_size) * (self.global_Q[h, s, a] - self.bonous_all[h,s,a]) + \
                                                     step_size * (rewards[h, s, a] + 
                                                                  self.agent_V_sum[ag_id, h, s, a])
                                    
                                    
                                    if self.count_variance[h, s, a] >= self.using_bern_samp and is_bern:
                                        temp = self.V2_sum_all[h, s, a]/self.count_variance[h, s, a]
                                        temp = temp - (self.V_sum_all[h, s, a] ** 2)/(self.count_variance[h, s, a] ** 2)
                                        self.bonous_all[h,s,a] = self.cb * temp * np.sqrt(self.theta2_sum[h,s,a])
                                    else: 
                                        self.bonous_all[h,s,a] = self.c * (H - h - 1) * np.sqrt(self.theta2_sum[h,s,a])
                                    self.global_Q[h, s, a] += self.bonous_all[h,s,a]

                        else:
                            t00 = N_h_k
                            alpha_agg_side = 1.0
                            self.global_Q[h, s, a] = self.global_Q[h, s, a] - self.bonous_all[h,s,a]
                            for i in range(n_h_k):
                                t00 = t00 + 1
                                step_size = (H + 1) / (H + t00)
                                self.theta2_sum[h,s,a] = ((1 - step_size)**2) * self.theta2_sum[h,s,a] + step_size**2
                                alpha_agg_side = alpha_agg_side*(1 - step_size)
                                
                                #ucb_bonus = self.c * (H - h - 1) * np.sqrt(H / t00)
                                #beta_temp = (1 - step_size)*beta_temp + step_size*ucb_bonus

                            self.global_Q[h, s, a] = alpha_agg_side * self.global_Q[h, s, a] + \
                                (1 - alpha_agg_side) * (rewards[h, s, a] + sum(self.agent_V_sum[:, h, s, a])/n_h_k)
                            
                            if self.count_variance[h, s, a] >= self.using_bern_samp and is_bern:
                                temp = self.V2_sum_all[h, s, a]/self.count_variance[h, s, a]
                                temp = temp - (self.V_sum_all[h, s, a] ** 2)/(self.count_variance[h, s, a] ** 2)
                                self.bonous_all[h,s,a] = self.cb * np.sqrt(self.theta2_sum[h,s,a] * temp)
                            else: 
                                self.bonous_all[h,s,a] = self.c * (H - h - 1) * np.sqrt(self.theta2_sum[h,s,a])
                            self.global_Q[h, s, a] += self.bonous_all[h,s,a]
                            #self.Q_upper[h, s, a] = temp
                            #self.global_Q[h, s, a] = min([self.global_Q[h, s, a], temp])

        # Update global visit counts
        
        
        self.V_sum_all += ((self.global_N >= self.using_bern_min)*(self.agent_V_sum.sum(axis=0)))
        self.V2_sum_all += ((self.global_N >= self.using_bern_min)*(self.agent_V_sum2.sum(axis=0)))
        self.count_variance += ((self.agent_N.sum(axis=0)) * (self.global_N >= self.using_bern_min))
        self.global_N += self.agent_N.sum(axis=0)


        # Reset the visit counts for each agent
        self.agent_N.fill(0)
        self.agent_V_sum.fill(0)
        self.agent_V_sum2.fill(0)

    def best_gen(self):
        if self.environnement == 0:
            Q = np.zeros([self.H, self.S, self.A])
            V = np.zeros([self.H + 1, self.S])
            actions = np.zeros([self.H, self.S, self.A])
            for h in range(self.H - 1, -1, -1):
                for s in range(self.S):
                    for a in range(self.A):
                        p = self.common_kernel[h,s, a]
                        EV = np.dot(p, V[h+1])
                        Q[h, s, a] = self.common_reward[h,s, a] + EV
                    actions[h, s, np.argmax(Q[h, s])] = 1
                    V[h, s] = np.max(Q[h, s])
            return V[0], actions, Q
        elif self.environnement == 1:
            Q = np.zeros([self.H, self.S, self.A])
            V = np.zeros([self.H + 1, self.S])
            actions = np.zeros([self.H, self.S, self.A])
            for h in range(self.H - 1, -1, -1):
                for s in range(self.S):
                    for a in range(self.A):
                        p = self.common_kernel[s, a]
                        EV = np.dot(p, V[h+1])
                        Q[h, s, a] = self.common_reward[s, a] + EV
                    actions[h, s, np.argmax(Q[h, s])] = 1
                    V[h, s] = np.max(Q[h, s])
            return V[0], actions, Q


    def value_gen(self, actions):
        if self.environnement == 0:
            Q = np.zeros([self.H, self.S, self.A])
            V = np.zeros([self.H + 1, self.S])
            for h in range(self.H - 1, -1, -1):
                for s in range(self.S):
                    for a in range(self.A):
                        p = self.common_kernel[h,s, a]
                        EV = np.dot(p, V[h+1])
                        Q[h, s, a] = self.common_reward[h,s, a] + EV
                    p = actions[h, s]
                    V[h, s] = np.dot(p, Q[h, s])
            return V[0]
        elif self.environnement == 1:
            Q = np.zeros([self.H, self.S, self.A])
            V = np.zeros([self.H + 1, self.S])
            for h in range(self.H - 1, -1, -1):
                for s in range(self.S):
                    for a in range(self.A):
                        p = self.common_kernel[s, a]
                        EV = np.dot(p, V[h+1])
                        Q[h, s, a] = self.common_reward[s, a] + EV
                    p = actions[h, s]
                    V[h, s] = np.dot(p, Q[h, s])
            return V[0]

    def learn(self):

        self.regret_cum = 0
        best_value , best_policy, best_Q = self.best_gen()
        # Event-triggered termination flag
        event_triggered = False
        # Initialize a structure to store rewards (deterministic reward)
        rewards = np.zeros((self.H, self.S, self.A))
        for h in range(self.H - 1):
            for s in range(self.S):
                self.V_func[h,s] = max(self.global_Q[h+1, s, :])
        actions_policy = self.choose_action()
        average_rewards = []
        for episode in range(self.total_episodes):

            # Run one episode for each agent
            value = self.value_gen(actions_policy)
            average_reward = 0
            for agent_id in range(self.num_agents):
                agent_reward, agent_event_triggered, state_init, cumulative_reward = self.run_episode(agent_id)
                average_reward += cumulative_reward / self.num_agents
                self.regret_cum = self.regret_cum + best_value[state_init] - value[state_init]
                self.regret.append(self.regret_cum/(episode+1))
                self.raw_gap.append(best_value[state_init] - value[state_init])
                if agent_id == self.num_agents - 1:
                    self.regret_federation.append(self.regret_cum/self.num_agents)

                for h in range(self.H):
                    for s in range(self.S):
                        a = np.argmax(actions_policy[h, s])
                        if rewards[h, s, a] == 0:
                            rewards[h, s, a] = agent_reward[h,s,a]

                if agent_event_triggered:
                    event_triggered = True

            average_rewards.append(average_reward)
            #print("Episode {} --> reward = {}".format(episode, average_reward))
            # Calculate regret

            
            #self.regret.append(best_value[initial_state] - value[initial_state])
            
            
            

            # Globally aggregate and update policy if event-triggered termination occurred
            if event_triggered:
                self.trigger_times += 1
                self.comm_episode_collection.append(episode)
                #actions_policy = self.choose_action()
#                 V_next = np.zeros(self.S)


#                 for s in range(self.S):
#                     # For each state, find the best action value at step h+1
#                     V_next[s] = np.max(self.global_Q[h+1, s])if h + 1 < self.H else 0


#                 agent_values = np.array([self.global_Q for _ in range(self.num_agents)])

                self.aggregate_data(actions_policy, rewards,  is_bern = self.is_bern, is_fed = self.is_fed)
                event_triggered = False
                actions_policy = self.choose_action()
                for h in range(self.H - 1):
                    for s in range(self.S):
                        self.V_func[h,s] = max(self.global_Q[h+1, s, :])
        
        return best_value, best_Q, value, self.global_Q, average_rewards, self.regret_federation, self.trigger_times ,self.comm_episode_collection