
import numpy as np
import matplotlib.pyplot as plt

class FedQlearning_gen_adv:
    def __init__(self, envs, total_episodes, num_agents, 
                 is_fed = False, using_adv_min = 100, is_adv = 1, is_ber = 0,  H = 5,
                 common_kernel = None, common_reward = None,environment = 0):
        self.envs = envs
        self.total_episodes = total_episodes # total_episodes * num_agents = all episodes
        self.num_agents = num_agents
        self.environnement = environment
        self.common_kernel = common_kernel
        self.common_reward = common_reward

        # this comes from the modified rlberry code
        self.transitions = [env.get_P() for env in self.envs]
        self.average_P = np.mean(self.transitions, axis=0)
        
        self.R = [env.get_r() for env in self.envs]
        self.average_R = np.mean(self.R, axis=0)


        # get states and actions
        if self.environnement == 0:
            self.states = list(range(self.envs[0].observation_dim()))
            self.actions = list(range(self.envs[0].action_dim()))
            self.H = self.common_kernel.shape[0]
        elif self.environnement == 1:
            self.states = list(range(self.envs[0].observation_space.n))
            self.actions = list(range(self.envs[0].action_space.n))
            self.H = H
        self.S = len(self.states)
        self.A = len(self.actions)
        
        self.V_func = np.zeros((self.H, self.S),dtype = np.float32) #estimated value function
        self.V_ref_func = np.zeros((self.H, self.S),dtype = np.float32) #used reference function
        self.trigger_times = 0 #number of round
        self.comm_episode_collection = []
        self.using_adv_min = using_adv_min
        self.is_adv = is_adv
        self.is_ber = is_ber

        self.V_sum_stage = np.zeros((self.H, self.S, self.A),dtype = np.float32)
        self.V2_sum_stage = np.zeros((self.H, self.S, self.A),dtype = np.float32)
        self.Vref_sum_all = np.zeros((self.H, self.S, self.A),dtype = np.float32)
        self.Vref2_sum_all = np.zeros((self.H, self.S, self.A),dtype = np.float32)
        self.Vadv_sum_stage = np.zeros((self.H, self.S, self.A),dtype = np.float32)
        self.Vadv2_sum_stage = np.zeros((self.H, self.S, self.A),dtype = np.float32)

        self.V_ref_trigger = np.zeros((self.H, self.S), dtype = np.int32)

        self.count_variance = np.zeros((self.H, self.S, self.A),dtype = np.float32)

        self.N = np.zeros((self.H, self.S, self.A),dtype = np.int32)
        self.n_previous_st = np.zeros((self.H, self.S, self.A),dtype = np.float32)
        self.n_current_st = np.zeros((self.H, self.S, self.A),dtype = np.float32)



        self.is_fed = is_fed


        self.global_Q = np.full((self.H, self.S, self.A), self.H, dtype=np.float32)
        for i in range(self.H):
            self.global_Q[i,:,:] = self.H - i


        self.agent_N = np.zeros((num_agents, self.H, self.S, self.A), dtype=np.int32)
        self.agent_V_sum = np.zeros((num_agents, self.H, self.S, self.A), dtype=np.float32)
        self.agent_V2_sum = np.zeros((num_agents, self.H, self.S, self.A), dtype=np.float32)
        self.agent_Vref_sum = np.zeros((num_agents, self.H, self.S, self.A), dtype=np.float32)
        self.agent_Vref2_sum = np.zeros((num_agents, self.H, self.S, self.A), dtype=np.float32)
        self.agent_Vadv_sum = np.zeros((num_agents, self.H, self.S, self.A), dtype=np.float32)
        self.agent_Vadv2_sum = np.zeros((num_agents, self.H, self.S, self.A), dtype=np.float32)



        self.regret = []
        self.raw_gap = []

    def run_episode(self, agent_id):
        # Get the policy (actions for all states and steps)
        #V_func[h,s]
        event_triggered = False
        actions_policy = self.choose_action()
        if self.environnement == 0:
            state =self.envs[agent_id].reset()
        elif self.environnement == 1:
            state, _ = self.envs[agent_id].reset()
        state_init = state
        rewards = np.zeros((self.H, self.S, self.A))  # To store rewards for each state-step pair
        cumulative_reward = 0
        for step in range(self.H):
            # Select the action based on the agent's policy
            action = np.argmax(actions_policy[step, state])

            if self.environnement == 0:
                next_state, reward = self.envs[agent_id].step(action)
            elif self.environnement == 1:
                next_state, reward, _, _, _ = self.envs[agent_id].step(action)

            cumulative_reward += reward
            # Increment visit count for the current state-action pair
            self.agent_N[agent_id, step, state, action] += 1

        #             self.V_func = np.zeros((self.H, self.S),dtype = np.float32)
        # self.V_ref_func = np.zeros((self.H, self.S),dtype = np.float32)
            
            if step < self.H - 1: #location shifting
                self.agent_V_sum[agent_id, step, state, action] += self.V_func[step, next_state]
                self.agent_V2_sum[agent_id, step, state, action] += self.V_func[step, next_state]**2
                self.agent_Vref_sum[agent_id, step, state, action] += self.V_ref_func[step, next_state]
                self.agent_Vref2_sum[agent_id, step, state, action] += (self.V_ref_func[step, next_state])**2
                self.agent_Vadv_sum[agent_id, step, state, action] += (
                    self.V_func[step, next_state] - self.V_ref_func[step, next_state])
                self.agent_Vadv2_sum[agent_id, step, state, action] += (
                    self.V_func[step, next_state] - self.V_ref_func[step, next_state])**2



            # Store the received reward
            rewards[step, state, action] = reward
            # Check if the event-triggered condition is met

            flag = self.check_sync_triggered(agent_id, step, state, action, self.is_fed)
            if flag:
                event_triggered = True
            state = next_state
        return rewards, event_triggered, state_init , cumulative_reward

    def choose_action(self):
        actions = np.zeros([self.H, self.S, self.A])

        for step in range(self.H):
            for state in range(self.S):
                best_action = np.argmax(self.global_Q[step, state])
                actions[step, state, best_action] = 1

        return actions


    def check_sync_triggered(self, agent_id, step, state, action, is_fed):
        # Calculate the threshold for triggering the event
        #         self.N = np.zeros((self.H, self.S, self.A),dtype = np.float32)
        # self.n_previous_st = np.zeros((self.H, self.S, self.A),dtype = np.float32)
        # self.n_current_st = np.zeros((self.H, self.S, self.A),dtype = np.float32)
        # self.n_current_rd = np.zeros((self.H, self.S, self.A),dtype = np.float32)

        previous_state_visit = self.n_previous_st[step, state, action]
        current_state_visit = self.n_current_st[step, state, action]
        threshold = 1
        if is_fed == 1 and previous_state_visit > 0:
            if current_state_visit > (1-1/self.H)*previous_state_visit:
                threshold = round(np.floor(previous_state_visit/self.num_agents/self.H))
            else:
                threshold = round(np.ceil((previous_state_visit - current_state_visit)/self.num_agents))

        # Check if the visit count exceeds the threshold
        return self.agent_N[agent_id, step, state, action] >= threshold
    
    def check_stage_triggered(self, step, state, action):
        # Calculate the threshold for triggering the event

        previous_state_visit = self.n_previous_st[step, state, action]
        current_state_visit = self.n_current_st[step, state, action]
        
        return current_state_visit >= (self.num_agents * self.H)*(previous_state_visit == 0) + (
            1+1/self.H)* previous_state_visit
    


    def aggregate_data(self, policy_k, rewards, is_fed): # after a round
        H, M = self.H, self.num_agents
        for h in range(H):
            for s in range(self.S):
                for a in range(self.A):
                    #print(policy_k[h, s])
                    if a != np.argmax(policy_k[h, s]) or self.agent_N[:, h, s, a].sum() == 0:
                        # No update required, retain previous Q-values
                        continue
                    else:
                        self.n_current_st[h, s, a] += self.agent_N[:, h, s, a].sum()

                        self.V_sum_stage[h, s, a] += self.agent_V_sum[:,h,s,a].sum()
                        self.V2_sum_stage[h, s, a] += self.agent_V2_sum[:,h,s,a].sum()
                        self.Vref_sum_all[h, s, a] += self.agent_Vref_sum[:, h, s, a].sum()
                        self.Vref2_sum_all[h, s, a] += self.agent_Vref2_sum[:, h, s, a].sum()
                        self.Vadv_sum_stage[h, s, a] += self.agent_Vadv_sum[:, h, s, a].sum()
                        self.Vadv2_sum_stage[h, s, a] += self.agent_Vadv2_sum[:, h, s, a].sum()
                        if self.check_stage_triggered(h,s,a):
                            self.N[h,s,a] += self.n_current_st[h, s, a]
                            Q1 = rewards[h,s,a] + self.V_sum_stage[h,s,a]/self.n_current_st[h, s, a] + np.sqrt(
                                2*(H-h-1)*(H-h-1)/self.n_current_st[h, s, a])
                            sigma2_v = self.V2_sum_stage[h,s,a]/self.n_current_st[h, s, a] - (
                                self.V_sum_stage[h,s,a]/self.n_current_st[h, s, a])**2
                            if sigma2_v < 0:
                                sigma2_v = 1e-8
                            Q2 = rewards[h,s,a] + self.V_sum_stage[h,s,a]/self.n_current_st[h, s, a] + 2*np.sqrt(
                                sigma2_v/self.n_current_st[h, s, a])
                                

                            Q2 = Q2*(self.n_current_st[h, s, a] > 10) + (H-h)*(self.n_current_st[h, s, a] <= 10)

                            if not self.is_ber:

                                Q2 = H-h

                            sigma2_vref = self.Vref2_sum_all[h,s,a]/self.N[h,s,a] - (self.Vref_sum_all[h,s,a]/self.N[h,s,a])**2
                            sigma2_vadv = self.Vadv2_sum_stage[h,s,a]/self.n_current_st[h,s,a] - (
                                self.Vadv_sum_stage[h,s,a]/self.n_current_st[h,s,a])**2
                            if sigma2_vref < 0:
                                sigma2_vref = 1e-8
                            if sigma2_vadv < 0:
                                sigma2_vadv = 1e-8
                            Q3 = rewards[h,s,a] + self.Vref_sum_all[h,s,a]/self.N[h, s, a] + (
                             self.Vadv_sum_stage[h,s,a]/self.n_current_st[h,s,a]) + 2*np.sqrt(
                                sigma2_vref/self.N[h, s, a]) + 2*np.sqrt(
                                sigma2_vadv/self.n_current_st[h, s, a])
                            
                            if self.is_adv == 0:
                                Q3 = H - h
                            # print(Q1)
                            # print(Q2)
                            # print(Q3)
                            # print(self.global_Q[h,s,a])
                            #Q3 = Q3*(self.n_current_st[h, s, a] > 10) + (H-h)*(self.n_current_st[h, s, a] <= 10)
                            self.global_Q[h,s,a] = min([Q1,Q2,Q3,self.global_Q[h,s,a]])
                        
                            self.n_previous_st[h,s,a] = self.n_current_st[h, s, a]
                            self.n_current_st[h, s, a] = 0.0
                            self.V_sum_stage[h, s, a] = 0.0
                            self.V2_sum_stage[h, s, a] = 0.0
                            self.Vadv_sum_stage[h, s, a] = 0.0
                            self.Vadv2_sum_stage[h, s, a] = 0.0
        
        self.agent_N.fill(0)
        self.agent_V_sum.fill(0)
        self.agent_V2_sum.fill(0)
        self.agent_Vref_sum.fill(0)
        self.agent_Vref2_sum.fill(0)
        self.agent_Vadv_sum.fill(0)
        self.agent_Vadv2_sum.fill(0)
    
    def update_reference(self, h, s):
        if h == 0 or self.V_ref_trigger[h,s] == 1:
            return
        if self.N[h,s,:].sum() >= self.using_adv_min:
            self.V_ref_trigger[h,s] = 1
            self.V_ref_func[h-1,s] = self.V_func[h-1,s]

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
        # cummulative regret per-agent
        self.regret_cum = 0
        best_value , _, best_Q = self.best_gen()
        # Event-triggered termination flag
        event_triggered = False
        # Initialize a structure to store rewards (deterministic reward)
        rewards = np.zeros((self.H, self.S, self.A))
        for h in range(self.H - 1):
            for s in range(self.S):
                self.V_func[h,s] = max(self.global_Q[h+1, s, :])
                self.V_ref_func[h,s] = self.V_func[h,s]

        for h in range(1,self.H):
            for s in range(self.S):
                self.update_reference(h, s)
        actions_policy = self.choose_action()
        average_rewards = []
        regret_federation = []
        comm_episode_collection = []
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
                    regret_federation.append(self.regret_cum/self.num_agents)

                for h in range(self.H):
                    for s in range(self.S):
                        a = np.argmax(actions_policy[h, s])
                        if rewards[h, s, a] == 0:
                            rewards[h, s, a] = agent_reward[h, s, a]

                if agent_event_triggered:
                    event_triggered = True
                    comm_episode_collection.append(episode)


            # Calculate regret
            average_rewards.append(average_reward)
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

                self.aggregate_data(actions_policy, rewards, is_fed = self.is_fed)
                event_triggered = False
                actions_policy = self.choose_action()
                for h in range(self.H - 1):
                    for s in range(self.S):
                        self.V_func[h,s] = max(self.global_Q[h+1, s, :])
                for h in range(1,self.H):
                    for s in range(self.S):
                        self.update_reference(h, s)
        return best_value, best_Q, value, self.global_Q,  average_rewards, regret_federation, self.trigger_times ,self.comm_episode_collection