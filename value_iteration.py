import numpy as np
from copy import deepcopy
# here we should import rlberry


class ValueIteration:
    def __init__(self, env, t, common_kernel, common_reward, ep = 0.0, er = 0.0, M=10, **kwargs):

        # state that remembers if doubled
        self.sync = False
        
        self.synccondition = kwargs.get('synchronisation_condition') 
        # initialize the environment
        self.env = env
        self.common_kernel = common_kernel
        self.common_reward = common_reward

        # initialize the value function
        #V = {s: 0 for s in states}
        self.H = self.common_kernel.shape[0]
        self.N = kwargs.get('N')
        self.environnement = kwargs.get('environment')

        # get hyperparameters
        self.T = kwargs.get('T')
        
        # get the confidence
        self.confidence = kwargs.get("confidence")
        
        # maybe this double notation is a bit confusing
        self.M = M
        self.epsilon_p = ep
        self.epsilon_r = er

        self.t = t
        self.l = 0

        if self.environnement == 0:
            self.states = list(range(self.env.observation_dim()))
            self.actions = list(range(self.env.action_dim()))
        elif self.environnement == 1:
            self.states = list(range(self.env.observation_space.n))
            self.actions = list(range(self.env.action_space.n))
        self.num_states = len(self.states)
        self.num_actions = len(self.actions)

        # initialize transitions
        self.transitions = self.init_transitions()

        # initialize visit trackers and the global tracker
        self.visit_trackers_pair, self.visit_trackers_triplet = self.init_visit_trackers()
        self.visit_trackers_pair_old = deepcopy(self.visit_trackers_pair)
        
        self.global_trackers = self.init_global_trackers()
        self.global_trackers_estimator = deepcopy(self.global_trackers)


        # initialize policy
        #self.policy = self.init_policy()
        #self.policy = policy
        self.V_hat = [np.full((self.num_states, 1), self.H, dtype=float) for h in range(self.H + 1)]
        self.policy = self.init_policy()

        # init r_hat
        self.r_hat = [np.zeros((self.num_states, self.num_actions), dtype=float) for h in range(self.H)]

        # init Q_hat
        self.Q_hat = [np.zeros((self.num_states, self.num_actions), dtype=float) for h in range(self.H)]
        self.products = [np.zeros((self.num_states, self.num_actions), dtype=float) for h in range(self.H)]
        self.products_squared = [np.zeros((self.num_states, self.num_actions), dtype=float) for h in range(self.H)]

        self.optimal_value_func = self.opt_value_func()
        self.actual_value_func = self.act_value_func()
        

        self.beta_c = np.log(6 * self.num_states * self.num_actions * self.H / self.confidence) + np.log(6 * np.exp(1) * (2 * self.T + 1))
        self.S = 14 * self.epsilon_p * self.T * self.H * self.M + 182 * self.M * self.beta_c
    def init_global_trackers(self):
        global_trackers = np.zeros((self.num_states,self.num_actions, self.H))
        return global_trackers
    
    def init_transitions(self):
        # initialize the transition matrix
        init_value = 1 / self.num_states
        return [np.full((self.num_states, self.num_actions, self.num_states), init_value) for _ in range(self.H)]
    
    def init_visit_trackers(self):
        # initialize the pair visit trackers (one for each h in H)
        visit_trackers_pair = np.zeros((self.num_states,self.num_actions, self.H))
        # initialize the triplet visit trackers (one for each h in H)
        visit_trackers_triplet =np.zeros((self.num_states,self.num_actions,self.num_states, self.H))
        return visit_trackers_pair, visit_trackers_triplet

    def init_policy(self):    
        # initialize the policy (one policy for every h in H)
        policy_ = []

        for _ in range(self.H):
            policy = np.zeros((self.num_states, self.num_actions))
            for state in range(self.num_states):
                action = np.random.choice(self.num_actions)
                policy[state, action] = 1
            policy_.append(policy)
        return policy_
    
    def opt_value_func(self):
        if self.environnement == 0:
            Q = [np.zeros((self.num_states, self.num_actions), dtype=float) for h in range(self.H)]
            V = [np.full((self.num_states, 1), self.H, dtype=float) for h in range(self.H + 1)]
            V[self.H] = np.zeros((self.num_states, 1), dtype=float) 
            P = self.common_kernel#self.env.get_P()
            R = self.common_reward#self.env.get_r()
            for h in range(self.H - 1, -1, -1):
                for s in range(self.num_states):
                    for a in range(self.num_actions):
                        p = P[h,s, a]
                        EV = p @ V[h+1]
                        Q[h][s, a] = R[h,s, a] + EV[0]
                    V[h][s] = np.max(Q[h][s])
            return V[0]#, actions, Q
        
        elif self.environnement == 1:
            Q = [np.zeros((self.num_states, self.num_actions), dtype=float) for h in range(self.H)]
            V = [np.full((self.num_states, 1), self.H, dtype=float) for h in range(self.H + 1)]
            V[self.H] = np.zeros((self.num_states, 1), dtype=float) 
            P = self.common_kernel#self.env.get_P()
            R = self.common_reward#self.env.get_r()
            for h in range(self.H - 1, -1, -1):
                for s in range(self.num_states):
                    for a in range(self.num_actions):
                        p = P[s, a]
                        EV = p @ V[h+1]
                        Q[h][s, a] = R[s, a] + EV[0]
                    V[h][s] = np.max(Q[h][s])
            return V[0]#, actions, Q
    def act_value_func(self):
        if self.environnement == 0:
            # Initialize Q and V arrays
            Q = np.zeros((self.H, self.num_states, self.num_actions), dtype=float)
            V = np.full((self.H + 1, self.num_states, 1), self.H, dtype=float)
            V[self.H] = np.zeros((self.num_states, 1), dtype=float)  # Terminal state value is zero

            P = self.common_kernel  # Transition probabilities (num_states, num_actions, num_next_states)
            R = self.common_reward  # Rewards (num_states, num_actions)

            # Loop over time steps h in reverse
            for h in range(self.H - 1, -1, -1):
                # Expected value calculation for all states and actions at once
                EV = np.einsum('sat,tn->sa', P[h,:,:,:], V[h+1])
                
                # Calculate Q-values for all states and actions
                Q[h] = R[h,:,:] + EV

                # Select the best action based on the policy for each state (vectorized)
                selected_actions = np.array([self.select_action(self.policy, s, h) for s in range(self.num_states)])

                # Update value function V based on the best action
                V[h, np.arange(self.num_states), 0] = Q[h, np.arange(self.num_states), selected_actions]
            return V[0]

        elif self.environnement == 1:
            # Initialize Q and V arrays
            Q = np.zeros((self.H, self.num_states, self.num_actions), dtype=float)
            V = np.full((self.H + 1, self.num_states, 1), self.H, dtype=float)
            V[self.H] = np.zeros((self.num_states, 1), dtype=float)  # Terminal state value is zero

            P = self.common_kernel  # Transition probabilities (num_states, num_actions, num_next_states)
            R = self.common_reward  # Rewards (num_states, num_actions)

            # Loop over time steps h in reverse
            for h in range(self.H - 1, -1, -1):
                # Expected value calculation for all states and actions at once
                EV = np.einsum('sat,tn->sa', P, V[h+1])
                
                # Calculate Q-values for all states and actions
                Q[h] = R + EV

                # Select the best action based on the policy for each state (vectorized)
                selected_actions = np.array([self.select_action(self.policy, s, h) for s in range(self.num_states)])

                # Update value function V based on the best action
                V[h, np.arange(self.num_states), 0] = Q[h, np.arange(self.num_states), selected_actions]
            return V[0]

    
    def update_actual_value_func(self):
        self.actual_value_func = self.act_value_func()
        return                    

    def select_action(self, policy, state, h):
        # select action for a specific
        #policy = self.policy[h]
        policy = policy[h]
        probs = policy[state, :]
        action = np.random.choice(self.num_actions, p=probs)
        return action

    def receive_policy(self, policy, h):
        # function to receive the policy from the central server
        self.policy[h] = policy
        self.visit_trackers_pair_old = deepcopy(self.visit_trackers_pair)
        self.sync = False
        return
    
    def receive_global_trackers(self, global_tracker, h):
        self.global_trackers[h] = global_tracker
        self.global_trackers_estimator[h] = deepcopy(self.global_trackers[h])#global_tracker
        return

    def receive_V_hat(self, V, h):
        # function to receive the value from the central server
        self.V_hat[h] = V
        return
    
    def get_t(self):
        return self.t
    
    def get_Q_hat(self, h):
        return self.Q_hat[h]
    
    def get_l(self):
        return self.l
    
    def get_r_hat(self, h):
        return self.r_hat[h]
    
    def get_transitions(self, h=0):
        return self.transitions[:,:,:,h]

    def rollout(self):
        if self.environnement == 0:
            state = self.env.reset()
        elif self.environnement == 1:
            state, info = self.env.reset()
        cumulative_reward = 0
        cumulative_reward = 0
        regret = self.optimal_value_func[state] - self.actual_value_func[state]
        for h in range(self.H):

            # select the action
            action = self.select_action(self.policy, state, h)
            # update the visit tracker for the pair
            self.visit_trackers_pair[state, action,h] += 1

            # perform the step
            if self.environnement == 0:
                next_state, reward = self.env.step(action)
            elif self.environnement == 1:
                next_state, reward, terminated, truncated, info = self.env.step(action)
            cumulative_reward += reward

            self.r_hat[h][state, action] = reward
            # update visit tracker for the triplet
            self.visit_trackers_triplet[state, action, next_state,h] += 1
            
            # update the global tracker estimator
            self.global_trackers_estimator[state, action,h] += self.M
            if  self.synccondition ==1:
                if self.global_trackers[state, action,h] < self.S and self.visit_trackers_pair[state, action,h] >= 2 * self.visit_trackers_pair_old[state, action,h]:
                    self.sync = True
                elif self.global_trackers[state, action,h] >= self.S and self.global_trackers_estimator[state, action,h] >= 2 * self.global_trackers[state, action,h]:
                    self.sync = True
            elif self.synccondition ==2:
                if self.visit_trackers_pair[state, action,h] >= 2 * self.visit_trackers_pair_old[state, action,h]:
                    self.sync = True
            elif self.synccondition ==3:
                if self.global_trackers_estimator[state, action,h] >= 2 * self.global_trackers[state, action,h]:
                    self.sync = True
            state = next_state
        # update l and t
        self.l += 1
        self.t += 1
        return cumulative_reward, regret, self.sync

    def update_transitions(self):
        #K = {key: value for key, value in self.visit_tracker_pair.items() if value > 0}
        self.transitions =  np.divide(
                                self.visit_trackers_triplet, 
                                self.visit_trackers_pair[:, :, np.newaxis, :],  # Add new axis to broadcast the division
                                out=np.full_like(self.visit_trackers_triplet, 1 / self.num_states),  # Output to store the result, initialized to zero
                                where=self.visit_trackers_pair[:, :, np.newaxis, :] != 0  # Only divide where visit_pair is non-zero
                            )
        return

    def update_Q_hat(self, h):
        # Calculate the dot product for all states and actions at once
        product = np.einsum('ijk,k->ij', self.transitions[:,:,:,h], self.V_hat[h+1].squeeze())
        product_squared = np.einsum('ijk,k->ij', self.transitions[:,:,:,h], np.square(self.V_hat[h+1]).squeeze())
        # Update Q_hat, products, and products_squared
        self.Q_hat[h] = self.r_hat[h] + product
        self.products[h] = product
        self.products_squared[h] = product_squared
        return self.Q_hat, self.products, self.products_squared, self.visit_trackers_pair

    def update(self, h, V):
        # update the transitions
        self.update_transitions()

        # update the Q_hat
        self.update_Q_hat(h, V)
        return #self.Q_hat, self.visit_trackers_pair
                     