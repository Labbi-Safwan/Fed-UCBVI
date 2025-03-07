import numpy as np
from  synthetic_env.Env import FiniteStateFiniteActionMDP
import pickle
from FedQ import FedQlearning_gen
from fed_adv import FedQlearning_gen_adv
import matplotlib.pyplot as plt
import os
import argparse
from gridword_env.gridworld import GridWorld
import fed_value_iteration as fvi
import concurrent.futures

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
            
def save_results_to_pickle(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def run_feducvbi(argument):
    full_path = argument[0]
    run = argument[1]

    rewards_file =  full_path + ',run_' +str(run) + ',rewards.pkl'
    regrets_file = full_path + ',run_'+ str(run)+ ',regrets.pkl'
    communications_file = full_path + ',run_'+ str(run)+ ',communications.pkl'
    aggregations_file = full_path +',run_'+ str(run) + ',aggregations.pkl'
    envs = argument[2]
    common_P = argument[3]
    common_r = argument[4]
    n = argument[5]
    er = argument[6]
    ep = argument[7]
    diction = argument[8]
    np.random.seed(run)
    federation = fvi.FedValueIteration(envs, common_kernel=common_P, common_reward=common_r, **diction)
    average_rewards, regrets, r, aggregations = federation.train()
    save_results_to_pickle(rewards_file, average_rewards)
    save_results_to_pickle(regrets_file, regrets)
    save_results_to_pickle(communications_file, r)
    save_results_to_pickle(aggregations_file, aggregations)
    return 

def run_fedqlearning(argument):
    c = 2.0 
    full_path = argument[0]
    run = argument[1]
    np.random.seed(run)
    rewards_file =  full_path + ',run_' +str(run) + ',rewards.pkl'
    regrets_file = full_path + ',run_'+ str(run)+ ',regrets.pkl'
    communications_file = full_path + ',run_'+ str(run)+ ',communications.pkl'
    aggregations_file = full_path +',run_'+ str(run) + ',aggregations.pkl'
    envs = argument[2]
    common_P = argument[3]
    common_r = argument[4]
    n = argument[5]
    diction = argument[6]
    fed_q = FedQlearning_gen(envs, c, diction['T'], n, is_bern= True, is_fed=True,
                                cb = 2.0, using_bern_min = 1000, using_bern_samp = 1000,
                                common_kernel = common_P, common_reward = common_r, environment=diction['environment'], H = diction['H'])
    _, _, _, _, average_rewards, regrets_federation, communications , aggregations_times = fed_q.learn()
    save_results_to_pickle(rewards_file, average_rewards)
    save_results_to_pickle(regrets_file, regrets_federation)
    save_results_to_pickle(communications_file, communications)
    save_results_to_pickle(aggregations_file, aggregations_times)
    return 

def run_fedqlearning_reference_advanatage(argument):
    full_path = argument[0]
    run = argument[1]
    np.random.seed(run)
    rewards_file =  full_path + ',run_' +str(run) + ',rewards.pkl'
    regrets_file = full_path + ',run_'+ str(run)+ ',regrets.pkl'
    communications_file = full_path + ',run_'+ str(run)+ ',communications.pkl'
    aggregations_file = full_path +',run_'+ str(run) + ',aggregations.pkl'
    envs = argument[2]
    common_P = argument[3]
    common_r = argument[4]
    n = argument[5]
    diction = argument[6]
    fed_q_advantage = FedQlearning_gen_adv(envs, diction['T'], n, common_kernel = common_P, common_reward = common_r,
                                                       environment=diction['environment'],  H = diction['H'])
    _, _, _, _, average_rewards, regrets_federation, communications , aggregations_times = fed_q_advantage.learn()
    save_results_to_pickle(rewards_file, average_rewards)
    save_results_to_pickle(regrets_file, regrets_federation)
    save_results_to_pickle(communications_file, communications)
    save_results_to_pickle(aggregations_file, aggregations_times)
    return 



if __name__ == '__main__':
    epsilons_p =  [0.0] + list(map(float, np.logspace(-3, -0.5, 6)))  
    #epsilons_p =  [0.0] + list(map(float, np.logspace(-3, -0.5, 7)))  
    epsilons_r = [0.]
    parser = argparse.ArgumentParser(description='launching the experiment')
    parser.add_argument("--alg", type=int, default=0, help="0 is for Fed-UCBVI, 1 is for Fed-Q learning and 2 for FedQ-Advantage")
    parser.add_argument("--environment", type=int, default=0, help="0 is for the synthetic environment and 1 is for Gridword")
    parser.add_argument("--runs", type=int, default=5, help="number of runs")
    parser.add_argument("--T", type=int, default=int(20), help="max number of episodes to collect")
    parser.add_argument("--N", type=int, nargs='+', default= 1, help="List of agents")
    parser.add_argument("--H", type=int, nargs='+', default=10, help="time horizon")
    parser.add_argument("--verbose", type=bool, default=False, help="time horizon")
    parser.add_argument("--confidence", type=float, default=0.05,help="confidence")
    parser.add_argument("--standard_aggregation", type=bool, default=False, help="Aggregate Fed-UCBVI using standard averaging")
    parser.add_argument("--synchronisation_condition", type=int, default=1, help="1 is for the mixture of the two conditions, 2 is for only local doubling and 3 if for using the local estimator")

    args = parser.parse_args()
    args_dict = vars(args)
    if isinstance(args.N, int):
        N = [args.N]    
    else:
        N = args.N
    runs = args.runs
    if args.alg ==2:
        algorithm = "Fed-Q_Advantage"
        aggregation_t = ""
        synchronisation_c = ""
    elif args.alg ==1:
        algorithm = "Fed-Q_learning"
        aggregation_t = ""
        synchronisation_c = ""
    else:
        algorithm = "Fed-UCBVI"
        if args.standard_aggregation:
            aggregation_t = '+regular_averaging'
            if args.synchronisation_condition ==1:
                synchronisation_c  = "+mixture_condition"
            elif args.synchronisation_condition ==2:
                synchronisation_c  = "+only_local_doubling"
            elif args.synchronisation_condition ==3:
                synchronisation_c  = "+only_local_estimator"
        else:
            aggregation_t = '+weighted_averaging'
            if args.synchronisation_condition ==1:
                synchronisation_c  = "+mixture_condition"
            elif args.synchronisation_condition ==2:
                synchronisation_c  = "+only_local_doubling"
            elif args.synchronisation_condition ==3:
                synchronisation_c  = "+only_local_estimator"
    if args.environment ==1:
        environment = "gridword"
    else:
        environment = "synthetic"
        
    parent_directory = './experiments/' + algorithm +aggregation_t +synchronisation_c +'/'+ environment
    create_folder_if_not_exists(parent_directory)
    args_dict = vars(args)
    np.random.seed(0)

    for n in N:
        for er in epsilons_r:
            for i, ep in enumerate(epsilons_p):
                full_path = parent_directory + '/' + 'T_' + str(args.T) +',N_'+ str(n) +',er_'+ str(er) + ',ep_'+ str(ep) 
                seeds = [k for k in range(args.runs)]
                if args.environment ==0:
                    S, A, H = 5, 5, 4
                    env = FiniteStateFiniteActionMDP(H=H, S=S, A=A)
                    common_P = env.get_P()
                    common_r = env.get_r()
                    envs = [FiniteStateFiniteActionMDP(H=H, S=S, A=A, epsilon_p=ep, common=common_P, common_reward=common_r) for _ in range(n)]
                    with concurrent.futures.ProcessPoolExecutor(max_workers=runs) as executor:
                        if args.alg ==0:
                            arguments = [[full_path,seed,envs, common_P,common_r,n,er,ep,args_dict] for seed in seeds]
                            results = list(executor.map(run_feducvbi, arguments))   
                        elif args.alg ==1:
                            arguments = [[full_path,seed,envs, common_P,common_r,n,args_dict] for seed in seeds]
                            results = list(executor.map(run_fedqlearning, arguments))
                        elif args.alg ==2:
                            arguments = [[full_path,seed,envs, common_P,common_r,n,args_dict] for seed in seeds]
                            results = list(executor.map(run_fedqlearning_reference_advanatage, arguments))
                        else:
                            print("The algorithm shoud be either 0, 1 or 2: 0 is for Fed-UCBVI, 1 is for Fed-Q learning and 2 for Federated Reference advantage decomposition") 
                elif args.environment ==1:
                    env = GridWorld(3, 3, walls=((1, 1),(1,1)), success_probability=0.8)
                    common_P = env.get_P() # get the common transition kernel
                    common_r = env.get_r() # get the common reward function
                    envs = [GridWorld(3, 3, walls=((1, 1),(1,1)), success_probability=1.0, common=common_P, epsilon_p=ep) for _ in range(n)]
                    with concurrent.futures.ProcessPoolExecutor(max_workers=runs) as executor:
                        if args.alg ==0:
                            arguments = [[full_path,seed,envs, common_P,common_r,n,er,ep,args_dict] for seed in seeds]
                            results = list(executor.map(run_feducvbi, arguments))   
                        elif args.alg ==1:
                            arguments = [[full_path,seed,envs, common_P,common_r,n,args_dict] for seed in seeds]
                            results = list(executor.map(run_fedqlearning, arguments)) 
                        elif args.alg ==2:
                            arguments = [[full_path,seed,envs, common_P,common_r,n,args_dict] for seed in seeds]
                            results = list(executor.map(run_fedqlearning_reference_advanatage, arguments))
                        else:
                            print("The algorithm shoud be either 0, 1 or 2: 0 is for Fed-UCBVI, 1 is for Fed-Q learning and 2 for Federated Reference advantage decomposition") 
                else:
                    print("The environnement shoud be either 0 or 1: 0 is for the synthetic environment and 1 is for Gridword")