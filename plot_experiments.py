import pickle

import matplotlib.pyplot as plt
import os
import numpy as np
import argparse
import sys
from matplotlib.lines import Line2D
import seaborn as sns



def load_results_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

if __name__ == '__main__':
    font = {'family' : 'serif'
         }
    T = 1000
    M = [10]
    M_second_experiment = 10
    epsilons_p = [0.1]

    epsilons_r = [0.]

    epsilons_p_third_experiment = [0.0,  0.01, 0.31622776601683794]
    runs = 5
    parser = argparse.ArgumentParser(description='plotting the experiments')
    parser.add_argument("--environment", type=int, default=1, help="0 is for the synthetic environment and 1 is for Gridword")
    parser.add_argument("--experiment", type=int, default=1, help="Choose a number between 1 and 3")
    parser.add_argument("--standard_aggregation", type=bool, default=False, help="Aggregate Fed-UCBVI using standard averaging")
    parser.add_argument("--synchronisation_condition", type=int, default=1, help="1 is for the mixture of the two conditions, 2 is for only local doubling and 3 if for using the local estimator")
    
    args = parser.parse_args()
    args_dict = vars(args)
    if args.environment ==1:
        environment = "gridword"
    else:
        environment = "synthetic"
        
    if args.experiment ==1:
        experiment = "experiment1"
    elif args.experiment ==2:
        experiment = "experiment2"
    elif args.experiment ==3:
        experiment = "experiment3"
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
    read_folder1 = './experiments/' + 'Fed-UCBVI' + aggregation_t +synchronisation_c + '/'+ environment
    read_folder2 = './experiments/' + 'Fed-Q_learning' +'/'+ environment
    save_plot_folder = './plots/' + experiment
    markers1 = [ '+','.', ',', 'o', 'v', '^', '<', '>']
    markers2 = ['x','s', 'D', 'h', 'p', '+', 'x', '|', '_']
    colors =  sns.color_palette("colorblind")
    fig = plt.figure(figsize=(4, 3))
    if args.experiment ==1:
        counter_marker = -1
        legend_elements = []
        for ep in epsilons_p:
            counter_marker+=1
            average_values_UCBVI = []
            average_values_fedq = []
            smaller_vales_UCBVI = []
            smaller_vales_fedq = []
            larger_vales_UCBVI =[]
            larger_vales_fedq = []
            std_values_UCBVI = []
            std_values_fedq = []
            for m in M:
                regrets_UCBVI = np.zeros((runs, T+1))
                regrets_fedq = np.zeros((runs, T))
                for run in range(runs):
                    regrets_file_fed_UCBVI = read_folder1 + '/' + 'T_' + str(T) +',N_'+ str(m) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run) +',regrets.pkl'
                    regrets_file_fed_Fedq = read_folder2 + '/' + 'T_' + str(T) +',N_'+ str(m) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run)+',regrets.pkl'
                    regrets_UCBVI[run,:] = np.mean(load_results_from_pickle(regrets_file_fed_UCBVI),axis=1)
                    regrets_fedq[run,:] = load_results_from_pickle(regrets_file_fed_Fedq)
                smaller_vales_UCBVI.append(np.min(regrets_UCBVI[:, -1]))
                smaller_vales_fedq.append(np.min(regrets_fedq[:, -1]))
                larger_vales_UCBVI.append(np.max(regrets_UCBVI[:, -1]))
                larger_vales_fedq.append(np.max(regrets_fedq[:, -1]))
                average_last_regret_UCVBI = np.mean(regrets_UCBVI[:, -1])
                average_last_regret_fedq = np.mean(regrets_fedq[:, -1])
                std_last_regret_UCBVI = np.std(regrets_UCBVI[:, -1])
                std_last_regret_fedq = np.std(regrets_fedq[:, -1])
                std_values_UCBVI.append(std_last_regret_UCBVI)
                std_values_fedq.append(std_last_regret_fedq)
                average_values_UCBVI.append(average_last_regret_UCVBI)
                average_values_fedq.append(average_last_regret_fedq)
            legend_elements.append(Line2D([0],[0], color = colors[counter_marker],label=r"$\epsilon_p = {:.3g}$".format(ep)))
            plt.plot(M, average_values_UCBVI, label = r"Fed-UCBVI: $\epsilon_p = {:.3g}$".format(ep), marker = 'x', color = colors[counter_marker],
                markersize = 8, linewidth = 0.8)
            plt.plot(M, average_values_fedq, label = r"Fed-Q-learning $\epsilon_p = {:.3g}$".format(ep), marker = 'o',color = colors[counter_marker],
                markersize = 8, linewidth = 0.8)
            plt.fill_between(M, np.maximum(np.array(average_values_UCBVI) - np.array(std_values_UCBVI),np.array(smaller_vales_UCBVI) ), np.minimum(np.array(average_values_UCBVI) + np.array(std_values_UCBVI),np.array(larger_vales_UCBVI)),color = colors[counter_marker] ,alpha=0.3)
            plt.fill_between(M, np.array(average_values_fedq) - np.array(std_values_fedq), np.array(average_values_fedq) + np.array(std_values_fedq), color = colors[counter_marker] , alpha=0.3)
        fontsize = 17
        plt.xlabel("Number of agents", fontsize=fontsize, **font)
        plt.ylabel("Last Regret", fontsize=fontsize, **font)
        plt.xscale('log')
        plt.yscale('log')
        plt.xticks(fontsize=fontsize-3)
        plt.yticks(fontsize=fontsize-3)
        # Add the custom legend to the plot
        if args.environment ==1:
            plt.legend(handles=legend_elements, fontsize=14, loc='upper right')
        plt.grid(linestyle = '--', alpha = 0.6) 
        plt.tight_layout()
        if args.environment ==1:
            plt.savefig("gridword_experiment_1.pdf", bbox_inches="tight")
        else:
            plt.savefig("garnett_experiment_1.pdf", bbox_inches="tight")

    elif args.experiment ==2:
        counter_marker = -1 
        legend_elements = []
        for ep in epsilons_p:
            counter_marker+=1
            regrets_UCBVI = np.zeros((runs, T+1))
            regrets_fedq = np.zeros((runs, T))
            aggregation_times = []
            for run in range(runs):
                regrets_file_fed_UCBVI = read_folder1 + '/' + 'T_' + str(T) +',N_'+ str(M_second_experiment) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run) +',regrets.pkl'
                regrets_file_fed_Fedq = read_folder2 + '/' + 'T_' + str(T) +',N_'+ str(M_second_experiment) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run) +',regrets.pkl'
                aggregation_times_feducbvi = read_folder1 + '/' + 'T_' + str(T) +',N_'+ str(M_second_experiment) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run) +',aggregations.pkl'
                list = np.mean(load_results_from_pickle(regrets_file_fed_UCBVI),axis=1)
                regrets_UCBVI[run,:] = np.mean(load_results_from_pickle(regrets_file_fed_UCBVI),axis=1)
                regrets_fedq[run,:] = load_results_from_pickle(regrets_file_fed_Fedq)
            legend_elements.append(Line2D([0],[0], color = colors[counter_marker],label=r"$\epsilon_p = {:.3g}$".format(ep)))
            average_values_UCBVI = np.mean(regrets_UCBVI, axis=0)
            average_values_fedq = np.mean(regrets_fedq, axis=0)
            std_regrets_UCBVI = np.std(regrets_UCBVI, axis=0)
            std_regrets_fedq = np.std(regrets_fedq, axis=0)
            plt.plot(range(T+1), average_values_UCBVI, label = r"Fed-UCBVI: $\epsilon_p = {:.3g}$".format(ep), marker = 'x', color = colors[counter_marker],  markevery=600, linewidth = 0.8, markersize=8)
            plt.plot(range(T), average_values_fedq, label = r"Fed-Q-learning $\epsilon_p = {:.3g}$".format(ep),marker = 'o', color = colors[counter_marker], markevery= 600, linewidth = 0.8, markersize=8)

            plt.fill_between(range(T+1), np.array(average_values_UCBVI) - np.array(std_regrets_UCBVI), np.array(average_values_UCBVI) + np.array(std_regrets_UCBVI), color = colors[counter_marker],alpha=0.3)
            plt.fill_between(range(T), np.array(average_values_fedq) - np.array(std_regrets_fedq), np.array(average_values_fedq) + np.array(std_regrets_fedq), color = colors[counter_marker],alpha=0.3)
        fontsize = 17
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlabel("T", fontsize=fontsize, **font)
        plt.ylabel("Regret(T)", fontsize=fontsize, **font)
        if args.environment ==1:
            plt.legend(handles=legend_elements, fontsize=14, loc='upper left')
        plt.xticks(fontsize=fontsize-3)
        plt.yticks(fontsize=fontsize-3)
        plt.grid(linestyle='--', alpha=0.6)
        plt.tight_layout()
        if args.environment ==1:
            plt.savefig("gridword_experiment_2.pdf", bbox_inches="tight")
        else:
            plt.savefig("garnett_experiment_2.pdf", bbox_inches="tight")

    elif args.experiment ==3:
        legend_elements = []
        counter_marker = -1 
        for ep in epsilons_p_third_experiment:
            counter_marker+=1
            average_values_UCBVI = []
            average_values_FEDQ = []
            std_values_UCBVI = []
            std_values_fedq = []
            for m in M:
                communications_UCBVI = np.zeros(runs)
                communication_fedq = np.zeros(runs)
                for run in range(runs):
                    communications_fed_UCBVI_file = read_folder1 + '/' + 'T_' + str(T) +',N_'+ str(m) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run) +',communications.pkl'
                    communications_fed_Fedq_file = read_folder2 + '/' + 'T_' + str(T) +',N_'+ str(m) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run) +',communications.pkl'
                    communications_UCBVI[run] = load_results_from_pickle(communications_fed_UCBVI_file)
                    communication_fedq[run] = load_results_from_pickle(communications_fed_Fedq_file)
                average_communication_UCVBI = np.mean(communications_UCBVI)
                average_communication_fedq = np.mean(communication_fedq)
                std_communication_UCBVI = np.std(communications_UCBVI)
                std_communication_fedq = np.std(communication_fedq)
                std_values_UCBVI.append(std_communication_UCBVI)
                std_values_fedq.append(std_communication_fedq)
                average_values_UCBVI.append(average_communication_UCVBI)
                average_values_FEDQ.append(average_communication_fedq)
            legend_elements.append(Line2D([0],[0], color = colors[counter_marker],label=r"$\epsilon_p = {:.3g}$".format(ep)))
            plt.plot(M, average_values_UCBVI, label = r"Fed-UCBVI: $\epsilon_p = {:.3g}$".format(ep), marker = 'x', color = colors[counter_marker],
                markersize = 8, linewidth = 0.8)
            plt.plot(M, average_values_FEDQ, label = r"Fed-Q-learning $\epsilon_p = {:.3g}$".format(ep), marker = 'o', color = colors[counter_marker],
                markersize = 8 , linewidth = 0.8)
            plt.fill_between(M, np.array(average_values_UCBVI) - np.array(std_values_UCBVI), np.array(average_values_UCBVI) + np.array(std_values_UCBVI), color = colors[counter_marker], alpha=0.3)
            plt.fill_between(M, np.array(average_values_FEDQ) - np.array(std_values_fedq), np.array(average_values_FEDQ) + np.array(std_values_fedq), color = colors[counter_marker], alpha=0.3)
        fontsize = 17
        plt.xlabel("Number of agents", fontsize=fontsize,  **font)
        plt.ylabel("Communications", fontsize=fontsize,  **font)
        plt.xticks(fontsize=fontsize-3)
        plt.yticks(fontsize=fontsize-3)
        if args.environment ==1:
            plt.legend(handles=legend_elements, fontsize=14, loc='upper left')
        plt.grid(linestyle = '--', alpha = 0.6) 
        plt.tight_layout()
        if args.environment ==1:
            plt.savefig("gridword_experiment_3.pdf", bbox_inches="tight")
        else:
            plt.savefig("garnett_experiment_3.pdf", bbox_inches="tight")