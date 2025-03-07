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
    T = 10000
    plt.rcParams['text.usetex'] = True
    M = [10,20,40,80,120]
    M_second_experiment = 20
    epsilons_p = [0.0,  0.01, 0.31622776601683794]

    epsilons_r = [0.]

    epsilons_p_third_experiment = [0.0,  0.01, 0.31622776601683794]#, 0.001, 0.0021544346900318843]
    runs = 4
    parser = argparse.ArgumentParser(description='plotting the experiments')
    parser.add_argument("--environment", type=int, default=0, help="0 is for the synthetic environment and 1 is for Gridword")
    parser.add_argument("--experiment", type=int, default=1, help="Choose a number between 1 and 3")
    parser.add_argument("--standard_aggregation", type=bool, default=False, help="Aggregate Fed-UCBVI using standard averaging")
    parser.add_argument("--synchronisation_condition", type=int , default=1, help="1 is for the mixture of the two conditions, 2 is for only local doubling and 3 if for using the local estimator")
    
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
    current_folder = os.getcwd().replace("\\", "/")
    read_folder1 = current_folder+ '/experiments/' + 'Fed-UCBVI' + aggregation_t +synchronisation_c + '/'+ environment
    read_folder2 = current_folder+  '/experiments/' + 'Fed-Q_learning' + '/'+ environment
    read_folder3 = current_folder+ '/experiments/' + 'Fed-Q_Advantage' +'/'+ environment
    save_plot_folder = current_folder+  '/plots/' + experiment
    markers1 = [ '+','.', ',', 'o', 'v', '^', '<', '>']
    markers2 = ['x','s', 'D', 'h', 'p', '+', 'x', '|', '_']
    colors =  sns.color_palette("colorblind")
    fig = plt.figure(figsize=(4, 3))
    if args.experiment ==1:
        file_name = "output1.txt"
        counter_marker = -1
        legend_elements = []
        for ep in epsilons_p:
            counter_marker+=1
            average_values_UCBVI = []
            average_values_fedq = []
            average_values_fedq_adv = []
            smaller_vales_UCBVI = []
            smaller_vales_fedq = []
            smaller_vales_fedq_adv = []
            larger_vales_UCBVI =[]
            larger_vales_fedq = []
            larger_vales_fedq_adv = []
            std_values_UCBVI = []
            std_values_fedq = []
            std_values_fedq_adv = []
            for m in M:
                regrets_UCBVI = np.zeros((runs, T+1))
                regrets_fedq = np.zeros((runs, T))
                regrets_fedq_adv = np.zeros((runs, T))
                for run in range(runs):
                    regrets_file_fed_UCBVI = read_folder1 + '/' + 'T_' + str(T) +',N_'+ str(m) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run) +",regrets.pkl"
                    regrets_file_fed_Fedq = read_folder2 + '/' + 'T_' + str(T) +',N_'+ str(m) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run)+",regrets.pkl"
                    regrets_file_fed_Fedq_adv = read_folder3 + '/' + 'T_' + str(T) +',N_'+ str(m) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run)+",regrets.pkl"
                    regrets_UCBVI[run,:] = np.mean(load_results_from_pickle(regrets_file_fed_UCBVI),axis=1)
                    #regrets_UCBVI[run,:] = load_results_from_pickle(regrets_file_fed_UCBVI)
                    #regrets_fedq[run,:] = np.mean(load_results_from_pickle(regrets_file_fed_UCBVI),axis=1)
                    regrets_fedq[run,:] = load_results_from_pickle(regrets_file_fed_Fedq)
                    regrets_fedq_adv[run,:] = load_results_from_pickle(regrets_file_fed_Fedq_adv)
                smaller_vales_UCBVI.append(np.min(regrets_UCBVI[:, -1]))
                smaller_vales_fedq.append(np.min(regrets_fedq[:, -1]))
                smaller_vales_fedq_adv.append(np.min(regrets_fedq_adv[:, -1]))
                larger_vales_UCBVI.append(np.max(regrets_UCBVI[:, -1]))
                larger_vales_fedq.append(np.max(regrets_fedq[:, -1]))
                larger_vales_fedq_adv.append(np.max(regrets_fedq_adv[:, -1]))
                average_last_regret_UCVBI = np.mean(regrets_UCBVI[:, -1])
                average_last_regret_fedq = np.mean(regrets_fedq[:, -1])
                average_last_regret_fedq_adv = np.mean(regrets_fedq_adv[:, -1])
                std_last_regret_UCBVI = np.std(regrets_UCBVI[:, -1])
                std_last_regret_fedq = np.std(regrets_fedq[:, -1])
                std_last_regret_fedq_adv = np.std(regrets_fedq_adv[:, -1])
                std_values_UCBVI.append(std_last_regret_UCBVI)
                std_values_fedq.append(std_last_regret_fedq)
                std_values_fedq_adv.append(std_last_regret_fedq_adv)
                average_values_UCBVI.append(average_last_regret_UCVBI)
                average_values_fedq.append(average_last_regret_fedq)
                average_values_fedq_adv.append(average_last_regret_fedq_adv)
            legend_elements.append(Line2D([0],[0], color = colors[counter_marker],label=r"$\epsilon_p = {:.3g}$".format(ep)))
            plt.plot(M, average_values_UCBVI, label = r"Fed-UCBVI: $\epsilon_p = {:.3g}$".format(ep), marker = 'x', color = colors[counter_marker],
                markersize = 8, linewidth = 0.8)
            plt.plot(M, average_values_fedq, label = r"Fed-Q $\epsilon_p = {:.3g}$".format(ep), marker = 'o',color = colors[counter_marker],
                markersize = 8, linewidth = 0.8)
            plt.plot(M, average_values_fedq_adv, label = r"Fed-Q-Adv $\epsilon_p = {:.3g}$".format(ep), marker = 'v',color = colors[counter_marker],
                markersize = 8, linewidth = 0.8)
            plt.fill_between(M, np.maximum(np.array(average_values_UCBVI) - np.array(std_values_UCBVI),np.array(smaller_vales_UCBVI) ), np.minimum(np.array(average_values_UCBVI) + np.array(std_values_UCBVI),np.array(larger_vales_UCBVI)),color = colors[counter_marker] ,alpha=0.3)
            plt.fill_between(M, np.array(average_values_fedq) - np.array(std_values_fedq), np.array(average_values_fedq) + np.array(std_values_fedq), color = colors[counter_marker] , alpha=0.3)
            plt.fill_between(M, np.array(average_values_fedq_adv) - np.array(std_values_fedq_adv), np.array(average_values_fedq_adv) + np.array(std_values_fedq_adv), color = colors[counter_marker] , alpha=0.3)
        fontsize = 17
        #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        plt.xlabel("Number of agents", fontsize=fontsize, **font)
        plt.ylabel("Last Regret", fontsize=fontsize, **font)
        #if args.environment ==1:
        #    plt.title("GridWorld environment", fontsize=fontsize)
        #else:
        #    plt.title("Synthetic environment", fontsize=fontsize)
        plt.xscale('log')
        plt.yscale('log')
        #plt.text(0.45, -0.4, 'Common regret (lower is better), at T = 10^4 as a function of M\n for different hetereogenity levels in a log-log scale:\n crosses represent Fed-UCBVI, circles represent\n FedQ-Bernstein and triangles represent FedQ-Advantage.', 
        #        fontsize=8, ha='center', va='center', transform=plt.gca().transAxes)
        #plt.tight_layout()
        plt.xticks(fontsize=fontsize-3)
        plt.yticks(fontsize=fontsize-3)
        # Add the custom legend to the plot
        #if args.environment ==1:
        #plt.legend(handles=legend_elements, fontsize=14, loc=6  ,framealpha = 0.4)
        plt.grid(linestyle = '--', alpha = 0.6) 
        plt.tight_layout()
        if args.environment ==1:
            plt.savefig("gridword_experiment_1.pdf", bbox_inches="tight")

        else:
            plt.savefig("garnett_experiment_1.pdf", bbox_inches="tight")

    elif args.experiment ==2:
        file_name = "output2.txt"
        counter_marker = -1 
        legend_elements = []
        for ep in epsilons_p:
            counter_marker+=1
            regrets_UCBVI = np.zeros((runs, T+1))
            regrets_fedq = np.zeros((runs, T))
            regrets_fedq_adv = np.zeros((runs, T))
            aggregation_times = []
            for run in range(runs):
                regrets_file_fed_UCBVI = read_folder1 + '/' + 'T_' + str(T) +',N_'+ str(M_second_experiment) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run) +',regrets.pkl'
                regrets_file_fed_Fedq = read_folder2 + '/' + 'T_' + str(T) +',N_'+ str(M_second_experiment) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run) +',regrets.pkl'
                regrets_file_fed_Fedq_adv = read_folder3 + '/' + 'T_' + str(T) +',N_'+ str(M_second_experiment) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run) +',regrets.pkl'
                aggregation_times_feducbvi = read_folder1 + '/' + 'T_' + str(T) +',N_'+ str(M_second_experiment) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run) +',aggregations.pkl'
                list = np.mean(load_results_from_pickle(regrets_file_fed_UCBVI),axis=1)
                regrets_UCBVI[run,:] = np.mean(load_results_from_pickle(regrets_file_fed_UCBVI),axis=1)
                regrets_fedq[run,:] = load_results_from_pickle(regrets_file_fed_Fedq)
                regrets_fedq_adv[run,:] = load_results_from_pickle(regrets_file_fed_Fedq_adv)
                #aggregation_times.append(load_results_from_pickle(aggregation_times_feducbvi))
            legend_elements.append(Line2D([0],[0], color = colors[counter_marker],label=r"$\epsilon_p = {:.3g}$".format(ep)))
            average_values_UCBVI = np.mean(regrets_UCBVI, axis=0)
            average_values_fedq = np.mean(regrets_fedq, axis=0)
            average_values_fedq_adv = np.mean(regrets_fedq_adv, axis=0)
            #aggregation_times_one_run = aggregation_times[0]
            std_regrets_UCBVI = np.std(regrets_UCBVI, axis=0)
            std_regrets_fedq = np.std(regrets_fedq, axis=0)
            std_regrets_fedq_adv = np.std(regrets_fedq_adv, axis=0)
            #plt.vlines(x=aggregation_times_one_run, ymin = 0,ymax=9, colors='teal', ls='--', lw=2)
            plt.plot(range(T+1), average_values_UCBVI, label = r"Fed-UCBVI: $\epsilon_p = {:.3g}$".format(ep), marker = 'x', color = colors[counter_marker],  markevery=600, linewidth = 0.8, markersize=8)
            plt.plot(range(T), average_values_fedq, label = r"Fed-Q $\epsilon_p = {:.3g}$".format(ep),marker = 'o', color = colors[counter_marker], markevery= 600, linewidth = 0.8, markersize=8)
            plt.plot(range(T), average_values_fedq_adv, label = r"Fed-Q-Adv $\epsilon_p = {:.3g}$".format(ep),marker = 'v', color = colors[counter_marker], markevery= 600, linewidth = 0.8, markersize=8)
            plt.fill_between(range(T+1), np.array(average_values_UCBVI) - np.array(std_regrets_UCBVI), np.array(average_values_UCBVI) + np.array(std_regrets_UCBVI), color = colors[counter_marker],alpha=0.3)
            plt.fill_between(range(T), np.array(average_values_fedq) - np.array(std_regrets_fedq), np.array(average_values_fedq) + np.array(std_regrets_fedq), color = colors[counter_marker],alpha=0.3)
            plt.fill_between(range(T), np.array(average_values_fedq_adv) - np.array(std_regrets_fedq_adv), np.array(average_values_fedq_adv) + np.array(std_regrets_fedq_adv), color = colors[counter_marker],alpha=0.3)

        fontsize = 17
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        #plt.text(0.45, -0.4, 'Common regret (lower is better) for M = 20 agents as a function \nof T for different hetereogeneity levels:crosses \n represent Fed-UCBVI, circles represent FedQ-Bernstein \n and triangles represent FedQ-Advantage.', 
        #        fontsize=8, ha='center', va='center', transform=plt.gca().transAxes)
        #plt.tight_layout()
        plt.xlabel("T", fontsize=fontsize, **font)
        plt.ylabel("Regret(T)", fontsize=fontsize, **font)
        #if args.environment ==1:
        #    plt.title("GridWorld environment, M =" + str(M_second_experiment), fontsize=fontsize)
        #else:
        #    plt.title("Synthetic environment, M =" + str(M_second_experiment), fontsize=fontsize)
        #plt.xscale('log')
        #plt.yscale('log')
        #if args.environment ==1:
        #plt.legend(handles=legend_elements, fontsize=14, loc=(1., 0.),framealpha = 0.4)
        plt.xticks(fontsize=fontsize-3)
        plt.yticks(fontsize=fontsize-3)
        plt.grid(linestyle='--', alpha=0.6)
        plt.tight_layout()
        if args.environment ==1:
            plt.savefig("gridword_experiment_2.pdf", bbox_inches="tight")
        else:
            plt.savefig("garnett_experiment_2.pdf", bbox_inches="tight")

    elif args.experiment ==3:
        file_name = "output3.txt"
        legend_elements = []
        counter_marker = -1 
        for ep in epsilons_p_third_experiment:
            counter_marker+=1
            average_values_UCBVI = []
            average_values_FEDQ = []
            average_values_FEDQ_adv = []
            std_values_UCBVI = []
            std_values_fedq = []
            std_values_fedq_adv = []
            for m in M:
                communications_UCBVI = np.zeros(runs)
                communication_fedq = np.zeros(runs)
                communication_fedq_adv = np.zeros(runs)
                for run in range(runs):
                    communications_fed_UCBVI_file = read_folder1 + '/' + 'T_' + str(T) +',N_'+ str(m) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run) +',communications.pkl'
                    communications_fed_Fedq_file = read_folder2 + '/' + 'T_' + str(T) +',N_'+ str(m) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run) +',communications.pkl'
                    communications_fed_Fedq_adv_file = read_folder2 + '/' + 'T_' + str(T) +',N_'+ str(m) +',er_'+ str(0.0) + ',ep_'+ str(ep)+ ',run_'+str(run) +',communications.pkl'
                    communications_UCBVI[run] = load_results_from_pickle(communications_fed_UCBVI_file)
                    communication_fedq[run] = load_results_from_pickle(communications_fed_Fedq_file)
                    communication_fedq_adv[run] = load_results_from_pickle(communications_fed_Fedq_adv_file)
                average_communication_UCVBI = np.mean(communications_UCBVI)
                average_communication_fedq = np.mean(communication_fedq)
                average_communication_fedq_adv = np.mean(communication_fedq_adv)
                std_communication_UCBVI = np.std(communications_UCBVI)
                std_communication_fedq = np.std(communication_fedq)
                std_communication_fedq_adv = np.std(communication_fedq_adv)
                std_values_UCBVI.append(std_communication_UCBVI)
                std_values_fedq.append(std_communication_fedq)
                std_values_fedq_adv.append(std_communication_fedq_adv)
                average_values_UCBVI.append(average_communication_UCVBI)
                average_values_FEDQ.append(average_communication_fedq)
                average_values_FEDQ_adv.append(average_communication_fedq_adv)
            legend_elements.append(Line2D([0],[0], color = colors[counter_marker],label=r"$\epsilon_p = {:.3g}$".format(ep)))
            plt.plot(M, average_values_UCBVI, label = r"Fed-UCBVI: $\epsilon_p = {:.3g}$".format(ep), marker = 'x', color = colors[counter_marker],
                markersize = 8, linewidth = 0.8)
            plt.plot(M, average_values_FEDQ, label = r"Fed-Q-learning $\epsilon_p = {:.3g}$".format(ep), marker = 'o', color = colors[counter_marker],
                markersize = 8 , linewidth = 0.8)
            plt.plot(M, average_values_FEDQ_adv, label = r"Fed-Q-learning $\epsilon_p = {:.3g}$".format(ep), marker = 'v', color = colors[counter_marker],
                markersize = 8 , linewidth = 0.8)
            plt.fill_between(M, np.array(average_values_UCBVI) - np.array(std_values_UCBVI), np.array(average_values_UCBVI) + np.array(std_values_UCBVI), color = colors[counter_marker], alpha=0.3)
            plt.fill_between(M, np.array(average_values_FEDQ) - np.array(std_values_fedq), np.array(average_values_FEDQ) + np.array(std_values_fedq), color = colors[counter_marker], alpha=0.3)
            plt.fill_between(M, np.array(average_values_FEDQ_adv) - np.array(std_values_fedq_adv), np.array(average_values_FEDQ_adv) + np.array(std_values_fedq_adv), color = colors[counter_marker], alpha=0.3)

        fontsize = 17
        #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        #plt.text(0.45, -0.4, 'Number of communication (lower is better) as a function of the \n number of agents for different hetereogenity levels:\n crosses  represent Fed-UCBVI, circles represent FedQ-Bernstein \n and triangles represent FedQ-Advantage.', 
        #        fontsize=8, ha='center', va='center', transform=plt.gca().transAxes)
        #plt.tight_layout()
        plt.xlabel("Number of agents", fontsize=fontsize,  **font)
        plt.ylabel("Communications", fontsize=fontsize,  **font)
        #if args.environment ==1:
        #    plt.title("GridWorld environment", fontsize=fontsize)
        #else:
        #    plt.title("Synthetic environment" , fontsize=fontsize)
        plt.xticks(fontsize=fontsize-3)
        plt.yticks(fontsize=fontsize-3)
        #if args.environment ==1:
        #plt.legend(handles=legend_elements, fontsize=14, loc=(1., 0.), framealpha = 0.4)
        plt.grid(linestyle = '--', alpha = 0.6) 
        plt.tight_layout()
        if args.environment ==1:
            plt.savefig("gridword_experiment_3.pdf", bbox_inches="tight")
        else:
            plt.savefig("garnett_experiment_3.pdf", bbox_inches="tight")