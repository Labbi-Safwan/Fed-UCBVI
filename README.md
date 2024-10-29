# Fed-UCBVI

To run the experiments, you need to specify the parameters from the command line: --T (number of episodes),
 --environment (0 for synthetic and 1 for GridWord), --alg (0 for Fed-UCBVI and 1 for Fed-Qlearning)
 --N  (number of agents). The hetereogeneity on the transition kernel and the number of runs
  can be manually set inside the file. 
```
	# for the simulation itself
	python main.py 
```

To make the plots you need to modify the corresponding paramters in the plot_experiments.py file
and specify from the command line: --environment (0 for synthetic and 1 for GridWord) and --experiment (from 1 to 3).

```
	# to make the plots of the paper
	python plots_main.py 
```
