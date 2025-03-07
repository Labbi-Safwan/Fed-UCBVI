# FederatedValueIteration

To run the experiments, you need to specify the parameters --T (number of episodes), --environment (0 for synthetic and 1 for GridWord) --alg (0 for Fed-UCBVI, 1 for Fed-Qlearning and 2 for FedQ-Advantage). To make the plots you just specify  --environment (0 for synthetic and 1 for GridWord) and --experiment (from 1 to 3)
```
	# for the simulation itself
	python main.py 

	# to make the plots of the paper
	python plots_main.py 
```
