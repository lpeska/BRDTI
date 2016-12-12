# BRDTI -- a Python library for Bayesian Ranking Prediction of Drug-Target Interactions

version 1.0, December 12, 2016

--------
This package is written by:

Ladislav Peska
Dept. of Software Engineering, Charles University in Prague, Czech Republic
Email: peska@ksi.mff.cuni.cz

This package is based on the PyDTI package by Yong Liu,
https://github.com/stephenliu0423/PyDTI

--------
BRDTI works on Python 2.7 (tested on Intel Python 2.7.12).
--------
BRDTI requires NumPy, scikit-learn and SciPy to run.
To get the results of different methods, please run eval_BRDTI.py by setting suitable values for the following parameters:

	--method 			set DTI prediction method
	--dataset: 			choose the benchmark dataset, i.e., nr, gpcr, ic, e
	--csv:				choose the cross-validation setting, 1 for predicting DTIs, 2 for prediction for novel drugs, and 3 for prediction for novel targets, (default 1)
	--specify-arg:		0 for choosing optimal arguments, 1 for using default/specified arguments (default 1)
	--method-opt:		set arguments for each method
	--predict-num:		a positive integer for predicting top-N novel DTIs for each drug and target (default 0)
        
	Here are some examples:

	(1) run a method with default arguments
		python eval_BRDTI.py --method="brdti" --dataset="nr"
		python eval_BRDTI.py --method="inv_brdti" --dataset="e" --cvs=2

	(2) run a method with specified arguments

		python eval_BRDTI.py --method="blmnii" --dataset="ic" --cvs=1 --specify-arg=1 --method-opt="alpha=1.0"

	(3) choose the optimal parameters for a method

		python eval_BRDTI.py --method="wnngip" --dataset="nr" --cvs=1 --specify-arg=0

	(4) predict the top-10 novel DTIs for each drug and target

		python eval_BRDTI.py --method="brdti" --dataset="gpcr" --predict-num=10 --method-opt="--method-opt=D=100 global_regularization=0.01 cb_alignment_regularization_user=0.5 learning_rate=0.1"

You can also use the same syntax via calling the main method within the eval_BRDTI.py code and thus run several predictions sequentionally:

            if __name__ == "__main__":  
                main(['--method=blmnii', '--dataset=gpcr', '--predict-num=20', '--method-opt=alpha=1.0'])
                main(['--method=wnngip', '--dataset=gpcr', '--predict-num=20', '--method-opt=alpha=0.9 T=0.2'])
                main(['--method=netlaprls', '--dataset=gpcr', '--predict-num=20', '--method-opt=beta_d=10**(-6) beta_t=10**(-6) gamma_d=100 gamma_t=100'])

The results can be analysed via result_sign_analysis.py
The predictions of each individual method can be combined via combined_approach.py
