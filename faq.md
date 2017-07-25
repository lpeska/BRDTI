# Frequently Asked Questions about BRDTI 

--------
## How to apply BRDTI on a novel dataset of drugs and targets?
--------

In order to apply BRDTI on a new dataset and predict most probable new drug-target interactions, you need to follow these steps:

- Prepare the dataset in the correct format: create 3 dataset matrices (drug-target interactions, drug-similarity and target-similarity) and store them in the tab-separated format in the folder /data/datasets (please follow the structure of the other datasets strictly)

- The dataset files should follow this naming convention:
  * [dataset]_admat_dgc.txt: matrix of drug-target interactions
  * [dataset]_simmat_dc.txt: matrix of drug similarities
  * [dataset]_simmat_dg.txt: matrix of target similarities

- Update the bottom of the eval_BRDTI.py file as follows:
  * insert after the 
" if __name__ == "__main__": "
line this code snipet
main(['--method=brdti', '--dataset=[dataset]', '--cvs=1', '--predict-num=[top-k predictions, e.g., 10]'])  
  * this code can be combined with hyperparameter setting or tuning as described in the readme.md file

- run eval_BRDTI.py
- results should be placed in the /output/newDTI folder

