# atom_centered_cgcnn_aws
Mimics Tian Xie cgcnn, altered for atomic feature prediction and for use in AWS

If you want to run deep learning jobs but you don't have access to GPUs, aws is an option. I ran into issues doing this for my modified version of a [cgcnn](https://github.com/txie-93/cgcnn) because I would need to install pymatgen whenever I wanted to train a model. This repository contains the code to pickle data which is then passed to a version of the cgcnn which is atom-centered. Therefore, the code in savedata has different requirements than the code in optunacgcnn.py and n the modified cgcnn. 

The main advantage here is the speed with which you can train things on aws, where you can dynamically add more storage/gpus, and the operations themselves are optmized to be faster than on traditional clusters. One drawback is that you have to run a conda install optuna on startup, but this takes far less time than training the models does. 
