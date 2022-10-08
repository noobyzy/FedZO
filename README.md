# FedZO
-----------------------------------------------------------------------------------------------
Black-box aderversarial attack experiments for  [Communication-Efficient Stochastic Zeroth-Order Optimization for Federated Learning](https://arxiv.org/abs/2201.09531).

For detailed information, please refer to the paper.

## Requirments
The environment for this experiment is 
* Python 3.8.5
* Pytorch

please download other required packages via `pip install -r requirements.txt`

## Dataset
The algorithm is expected to work on multiple datsets, not limited to `cifar` `fashion mnist` `mnist` which are standard datasets

## Files
```
FedZO/
│   README.md
│   requirements.txt    
│
└─── blackbox_attack/
│   │   models/
│   │   save/
│   │   src/
│   
└─── dataset/
    │   cifar/
    │   fmnist/
    │   mnist/
```
- `blackbox_attack/` the main folder
-- `models/` the well-trained classification models
-- `save/` the running result of the algorithm, metrics including *training loss* and *testing accuracy*
-- `src/` the algorithm code

- `dataset/` the folder stores all datasets
-- `cifar/` `fmnist/` `mnist/` folders to store different datasets

`src/` contains multiple files:
- `alg_*.py` the distributed zeroth order optimization algorithm
- `attack_main.py` the main function to perform aderversarial attack
- `load_file.py`  load *dataset* and *model*
- `models.py` pytorch model structure
- `ObjFunc.py` blackbox attack objective function
- `options.py` defining the arguments of the experiments/ hyperparameters of the algorithms
- `run_*.sh` shell file to excute the experiment (contain default argument setting)
- `train_model.py` train the DNN classification model from initilization
- `utils.py` other util functions

## Perform the experiment
Once you have configured the environment, run the experiement via 
```
bash run_[your_algorithm_name].sh
```
and replace `[your_algorithm_name]` with your desired experiment name.

If everything goes well, you will find the results in `save/` folder, each execute of any of the algorithms should yield two products:
- a `*.pkl` file, produced by *pickle* package, contains the the algorithm's per-iteration training loss / testing accuracy
- a `*/` folder sharing the same prefix as the previous file, containing the visualization of the optimized adversarial image, and a mini-batch of samples: the original images and the images perturbed by the adversarial image (aims at showing that the perturbation is imperceptible by human)

## Contact Us
If you encounter any problem, feel free to post it in `issue` of this repository. You can also contact us via the following email:

Ziyi Yu
```
yuzy@shanghaitech.edu.cn
yuziyi@mail.ustc.edu.cn
```

Wenzhi Fang
```
fangwzh1@shanghaitech.edu.cn
```