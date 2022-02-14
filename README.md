# CSP-DRL
Implementation of the EAAI paper [Learning Variable Ordering Heuristics for Solving Constraint Satisfaction Problems](https://www.sciencedirect.com/science/article/pii/S0952197621004255). *Engineering Applications of Artificial Intelligence (EAAI)*, 2022. 

ArXiv version is [here](https://arxiv.org/pdf/1912.10762.pdf).
```
@article{song2022learning,
  title={Learning variable ordering heuristics for solving Constraint Satisfaction Problems},
  author={Song, Wen and Cao, Zhiguang and Zhang, Jie and Xu, Chi and Lim, Andrew},
  journal={Engineering Applications of Artificial Intelligence},
  volume={109},
  pages={104603},
  year={2022},
  publisher={Elsevier}
}
```
The implementation is based on Google [Or-Tools](https://github.com/google/or-tools) and [graphnn](https://github.com/Hanjun-Dai/graphnn). Please see the respective repo for prerequisites. The code is complied under Windows 10, Visual studio 2017, and Cmake 3.13.2. Please following the below steps to compile and run.

## Compile Instructions
### 1. Clone the repo
```
git clone https://github.com/songwenas12/csp-drl.git
```

### 2. Build drlDLL
NOTE: we provide [pre-complied drlDLL files](https://github.com/songwenas12/csp-drl/tree/main/csp_DRL/pre_compiled_libs), which can be directly used if you do not want to modify the source code. To use these pre-complied libs, simply cd to ./csp_DRL and run
```
cd csp_DRL
Python3 copy_pre_complied_files.py
```
If you want to build drlDLL, cd to ./drlDLL and open drlDLL.sln using Visual Studio 2017, and complie. Then, copy the generated drlDLL.dll and drlDLL.lib files to ./csp_drl/pre_compiled_libs and run ```copy_pre_complied_files.py``` as above, or manually copy these files.

### 3. Build OR-Tools
Please follow the [official complie instructions of OR-Tools](https://developers.google.com/optimization/install/python/source_windows).


## Run the Pre-trained Models
We provide [pre-trained models](https://github.com/songwenas12/csp-drl/tree/main/csp_DRL/pre_trained_models) and [instances](https://github.com/songwenas12/csp-drl/blob/main/csp_DRL/evaluation_pool_randCSP.7z) to reproduce the results in the paper. Please cd to ./csp_DRL, and follow the below instructions.
### 1. Unzip evaluation_pool_randCSP.7z
### 2. Evaluate the pre-trained models
Modify ```pretrained_model_name``` and ```exising_pool_name``` in ```main_randCSP_Evaluation.py``` according to your request, and run
```
Python3 main_randCSP_Evaluation.py
```

## Train New Models
The training instances are generated on-the-fly. To train new models, open ```main_randCSP_Training.py```, modify the corresponding parameters, and run
```
Python3 main_randCSP_Training.py
```
