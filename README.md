# HGA_CDM
The Model Zoo of Cognitive Diagnosis Models,including classic Deterministic Inputs, Noisy And gates `(DINA)`, Genetic Algorithm for Cognitive Diagnosis Models `(GA_CDM)`, the hybrid GA with the local search (LS) operator `HGA_CDM(LS)`, the hybrid GA with the LS operator, where `ALS` is adopted to decide whether to conduct a LS on an individual,and the Hybrid GA with the LS operator, where `IALS` is adopted to decide whether to conduct a LS on an individual.  
Cognitive diagnosis has attracted increasing attention owing to the flourishing development of online education. As one of the most widely used cognitive diagnostic models, `DINA (Deterministic Inputs, Noisy And gate)` evaluates students’ knowledge mastery based on their performance of the exercises. However, the traditional DINA model and its variants face the problem of exponential explosion with respect to the number of knowledge components. The running time of these models
increases exponentially with the number of knowledge components, limiting their practical use. To make cognitive diagnosis more practical, an effective memetic algorithm composed of a genetic algorithm and a local search operator is applied to DINA to address the exponential explosion problem of the traditional model. Moreover, an improved adaptive local search method without the need of specifying any parameters is proposed to reduce redundant local searches and accelerate the running time. Experiments on real-world datasets demonstrate the effectiveness of the proposed models with respect to both time and accuracy.  
# Cognitive Diagnostic Model Made More Practical by Genetic Algorithm
Source code and data set for the paper *Cognitive Diagnostic Model Made More Practical by Genetic Algorithm*.  
If this code helps with your studies, please kindly cite the following publication:  
```
@ARTICLE{9812476,  
author={Bu, Chenyang and Liu, Fei and Cao, Zhiyong and Li, Lei and Zhang, Yuhong and Hu, Xuegang and Luo, Wenjian},  
journal={IEEE Transactions on Emerging Topics in Computational Intelligence},   
title={Cognitive Diagnostic Model Made More Practical by Genetic Algorithm},   
year={2022},  
pages={1-15},  
doi={10.1109/TETCI.2022.3182692}}
```

# Dependencies:
  * python 3.6
  * matplotlib
  * numpy
  * pandas
  * sklearn
  * math
  * time


# Usage
The dataset of the project is saved in the *dataSets* folder, and the running results are saved in the *czy_result* folder.  
If you want to run the program, just run the main.py file.  
` python main.py`  
* The `func.py` program mainly includes some basic functions, such as acquiring the knowledge point inspection matrix, etc. 
* The `tools.py` file mainly contains functions to save the training results.  
* The `myAlgorithms.py` file mainly contains the functions of the GA_NBA model and the GA model.  
* The `train_model.py` file is used to train the model.  
* The `test_model.py` file is used to test the model.  


# Data Set
The datasets used for training in this project are *FrcSub, Math1, Math2, Scores*.  
For each dataset, the above five models were used for training and testing, and the effects were compared on the five evaluation indicators of *AUC, Accuracy, Recall, F1, and N_LS*.  


# Details
* The results of the comparison experiments are presented from the following aspects: the prediction performance, the times of local search, and the running time. 
* To avoid accidental results, the experiments used the five-fold cross-validation method, and all the experimental results were averaged from 20 independent replicate experiments for each fold.
* The article analyzes the influence of *the number of knowledge points, crossover probability Pc, mutation probability Pm and population size n_pop* on the experimental performance.  

