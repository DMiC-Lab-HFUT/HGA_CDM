# HGA_CDM
The Model Zoo of Cognitive Diagnosis Models,including classic Deterministic Inputs, Noisy And gates (DINA), Genetic Algorithm for Cognitive Diagnosis Models (GA_CDM), the hybrid GA with the local search (LS) operator HGA_CDM(LS), the hybrid GA with the LS operator, where ALS is adopted to decide whether to conduct a LS on an individual,and the Hybrid GA with the LS operator, where IALS is adopted to decide whether to conduct a LS on an individual. 
# Cognitive Diagnostic Model Made More Practical by Genetic Algorithm
Cognitive diagnosis has attracted increasing attention owing to the flourishing development of online education. As one of the most widely used cognitive diagnostic models, DINA (Deterministic Inputs, Noisy And gate) evaluates students’ knowledge mastery based on their performance of the exercises. However, the traditional DINA model and its variants face the problem of exponential explosion with respect to the number of knowledge components. The running time of these models
increases exponentially with the number of knowledge components, limiting their practical use. To make cognitive diagnosis more practical, an effective memetic algorithm composed of a genetic algorithm and a local search operator is applied to DINA to address the exponential explosion problem of the traditional model. Moreover, an improved adaptive local search method without the need of specifying any parameters is proposed to reduce redundant local searches and accelerate the running time. Experiments on real-world datasets demonstrate the effectiveness of the proposed models with respect to both time and accuracy.  
Source code and data set for the paper *Cognitive Diagnostic Model Made More Practical by Genetic Algorithm*.  
If this code helps with your studies, please kindly cite the following publication:
```

@article{
  title={Cognitive Diagnostic Model Made More Practical by Genetic Algorithm},
  author={Chenyang Bu, Member, IEEE, Fei Liu, Zhiyong Cao, Lei Li, Senior Member, IEEE, Yuhong Zhang, Xuegang Hu, Wenjian Luo, Senior Member, IEEE},
  year={2021}
}
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
The *func.py* program mainly includes some basic functions, such as acquiring the knowledge point inspection matrix, etc. 
The *tools.py* file mainly contains functions to save the training results.  
The *myAlgorithms.py* file mainly contains the functions of the GA_NBA model and the GA model.  
The *train_model.py* file is used to train the model.  
The *test_model.py* file is used to test the model.  

 


# Data Set
The datasets used for training in this project are *FrcSub, Math1, Math2, Scores*.  
For each dataset, the above five models were used for training and testing, and the effects were compared on the five evaluation indicators of *AUC, Accuracy, Recall, F1, and N_LS*.  
# Details

