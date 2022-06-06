import numpy as np
import pandas as pd
import time
import math
from multiprocessing import Pool
from sklearn.model_selection import KFold
import csv
import random


'''
use math2015 data,including FrcSub,Math1,Math2
training data use 80% of total data
'''
threshold_slip = 0.3
threshold_guess = 0.1

def EStep(IL,sg,n,r,k,i):
    base = 2**(k-2)
    for l in range(i*base,(i+1)*base):
        # student number
        lll = ((1 - sg[:, 0]) ** n * sg[:, 0] ** (1 - n)) ** r.T.A[l] * (sg[:, 1] ** n * (
            1 - sg[:, 1]) ** (1 - n)) ** (1 - r.T.A[l])
        IL[:, l] = lll.prod(axis=1)
    return IL

def MStep(IL,n,r,k,i):
    base = 2**(k-2)
    ni,nj=n.shape
    IR = np.zeros((4, nj))
    n1 = np.ones(n.shape)
    for l in range(i*base,(i+1)*base):
        IR[0] += np.sum(((1 - r.A[:, l]) * n1).T * IL[:, l], axis=1)
        IR[1] += np.sum(((1 - r.A[:, l]) * n).T * IL[:, l], axis=1)
        IR[2] += np.sum((r.A[:, l] * n1).T * IL[:, l], axis=1)
        IR[3] += np.sum((r.A[:, l] * n).T * IL[:, l], axis=1)
    return IR
def trainDINAModel(n,Q,threshold):
    startTime = time.time()
    ni, nj = n.shape
    Qi, Qj = Q.shape   #
    # The K matrix represents the matrix of possible skill patterns composed of k skills
    K = np.mat(np.zeros((Qj, 2 ** Qj), dtype=int))
    for j in range(2 ** Qj):
        l = list(bin(j).replace('0b', ''))  # l stands for each skill mode.
        for i in range(len(l)):
            K[Qj - len(l) + i, j] = l[i]
    std = np.sum(Q, axis=1)
    r = (Q * K == std) * 1    # The R matrix indicates whether the theoretical jth question can be correct for the pattern l.
    sg = 0.01 * np.ones((nj, 2))

    continueSG = True
    kk =1
    lastLX = 1
    # count iteration times
    # student*pattern = student* problem       problem*skill         skill*pattern
    #  E-M step
    while continueSG == True:
        # E step，calculate likelihood matrix
        IL = np.zeros((ni, 2 ** Qj))
        IR = np.zeros((4, nj))
        # skill pattern number
        if multi==True:
            #print('multi 4 processes')
            with Pool(processes=4) as pool:
                multiple_results = [pool.apply_async(EStep, (IL, sg, n, r, Qj, i)) for i in range(4)]
                for item in ([res.get(timeout=1000) for res in multiple_results]):
                    IL += item

                sumIL = IL.sum(axis=1)
                LX = np.sum([i for i in map(math.log2, sumIL)])

                IL = (IL.T / sumIL).T * aPrior

                multiple_results = [pool.apply_async(MStep, (IL, n, r, Qj, i)) for i in range(4)]
                for item in ([res.get(timeout=1000) for res in multiple_results]):
                    IR += item
        else:
            for l in range(2 ** Qj):
                lll = ((1 - sg[:, 0]) ** n * sg[:, 0] ** (1 - n)) ** r.T.A[l] * (sg[:, 1] ** n * (
                    1 - sg[:, 1]) ** (1 - n)) ** (1 - r.T.A[l])
                IL[:, l] = lll.prod(axis=1)
            sumIL = IL.sum(axis=1)
            LX = np.sum([i for i in map(math.log2, sumIL)])
            IL = (IL.T / sumIL).T* aPrior
            n1 = np.ones(n.shape)
            for l in range(2 ** Qj):
                IR[0] += np.sum(((1 - r.A[:, l]) * n1).T * IL[:, l], axis=1)
                IR[1] += np.sum(((1 - r.A[:, l]) * n).T * IL[:, l], axis=1)
                IR[2] += np.sum((r.A[:, l] * n1).T * IL[:, l], axis=1)
                IR[3] += np.sum((r.A[:, l] * n).T * IL[:, l], axis=1)
        if abs(LX-lastLX)<threshold:
            continueSG = False
        lastLX = LX
        sg[:,1] = IR[1] / IR[0]            # guessing
        sg[:,0] = (IR[2]-IR[3]) / IR[2]   # slipping
        kk +=1   # Record the number of iterations
    endTime = time.time()
    print('Training time of DINA :[%.3f] s'%(endTime-startTime))

    for i in range(Qi):
        if sg[i, 0] > threshold_slip:
           sg[i, 0] = threshold_slip
    for j in range(Qi):
        if sg[j, 1] > threshold_guess:
            sg[j, 1] = threshold_guess
    return sg,r

def predictDINA(n,Q,sg,r,):
    startTime = time.time()
    ni, nj = n.shape
    Qi, Qj = Q.shape
    IL = np.zeros((ni, 2**Qj))
    if multi == True:
        with Pool(processes=4) as pool:
            multiple_results = [pool.apply_async(EStep, (IL, sg, n, r, Qj, i)) for i in range(4)]
            for item in ([res.get(timeout=1000) for res in multiple_results]):
                IL += item
    else:
        for l in range(2 ** Qj):
            lll = ((1 - sg[:, 0]) ** n * sg[:, 0] ** (1 - n)) ** r.T.A[l] * (sg[:, 1] ** n * (
                1 - sg[:, 1]) ** (1 - n)) ** (1 - r.T.A[l])
            IL[:, l] = lll.prod(axis=1)
    a = IL.argmax(axis=1)
    unique, counts = np.unique(a, return_counts=True)
    aPrior[unique] = counts/len(a)
    K = np.mat(np.zeros((Qj, 2 ** Qj), dtype=int))
    for j in range(2 ** Qj):
        l = list(bin(j).replace('0b', ''))
        for i in range(len(l)):
            K[Qj - len(l) + i, j] = l[i]
    std = np.sum(Q, axis=1)
    r = (Q * K == std) * 1
    i, j = n.shape
    rrrSum = []
    nnnSum = []
    for rrr, nnn in zip(r[:, a], n.T):
        rrrSum.extend(rrr.tolist()[0])
        nnnSum.extend(nnn)

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    C2 = confusion_matrix(nnnSum, rrrSum, labels=[0, 1])
    TP = C2[0][0]
    FP = C2[0][1]
    FN = C2[1][0]
    TN = C2[1][1]
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    AUC = roc_auc_score(nnnSum, rrrSum)
    Accuracy =(TP + TN) / (TP+TN+FP+FN)
    return Accuracy, Precision, Recall, F1, AUC

def trainAndPredict(model,dataSet,threshold):
    print('model:[%s]   dataset:[%s]' %(model,dataSet))
    if dataSet == 'FrcSub':
        n = pd.read_csv('math2020/FrcSub/data.csv').values    # Read the student response matrix X
        Q = np.mat(pd.read_csv('math2020/FrcSub/q.csv'))    # Read Knowledge Mastery Matrix Q
    elif dataSet == 'Math1':
        n = pd.read_csv('math2020/Math1/data.csv').values
        Q = np.mat(pd.read_csv('math2020/Math1/q.csv').head(15).values)
    elif dataSet == 'Math2':
        n = pd.read_csv('math2020/Math2/data.csv').head(500).values
        Q = np.mat(pd.read_csv('math2020/Math2/q.csv'))
    elif dataSet == 'Math2':
        n = pd.read_csv('math2020/scores/data.csv').head(500).values
        Q = np.mat(pd.read_csv('math2020/scores/q.csv'))
    else:
        print('dataSet not exist!')
        exit(0)

    n_splits = 5
    KF = KFold(n_splits=n_splits,shuffle=False)  # Fifty-fold cross-validation function
    s1=0
    s2=0
    s3=0
    s4=0
    s5=0
    for train_index, test_index in KF.split(n):
        X_train, X_test = n[train_index], n[test_index]
        sg,r = trainDINAModel(X_train,Q,threshold)
        Accuracy, Precision, Recall, F1, AUC =predictDINA(X_test, Q, sg, r,)
        f_record =open(dataSet+".txt","a")
        f_record.writelines(str(Accuracy)+","+str(Precision)+","+str(Recall)+","+str(F1)+","+str(AUC)+'\n')
        s1 +=Accuracy
        s2 +=Precision
        s3 +=Recall
        s4 +=F1
        s5 +=AUC
    s1 = s1/n_splits
    s2 = s2/n_splits
    s3 = s3/n_splits
    s4 = s4/n_splits
    s5 = s5/n_splits
    print('test:accuracy=[%s]  precision=[%s] recall=[%s]  f1=[%s]  auc=[%s] ' %(s1,s2,s3,s4,s5))
    return s1, s2,s3,s4,s5

def main():
    startTime = time.time()
    global  multi, aPrior,threshold

    threshold = 0.001   # termination threshold

    multi =True
    aPrior = np.ones(2 ** 8) / 10 ** 8   #FruSub
    #aPrior = np.ones(2 ** 11) / 10 ** 8     #math1
    #aPrior = np.ones(2 ** 16) / 10 ** 8  # math2
    dataSet = ('FrcSub', 'Math1', 'Math2', 'scores')
    model = ('DINA')
    s1=0
    s2=0
    s3=0
    s4=0
    s5=0
    for js in range(0,20):
        Accuracy_1, Precision_1, Recall_1, F1_1, AUC_1 = trainAndPredict(model[0], dataSet[0], threshold) # train part
        s1 += Accuracy_1
        s2 += Precision_1
        s3 += Recall_1
        s4 += F1_1
        s5 += AUC_1

    accuracy_average = s1 / 20
    precision_average = s2 / 20
    recall_average = s3 /20
    f1_average = s4 / 20
    auc_average = s5 / 20

    print('---------Dataset: Results after [%s] rounds of operation:------' % (dataSet[2]))
    print('main_accuracy=[%.6f]  main_precision=[%.6f] main_recall=[%.6f]  mean_f1=[%.6f]  mean_auc=[%.6f]  ' % (
        accuracy_average, precision_average, recall_average, f1_average, auc_average))
    print('Total time consuming :[%.3f] s' %(time.time()-startTime))


if __name__ == "__main__":
    main()
