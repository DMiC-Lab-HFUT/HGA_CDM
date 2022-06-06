import csv
import math

import numpy as np
import time
import igraph
import pandas as pd

threshold_slip = 0.3
threshold_guess = 0.1



def acquireQ(Qpath):
    with open(Qpath, "r") as csvfile:
        reader = csv.reader(csvfile)
        Q = []
        temp = []
        for line in reader:
            temp.append(line)
        del temp[0]


        for line in temp:
            Qlie = []
            for i in line:
                i = int(float(i))
                Qlie.append(i)
            Q.append(Qlie)
    return Q



def acquireData(dataPath):

    with open(dataPath, "r") as csvfile:
        reader = csv.reader(csvfile)
        data = []
        temp = []
        for line in reader:
            temp.append(line)
        del temp[0]

        for line in temp:
            data_per_line = []
            for i in line:
                i = int(float(i))
                data_per_line.append(i)
            data.append(data_per_line)
    return data



def acquireA(individual, student, knowledge):
    A = individual[0:student * knowledge]
    A = np.array(A)
    A = A.reshape(student, knowledge)

    A = A.tolist()
    return A



def acquireYITA(A, Q, student, question, knowledge):
    YITA = [[0] * question for i in range(student)]
    for i in range(student):
        for j in range(question):
            yita = 1
            for k in range(knowledge):
                yita = yita * pow(A[i][k], Q[j][k])
            YITA[i][j] = yita
    return YITA



def acquireSandG(student, individual, knowledge, GENE_LENGTH, len_s_g):
    S = []
    G = []
    num_points = pow(2, len_s_g)

    for i in range(knowledge * student, GENE_LENGTH, len_s_g * 2):
        s = individual[i:i + len_s_g]
        s = list(map(str, s))
        s = "".join(s)
        sdec = int(s, 2)
        x = 0 + sdec * (threshold_slip / (num_points - 1))
        S.append(x)

        s = individual[i + len_s_g:i + (len_s_g * 2)]
        s = list(map(str, s))
        s = "".join(s)
        sdec = int(s, 2)
        x = 0 + sdec * (threshold_guess / (num_points - 1))
        G.append(x)
    return S, G



def acquireX(student, question, YITA, S, G):
    X = [[0] * question for i in range(student)]
    Xscore = [[0] * question for i in range(student)]
    for i in range(student):
        for j in range(question):
            x = pow(G[j], (1 - YITA[i][j])) * pow((1 - S[j]), YITA[i][j])
            Xscore[i][j] = x

            if x >= 0.5341:
                X[i][j] = 1
            else:
                X[i][j] = 0
    return X, Xscore



def hammingDis(invalid_ind):
    Matrix = []
    for i in invalid_ind:
        sonMatrix = []
        for j in invalid_ind:
            dis = sum([ch1 != ch2 for ch1, ch2 in zip(i, j)])
            sonMatrix.append(dis)
        Matrix.append(sonMatrix)
    return Matrix


def distance(invalid_ind1,invalid_ind2):
    dis = sum([ch1 !=ch2 for ch1,ch2 in zip(invalid_ind1,invalid_ind2)])
    return dis



def getMultiPopList(invalid_ind, disMatrix, GENE_LENGTH):

    fitnessesList = [ind.fitness.values[0] for ind in invalid_ind]
    indDict = dict(zip(range(len(invalid_ind)), fitnessesList))
    indDict = dict(sorted(indDict.items(), key=lambda x: x[1], reverse=True))

    sortInd = [i for i in indDict]


    g = igraph.Graph(directed=True)
    g.add_vertices(sortInd)
    g.vs['label'] = sortInd
    g.es['weight'] = 1.0


    index = 0
    weightEdgesDict = {}

    for i in sortInd[1:]:
        newsortInd = sortInd[index + 1:]

        idisListTemp = disMatrix[i]

        for j in newsortInd:
            idisListTemp[j] = GENE_LENGTH + 1   #601

        minDisIndex = idisListTemp.index(min(idisListTemp))
        minDis = idisListTemp[minDisIndex]

        nodeIdSource = sortInd.index(i)
        nodeIdTarget = sortInd.index(minDisIndex)

        g.add_edge(nodeIdSource, nodeIdTarget)
        if (minDis == 0):
            g[nodeIdSource, nodeIdTarget] = 1
        else:
            g[nodeIdSource, nodeIdTarget] = minDis

        weightEdgesDict[(i, minDisIndex)] = minDis

        index += 1

    meanDis = sum(g.es['weight']) / len(g.es['weight'])

    numbers = g.indegree()

    neighbors = dict(zip(g.vs['label'], numbers))

    for node in weightEdgesDict:

        ni = neighbors[node[0]]

        nodeIdSource = sortInd.index(node[0])
        nodeIdTarget = sortInd.index(node[1])
        if (weightEdgesDict[node] > meanDis and ni > 2):

            g.delete_edges((nodeIdSource, nodeIdTarget))


    newg = g.as_undirected().decompose()
    return newg


def computeTime():
    now = time.time()
    local_time = time.localtime(now)
    date_format_localtime = time.strftime('%Y-%m-%d %H:%M:%S', local_time)
    return date_format_localtime


def decode_slip(x, len_s_g=7):

    min_x = 0.0
    num_points = pow(2, len_s_g)
    n = (x - min_x) * (num_points - 1) / threshold_slip
    if(math.isnan(n)):
        print('here a nan')

    num_int = int(n)
    tmp_str = '{:0' + str(len_s_g) + 'b}'
    numbers = tmp_str.format(num_int)
    n = list(map(int, numbers))
    return n


def decode_guess(x, len_s_g=7):
    min_x = 0.0
    num_points = pow(2, len_s_g)
    n = (x - min_x) * (num_points - 1) / threshold_guess
    num_int = int(n)
    tmp_str = '{:0' + str(len_s_g) + 'b}'
    numbers = tmp_str.format(num_int)
    n = list(map(int, numbers))
    return n


def txt2csv(file):
    txt = np.loadtxt(file+".txt")
    print(txt.shape)
    txtDF = pd.DataFrame(txt)
    txtDF.to_csv(file+".csv", index=False)
