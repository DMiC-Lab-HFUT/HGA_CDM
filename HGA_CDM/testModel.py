from myAlgorithms import *
from multiprocessing import Pool



def testModel(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g, data, q_matrix, slip, guess, data_patch_id, run_id=1,
              n_pop=50, max_generations=100, alg_name='GA_NBC', data_name='Math_DMiC'):

    multi = False

    Accuracy, Precision, Recall, F1, AUC = predictDINA(data, q_matrix, slip, guess, multi,data_patch_id, run_id=1)


    return Accuracy, Precision, Recall, F1, AUC


def predictDINA(data, q_matrix, slip, guess, multi,data_patch_id, run_id=1):

    startTime = time.time()
    slip = np.array(slip)
    guess = np.array(guess)
    data = np.array(data)
    q_matrix = np.mat(q_matrix)
    n_questions, n_knowledge = q_matrix.shape
    # crate K matrix，indict k skill could get how many vector
    k_matrix = np.mat(np.zeros((n_knowledge, 2 ** n_knowledge), dtype=int))
    for j in range(2 ** n_knowledge):
        l = list(bin(j).replace('0b', ''))
        for i in range(len(l)):
            k_matrix[n_knowledge - len(l) + i, j] = l[i]
    std = np.sum(q_matrix, axis=1)
    r_matrix = (q_matrix * k_matrix == std) * 1
    ni, nj = data.shape
    Qi, Qj = q_matrix.shape

    IL = np.zeros((ni, 2 ** Qj))
    k = Qj

    if multi == True:
        print(' multi 4 processes')
        with Pool(processes=4) as pool:
            multiple_results = [pool.apply_async(EStep_multi, (IL, slip, guess, data, r_matrix, k, i)) for i in range(4)]
            for item in ([res.get(timeout=1000) for res in multiple_results]):
                IL += item

    else:
        for l in range(2 ** Qj):

            lll = ((1 - slip) ** data * slip ** (1 - data)) ** r_matrix.T.A[l] * (guess ** data * (
                1 - guess) ** (1 - data)) ** (1 - r_matrix.T.A[l])
            IL[:, l] = lll.prod(axis=1)

    a = IL.argmax(axis=1)

    rrrSum = []
    nnnSum = []
    for rrr, nnn in zip(r_matrix[:, a], data.T):
        rrrSum.extend(rrr.tolist()[0])
        nnnSum.extend(nnn)

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import roc_auc_score
    print(len(nnnSum),nnnSum)
    print(len(rrrSum),rrrSum)

    C2 = confusion_matrix(nnnSum, rrrSum, labels=[0, 1])
    TP = C2[0][0]
    FP = C2[0][1]
    FN = C2[1][0]
    TN = C2[1][1]

    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    AUC = roc_auc_score(nnnSum, rrrSum)

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    print('The predicted model time is.：' + str(int(time.time()) - int(startTime)) + 's')
    print('------------------Prediction End-----------------------------')
    return Accuracy, Precision, Recall, F1, AUC




def EStep_multi(IL, slip, guess, n, r, k, i):
    base = 2**(k-2)
    for l in range(i*base,(i+1)*base):

        lll = ((1 - slip) ** n * slip ** (1 - n)) ** r.T.A[l] * (guess ** n * (
            1 - guess) ** (1 - n)) ** (1 - r.T.A[l])
        IL[:, l] = lll.prod(axis=1)
    return IL





def MStep_multi(IL,n,r,k,i):
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

