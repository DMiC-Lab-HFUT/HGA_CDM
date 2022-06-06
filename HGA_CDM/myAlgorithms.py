import copy
import math
import random
from queue import Queue
import pandas as pd
from deap import base, creator, tools
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
from sklearn.metrics import roc_auc_score

from tools import *
import os




def HGA_CDM_IALS(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, data, q_matrix, slip, guess, data_patch_id,
           run_id, n_pop,
           flag_train, max_generations, len_s_g, alg_name='GA_NBC', data_name='Math_DMiC'):

    n_knowledge = n_knowledge_fine
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    GENE_LENGTH = n_students * n_knowledge + n_questions * 2 * len_s_g
    toolbox = base.Toolbox()
    toolbox.register('Binary', bernoulli.rvs, 0.5)
    toolbox.register('Individual', tools.initRepeat, creator.Individual, toolbox.Binary, n=GENE_LENGTH)
    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)
    pop = toolbox.Population(n=n_pop)  # Generate initialized populations

    def evaluate(individual):  # Define the evaluation function
        A = acquireA(individual, n_students, n_knowledge)  # Knowledge mastery matrix from chromosomes
        YITA = acquireYITA(A, q_matrix, n_students, n_questions, n_knowledge)  # Obtain the matrix YITA of students' answers without considering S and G
        # slip_ratio = slip
        # guess_ratio = guess
        s, g = acquireSandG(n_students, individual, n_knowledge, GENE_LENGTH, len_s_g)
        slip_ratio = s
        guess_ratio = g
        X, Xscore = acquireX(n_students, n_questions, YITA, slip_ratio, guess_ratio)

        sum_ = 0
        for i in range(n_students):
            for j in range(n_questions):
                sum_ = sum_ + (1 - abs(data[i][j] - X[i][j]))  # Obtain the fitness value

        return (sum_),

    toolbox.register('evaluate', evaluate)
    toolbox.register('select', tools.selTournament, tournsize=2)  # Tournament Selection
    toolbox.register('mate', tools.cxUniform, indpb=0.5)  # Uniform crossover
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.02)  # 1-bit-flip mutation

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)  # Data to be recorded during the registration calculation
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    tempTotalPop = []
    LSmax = 100  # Maximum number of LS
    Nls = 0
    LSMcm = []  # Memory
    LSMdis = [] # Radius of Influence
    LSM_bestFitness_pre = []  # pre-local-search solutions
    LSM_bestFitness_post = []  # post-local-search solutions
    for gen in range(1, max_generations + 1):
        if (gen == 1):
            totalPop = invalid_ind
        else:
            totalPop = tempTotalPop
            tempTotalPop = []
        disMatrix = hammingDis(totalPop)
        try:
            newg = getMultiPopList(totalPop, disMatrix, GENE_LENGTH)
        except:
            pass
        Popth = 0
        for PopNodeList in newg:
            local_flag = True
            Popth += 1
            subPop = []

            for nodeId in PopNodeList.vs['label']:
                try:
                    subPop.append(totalPop[nodeId])
                except :
                    pass

            cxpb = 0.4329  # crossover probability (Pc)
            mutpb = 0.09351  # mutation probability (Pm)

            N_subPop = len(subPop)
            ind_seed = subPop[0]
            seed_LS = toolbox.clone(ind_seed)
            seed_LS_tem = toolbox.clone(ind_seed)
            if alg_name == 'GA_NBC' :
                PMatrix = hammingDis(subPop)
                PRadius_tem = []
                for i in PMatrix:
                    PRadius_tem.append(max(i))
                PRadius = max(PRadius_tem)
                if Popth == 1 and gen==1:
                    local_flag = True
                else:
                    if Nls > LSmax:
                        local_flag = False
                    elif int(seed_LS.fitness.values[0]) < max(LSM_bestFitness_pre):
                        local_flag = False
                    elif int(seed_LS.fitness.values[0]) > max(LSM_bestFitness_post):
                        local_flag = True
                    else:
                        for  i   in LSMcm:
                            if distance(seed_LS,i) < LSMdis[LSMcm.index(i)] :
                                local_flag = False

                if local_flag and Nls < LSmax:
                    Nls = Nls + 1
                    LSMcm.append(seed_LS_tem)
                    seed_LS = local_search_train_0(data, seed_LS, q_matrix, n_students, n_knowledge_fine,
                                                           n_questions, GENE_LENGTH, len_s_g)

                    seed_LS.fitness.values = evaluate(seed_LS)

                    tem_distance = distance(seed_LS_tem, seed_LS)
                    tem = max(tem_distance, PRadius)
                    LSMdis.append(tem)
                    LSMcm.append(seed_LS)
                    LSMdis.append(tem)
                    if len(LSM_bestFitness_post) == 0:
                        LSM_bestFitness_post.append(int(seed_LS.fitness.values[0]))
                    elif max(LSM_bestFitness_post) < int(seed_LS.fitness.values[0]):
                        LSM_bestFitness_post.append(int(seed_LS.fitness.values[0]))
                    if len(LSM_bestFitness_pre) == 0:
                        LSM_bestFitness_pre.append(int(seed_LS_tem.fitness.values[0]))
                    elif max(LSM_bestFitness_pre) < int(seed_LS_tem.fitness.values[0]):
                        LSM_bestFitness_pre.append(int(seed_LS_tem.fitness.values[0]))

            offspring = toolbox.select(subPop, N_subPop)
            offspring_Xor = []
            for i in range(N_subPop):
                offspring_Xor.append(copy.deepcopy(offspring[i]))
                offspring_Xor.append(copy.deepcopy(seed_LS))

            for child1, child2 in zip(offspring_Xor[::2], offspring_Xor[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            invalid_ind = [ind for ind in offspring_Xor if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            offspring_mut = [toolbox.clone(ind) for ind in offspring]
            for mutant in offspring_mut:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values


            invalid_ind = [ind for ind in offspring_mut if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            offspring = offspring_Xor + offspring_mut + subPop
            offspring.append(seed_LS)

            offspring = tools.selBest(offspring, len(offspring), fit_attr='fitness')
            pop_selected = []
            pop_selected.append(offspring[0])
            num_selected = 1
            for i in range(1, len(offspring)):
                idx = pop_selected[num_selected - 1]
                dis = sum([ch1 != ch2 for ch1, ch2 in zip(offspring[i], idx)])
                if dis >= 5:
                    pop_selected.append(offspring[i])
                    num_selected = num_selected + 1
            pop_selected = tools.selBest(pop_selected, N_subPop, fit_attr='fitness')

            tempTotalPop.extend(pop_selected)


        record = stats.compile(tempTotalPop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)


    resultPop = tempTotalPop

    def decode(resultPopx):
        if flag_train:
            resultS, resultG = acquireSandG(n_students, resultPopx, n_knowledge, GENE_LENGTH, len_s_g)  # func79行
            A = acquireA(resultPopx, n_students, n_knowledge)

            return resultS, resultG
        else:
            A = acquireA(resultPopx, n_students, n_knowledge)
            return A


    index = np.argmax([ind.fitness for ind in resultPop])
    if flag_train:
        slip, guess = decode(resultPop[index])

    fit_max = resultPop[index].fitness
    fit_max = fit_max.values[0]
    Accuracy = fit_max / (n_students * n_questions)

    gen = logbook.select('gen')
    fit_maxs = logbook.select('max')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(gen[1:], fit_maxs[1:], 'b-', linewidth=2.0, label='Max Fitness')
    ax.legend(loc='best')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    fig.tight_layout()

    if not flag_train:
        path_pre = '_results/pict/test/MaxFitness_perGen_alg'
    else:
        path_pre = '_results/pict/train/MaxFitness_perGen_alg'
    os.makedirs(path_pre, exist_ok=True)
    fig.savefig(path_pre + alg_name + '_data：' + data_name + ' total' + str(max_generations) + 'geb：   NO.' + str(
        data_patch_id) + 'patch' + 'NO.' + str(run_id) + 'run' + '.png')
    return resultPop, logbook, slip, guess


def HGA_CDM_ALS(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, data, q_matrix, slip, guess, data_patch_id,
           run_id, n_pop,
           flag_train, max_generations, len_s_g, alg_name='GA_NBC', data_name='Math_DMiC'):

    n_knowledge = n_knowledge_fine

    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    if flag_train:
        GENE_LENGTH = n_students * n_knowledge + n_questions * 2 * len_s_g
    else:
        GENE_LENGTH = n_students * n_knowledge

    toolbox = base.Toolbox()
    toolbox.register('Binary', bernoulli.rvs, 0.5)

    toolbox.register('Individual', tools.initRepeat, creator.Individual, toolbox.Binary, n=GENE_LENGTH)

    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)

    pop = toolbox.Population(n=n_pop)

    def evaluate(individual):
        A = acquireA(individual, n_students, n_knowledge)

        YITA = acquireYITA(A, q_matrix, n_students, n_questions, n_knowledge)
        s, g = acquireSandG(n_students, individual, n_knowledge, GENE_LENGTH, len_s_g)

        slip_ratio = s
        guess_ratio = g

        X, Xscore = acquireX(n_students, n_questions, YITA, slip_ratio, guess_ratio)
        sum_ = 0
        for i in range(n_students):
            for j in range(n_questions):
                sum_ = sum_ + (1 - abs(data[i][j] - X[i][j]))

        return (sum_),


    toolbox.register('evaluate', evaluate)

    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('mate', tools.cxUniform, indpb=0.5)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.5)

    #
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)  #
    stats.register("std", np.std)  #
    stats.register("min", np.min)  #
    stats.register("max", np.max)  #

    #
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields


    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)


    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    tempTotalPop = []
    LSmax = 100
    LS = 0


    for gen in range(1, max_generations + 1):
        LSMcm = []
        LSMdis = []

        if (gen == 1):
            totalPop = invalid_ind
        else:
            totalPop = tempTotalPop
            tempTotalPop = []
        disMatrix = hammingDis(totalPop)

        try:
            newg = getMultiPopList(totalPop, disMatrix, GENE_LENGTH)
        except:
            pass

        Popth = 0

        for PopNodeList in newg:
            local_flag = True
            Popth += 1

            subPop = []
            for nodeId in PopNodeList.vs['label']:
                try:
                    subPop.append(totalPop[nodeId])
                except:
                    pass

            cxpb = 0.4329
            mutpb = 0.09351

            N_subPop = len(subPop)
            ind_seed = subPop[0]  #

            seed_LS = toolbox.clone(ind_seed)
            seed_LS_tem = toolbox.clone(ind_seed)
            if Popth == 1:
                local_flag = True

            else:
                for i in LSMcm:
                    for j in LSMdis:
                        if distance(seed_LS, i) < j:
                            local_flag = False
            if alg_name == 'GA_NBC_multi' or alg_name == 'GA_NBC':
                if LS < LSmax and local_flag:
                    LS = LS + 1
                    LSMcm.append(seed_LS_tem)

                    seed_LS = local_search_train_0(data, seed_LS, q_matrix, n_students, n_knowledge_fine,
                                                           n_questions, GENE_LENGTH, len_s_g)

                    seed_LS.fitness.values = evaluate(seed_LS)
                    tem_distance = distance(seed_LS_tem, seed_LS)

                    LSMdis.append(tem_distance)
                    LSMcm.append(seed_LS)
                    LSMdis.append(tem_distance)


            offspring = toolbox.select(subPop, N_subPop)
            offspring_Xor = []  # [toolbox.clone(ind) for ind in offspring]
            for i in range(N_subPop):
                offspring_Xor.append(copy.deepcopy(offspring[i]))
                offspring_Xor.append(copy.deepcopy(seed_LS))

            for child1, child2 in zip(offspring_Xor[::2], offspring_Xor[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            invalid_ind = [ind for ind in offspring_Xor if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            offspring_mut = [toolbox.clone(ind) for ind in offspring]
            for mutant in offspring_mut:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring_mut if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            offspring = offspring_Xor + offspring_mut + subPop
            offspring.append(seed_LS)

            offspring = tools.selBest(offspring, len(offspring), fit_attr='fitness')
            pop_selected = []
            pop_selected.append(offspring[0])
            num_selected = 1

            for i in range(1, len(offspring)):
                idx = pop_selected[num_selected - 1]
                dis = sum([ch1 != ch2 for ch1, ch2 in zip(offspring[i], idx)])
                if dis >= 5:
                    pop_selected.append(offspring[i])
                    num_selected = num_selected + 1
            pop_selected = tools.selBest(pop_selected, N_subPop, fit_attr='fitness')

            tempTotalPop.extend(pop_selected)

        record = stats.compile(tempTotalPop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)



    resultPop = tempTotalPop


    def decode(resultPopx):
        if flag_train:
            resultS, resultG = acquireSandG(n_students, resultPopx, n_knowledge, GENE_LENGTH, len_s_g)

            return resultS, resultG
        else:
            A = acquireA(resultPopx, n_students, n_knowledge)
            return A


    index = np.argmax([ind.fitness for ind in resultPop])
    if flag_train:
        slip, guess = decode(resultPop[index])


    fit_max = resultPop[index].fitness
    fit_max = fit_max.values[0]



    gen = logbook.select('gen')
    fit_maxs = logbook.select('max')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(gen[1:], fit_maxs[1:], 'b-', linewidth=2.0, label='Max Fitness')
    ax.legend(loc='best')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')

    fig.tight_layout()

    if not flag_train:
        path_pre = '_results/pict/test/MaxFitness_perGen_alg'
    else:
        path_pre = '_results/pict/train/MaxFitness_perGen_alg'
    os.makedirs(path_pre, exist_ok=True)

    fig.savefig(path_pre + alg_name + '_data：' + data_name + ' total' + str(max_generations) + 'geb：   NO.' + str(
        data_patch_id) + 'patch' + 'NO.' + str(run_id) + 'run' + '.png')
    return resultPop, logbook, slip, guess

def HGA_CDM_LS(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, data, q_matrix, slip, guess, data_patch_id,
           run_id, n_pop,
           flag_train, max_generations, len_s_g, alg_name='GA_NBC', data_name='Math_DMiC'):

    n_knowledge = n_knowledge_fine
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    if flag_train:
        GENE_LENGTH = n_students * n_knowledge + n_questions * 2 * len_s_g
    else:
        GENE_LENGTH = n_students * n_knowledge

    toolbox = base.Toolbox()
    toolbox.register('Binary', bernoulli.rvs, 0.5)
    toolbox.register('Individual', tools.initRepeat, creator.Individual, toolbox.Binary, n=GENE_LENGTH)
    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)

    pop = toolbox.Population(n=n_pop)
    def evaluate(individual):
        A = acquireA(individual, n_students, n_knowledge)
        YITA = acquireYITA(A, q_matrix, n_students, n_questions, n_knowledge)
        s, g = acquireSandG(n_students, individual, n_knowledge, GENE_LENGTH, len_s_g)
        slip_ratio = s
        guess_ratio = g
        X, Xscore = acquireX(n_students, n_questions, YITA, slip_ratio, guess_ratio)

        sum_ = 0
        for i in range(n_students):
            for j in range(n_questions):
                sum_ = sum_ + (1 - abs(data[i][j] - X[i][j]))

        return (sum_),

    toolbox.register('evaluate', evaluate)
    toolbox.register('select', tools.selTournament, tournsize=2)
    toolbox.register('mate', tools.cxUniform, indpb=0.5)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.02)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)  #
    stats.register("max", np.max)  #


    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields


    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    tempTotalPop = []
    Nls = 0

    for gen in range(1, max_generations + 1):


        if (gen == 1):
            totalPop = invalid_ind
        else:
            totalPop = tempTotalPop
            tempTotalPop = []
        disMatrix = hammingDis(totalPop)
        try:
            newg = getMultiPopList(totalPop, disMatrix, GENE_LENGTH)
        except:
            pass

        Popth = 0
        for PopNodeList in newg:
            Popth += 1
            subPop = []

            for nodeId in PopNodeList.vs['label']:
                try:
                    subPop.append(totalPop[nodeId])
                except :
                    pass

            cxpb = 0.4329
            mutpb = 0.09351

            N_subPop = len(subPop)
            ind_seed = subPop[0]

            seed_LS = toolbox.clone(ind_seed)
            if alg_name == 'HGA_CDM_LS':

                        seed_LS = local_search_train_0(data, seed_LS, q_matrix, n_students, n_knowledge_fine,
                                                       n_questions, GENE_LENGTH, len_s_g)

                        seed_LS.fitness.values = evaluate(seed_LS)


            offspring = toolbox.select(subPop, N_subPop)
            offspring_Xor = []
            for i in range(N_subPop):
                offspring_Xor.append(copy.deepcopy(offspring[i]))
                offspring_Xor.append(copy.deepcopy(seed_LS))


            for child1, child2 in zip(offspring_Xor[::2], offspring_Xor[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values


            invalid_ind = [ind for ind in offspring_Xor if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit


            offspring_mut = [toolbox.clone(ind) for ind in offspring]
            for mutant in offspring_mut:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            invalid_ind = [ind for ind in offspring_mut if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            offspring = offspring_Xor + offspring_mut + subPop
            offspring.append(seed_LS)

            offspring = tools.selBest(offspring, len(offspring), fit_attr='fitness')

            pop_selected = []
            pop_selected.append(offspring[0])
            num_selected = 1

            for i in range(1, len(offspring)):
                idx = pop_selected[num_selected - 1]
                dis = sum([ch1 != ch2 for ch1, ch2 in zip(offspring[i], idx)])
                if dis >= 5:
                    pop_selected.append(offspring[i])
                    num_selected = num_selected + 1
            pop_selected = tools.selBest(pop_selected, N_subPop, fit_attr='fitness')


            tempTotalPop.extend(pop_selected)


        record = stats.compile(tempTotalPop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)

    with open('./_results/ALS/als.txt', 'a') as f:
        f.write(str(Nls) + '\n')
    print("第%s轮局部搜索次数为： %s" % (run_id, Nls))
    resultPop = tempTotalPop

    def decode(resultPopx):
        if flag_train:
            resultS, resultG = acquireSandG(n_students, resultPopx, n_knowledge, GENE_LENGTH, len_s_g)
            A = acquireA(resultPopx, n_students, n_knowledge)
            return resultS, resultG
        else:
            A = acquireA(resultPopx, n_students, n_knowledge)
            return A


    index = np.argmax([ind.fitness for ind in resultPop])
    if flag_train:
        slip, guess = decode(resultPop[index])

    fit_max = resultPop[index].fitness
    fit_max = fit_max.values[0]
    gen = logbook.select('gen')
    fit_maxs = logbook.select('max')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(gen[1:], fit_maxs[1:], 'b-', linewidth=2.0, label='Max Fitness')
    ax.legend(loc='best')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    fig.tight_layout()

    if not flag_train:
        path_pre = '_results/pict/test/MaxFitness_perGen_alg'
    else:
        path_pre = '_results/pict/train/MaxFitness_perGen_alg'
    os.makedirs(path_pre, exist_ok=True)

    fig.savefig(path_pre + alg_name + '_data：' + data_name + ' total' + str(max_generations) + 'geb：   NO.' + str(
        data_patch_id) + 'patch' + 'NO.' + str(run_id) + 'run' + '.png')
    return resultPop, logbook, slip, guess

def GA_CDM(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, data, q_matrix, slip, guess, data_patch_id,
           run_id, n_pop,
           flag_train, max_generations, len_s_g, alg_name='GA_NBC', data_name='Math_DMiC'):

    n_knowledge = n_knowledge_fine
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    if flag_train:
        GENE_LENGTH = n_students * n_knowledge + n_questions * 2 * len_s_g

    else:
        GENE_LENGTH = n_students * n_knowledge

    toolbox = base.Toolbox()
    toolbox.register('Binary', bernoulli.rvs, 0.5)

    toolbox.register('Individual', tools.initRepeat, creator.Individual, toolbox.Binary, n=GENE_LENGTH)

    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)


    pop = toolbox.Population(n=n_pop)


    def evaluate(individual):
        A = acquireA(individual, n_students, n_knowledge)


        YITA = acquireYITA(A, q_matrix, n_students, n_questions, n_knowledge)
        slip_ratio = slip
        guess_ratio = guess
        if flag_train:

            s, g = acquireSandG(n_students, individual, n_knowledge, GENE_LENGTH, len_s_g)

            slip_ratio = s
            guess_ratio = g

        X, Xscore = acquireX(n_students, n_questions, YITA, slip_ratio, guess_ratio)

        sum_ = 0
        for i in range(n_students):
            for j in range(n_questions):
                sum_ = sum_ + (1 - abs(data[i][j] - X[i][j]))

        return (sum_),

    toolbox.register('evaluate', evaluate)

    toolbox.register('select', tools.selTournament, tournsize=2)

    toolbox.register('mate', tools.cxUniform, indpb=0.5)
    toolbox.register('mutate', tools.mutFlipBit, indpb=0.02)


    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)


    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields


    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)


    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit


    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    tempTotalPop = []
    for gen in range(1, max_generations + 1):


        if (gen == 1):
            totalPop = invalid_ind
        else:
            totalPop = tempTotalPop
            tempTotalPop = []
        disMatrix = hammingDis(totalPop)
        try:
            newg = getMultiPopList(totalPop, disMatrix, GENE_LENGTH)
        except:
            pass

        Popth = 0
        for PopNodeList in newg:

            Popth += 1

            subPop = []

            for nodeId in PopNodeList.vs['label']:
                try:
                    subPop.append(totalPop[nodeId])
                except :
                    pass

            cxpb = 0.4329

            mutpb = 0.09351

            N_subPop = len(subPop)
            ind_seed = subPop[0]  #


            seed_LS = toolbox.clone(ind_seed)

            offspring = toolbox.select(subPop, N_subPop)
            offspring_Xor = []
            for i in range(N_subPop):
                offspring_Xor.append(copy.deepcopy(offspring[i]))
                offspring_Xor.append(copy.deepcopy(seed_LS))


            for child1, child2 in zip(offspring_Xor[::2], offspring_Xor[1::2]):
                if random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values


            invalid_ind = [ind for ind in offspring_Xor if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit


            offspring_mut = [toolbox.clone(ind) for ind in offspring]
            for mutant in offspring_mut:
                if random.random() < mutpb:
                    toolbox.mutate(mutant)  #
                    del mutant.fitness.values


            invalid_ind = [ind for ind in offspring_mut if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            offspring = offspring_Xor + offspring_mut + subPop
            offspring.append(seed_LS)

            offspring = tools.selBest(offspring, len(offspring), fit_attr='fitness')

            pop_selected = []
            pop_selected.append(offspring[0])
            num_selected = 1

            for i in range(1, len(offspring)):
                idx = pop_selected[num_selected - 1]
                dis = sum([ch1 != ch2 for ch1, ch2 in zip(offspring[i], idx)])
                if dis >= 5:
                    pop_selected.append(offspring[i])
                    num_selected = num_selected + 1
            pop_selected = tools.selBest(pop_selected, N_subPop, fit_attr='fitness')


            tempTotalPop.extend(pop_selected)

        record = stats.compile(tempTotalPop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)


    resultPop = tempTotalPop


    def decode(resultPopx):
        if flag_train:
            resultS, resultG = acquireSandG(n_students, resultPopx, n_knowledge, GENE_LENGTH, len_s_g)
            return resultS, resultG
        else:
            A = acquireA(resultPopx, n_students, n_knowledge)
            return A


    index = np.argmax([ind.fitness for ind in resultPop])
    if flag_train:
        slip, guess = decode(resultPop[index])

    fit_max = resultPop[index].fitness
    fit_max = fit_max.values[0]


    gen = logbook.select('gen')
    fit_maxs = logbook.select('max')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(gen[1:], fit_maxs[1:], 'b-', linewidth=2.0, label='Max Fitness')
    ax.legend(loc='best')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    fig.tight_layout()

    if not flag_train:
        path_pre = '_results/pict/test/MaxFitness_perGen_alg'
    else:
        path_pre = '_results/pict/train/MaxFitness_perGen_alg'
    os.makedirs(path_pre, exist_ok=True)
    fig.savefig(path_pre + alg_name + '_data：' + data_name + ' sum' + str(max_generations) + 'gen_iter：   No' + str(
        data_patch_id) + 'patch' + 'NO.' + str(run_id) + 'train' + '.png')
    return resultPop, logbook, slip, guess


def local_search_train_0(data, individual, q_matrix, n_students, n_knowledge, n_questions, GENE_LENGTH, len_s_g):
    slip, guess = acquireSandG(n_students, individual, n_knowledge, GENE_LENGTH, len_s_g)
    A, IL, k_matrix, r_matrix = EStep(slip, guess, data, q_matrix, n_knowledge, n_students)
    slip, guess = MStep(IL, r_matrix, data, n_knowledge, n_students, n_questions)
    individual = updateIndividual_A_0(individual, A, n_students, n_knowledge)  # 656行


    if not math.isnan(slip[0]) or not math.isnan(guess[0]):
        individual = updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge, GENE_LENGTH,
                                          len_s_g)
    return individual




def updateIndividual_A_0(individual, A, n_students, n_knowledge):
    individual[0:n_students * n_knowledge] = A
    return individual


def updateIndividual_A(individual, A, n_students, n_knowledge_coarse, n_knowledge_fine, groups):

    for j in range(n_students):
        for i in range(n_knowledge_coarse):
            for idx in groups[i]:
                individual[j * n_knowledge_fine + idx] = A[j * n_knowledge_coarse + i]
    return individual


def updateIndividual_s_g(individual, slip, guess, n_students, n_knowledge, GENE_LENGTH, len_s_g):

    loop = 0
    for i in range(n_knowledge * n_students, GENE_LENGTH, len_s_g * 2):

        if not math.isnan(slip[loop]) and not math.isnan(guess[loop]):
            individual[i:i + len_s_g] = decode_slip(slip[loop], len_s_g)
            individual[i + len_s_g:i + (len_s_g * 2)] = decode_guess(guess[loop], len_s_g)
        loop = loop + 1

    return individual





def EStep(slip, guess, data, q_matrix, n_knowledge, n_students):
    slip = np.array(slip)
    guess = np.array(guess)
    data = np.array(data)
    q_matrix = np.mat(q_matrix)
    # crate K matrix，indict k skill could get how many vector

    k_matrix = np.mat(np.zeros((n_knowledge, 2 ** n_knowledge), dtype=int))
    for j in range(2 ** n_knowledge):
        l = list(bin(j).replace('0b', ''))
        for i in range(len(l)):
            k_matrix[n_knowledge - len(l) + i, j] = l[i]

    aPrior = np.ones(2 ** n_knowledge) / 10 ** 8
    std = np.sum(q_matrix, axis=1)

    r_matrix = (q_matrix * k_matrix == std) * 1
    IL = np.zeros((n_students, 2 ** n_knowledge))
    for s in range(0,3):
        for l in range(2 ** n_knowledge):

            lll = ((1 - slip) ** data * slip ** (1 - data)) ** r_matrix.T.A[l] * (guess ** data * (
                1 - guess) ** (1 - data)) ** (1 - r_matrix.T.A[l])
            IL[:, l] = lll.prod(axis=1)
        sumIL = IL.sum(axis=1)
        IL = (IL.T / sumIL).T* aPrior


    A = []
    for i in range(n_students):
        idx = IL[i].argmax()
        tmp = k_matrix[:, idx].data.tolist()
        tmp_array = [i[0] for i in tmp]
        A = A + tmp_array

    return A, IL, k_matrix, r_matrix



def MStep(IL, r_matrix, data, n_knowledge, n_students, n_questions):
    data = np.array(data)

    IR = np.zeros((4, n_questions))
    n1 = np.ones((n_students, n_questions))
    for l in range(2 ** n_knowledge):
        IR[0] += np.sum(((1 - r_matrix.A[:, l]) * n1).T * IL[:, l], axis=1)
        IR[1] += np.sum(((1 - r_matrix.A[:, l]) * data).T * IL[:, l], axis=1)
        IR[2] += np.sum((r_matrix.A[:, l] * n1).T * IL[:, l], axis=1)
        IR[3] += np.sum((r_matrix.A[:, l] * data).T * IL[:, l], axis=1)
    guess = IR[1] / IR[0]
    slip = (IR[2] - IR[3]) / IR[2]

    for i in range(n_questions):
        if slip[i] > threshold_slip:
            slip[i] = threshold_slip
        if guess[i] > threshold_guess:
            guess[i] = threshold_guess


    return slip, guess
