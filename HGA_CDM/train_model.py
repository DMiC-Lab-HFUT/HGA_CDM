from myAlgorithms import *


def trainModel(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, len_s_g, data, q_matrix, data_patch_id, run_id=1, n_pop=50,
               max_generations=100, alg_name='GA_NBC', data_name='Math_DMiC'):

    flag_train = True
    slip = []
    guess = []



    if alg_name == "HGA_CDM_IALS":
        resultPop, logbook, slip, guess = HGA_CDM_IALS(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine, data,
                                                 q_matrix, slip, guess, data_patch_id, run_id, n_pop,
                                                 flag_train, max_generations, len_s_g, alg_name, data_name)
    elif alg_name == "HGA_CDM_ALS":
        resultPop, logbook, slip, guess = HGA_CDM_ALS(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine,
                                                       data,
                                                       q_matrix, slip, guess, data_patch_id, run_id, n_pop,
                                                       flag_train, max_generations, len_s_g, alg_name, data_name)
    elif alg_name == "HGA_CDM_LS":
        resultPop, logbook, slip, guess = HGA_CDM_LS(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine,
                                                      data,
                                                      q_matrix, slip, guess, data_patch_id, run_id, n_pop,
                                                      flag_train, max_generations, len_s_g, alg_name, data_name)
    elif alg_name == "GA_CDM":
        resultPop, logbook, slip, guess = GA_CDM(n_students, n_questions, n_knowledge_coarse, n_knowledge_fine,
                                                     data,
                                                     q_matrix, slip, guess, data_patch_id, run_id, n_pop,
                                                     flag_train, max_generations, len_s_g, alg_name, data_name)




    return slip, guess

