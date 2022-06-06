import os
import random

import pandas as pd
from func import *


def saveC(array, path):
    data = pd.DataFrame(array)
    data.to_csv(path, index=False)


def data_reader(data_name):
    q_path = "dataSets/" + data_name + "/q.csv"

    data_path = "dataSets/" + data_name + "/data.csv"

    if os.path.exists(q_path) == False:
        q_path_pre = "dataSets/" + data_name + "/q"
        txt2csv(q_path_pre)

    if os.path.exists(data_path) == False:
        data_path_pre = "dataSets/" + data_name + "/data"
        txt2csv(data_path_pre)

    q_matrix = acquireQ(q_path)
    data = acquireData(data_path)

    data = np.array(data)

    return q_matrix, data

def files_open(alg, data_name, data_patch_i, runID):
    filename = "_results/Frusub_GA/" + alg + "_" + data_name + "_" + str(data_patch_i) + "_th_" + str(runID) + ".txt"
    f_record = open(filename, 'a')
    filename = "_results/Frusub_GA/" + alg + "_score_" + data_name + "_" + str(data_patch_i) + "_th_" + str(runID) + ".txt"
    f_record_data = open(filename, 'a')
    return f_record, f_record_data


def files_close(f_record, f_record_data):
    f_record.close()
    f_record_data.close()


def save_final_results(results_per_run, data_patch_id, mean_accuracy, mean_precision, mean_recall, mean_f1, mean_auc,
                       max_runs, alg_name,
                       data_name):
    filename = "_results/Frusub_GA/alg_" + alg_name + "_data_" + data_name + "_" + str(data_patch_id) + "_max_run" + str(max_runs) + "_.txt"
    print(filename)
    str_results = str(mean_accuracy) + "    " + str(mean_precision) + "  " + str(mean_recall) + "   " + str(
        mean_f1) + "       " + str(mean_auc)
    str_index = " mean_accuracy       mean_precision       mean_recall      mean_f1         mean_auc  "
    print("accuracy, precision, recall, f1, auc  " + str_results)
    f_record = open(filename, 'a')
    f_record.writelines(str_index + '\n')
    f_record.writelines(str_results + '\n')
    str_index_1 = " run_id   mean_accuracy        mean_precision       mean_recall      mean_f1         mean_auc  "
    f_record.writelines(str_index_1 + '\n')
    for i in range(max_runs):

        f_record.writelines(str(results_per_run[i]) + '\n')

    f_record.close()


def save_final_results_average(average_accuracy, average_precision, average_recall, average_f1, average_auc, max_runs,
                               alg_name, data_name):
    filename = "_results/Frusub_GA/_average_alg_" + alg_name + "_data_" + data_name + "_max_run ：" + str(max_runs) + "_.txt"
    print(filename)
    str_average = str(average_accuracy) + "      " + str(average_precision) + "      " + str(
        average_recall) + "     " + str(average_f1) + "       " + str(average_auc)
    str_index = " mean_accuracy           mean_precision               mean_recall            mean_f1        " \
                "      mean_auc  "
    print("accuracy, precision, recall, f1, auc  " + str_average)
    f_record = open(filename, 'a')
    f_record.writelines(str_index + '\n')
    f_record.writelines(str_average + '\n')
    f_record.close()

def saveC(array, path):
    data = pd.DataFrame(array)
    data.to_csv(path, index=False)

