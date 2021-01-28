import numpy as np
from sklearn.datasets import load_svmlight_file


def getParamsByDataset(dataset):
    if dataset == "2003_td_dataset":
        _TRAIN_DATA_PATH = "D:\\Colecoes\\2003_td_dataset\\Fold1\\trainn.txt"
        _TEST_DATA_PATH = "D:\\Colecoes\\2003_td_dataset\\Fold1\\testn.txt"
        _VALI_DATA_PATH = "D:\\Colecoes\\2003_td_dataset\\Fold1\\valin.txt"
        _LIST_SIZE = 100
        _NUM_FEATURES = 64
        return (_TRAIN_DATA_PATH, _TEST_DATA_PATH, _VALI_DATA_PATH, _LIST_SIZE, _NUM_FEATURES)

# TODO todas as entradas devem ser formatadas , remover #DOC-id do fim de cada linha do arquivo lib_svm


def getData(path):
    data = load_svmlight_file(path, query_id=True)

    queries_id = np.array(data[2])

    ant = queries_id[0]
    all_trues = []
    true_relevance = []

    for i in range(queries_id.size):
        if ant != queries_id[i]:
            all_trues.append(true_relevance)
            true_relevance = []

        true_relevance.append(data[1][i])
        ant = queries_id[i]

    all_trues.append(true_relevance)

    # X == data[0]
    # y == data[1]
    # queries_id_array data == [2]
    # all_trues == y (cada linha em all_trues contém os y de uma mesma query)
    return data[0], data[1], data[2], all_trues
