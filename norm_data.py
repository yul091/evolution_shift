import pickle
import json
import os
from preprocess.utils import BASEDICT


def build_dict(dataset):
    base_dict = BASEDICT.copy()
    token2index, path2index, func2index = base_dict.copy(), base_dict.copy(), base_dict.copy()
    for i, data in enumerate(dataset):
        data = data.strip().lower()
        target, code_context = data.split()[0], data.split()[1:]
        target = target.split('|')
        func_name = target[0]
        if func_name not in func2index:
            func2index[func_name] = len(func2index)
        for context in code_context:
            st, path, ed = context.split(',')
            if st not in token2index:
                token2index[st] = len(token2index)
            if ed not in token2index:
                token2index[ed] = len(token2index)
            if path not in path2index:
                path2index[path] = len(path2index)
    with open(DIR + '/' + 'tk.pkl', 'wb') as f:
        pickle.dump([token2index, path2index, func2index], f)
    print("finish dictionary build", len(token2index), len(path2index), len(func2index))


def tk2index(tk_dict, k):
    if k not in tk_dict:
        return tk_dict['____UNKNOW____']
    return tk_dict[k]


def norm_data(data_type):
    file_name = DATA_DIR + '/'+ FILENAME + '.' + data_type + '.c2v'
    with open(file_name, 'r') as f:
        dataset = f.readlines()
    with open(DIR + '/' + 'tk.pkl', 'rb') as f:
        token2index, path2index, func2index = pickle.load(f)
    newdataset = []
    for i, data in enumerate(dataset):
        data = data.strip().lower()
        target, code_context = data.split()[0], data.split()[1:]
        target = target.split('|')
        func_name = target[0]
        label = tk2index(func2index, func_name)
        newdata = []
        for context in code_context:
            st, path, ed = context.split(',')
            newdata.append(
                [tk2index(token2index, st), tk2index(path2index, path), tk2index(token2index, ed)]
            )
        newdataset.append([newdata, label])
    save_file = DIR + '/' + data_type + '.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(newdataset, f)
    print("finish normalize dataset", data_type)


def main():
    with open(DATA_DIR + '/' + FILENAME + '.train.c2v', 'r') as f:
        dataset = f.readlines()
        print('dataset number is ', len(dataset))
    if not os.path.isdir(DIR):
        os.mkdir(DIR)
    build_dict(dataset)
    norm_data('train')
    norm_data('test1')
    norm_data('test2')
    norm_data('test3')
    norm_data('test4')
    # norm_data('test')
    # norm_data('val')


if __name__ == '__main__':
    FILENAME = 'java_projects'
    GEN_FILE = 'java_project_files'
    DIR = os.path.join('data', GEN_FILE)
    DATA_DIR = os.path.join('data', FILENAME)
    main()
