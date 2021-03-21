import pickle
from random import sample
import torch
from torch import optim
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import datetime
import argparse
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, sampler

from preprocess.checkpoint import Checkpoint
from preprocess.utils import set_random_seed
from program_tasks.code_summary.CodeLoader import CodeLoader
from program_tasks.code_summary.Code2VecModule import Code2Vec


def my_collate(batch):
    x, y = zip(*batch)
    sts, paths, eds = [], [], []
    for data in x:
        st, path, ed = zip(*data)
        sts.append(torch.tensor(st, dtype=torch.int))
        paths.append(torch.tensor(path, dtype=torch.int))
        eds.append(torch.tensor(ed, dtype=torch.int))

    length = [len(i) for i in sts]
    sts = rnn_utils.pad_sequence(sts, batch_first=True, padding_value=1).long()
    eds = rnn_utils.pad_sequence(eds, batch_first=True, padding_value=1).long()
    paths = rnn_utils.pad_sequence(paths, batch_first=True, padding_value=1).long()
    return (sts, paths, eds), y, length


def dict2list(tk2index):
    res = {}
    for tk in tk2index:
        res[tk2index[tk]] = tk
    return res


def new_acc(pred, y, index2func):
    pred = pred.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    tp, fp, fn = 0, 0, 0
    acc = np.sum(pred == y)
    for i, pred_i in enumerate(pred):
        pred_i = set(index2func[pred_i].split('|'))
        y_i = set(index2func[y[i]].split('|'))
        tp += len(pred_i & y_i)
        fp += len(pred_i - y_i)
        fn = len(y_i - pred_i)
    return acc, tp, fp, fn


def perpare_train(tk_path, embed_type, vec_path, embed_dim, out_dir):
    tk2num = None
    with open(tk_path, 'rb') as f:
        token2index, path2index, func2index = pickle.load(f)
        embed = None
    if embed_type == 0: # pretrained embedding
        tk2num, embed = torch.load(vec_path)
        print('load existing embedding vectors, name is ', vec_path)
    elif embed_type == 1: # train with embedding updated
        tk2num = token2index
        print('create new embedding vectors, training from scratch')
    elif embed_type == 2: # train random embedding
        tk2num = token2index
        embed = torch.randn([len(token2index), embed_dim])
        print('create new embedding vectors, training the random vectors')
    else:
        raise ValueError('unsupported type')
    if embed is not None:
        if type(embed) is np.ndarray:
            embed = torch.tensor(embed, dtype=torch.float)
        assert embed.size()[1] == embed_dim
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    return token2index, path2index, func2index, embed, tk2num


def train_model(model, cur_epoch, train_loader, device,
                criterian, optimizer, index2func):
    model.train()
    acc, tp, fp, fn = 0, 0, 0, 0 
    total_samples = 0

    for i, ((sts, paths, eds), y, length) in tqdm(enumerate(train_loader)):
        sts = sts.to(device)
        paths = paths.to(device)
        eds = eds.to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        total_samples += sts.size(0)
        pred_y = model(sts, paths, eds, length, device)
        loss = criterian(pred_y, y)
        loss.backward()
        optimizer.step()
        pos, pred_y = torch.max(pred_y, 1)
        acc_add, tp_add, fp_add, fn_add = new_acc(pred_y, y, index2func)
        tp += tp_add
        fp += fp_add
        fn += fn_add
        acc += acc_add
    acc = acc / total_samples
    prec = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = prec * recall * 2 / (prec + recall + 1e-8) 

    res = {
        'epoch': cur_epoch, 
        'train acc:': acc, 
        'train p': prec, 
        'train r': recall,
        'train f1':f1,
    }
    print(res)
    return model


def test_model(val_loader, model, device, index2func, val_name):
    model.eval()
    acc, tp, fn, fp = 0, 0, 0, 0
    total_samples = 0

    for i, ((sts, paths, eds), y, length) in enumerate(val_loader):
        sts = sts.to(device)
        paths = paths.to(device)
        eds = eds.to(device)
        y = torch.tensor(y, dtype=torch.long).to(device)
        total_samples += sts.size(0)
        pred_y = model(sts, paths, eds, length, device)
        pos, pred_y = torch.max(pred_y, 1)
        acc_add, tp_add, fp_add, fn_add = new_acc(pred_y, y, index2func)
        tp += tp_add
        fp += fp_add
        fn += fn_add
        acc += acc_add
    acc = acc / total_samples
    prec = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = prec * recall * 2 / (prec + recall + 1e-8) 

    res = {
        f'{val_name} acc': acc, 
        f'{val_name} p': prec, 
        f'{val_name} r': recall,
        f'{val_name} f1':f1,
    }
    print(res)
    return res


def main(args_set):
    # parameters setting
    tk_path = args_set.tk_path
    train_path = args_set.train_data
    test_path1 = args_set.test_data1
    test_path2 = args_set.test_data2
    test_path3 = args_set.test_data3
    # test_path4 = args_set.test_data4
    embed_dim = args_set.embed_dim
    embed_type = args_set.embed_type
    vec_path = args_set.embed_path
    out_dir = args_set.res_dir
    experiment_name = args_set.experiment_name
    train_batch = args_set.batch
    epochs = args_set.epochs
    lr = args_set.lr
    weight_decay=args_set.weight_decay
    max_size = args_set.max_size
    load_ckpt = args_set.load_ckpt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load data and preparation
    token2index, path2index, func2index, embed, tk2num =\
        perpare_train(tk_path, embed_type, vec_path, embed_dim, out_dir)
    nodes_dim, paths_dim, output_dim = len(tk2num), len(path2index), len(func2index)
    index2func = dict2list(func2index)
    model = Code2Vec(nodes_dim, paths_dim, embed_dim, output_dim, embed) # modified!

    criterian = nn.CrossEntropyLoss()  # loss

    # load ckpt if necessary
    if load_ckpt:
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(out_dir)
        resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        model = resume_checkpoint.model
        optimizer = resume_checkpoint.optimizer
        start_epoch = resume_checkpoint.epoch
    else:
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        start_epoch = 1

    # build test loader
    # random sample max_size training samples
    st_time = datetime.datetime.now()
    train_dataset = CodeLoader(train_path, None, token2index, tk2num)
    # train_loader = DataLoader(train_dataset, batch_size=train_batch, collate_fn=my_collate)
    ed_time = datetime.datetime.now()
    print('train dataset size is ', len(train_dataset), 'cost time', ed_time - st_time)

    st_time = datetime.datetime.now()
    test_dataset1 = CodeLoader(test_path1, None, token2index, tk2num)
    # test_loader1 = DataLoader(test_dataset1, batch_size=train_batch, collate_fn=my_collate)
    ed_time = datetime.datetime.now()
    print('test dataset1 size is ', len(test_dataset1), 'cost time', ed_time - st_time)

    st_time = datetime.datetime.now()
    test_dataset2 = CodeLoader(test_path2, None, token2index, tk2num)
    # test_loader2 = DataLoader(test_dataset2, batch_size=train_batch, collate_fn=my_collate)
    ed_time = datetime.datetime.now()
    print('test dataset2 size is ', len(test_dataset2), 'cost time', ed_time - st_time)

    st_time = datetime.datetime.now()
    test_dataset3 = CodeLoader(test_path3, None, token2index, tk2num)
    # test_loader3 = DataLoader(test_dataset3, batch_size=train_batch, collate_fn=my_collate)
    ed_time = datetime.datetime.now()
    print('test dataset3 size is ', len(test_dataset3), 'cost time', ed_time - st_time)

    # st_time = datetime.datetime.now()
    # test_dataset4 = CodeLoader(test_path4, None, token2index, tk2num)
    # # test_loader4 = DataLoader(test_dataset4, batch_size=train_batch, collate_fn=my_collate)
    # ed_time = datetime.datetime.now()
    # print('test dataset4 size is ', len(test_dataset4), 'cost time', ed_time - st_time)

    # training
    print('begin training experiment {} ...'.format(experiment_name))

    model.to(device)
    best_val_acc = 0
    total_st_time = datetime.datetime.now()

    for epoch in range(start_epoch, epochs+1):
        # print('max size: {}'.format(max_size))
        idx = np.random.randint(0, max_size, max_size)
        train_sampler = sampler.SubsetRandomSampler(idx)
        train_loader = DataLoader(train_dataset, batch_size=train_batch, 
                                  collate_fn=my_collate, sampler=train_sampler)
        test_loader1 = DataLoader(test_dataset1, batch_size=train_batch, 
                                  collate_fn=my_collate, sampler=train_sampler)
        test_loader2 = DataLoader(test_dataset2, batch_size=train_batch, 
                                  collate_fn=my_collate, sampler=train_sampler)
        test_loader3 = DataLoader(test_dataset3, batch_size=train_batch, 
                                  collate_fn=my_collate, sampler=train_sampler)
        # test_loader4 = DataLoader(test_dataset4, batch_size=train_batch, 
        #                           collate_fn=my_collate, sampler=train_sampler)


        model = train_model(model, epoch, train_loader, device,
                            criterian, optimizer, index2func)
        res1 = test_model(test_loader1, model, device, index2func, 'test1')
        res2 = test_model(test_loader2, model, device, index2func, 'test2')
        res3 = test_model(test_loader3, model, device, index2func, 'test3')
        # res4 = test_model(test_loader4, model, device, index2func, 'test4')

        # save model checkpoint
        if res1['test1 acc'] > best_val_acc:
            Checkpoint(model, optimizer, epoch, res1).save(out_dir)
            best_val_acc = res1['test1 acc']

    total_ed_time = datetime.datetime.now()
    print('training experiment {} finished! Total cost time: {}'.format(
        experiment_name, total_ed_time - total_st_time
    ))




if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch', default=256, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', default=0.005, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--hidden-size', default=128, type=int, metavar='N', help='rnn hidden size')
    parser.add_argument('--layers', default=2, type=int, metavar='N', help='number of rnn layers')
    parser.add_argument('--classes', default=250, type=int, metavar='N', help='number of output classes')
    parser.add_argument('--min-samples', default=5, type=int, metavar='N', help='min number of tokens')
    parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
    parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
    parser.add_argument('--mean_seq', default=False, action='store_true', help='use mean of rnn output')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')

    parser.add_argument('--embed_dim', default=100, type=int, metavar='N', help='embedding size')
    parser.add_argument('--embed_path', type=str, default='vec/100_2/Doc2VecEmbedding0.vec')
    parser.add_argument('--train_data', type=str, default='data/java_pkl_files/train.pkl')
    parser.add_argument('--test_data1', type=str, default='data/java_pkl_files/test1.pkl')
    parser.add_argument('--test_data2', type=str, default='data/java_pkl_files/test2.pkl')
    parser.add_argument('--test_data3', type=str, default='data/java_pkl_files/test3.pkl')
    # parser.add_argument('--test_data4', type=str, default='data/java_pkl_files/test4.pkl')
    parser.add_argument('--tk_path', type=str, default='data/java_pkl_files/tk.pkl')
    parser.add_argument('--embed_type', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--experiment_name', type=str, default='code summary')
    parser.add_argument('--load_ckpt', default=False, action='store_true', help='load checkpoint')
    parser.add_argument('--res_dir', type=str, default='program_tasks/code_summary/result')
    parser.add_argument('--max_size', type=int, default=None, help='if not None, then use maxsize of the training data')

    args = parser.parse_args()
    options = vars(args)
    print(options)
    # set_random_seed(10)
    main(args)