from __future__ import print_function

import gc
import os
import argparse
import datetime
import numpy as np
import joblib
import torch
from torch import nn
# import torch.backends.cudnn as cudnn
from gensim.models import word2vec

from preprocess.checkpoint import Checkpoint
from program_tasks.code_completion.vocab import VocabBuilder
from program_tasks.code_completion.dataloader import Word2vecLoader
from program_tasks.code_completion.util import AverageMeter, accuracy
from program_tasks.code_completion.util import adjust_learning_rate
from program_tasks.code_completion.model import Word2vecPredict


def preprocess_data():
    print("===> creating vocabs ...")
    train_path = args.train_data
    val_path = args.val_data
    test_path1 = args.test_data1
    test_path2 = args.test_data2
    test_path3 = args.test_data3

    pre_embedding_path = args.embedding_path
    if args.embedding_type == 0:
        d_word_index, embed = torch.load(pre_embedding_path)
        print('load existing embedding vectors, name is ', pre_embedding_path)
    elif args.embedding_type == 1:
        v_builder = VocabBuilder(path_file=train_path)
        d_word_index, embed = v_builder.get_word_index(min_sample=args.min_samples)
        print('create new embedding vectors, training from scratch')
    elif args.embedding_type == 2:
        v_builder = VocabBuilder(path_file=train_path)
        d_word_index, embed = v_builder.get_word_index(min_sample=args.min_samples)
        embed = torch.randn([len(d_word_index), args.embedding_dim]).cuda()
        print('create new embedding vectors, training the random vectors')
    else:
        raise ValueError('unsupported type')

    if embed is not None:
        if type(embed) is np.ndarray:
            embed = torch.tensor(embed, dtype=torch.float).cuda()
        assert embed.size()[1] == args.embedding_dim

    if not os.path.exists('program_tasks/code_completion/result'):
        os.mkdir('program_tasks/code_completion/result')

    train_loader = Word2vecLoader(train_path, d_word_index, batch_size=args.batch_size)
    val_loader = Word2vecLoader(val_path, d_word_index, batch_size=args.batch_size)
    val_loader1 = Word2vecLoader(test_path1, d_word_index, batch_size=args.batch_size)
    val_loader2 = Word2vecLoader(test_path2, d_word_index, batch_size=args.batch_size)
    val_loader3 = Word2vecLoader(test_path3, d_word_index, batch_size=args.batch_size)

    return d_word_index, embed, train_loader, val_loader, \
        val_loader1, val_loader2, val_loader3


def train(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    for i, (input, target, _) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0][0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()


def test(val_loader, model, criterion, val_name):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    for i, (input, target, _) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0][0], input.size(0))

    res = {f'{val_name} acc': top1.avg.item()}
    print(res)
    return res


def main(args):
    d_word_index, embed, train_loader, val_loader, \
        val_loader1, val_loader2, val_loader3 = preprocess_data()
    vocab_size = len(d_word_index)
    print('vocab_size is', vocab_size)

    # load ckpt if necessary
    if args.load_ckpt:
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(args.res_dir)
        resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        model = resume_checkpoint.model
        optimizer = resume_checkpoint.optimizer
        start_epoch = resume_checkpoint.epoch
    else:
        model = Word2vecPredict(d_word_index, embed)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        start_epoch = 1

    model = model.cuda()
    criterion = nn.CrossEntropyLoss()

    print('training dataset size is ', train_loader.n_samples)
    t1 = datetime.datetime.now()
    time_cost = None
    best_val_acc = 0

    for epoch in range(start_epoch, args.epochs + 1):
        st = datetime.datetime.now()
        train(train_loader, model, criterion, optimizer)
        ed = datetime.datetime.now()
        if time_cost is None:
            time_cost = ed - st
        else:
            time_cost += (ed - st)

        print(epoch, 'cost time', ed - st)
        res1 = test(val_loader1, model, criterion, 'test1')
        res2 = test(val_loader2, model, criterion, 'test2')
        res3 = test(val_loader3, model, criterion, 'test3')
        res4 = test(val_loader, model, criterion, 'val')
        merge_res = {**res4, **res1, **res2, **res3} # merge all the test results

        # save model checkpoint
        if res1['test1 acc'] > best_val_acc:
            Checkpoint(model, optimizer, epoch, merge_res).save(args.res_dir)
            best_val_acc = res1['test1 acc']


    print('time cost', time_cost / args.epochs)
    t2 = datetime.datetime.now()

    weight_save_model = os.path.join('program_tasks/code_completion', args.weight_name)
    torch.save(model.encoder.weight, weight_save_model)
    print('result is ', res1, res2, res3)
    print('cost time', t2 - t1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', default=10, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=2048, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
    parser.add_argument('--embedding_dim', default=100, type=int, metavar='N', help='embedding size')
    parser.add_argument('--hidden-size', default=128, type=int, metavar='N', help='rnn hidden size')
    parser.add_argument('--layers', default=2, type=int, metavar='N', help='number of rnn layers')
    parser.add_argument('--min_samples', default=5, type=int, metavar='N', help='min number of tokens')
    parser.add_argument('--cuda', default=True, action='store_true', help='use cuda')
    parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
    parser.add_argument('--mean_seq', default=False, action='store_true', help='use mean of rnn output')
    parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--weight_name', type=str, default='1', help='model name')
    parser.add_argument('--embedding_path', type=str, default='embedding_vec100_1/fasttext.vec')
    parser.add_argument('--train_data', type=str, default='program_tasks/code_completion/dataset/train.tsv',)
    parser.add_argument('--val_data', type=str, default='program_tasks/code_completion/dataset/val.tsv', help='model name')
    parser.add_argument('--test_data1', type=str, default='program_tasks/code_completion/dataset/test1.tsv', help='model name')
    parser.add_argument('--test_data2', type=str, default='program_tasks/code_completion/dataset/test2.tsv', help='model name')
    parser.add_argument('--test_data3', type=str, default='program_tasks/code_completion/dataset/test3.tsv', help='model name')
    parser.add_argument('--embedding_type', type=int, default=1, choices=[0, 1, 2])
    parser.add_argument('--experiment_name', type=str, default='code_completion')
    parser.add_argument('--res_dir', type=str, default='program_tasks/code_completion/result/')
    parser.add_argument('--load_ckpt', default=False, action='store_true', help='use pretrained model')

    args = parser.parse_args()
    main(args)