import os
import numpy as np
from torch.cuda import is_available
import torch
import torch.nn as nn
from BasicalClass.common_function import *
from program_tasks.code_summary.CodeLoader import CodeLoader
from program_tasks.code_summary.main import perpare_train, my_collate, test_model, dict2list
from preprocess.checkpoint import Checkpoint
from tqdm import tqdm


class Filter:
    def __init__(self, res_dir, data_dir, metric_dir, save_dir, 
                 device, module_id, shift, max_size, batch_size):

        self.res_dir = res_dir
        self.data_dir = data_dir
        self.device = device
        self.shift = shift
        self.module_id = module_id
        self.embed_type = 1
        self.vec_path = None
        self.embed_dim = 100
        self.max_size = max_size
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.vanilla = torch.load(os.path.join(metric_dir, 'Vanilla.res'))
        self.temp = torch.load(os.path.join(metric_dir, 'ModelWithTemperature.res'))
        self.pv = torch.load(os.path.join(metric_dir, 'PVScore.res'))
        self.dropout = torch.load(os.path.join(metric_dir, 'ModelActivateDropout.res'))
        self.mutation = torch.load(os.path.join(metric_dir, 'Mutation.res'))
        self.batch_size = batch_size

        if module_id == 0: # code summary
            self.tk_path = os.path.join(self.data_dir, 'tk.pkl')
            self.train_path = os.path.join(self.data_dir, 'train.pkl')
            self.val_path = os.path.join(self.data_dir, 'val.pkl')
            if shift:
                self.test_path = None
                self.test1_path = os.path.join(self.data_dir, 'test1.pkl')
                self.test2_path = os.path.join(self.data_dir, 'test2.pkl')
                self.test3_path = os.path.join(self.data_dir, 'test3.pkl')
            else:
                self.test_path = os.path.join(self.data_dir, 'test.pkl')
                self.test_path1 = None
                self.test_path2 = None
                self.test_path3 = None
        else: # code completion
            self.tk_path = None
            self.train_path = os.path.join(self.data_dir, 'train.tsv')
            self.val_path = os.path.join(self.data_dir, 'val.tsv')
            if shift:
                self.test_path = None
                self.test1_path = os.path.join(self.data_dir, 'test1.tsv')
                self.test2_path = os.path.join(self.data_dir, 'test2.tsv')
                self.test3_path = os.path.join(self.data_dir, 'test3.tsv')
            else:
                self.test_path = os.path.join(self.data_dir, 'test.tsv')
                self.test_path1 = None
                self.test_path2 = None
                self.test_path3 = None

        if module_id == 0:
            # load data and preparation
            self.token2index, path2index, func2index, embed, self.tk2num = perpare_train(
                self.tk_path, self.embed_type, self.vec_path, 
                self.embed_dim, self.res_dir
            )
            # nodes_dim, paths_dim, output_dim = len(tk2num), len(path2index), len(func2index)
            self.index2func = dict2list(func2index)
            # criterian = nn.CrossEntropyLoss()  # loss

            # load ckpt 
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.res_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            self.model = resume_checkpoint.model
            self.model.to(self.device)
            self.model.eval()
            # optimizer = resume_checkpoint.optimizer
            # start_epoch = resume_checkpoint.epoch
            
            # build test loader
            if shift:
                self.test_dataset1 = CodeLoader(
                    self.test_path1, self.max_size, self.token2index, self.tk2num
                )
                self.test_dataset2 = CodeLoader(
                    self.test_path2, self.max_size, self.token2index, self.tk2num
                )
                self.test_dataset3 = CodeLoader(
                    self.test_path3, self.max_size, self.token2index, self.tk2num
                )
            else:
                self.test_dataset = CodeLoader(
                    self.test_path, max_size, self.token2index, self.tk2num
                )

    def run(self):
        # evaluate on test dataset
        res = {
            'x': np.arange(0,1,0.01), 
            'va_acc': [], 
            'temp_acc': [], 
            'mutation_acc': [],
            'dropout_acc': [],
            'pv_acc': [],
        }
        for threshold in tqdm(np.arange(0,1,0.01)):
            # vanilla
            va_idx = np.arange(len(self.test_dataset))[self.vanilla['test'] > threshold]
            va_dataset = CodeLoader(
                self.test_path, self.max_size, self.token2index, self.tk2num, idx=va_idx
            )
            va_test_loader = DataLoader(va_dataset, batch_size=self.batch_size, collate_fn=my_collate)
            va_acc = test_model(va_test_loader, self.model, self.device, self.index2func, 'test')['test acc']
            res['va_acc'].append(va_acc)
            # temp scaling
            temp_idx = np.arange(len(self.test_dataset))[self.temp['test'] > threshold]
            temp_dataset = CodeLoader(
                self.test_path, self.max_size, self.token2index, self.tk2num, idx=temp_idx
            )
            temp_test_loader = DataLoader(temp_dataset, batch_size=self.batch_size, collate_fn=my_collate)
            temp_acc = test_model(temp_test_loader, self.model, self.device, self.index2func, 'test')['test acc']
            res['temp_acc'].append(temp_acc)
            # mutation
            mutation_idx = np.arange(len(self.test_dataset))[self.mutation['test'][0] > threshold]
            mutation_dataset = CodeLoader(
                self.test_path, self.max_size, self.token2index, self.tk2num, idx=mutation_idx
            )
            mutation_test_loader = DataLoader(mutation_dataset, batch_size=self.batch_size, collate_fn=my_collate)
            mutation_acc = test_model(mutation_test_loader, self.model, self.device, self.index2func, 'test')['test acc']
            res['mutation_acc'].append(mutation_acc)
            # dropout
            dropout_idx = np.arange(len(self.test_dataset))[self.dropout['test'] > threshold]
            dropout_dataset = CodeLoader(
                self.test_path, self.max_size, self.token2index, self.tk2num, idx=dropout_idx
            )
            dropout_test_loader = DataLoader(dropout_dataset, batch_size=self.batch_size, collate_fn=my_collate)
            dropout_acc = test_model(dropout_test_loader, self.model, self.device, self.index2func, 'test')['test acc']
            res['dropout_acc'].append(dropout_acc)
            # dissector
            pv_idx = np.arange(len(self.test_dataset))[self.pv['test'][0] > threshold]
            pv_dataset = CodeLoader(
                self.test_path, self.max_size, self.token2index, self.tk2num, idx=pv_idx
            )
            pv_test_loader = DataLoader(pv_dataset, batch_size=self.batch_size, collate_fn=my_collate)
            pv_acc = test_model(pv_test_loader, self.model, self.device, self.index2func, 'test')['test acc']
            res['pv_acc'].append(pv_acc)

            print('threshold {}: vanilla test acc {}, temp test acc {}, mutation test acc {}, dropout test acc {}, dissector test acc {}'.format(
                threshold, va_acc, temp_acc, mutation_acc, dropout_acc, pv_acc
            ))

        # save file 
        torch.save(res, os.path.join(self.save_dir, 'filter.res'))



if __name__ == "__main__":

    project = 'java_project3'
    res_dir = 'program_tasks/code_summary/result/different_project/'+project
    data_dir = 'java_data/different_project/java_pkl3'
    metric_dir = 'Uncertainty_Results/different_project/'+project+'/CodeSummary_Module'
    save_dir = 'Uncertainty_Eval/filter'
    module_id = 0
    shift = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_size = None
    batch_size = 256


    filter = Filter(
        res_dir=res_dir, data_dir=data_dir, metric_dir=metric_dir,
        save_dir=save_dir, device=device, module_id=module_id, shift=shift,
        max_size=max_size, batch_size=batch_size
    )

    filter.run()






