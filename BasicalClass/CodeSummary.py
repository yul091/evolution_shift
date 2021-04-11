import os
import torch.nn as nn
import torch.optim as optim
from BasicalClass.common_function import *
from BasicalClass.BasicModule import BasicModule
from program_tasks.code_summary.Code2VecModule import Code2Vec
from program_tasks.code_summary.CodeLoader import CodeLoader
from program_tasks.code_summary.main import perpare_train, my_collate
from preprocess.checkpoint import Checkpoint


class CodeSummary_Module(BasicModule):

    def __init__(self, device, res_dir, save_dir, data_dir, 
                 module_id, train_batch_size, test_batch_size, 
                 max_size, load_poor=False):
        super(CodeSummary_Module, self).__init__(
            device, res_dir, save_dir, data_dir, module_id,
            train_batch_size, test_batch_size, max_size, load_poor
        )

        self.train_loader, self.val_loader, self.test_loader = self.load_data()
        self.get_information()
        self.test_acc = common_cal_accuracy(self.test_pred_y, self.test_y)
        self.val_acc = common_cal_accuracy(self.val_pred_y, self.val_y)
        self.train_acc = common_cal_accuracy(self.train_pred_y, self.train_y)
        self.save_truth()
        print(
            'construct the module {}: '.format(self.__class__.__name__), 
            'train acc %0.4f, val acc %0.4f, test acc %0.4f' % (
                self.train_acc, self.val_acc, self.test_acc)
        )


    def load_model(self):
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.res_dir)
        resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        model = resume_checkpoint.model

        return model


    def load_poor_model(self):
        oldest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.res_dir)
        resume_checkpoint = Checkpoint.load(oldest_checkpoint_path)
        model = resume_checkpoint.model
        return model

    def load_data(self):
        token2index, path2index, func2index, embed, tk2num = perpare_train(
            self.tk_path, self.embed_type, self.vec_path, self.embed_dim, self.res_dir
        )
        # print('train path: {}'.format(self.train_path))
        train_db = CodeLoader(self.train_path, self.max_size, token2index, tk2num)
        val_db = CodeLoader(self.val_path, self.max_size, token2index, tk2num)
        test_db = CodeLoader(self.test_path, self.max_size, token2index, tk2num)
    
        print('train data length: {}, val data length: {}, test data length: {}'.format(
            len(train_db), len(val_db), len(test_db)
        ))

        train_loader = DataLoader(
            train_db, batch_size=self.train_batch_size, 
            collate_fn=my_collate, shuffle=False
        )
        val_loader = DataLoader(
            val_db, batch_size=self.test_batch_size, 
            collate_fn=my_collate, shuffle=False
        )
        test_loader = DataLoader(
            test_db, batch_size=self.test_batch_size, 
            collate_fn=my_collate, shuffle=False
        )
        
        print('train loader size: {}, val loader size: {}, test loader size: {}'.format(
            len(train_loader), len(val_loader), len(test_loader)
        ))

        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    CodeSummary_Module(DEVICE)