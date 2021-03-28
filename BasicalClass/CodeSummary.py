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

    def __init__(self, device, res_dir, save_dir, data_dir, load_poor=False):
        super(CodeSummary_Module, self).__init__(device, res_dir, save_dir, data_dir, load_poor)

        self.train_loader, self.val_loader, self.shift1_loader, \
            self.shift2_loader, self.shift3_loader = self.load_data()
        self.get_information()
        self.shift1_acc = common_cal_accuracy(self.shift1_pred_y, self.shift1_y)
        self.shift2_acc = common_cal_accuracy(self.shift2_pred_y, self.shift2_y)
        self.shift3_acc = common_cal_accuracy(self.shift3_pred_y, self.shift3_y)
        self.val_acc = common_cal_accuracy(self.val_pred_y, self.val_y)
        self.train_acc = common_cal_accuracy(self.train_pred_y, self.train_y)

        self.save_truth()
        print(
            'construct the module', self.__class__.__name__, 
            'train acc %0.4f, val acc %0.4f, shift1 acc %0.4f, shift2 acc %0.4f, shift3 acc %0.4f' % (
                self.train_acc, self.val_acc, self.shift1_acc, self.shift2_acc, self.shift3_acc
            )
        )

    def load_model(self):
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.res_dir)
        resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        model = resume_checkpoint.model
        # token2index, path2index, func2index, embed, tk2num =\
        #     perpare_train(self.tk_path, self.embed_type, self.vec_path, self.embed_dim, self.out_dir)
        # nodes_dim, paths_dim, output_dim = len(tk2num), len(path2index), len(func2index)
        # model = Code2Vec(nodes_dim, paths_dim, self.embed_dim, output_dim, embed)
        # # get latest ckpt model name
        # checkpoints_path = os.path.join(self.res_dir, self.CHECKPOINT_DIR_NAME)
        # all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        # model_dir = os.path.join(checkpoints_path, all_times[0])
        # model_path = os.path.join(model_dir, self.MODEL_NAME)

        # model.load_state_dict(
        #     torch.load(model_path, map_location=self.device)
        # )
        return model

    def load_poor_model(self):
        oldest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.res_dir)
        resume_checkpoint = Checkpoint.load(oldest_checkpoint_path)
        model = resume_checkpoint.model
        return model

    def load_data(self):
        token2index, path2index, func2index, embed, tk2num = \
            perpare_train(
                self.tk_path, self.embed_type, self.vec_path, 
                self.embed_dim, self.res_dir)

        train_db = CodeLoader(self.train_path, self.max_size, token2index, tk2num)
        val_db = CodeLoader(self.val_path, self.max_size, token2index, tk2num)
        shift1_db = CodeLoader(self.shift1_path, self.max_size, token2index, tk2num)
        shift2_db = CodeLoader(self.shift2_path, self.max_size, token2index, tk2num)
        shift3_db = CodeLoader(self.shift3_path, self.max_size, token2index, tk2num)
        print('train data length: {}, val data length: {}, shift1 length: {}, shift2 length: {}, shift3 length: {}'.format(
            len(train_db), len(val_db), len(shift1_db), len(shift2_db), len(shift3_db)
        ))

        train_loader = DataLoader(
            train_db, batch_size=self.train_batch_size, 
            collate_fn=my_collate, shuffle=False
        )
        val_loader = DataLoader(
            val_db, batch_size=self.test_batch_size, 
            collate_fn=my_collate, shuffle=False
        )
        shift1_loader = DataLoader(
            shift1_db, batch_size=self.test_batch_size, 
            collate_fn=my_collate, shuffle=False
        )
        shift2_loader = DataLoader(
            shift2_db, batch_size=self.test_batch_size, 
            collate_fn=my_collate, shuffle=False
        )
        shift3_loader = DataLoader(
            shift3_db, batch_size=self.test_batch_size, 
            collate_fn=my_collate, shuffle=False
        )

        print('train loader size: {}, val loader size: {}, shift1 size: {}, shift2 size: {}, shift3 size: {}'.format(
            len(train_loader), len(val_loader), len(shift1_loader), len(shift2_loader), len(shift3_loader)
        ))
        # return self.get_loader(train_db, val_db, test_db)
        return train_loader, val_loader, shift1_loader, shift2_loader, shift3_loader


if __name__ == '__main__':
    
    CodeSummary_Module(DEVICE)