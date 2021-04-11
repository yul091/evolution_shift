import os
import torch.nn as nn
import torch.optim as optim
from BasicalClass.common_function import *
from BasicalClass.BasicModule import BasicModule
from preprocess.checkpoint import Checkpoint
from program_tasks.code_completion.model import Word2vecPredict
from program_tasks.code_completion.vocab import VocabBuilder
from program_tasks.code_completion.dataloader import TextClassDataLoader, Word2vecLoader


class CodeCompletion_Module(BasicModule):

    def __init__(self, device, res_dir, save_dir, data_dir, 
                 module_id, train_batch_size, test_batch_size, 
                 max_size, load_poor=False):
        super(CodeCompletion_Module, self).__init__(
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
            'construct the module', self.__class__.__name__, 
            'train acc %0.4f, val acc %0.4f, test acc %0.4f' % (
                self.train_acc, self.val_acc, self.test_acc)
        )


    def load_model(self):
        latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.res_dir)
        resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
        model = resume_checkpoint.model

        return model

    def load_data(self):
        v_builder = VocabBuilder(path_file=self.train_path)
        d_word_index, embed = v_builder.get_word_index(min_sample=self.min_samples)

        if embed is not None:
            if type(embed) is np.ndarray:
                embed = torch.tensor(embed, dtype=torch.float).cuda()
            assert embed.size()[1] == self.embed_dim

        train_loader = Word2vecLoader(self.train_path, d_word_index, 
                                      batch_size=self.train_batch_size,
                                      max_size=self.max_size)
        val_loader = Word2vecLoader(self.val_path, d_word_index, 
                                    batch_size=self.train_batch_size,
                                    max_size=self.max_size)
        test_loader = Word2vecLoader(self.test_path, d_word_index, 
                                     batch_size=self.train_batch_size,
                                     max_size=self.max_size)
        
        print('train loader size: {}, val loader size: {}, test loader size: {}'.format(
            len(train_loader), len(val_loader), len(test_loader)
        ))

        return train_loader, val_loader, test_loader


