import os
import torch
# import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import roc_curve, auc, average_precision_score, brier_score_loss, precision_recall_curve


class Uncertainty_Eval():
    def __init__(self, res_dir, project, task='CodeSummary_Module'):
        """
        res_dir (str): path of uncertainty result, Default: "Uncertainty_Results".
        project (str): project name like gradle, elasticsearch, etc.
        task (str): task name like CodeSummary_Module.
        """
        self.res_dir = res_dir
        self.project = project
        self.task = task

        
    def common_get_auc(self, y_test, y_score, name=None):
        fpr, tpr, threshold = roc_curve(y_test, y_score)  # calculate true positive & false positive
        roc_auc = auc(fpr, tpr)  # calculate AUC
        if name is not None:
            print(name, 'auc is ', roc_auc)
        return roc_auc 

    def common_get_aupr(self, y_test, y_score, name=None):
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        area = auc(recall, precision)
        # aupr = average_precision_score(y_test, y_score)
        if name is not None:
            print(name, 'aupr is ', area)
        return area

    def common_get_nll(self, y_test, y_score):
        pred_logits = torch.cat((
            torch.tensor(y_score).unsqueeze(1), 
            torch.tensor(1-y_score).unsqueeze(1)
        ), dim=-1)
        nll = torch.nn.NLLLoss()
        return nll(pred_logits, torch.tensor(y_test).long()).item()

    def common_get_brier(self, y_test, y_score, name=None):
        brier = brier_score_loss(y_test, y_score)
        if name is not None:
            print(name, 'brier is ', brier)
        return brier


    def evaluation(self):
        
        print('Evaluating project {} ...'.format(self.project))
        trg_dir = os.path.join(self.res_dir, self.project, self.task)
        truth = torch.load(os.path.join(trg_dir,'truth.res'))
        uncertainty_res = [f for f in os.listdir(trg_dir) if f.endswith('.res') and f != 'truth.res']
        # print(uncertainty_res)
        print('train_acc: %.4f, val_acc: %.4f, shift1_acc: %.4f, shift2_acc: %.4f, shift3_acc: %.4f' % (
            np.mean(truth['train']), np.mean(truth['val']), 
            np.mean(truth['shift1']), np.mean(truth['shift2']),
            np.mean(truth['shift3'])
        ))
        for metric in uncertainty_res:
            metric_res = torch.load(os.path.join(trg_dir, metric))
            metric_name = metric[:-4] # get rid of endswith '.res'
            if metric_name not in ['Mutation', 'PVScore']:
                # average uncertainty
                print('%s: \nmUncertainty: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % (
                    metric_name,
                    np.mean(metric_res['val']), np.mean(metric_res['shift1']), 
                    np.mean(metric_res['shift2']), np.mean(metric_res['shift3'])
                ))
                # AUC
                print('AUC: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % ( 
                    self.common_get_auc(truth['val'], metric_res['val']), 
                    self.common_get_auc(truth['shift1'], metric_res['shift1']), 
                    self.common_get_auc(truth['shift2'], metric_res['shift2']), 
                    self.common_get_auc(truth['shift3'], metric_res['shift3']),
                ))
                # AUPR
                print('AUPR: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % ( 
                    self.common_get_aupr(truth['val'], metric_res['val']), 
                    self.common_get_aupr(truth['shift1'], metric_res['shift1']), 
                    self.common_get_aupr(truth['shift2'], metric_res['shift2']), 
                    self.common_get_aupr(truth['shift3'], metric_res['shift3']),
                ))
                # Brier score
                print('Brier: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % ( 
                    self.common_get_brier(truth['val'], metric_res['val']), 
                    self.common_get_brier(truth['shift1'], metric_res['shift1']), 
                    self.common_get_brier(truth['shift2'], metric_res['shift2']), 
                    self.common_get_brier(truth['shift3'], metric_res['shift3']),
                ))
            else:
                # average uncertainty
                print('%s: \nmUncertainty: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % (
                    metric_name,
                    np.mean(metric_res['val'][0]), np.mean(metric_res['shift1'][0]), 
                    np.mean(metric_res['shift2'][0]), np.mean(metric_res['shift3'][0])
                ))
                # AUC
                print('AUC: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % ( 
                    self.common_get_auc(truth['val'], metric_res['val'][0]), 
                    self.common_get_auc(truth['shift1'], metric_res['shift1'][0]), 
                    self.common_get_auc(truth['shift2'], metric_res['shift2'][0]), 
                    self.common_get_auc(truth['shift3'], metric_res['shift3'][0]),
                ))
                # AUPR
                print('AUPR: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % ( 
                    self.common_get_aupr(truth['val'], metric_res['val'][0]), 
                    self.common_get_aupr(truth['shift1'], metric_res['shift1'][0]), 
                    self.common_get_aupr(truth['shift2'], metric_res['shift2'][0]), 
                    self.common_get_aupr(truth['shift3'], metric_res['shift3'][0]),
                ))
                # Brier score
                print('Brier: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % ( 
                    self.common_get_brier(truth['val'], metric_res['val'][0]), 
                    self.common_get_brier(truth['shift1'], metric_res['shift1'][0]), 
                    self.common_get_brier(truth['shift2'], metric_res['shift2'][0]), 
                    self.common_get_brier(truth['shift3'], metric_res['shift3'][0]),
                ))



if __name__ == "__main__":
    from preprocess.train_split import JAVA_PROJECTS

    for project in JAVA_PROJECTS:

        eval_m = Uncertainty_Eval(res_dir="Uncertainty_Results", project=project)
        eval_m.evaluation()


