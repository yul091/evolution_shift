import os
from re import M
import torch
# import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import roc_curve, auc, average_precision_score, brier_score_loss, precision_recall_curve


class Uncertainty_Eval():
    def __init__(self, res_dir, projects, save_dir, task='CodeSummary_Module'):
        """
        res_dir (str): path of uncertainty result, Default: "Uncertainty_Results".
        projects (list): list of project names like [gradle, elasticsearch, etc.]
        save_dir (str): path of saving evaluation res, Default: "Uncertainty_Eval/java".
        task (str): task name like CodeSummary_Module.
        """
        self.res_dir = res_dir
        self.projects = projects
        self.task = task
        self.save_dir = save_dir
        
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
        eval_res = {}
        if isinstance(self.projects, list):
            for project in self.projects:
                eval_res[project] = {}
                print('Evaluating project {} ...'.format(project))
                trg_dir = os.path.join(self.res_dir, project, self.task)
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
                    eval_res[project][metric_name] = {}

                    if metric_name not in ['Mutation', 'PVScore']:
                        # average uncertainty
                        mU_val = np.mean(metric_res['val'])
                        mU_shift1 = np.mean(metric_res['shift1'])
                        mU_shift2 = np.mean(metric_res['shift2'])
                        mU_shift3 = np.mean(metric_res['shift3'])
                        print('%s: \nmUncertainty: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % (
                            metric_name, mU_val, mU_shift1, mU_shift2, mU_shift3,
                        ))
                        # AUC
                        AUC_val = self.common_get_auc(truth['val'], metric_res['val'])
                        AUC_shift1 = self.common_get_auc(truth['shift1'], metric_res['shift1'])
                        AUC_shift2 = self.common_get_auc(truth['shift2'], metric_res['shift2'])
                        AUC_shift3 = self.common_get_auc(truth['shift3'], metric_res['shift3'])
                        print('AUC: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % ( 
                            AUC_val, AUC_shift1, AUC_shift2, AUC_shift3,
                        ))
                        # AUPR
                        AUPR_val = self.common_get_aupr(truth['val'], metric_res['val'])
                        AUPR_shift1 = self.common_get_aupr(truth['shift1'], metric_res['shift1'])
                        AUPR_shift2 = self.common_get_aupr(truth['shift2'], metric_res['shift2'])
                        AUPR_shift3 = self.common_get_aupr(truth['shift3'], metric_res['shift3'])
                        print('AUPR: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % ( 
                            AUPR_val, AUPR_shift1, AUPR_shift2, AUPR_shift3,
                        ))
                        # Brier score
                        Brier_val = self.common_get_brier(truth['val'], metric_res['val'])
                        Brier_shift1 = self.common_get_brier(truth['shift1'], metric_res['shift1'])
                        Brier_shift2 = self.common_get_brier(truth['shift2'], metric_res['shift2'])
                        Brier_shift3 = self.common_get_brier(truth['shift3'], metric_res['shift3'])
                        print('Brier: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % ( 
                            Brier_val, Brier_shift1, Brier_shift2, Brier_shift3,
                        ))
                    else:
                        # average uncertainty
                        mU_val = np.mean(metric_res['val'][0])
                        mU_shift1 = np.mean(metric_res['shift1'][0])
                        mU_shift2 = np.mean(metric_res['shift2'][0])
                        mU_shift3 = np.mean(metric_res['shift3'][0])
                        print('%s: \nmUncertainty: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % (
                            metric_name, mU_val, mU_shift1, mU_shift2, mU_shift3,
                        ))
                        # AUC
                        AUC_val = self.common_get_auc(truth['val'], metric_res['val'][0])
                        AUC_shift1 = self.common_get_auc(truth['shift1'], metric_res['shift1'][0])
                        AUC_shift2 = self.common_get_auc(truth['shift2'], metric_res['shift2'][0])
                        AUC_shift3 = self.common_get_auc(truth['shift3'], metric_res['shift3'][0])
                        print('AUC: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % ( 
                            AUC_val, AUC_shift1, AUC_shift2, AUC_shift3,
                        ))
                        # AUPR
                        AUPR_val = self.common_get_aupr(truth['val'], metric_res['val'][0])
                        AUPR_shift1 = self.common_get_aupr(truth['shift1'], metric_res['shift1'][0])
                        AUPR_shift2 = self.common_get_aupr(truth['shift2'], metric_res['shift2'][0])
                        AUPR_shift3 = self.common_get_aupr(truth['shift3'], metric_res['shift3'][0])
                        print('AUPR: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % ( 
                            AUPR_val, AUPR_shift1, AUPR_shift2, AUPR_shift3,
                        ))
                        # Brier score
                        Brier_val = self.common_get_brier(truth['val'], metric_res['val'][0])
                        Brier_shift1 = self.common_get_brier(truth['shift1'], metric_res['shift1'][0])
                        Brier_shift2 = self.common_get_brier(truth['shift2'], metric_res['shift2'][0])
                        Brier_shift3 = self.common_get_brier(truth['shift3'], metric_res['shift3'][0])
                        print('Brier: val: %.4f, shift1: %.4f, shift2: %.4f, shift3: %.4f' % ( 
                            Brier_val, Brier_shift1, Brier_shift2, Brier_shift3,
                        ))

                    eval_res[project][metric_name]['mUncertain'] = {
                        'val': mU_val, 'shift1': mU_shift1, 'shift2': mU_shift2, 'shift3': mU_shift3,
                    }
                    eval_res[project][metric_name]['AUC'] = {
                        'val': AUC_val, 'shift1': AUC_shift1, 'shift2': AUC_shift2, 'shift3': AUC_shift3,
                    }
                    eval_res[project][metric_name]['AUPR'] = {
                        'val': AUPR_val, 'shift1': AUPR_shift1, 'shift2': AUPR_shift2, 'shift3': AUPR_shift3,
                    }
                    eval_res[project][metric_name]['Brier'] = {
                        'val': Brier_val, 'shift1': Brier_shift1, 'shift2': Brier_shift2, 'shift3': Brier_shift3,
                    }

        else: # all java files trained together
            project = self.projects
            trg_dir = os.path.join(self.res_dir, project, self.task)
            truth = torch.load(os.path.join(trg_dir,'truth.res'))
            uncertainty_res = [f for f in os.listdir(trg_dir) if f.endswith('.res') and f != 'truth.res']
            # print(uncertainty_res)
            print('train_acc: %.4f, val_acc: %.4f, test_acc: %.4f' % (
                np.mean(truth['train']), np.mean(truth['val']), np.mean(truth['test'])
            ))
            for metric in uncertainty_res:
                metric_res = torch.load(os.path.join(trg_dir, metric))
                metric_name = metric[:-4] # get rid of endswith '.res'
                eval_res[metric_name] = {}

                if metric_name not in ['Mutation', 'PVScore']:
                    # average uncertainty
                    mU_val = np.mean(metric_res['val'])
                    mU_test = np.mean(metric_res['test'])
                    print('%s: \nmUncertainty: val: %.4f, test: %.4f' % (
                        metric_name, mU_val, mU_test
                    ))
                    # AUC
                    AUC_val = self.common_get_auc(truth['val'], metric_res['val'])
                    AUC_test = self.common_get_auc(truth['test'], metric_res['test'])
                    print('AUC: val: %.4f, test: %.4f' % (AUC_val, AUC_test))
                    # AUPR
                    AUPR_val = self.common_get_aupr(truth['val'], metric_res['val'])
                    AUPR_test = self.common_get_aupr(truth['test'], metric_res['test'])
                    print('AUPR: val: %.4f, test: %.4f' % (AUPR_val, AUPR_test))
                    # Brier score
                    Brier_val = self.common_get_brier(truth['val'], metric_res['val'])
                    Brier_test = self.common_get_brier(truth['test'], metric_res['test'])
                    print('Brier: val: %.4f, test: %.4f' % (Brier_val, Brier_test))
                else:
                    # average uncertainty
                    mU_val = np.mean(metric_res['val'][0])
                    mU_test = np.mean(metric_res['test'][0])
                    print('%s: \nmUncertainty: val: %.4f, test: %.4f' % (
                        metric_name, mU_val, mU_test
                    ))
                    # AUC
                    AUC_val = self.common_get_auc(truth['val'], metric_res['val'][0])
                    AUC_test = self.common_get_auc(truth['test'], metric_res['test'][0])
                    print('AUC: val: %.4f, test: %.4f' % (AUC_val, AUC_test))
                    # AUPR
                    AUPR_val = self.common_get_aupr(truth['val'], metric_res['val'][0])
                    AUPR_test = self.common_get_aupr(truth['test'], metric_res['test'][0])
                    print('AUPR: val: %.4f, test: %.4f' % (AUPR_val, AUPR_test))
                    # Brier score
                    Brier_val = self.common_get_brier(truth['val'], metric_res['val'][0])
                    Brier_test = self.common_get_brier(truth['test'], metric_res['test'][0])
                    print('Brier: val: %.4f, test: %.4f' % (Brier_val, Brier_test))

                eval_res[metric_name]['mUncertain'] = {'val': mU_val, 'test': mU_test}
                eval_res[metric_name]['AUC'] = {'val': AUC_val, 'test': AUC_test}
                eval_res[metric_name]['AUPR'] = {'val': AUPR_val, 'test': AUPR_test}
                eval_res[metric_name]['Brier'] = {'val': Brier_val, 'test': Brier_test}

        # save evaluation res
        if not os.path.exists(os.path.join(self.save_dir, self.task)):
            os.makedirs(os.path.join(self.save_dir, self.task))
        save_name = os.path.join(self.save_dir, self.task, 'uncertainty_eval.res')
        torch.save(eval_res, save_name)



if __name__ == "__main__":
    # from preprocess.train_split import JAVA_PROJECTS
    # projects = ['java_project1', 'java_project2', 'java_project3']

    # for project in projects:
    #     eval_m = Uncertainty_Eval(
    #         res_dir='Uncertainty_Results/different_project', projects=project, 
    #         save_dir='Uncertainty_Eval/different_project/'+project, task='CodeCompletion_Module'
    #     )
    #     eval_m.evaluation()

    eval_m = Uncertainty_Eval(
        res_dir='Uncertainty_Results/different_project', projects='java_project3', 
        save_dir='Uncertainty_Eval/different_project/java_project3', task='CodeCompletion_Module'
    )
    eval_m.evaluation()
