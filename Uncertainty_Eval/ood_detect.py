import os
import torch
import numpy as np
from Uncertainty_Eval.evaluation import Uncertainty_Eval

class OODdetect(Uncertainty_Eval):

    def __init__(self, res_dir, projects, save_dir, task='CodeSummary_Module'):
        super(OODdetect, self).__init__(res_dir, projects, save_dir, task)

    def evaluate(self):
        eval_res = {}
        project = self.projects
        trg_dir = os.path.join(self.res_dir, project, self.task)
        truth = torch.load(os.path.join(trg_dir,'truth.res'))
        uncertainty_res = [
            f for f in os.listdir(trg_dir) if f.endswith('.res') and f != 'truth.res'
        ]
        print('train_acc: %.4f, val_acc: %.4f, test_acc: %.4f' % (
            np.mean(truth['train']), np.mean(truth['val']), np.mean(truth['test'])
        ))
        # val as in-distribution, test as out-of-distribution
        oracle = np.array([1]*len(truth['val']) + [0]*len(truth['test'])) 

        for metric in uncertainty_res:
            metric_res = torch.load(os.path.join(trg_dir, metric))
            metric_name = metric[:-4] # get rid of endswith '.res'
            print('\n%s:' % (metric_name))
            eval_res[metric_name] = {}

            if metric_name not in ['Mutation', 'PVScore']:
                pred = np.concatenate((metric_res['val'], metric_res['test']))

            else:
                pred = np.concatenate((metric_res['val'][0], metric_res['test'][0]))

            # AUC
            AUC = self.common_get_auc(oracle, pred)
            print('AUC: %.4f' % (AUC))
            # AUPR
            AUPR = self.common_get_aupr(oracle, pred)
            print('AUPR: %.4f' % (AUPR))
            # Brier score
            Brier = self.common_get_brier(oracle, pred)
            print('Brier: %.4f' % (Brier))

            eval_res[metric_name]['AUC'] = AUC
            eval_res[metric_name]['AUPR'] = AUPR
            eval_res[metric_name]['Brier'] = Brier

        # save evaluation res
        if not os.path.exists(os.path.join(self.save_dir, self.task)):
            os.makedirs(os.path.join(self.save_dir, self.task))
        save_name = os.path.join(self.save_dir, self.task, 'uncertainty_ood_eval.res')
        torch.save(eval_res, save_name)

            

if __name__ == "__main__":
    # from preprocess.train_split import JAVA_PROJECTS

    # eval_m = Uncertainty_Eval(
    #     res_dir='Uncertainty_Results', projects=JAVA_PROJECTS, 
    #     save_dir='Uncertainty_Eval/java', task='CodeSummary_Module'
    # )
    eval_m = OODdetect(
        res_dir='Uncertainty_Results/different_project', 
        projects='java_project3', 
        save_dir='Uncertainty_Eval/different_project/java_project3', 
        # task='CodeSummary_Module'
        task='CodeCompletion_Module',
    )
    eval_m.evaluate()
        
