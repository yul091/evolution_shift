from Metric import *
from BasicalClass import Fashion_Module, CIFAR10_Module, CodeSummary_Module
import torch
from BasicalClass import common_get_auc
import argparse

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# module_instance = CIFAR10_Module(device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=int, default=0, help='the gpu id')
    parser.add_argument('-module_id', type=int, default=0, help='the task id')
    parser.add_argument('-res_dir', type=str, default='program_tasks/code_summary/result/wildfly')
    parser.add_argument('-data_dir', type=str, default='data/wildfly_pkl')
    parser.add_argument('-save_dir', type=str, default='Uncertainty_Results/wildfly')

    args = parser.parse_args()
    print(vars(args))
    ModuleList = [
        # Fashion_Module,
        # CIFAR10_Module,
        CodeSummary_Module
    ]
    MetricList = [
        Vanilla,
        ModelWithTemperature,
        PVScore,
        Mahalanobis,
        ModelActivateDropout,
        Mutation,
        Entropy,
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res_dir = args.res_dir
    save_dir = args.save_dir
    data_dir = args.data_dir
    Module = ModuleList[args.module_id] # code_summary

    module_instance = Module(device=device, res_dir=res_dir, data_dir=data_dir, save_dir=save_dir)
    for i, metric in enumerate(MetricList):
        print(f'metric name: {metric.__name__}')
        if metric.__name__ not in ['ModelActivateDropout', 'Mutation']:
            v = metric(module_instance, device)
        else:
            v = metric(module_instance, device, iter_time=50)
        v.run()