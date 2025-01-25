import random
from warnings import simplefilter
from dataloader import Dataset
from model import *
import warnings
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description='FaGSP')
    parser.add_argument('--dataset', default='ml1m', type=str)
    parser.add_argument('--pri_factor1', default=256, type=int)
    parser.add_argument('--pri_factor2', default=128, type=int)
    parser.add_argument('--alpha1', default=0.3, type=float)
    parser.add_argument('--alpha2', default=0.5, type=float)
    parser.add_argument('--order1', default=12, type=int)
    parser.add_argument('--order2', default=14, type=int)
    parser.add_argument('--q', default=0.7, type=float)

    args = parser.parse_args()

    random.seed(2020)
    np.random.seed(2020)
    dataloader = Dataset(args.dataset)

    init_adj_mat, init_rat_mat = dataloader.get_init_mats()
    user_interacted_items_dict, valid_user_items_dict, test_user_items_dict = dataloader.get_train_test_data()
    lm = FaGSP(init_adj_mat, init_rat_mat,
               pri_factor1=args.pri_factor1, pri_factor2=args.pri_factor2, alpha1=args.alpha1, alpha2=args.alpha2,
               order1=args.order1, order2=args.order2,
               user_interacted_items_dict=user_interacted_items_dict)

    ratings = lm.train(quan=args.q)
    valid_f1s, valid_mrrs, valid_ndcgs = lm.eval_test(valid_user_items_dict, ratings)  # for hyper-parameter tuning
    test_metrics = lm.eval_test(test_user_items_dict, ratings)

    (best_f1s, best_mrrs, best_ndcgs) = test_metrics
    test_metric_info = '\tF1@10:{:.4f}\tMRR@10:{:.4f}\tNDCG@10:{:.4f}\n' \
                       '\tF1@20:{:.4f}\tMRR@20:{:.4f}\tNDCG@20:{:.4f}'.format(
        best_f1s[0], best_mrrs[0], best_ndcgs[0], best_f1s[1], best_mrrs[1], best_ndcgs[1])

    print('[Test]\n{}'.format(test_metric_info))


if __name__ == '__main__':
    main()
