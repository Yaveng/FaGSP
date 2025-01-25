import torch
import metric
import numpy as np
import scipy.sparse as sp


class FaGSP(object):
    def __init__(self, init_adj_mat, init_rat_mat,
                 pri_factor1=256, pri_factor2=256, alpha1=0.3, alpha2=0.3, order1=2, order2=2,
                 user_interacted_items_dict=None):
        self.adj_mat = init_adj_mat
        self.rat_mat = init_rat_mat
        self.pri_factor1 = pri_factor1
        self.pri_factor2 = pri_factor2
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.order1 = order1
        self.order2 = order2

        self.user_interacted_items_dict = user_interacted_items_dict
        self.topks = [10, 20]

    def normalize_adj_mat(self, adj_mat):
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        colsum = np.array(adj_mat.sum(axis=0))
        d_inv = np.power(colsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        d_mat_i = d_mat
        d_inv = 1.0 / d_inv
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_i_inv = sp.diags(d_inv)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsc()
        return norm_adj, d_mat_i, d_mat_i_inv

    def matpow(self, mat, order):
        R = mat
        if order == 1:
            return R
        for ord in range(2, order+1):
            R = R.T @ mat
        return R

    def get_user_rating(self, quan):
        rat_mat = self.rat_mat
        batch_test = np.array(rat_mat.todense())

        P = 0

        P11 = batch_test @ (np.eye(self.RTR1_pow.shape[0]) - self.RTR1_pow)
        P += P11
        P12 = (np.eye(self.RTR2_pow.shape[0]) - self.RTR2_pow) @ batch_test
        P += P12

        vt2 = self.vt2[-self.pri_factor2:]
        P30 = batch_test @ self.d_mat_i @ vt2.T @ vt2 @ self.d_mat_i_inv
        quan = np.quantile(P30, q=quan, axis=0, keepdims=True)
        P30[P30 > quan] = 1.0
        P30[P30 <= quan] = 0.0
        P30[batch_test<1] = 0.0
        P3 = batch_test + self.alpha2 * P30
        P3 = sp.csr_matrix(P3)

        norm_P3, d_mat_i_P3, d_mat_i_inv_P3 = self.normalize_adj_mat(P3)
        _, _, vt1 = np.linalg.svd(norm_P3.A, full_matrices=False)
        vt1 = vt1[:self.pri_factor1]
        P2 = P3 @ d_mat_i_P3 @ vt1.T @ vt1 @ d_mat_i_inv_P3
        P += self.alpha1 * P2

        return P
    
    def train(self, quan):
        self.norm_adj, self.d_mat_i, self.d_mat_i_inv = self.normalize_adj_mat(self.adj_mat)
        _, _, self.vt2 = np.linalg.svd(self.norm_adj.A, full_matrices=False)

        RTR1 = self.norm_adj.T @ self.norm_adj
        self.RTR1_pow = self.matpow(np.eye(RTR1.shape[0]) - RTR1, self.order1)
        RTR2 = self.norm_adj @ self.norm_adj.T
        self.RTR2_pow = self.matpow(np.eye(RTR2.shape[0]) - RTR2, self.order2)

        ratings = self.get_user_rating(quan)
        return ratings

    def eval_test(self, user_items_dict, ratings):
        users = list(user_items_dict.keys())
        user_prediction_items_list = []
        user_truth_items_list = []
        for user in users:
            rating = ratings[user, :]
            rating = torch.from_numpy(rating).view(1, -1)
            user_interacted_items = list(self.user_interacted_items_dict[user])
            rating[0, user_interacted_items] = -999999.0
            ranked_items = torch.topk(rating, k=max(self.topks))[1].numpy()[0]
            user_prediction_items_list.append(ranked_items)
            user_truth_items_list.append(user_items_dict[user])
        _, _, f1_scores, mrrs, ndcgs = metric.calculate_all(user_truth_items_list,user_prediction_items_list, self.topks)
        return f1_scores, mrrs, ndcgs
