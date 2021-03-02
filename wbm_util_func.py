import numpy as np
import scipy.linalg as linalg
import os
import pickle
import time
import sys
import matplotlib.pyplot as plt
import scipy.stats as stats
import wbm_class

from scipy.spatial.distance import pdist
from datetime import datetime
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from tqdm import *


# def get_similarity_euclidean(model_obj, target_wf_list):
#     time_start = time.time()
#     model_obj.update_dict_sim_val(target_wf_list, sim_method='EUC')
#     time_end = time.time()
#     time_EUC = np.round(time_end - time_start, 3)
#     model_obj.runtime_dict['runtime_EUC'] = time_EUC
#     print(f'runtime_EUC : {time_EUC}')
#
#
# def get_similarity_JSD(model_obj, target_wf_list, n_cg=9, cov_type='real', linkage_method='complete'):
#     model_obj.get_dpgmm()
#     model_obj.get_skldm(linkage_method=linkage_method)
#     model_obj.get_cg(n_cg=n_cg, cov_type=cov_type)
#     time_start = time.time()
#     model_obj.update_dict_sim_val(target_wf_list, sim_method='JSD')
#     time_end = time.time()
#
#     if not model_obj.load_check_dpgmm:
#         time_JSD = np.round(time_end - time_start
#                             + model_obj.runtime_dict['runtime_dpgmm']
#                             + model_obj.runtime_dict['runtime_skldm']
#                             + model_obj.runtime_dict[f'runtime_cg_{n_cg}'], 3)
#         model_obj.runtime_dict[f'runtime_JSD_nCG_{n_cg}'] = time_JSD
#         fname_time = model_obj.wbm_obj.save_folder_runtime + f'runtime_JSD_nCG_{n_cg}_{time_JSD}.csv'
#         np.savetxt(fname_time, np.array([model_obj.time_JSD]), fmt='%1.8f')
#         print(f'runtime_JSD : {time_JSD}')
#
#
# def get_similarity_SKL(model_obj, target_wf_list, n_cg=9, linkage_method='complete'):
#     model_obj.get_dpgmm()
#     model_obj.get_skldm(linkage_method=linkage_method)
#     model_obj.get_cg(n_cg=n_cg)
#     time_start = time.time()
#     model_obj.update_dict_sim_val(target_wf_list, sim_method='SKL')
#     time_end = time.time()
#
#     if not model_obj.load_check_dpgmm:
#         time_SKL = np.round(time_end - time_start
#                             + model_obj.runtime_dict['runtime_dpgmm']
#                             + model_obj.runtime_dict['runtime_skldm']
#                             + model_obj.runtime_dict[f'runtime_cg_{n_cg}'], 3)
#         model_obj.runtime_dict[f'runtime_SKL_nCG_{n_cg}'] = time_SKL
#         fname_time = model_obj.wbm_obj.save_folder_runtime + f'runtime_SKL_nCG_{n_cg}_{time_SKL}.csv'
#         np.savetxt(fname_time, np.array([model_obj.time_SKL]), fmt='%1.8f')
#         print(f'runtime_SKL : {time_SKL}')
#
#
# def get_similarity_WMHD(model_obj, target_wf_list, weight_type='type_2', m=1, s_out_rate=0.1):
#     param_str_key = f'{weight_type}, {m}, {s_out_rate}'
#     model_obj.update_dict_sim_val(target_wf_list,
#                                   sim_method='WMH',
#                                   weight_type=weight_type, m=m, s_out_rate=s_out_rate)
#     if not model_obj.load_check_wmhd:
#         runtime_wmhd = model_obj.runtime_dict['runtime_wmhd_'+param_str_key]
#         fname_time = f'runtime_WMHD_{weight_type}_{m}_{s_out_rate}_{runtime_wmhd}.csv'
#         fname_time = model_obj.wbm_obj.save_folder_runtime + fname_time
#         np.savetxt(fname_time, np.array([runtime_wmhd]))
#         print(f'runtime_WMHD (weight_type: {weight_type} m:{m} s_out_rate: {s_out_rate}) {runtime_wmhd}')


def save_list(list_obj, fname):
    outfile = open(fname, 'wb')
    pickle.dump(list_obj, outfile)
    outfile.close()


def get_now_str(second=True):
    now = datetime.now()
    if second:
        dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    else:
        dt_string = now.strftime("%Y_%m_%d")
    return dt_string


def make_sub_folder(*args, printout=False):
    sub_folder = ''
    for arg in args:
        sub_folder += arg + '/'
    final_dir = os.path.join(os.getcwd(), sub_folder)
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
        if printout:
            print('sub folder has been created: ', final_dir)
    else:
        if printout:
            print('sub folder exists: ', final_dir)
    return final_dir


def KLD(mu1, sigma1, mu2, sigma2, dim=2):
    blk_1 = np.log(linalg.det(sigma2) / linalg.det(sigma1)) - dim
    blk_2 = np.trace(linalg.inv(sigma2) @ sigma1)
    blk_3 = np.transpose(mu2 - mu1) @ linalg.inv(sigma2) @ (mu2 - mu1)
    DKL = 0.5 * (blk_1 + blk_2 + blk_3)
    return DKL


def KLD_sym(mu1, sigma1, mu2, sigma2):
    return KLD(mu1, sigma1, mu2, sigma2, dim=2) + KLD(mu2, sigma2, mu1, sigma1, dim=2)


def get_SKLDM(mean_list, cov_list):
    KL_mtx = np.empty((len(mean_list), len(mean_list)))
    for i in tqdm(range(len(mean_list)), desc='getting KL_mtx...'):
        mu1 = np.array(mean_list[i])
        sigma1 = np.array(cov_list[i])

        for j in range(len(mean_list)):
            mu2 = np.array(mean_list[j])
            sigma2 = np.array(cov_list[j])
            DKL = KLD_sym(mu1, sigma1, mu2, sigma2)
            KL_mtx[i, j] = DKL
    for i in range(len(mean_list)):
        KL_mtx[i, i] = 0
    return KL_mtx


def KLD_cat(p, q):
    return (p * np.log(p / q)).sum()


def get_KLD_cat_mtx(wv):
    data_len = len(wv)
    KLDM_cat = np.zeros((data_len, data_len))

    for i in range(data_len):
        p = wv[i]
        for j in range(data_len):
            q = wv[j]
            KLDM_cat[i][j] = KLD_cat(p, q)
    return KLDM_cat


def sym_KLD_cat(p, q):
    return KLD_cat(p, q) + KLD_cat(q, p)


def get_sym_KLD_cat_mtx(wv):
    data_len = len(wv)
    sKLDM_cat = np.zeros((data_len, data_len))

    for i in range(data_len):
        p = wv[i]
        for j in range(data_len):
            q = wv[j]
            sKLDM_cat[i][j] = sym_KLD_cat(p, q)
    return sKLDM_cat


def JSD(p, q):
    return 0.5 * KLD_cat(p, (0.5 * (p + q))) + 0.5 * KLD_cat(q, (0.5 * (p + q)))


def get_JSD_cat_mtx(wv_arr):
    data_len = len(wv_arr)
    JSD_cat = np.zeros((data_len, data_len))

    for i in range(data_len):
        p = wv_arr[i]
        for j in range(data_len):
            q = wv_arr[j]
            if i == j:
                JSD_cat[i][j] = 0
            else:
                JSD_cat[i][j] = JSD(p, q)
                # print(f'{i:3}_{j:3}, {JSD_cat[i][j]:.4f}')
    return JSD_cat


def compute_serial_matrix(dist_mat, linkage_method, optimal=False):
    """
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    """

    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=linkage_method, optimal_ordering=optimal)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage


def seriation(Z, N, cur_index):
    """
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    """
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return seriation(Z, N, left) + seriation(Z, N, right)


def plot_roc_curve(sim_score_dict, target_wf_list):
    n_target_wf = len(target_wf_list)
    fig, axs = plt.subplots(1, n_target_wf, figsize=(4 * n_target_wf, 4))
    for sim_method in sim_score_dict.keys():
        for para in sim_score_dict[sim_method].keys():
            for i, target_wf in enumerate(target_wf_list):
                tpr = sim_score_dict[sim_method][para]['tpr'][target_wf]
                fpr = sim_score_dict[sim_method][para]['fpr'][target_wf]
                AUC = sim_score_dict[sim_method][para]['auc'][i].round(2)
                roc_label = f'{sim_method}_{AUC}'
                axs[i].plot(fpr, tpr, label=roc_label)
    plt.show()


# OLD Functions : not using
def plot_dpmm_points_single_wafer(dpmm_obj, wbm_obj, wbm_idx, res_folder, n_exp, save=True):
    dpmm_data = dpmm_obj.data
    wbm_data = wbm_obj.wbm_without_nan[wbm_idx]

    fig, axs = plt.subplots(1, 4, figsize=(12, 3), facecolor='none', tight_layout=True)
    fig.suptitle(f'WBM ID : {wbm_idx}')

    axs[0].imshow(wbm_obj.wbm_with_nan[wbm_idx].reshape(22, 44), aspect='auto', interpolation='none')

    axs[1].scatter(wbm_obj.t_758, wbm_obj.r_758, c=wbm_data, marker='.')
    axs[1].scatter(dpmm_data.T[0], dpmm_data.T[1], c='#fff600', marker='.')
    axs[1].set_xlim(wbm_obj.t_range)
    axs[1].set_ylim(wbm_obj.r_range)
    axs[1].set_facecolor('#a9a9a9')  # Dark Gray

    axs[2].scatter(dpmm_data.T[0], dpmm_data.T[1], c=dpmm_obj.idxClusterAssignment, marker='.', cmap='rainbow')
    axs[2].set_facecolor('#a9a9a9')
    axs[2].set_xlim(wbm_obj.t_range)
    axs[2].set_ylim(wbm_obj.r_range)

    for clst, cnt in enumerate(dpmm_obj.cntClusterAssignment):
        if cnt > 2:
            mu = dpmm_obj.paramClusterMu[clst]
            cov = dpmm_obj.paramClusterSigma[clst]
            gridX = np.arange(min(wbm_obj.t_range), max(wbm_obj.t_range), max(wbm_obj.t_range) / 100)
            gridY = np.arange(min(wbm_obj.t_range), max(wbm_obj.t_range), max(wbm_obj.t_range) / 100)
            meshX, meshY = np.meshgrid(gridX, gridY)

            Z = np.zeros(shape=(len(gridY), len(gridX)), dtype=float)
            for itr1 in range(len(meshX)):
                for itr2 in range(len(meshX[itr1])):
                    Z[itr1][itr2] = stats.multivariate_normal.pdf([meshX[itr1][itr2], meshY[itr1][itr2]],
                                                                  mean=mu, cov=cov)
            try:
                CS = axs[2].contour(meshX, meshY, Z, 1, colors='k', linewidths=1)
                axs[2].clabel(CS, inline=1, fontsize=10)
            except:
                continue

    axs[3].plot(dpmm_obj.cntClusters_list_over_iter, 'b.-')

    if save:
        fname = (res_folder + f'dpmm_result_single_waferID_{wbm_idx}_expNo_{n_exp}')
        fig.savefig(fname)
    plt.close(fig)


def plotPoints(data, affiliation, means, covs, delta=1, numLine=3):
    plt.figure()

    types = []
    for i in range(len(affiliation)):
        if affiliation[i] in types:
            pass
        else:
            types.append(affiliation[i])

    colors = ['r', 'g', 'b', 'y', 'k', 'coral', 'magenta', 'pink', 'darkcyan', 'gold']
    totalX = []
    totalY = []
    for j in range(len(types)):
        x = []
        y = []
        for i in range(len(data)):
            if affiliation[i] == types[j]:
                x.append(data[i][0])
                y.append(data[i][1])
                totalX.append(data[i][0])
                totalY.append(data[i][1])
        plt.plot(x, y, color=colors[j % len(colors)], marker='o', linewidth=0)

    gridX = np.arange(min(totalX), max(totalX), delta)
    gridY = np.arange(min(totalY), max(totalY), delta)
    meshX, meshY = np.meshgrid(gridX, gridY)

    for j in range(len(types)):
        Z = np.zeros(shape=(len(gridY), len(gridX)), dtype=float)
        for itr1 in range(len(meshX)):
            for itr2 in range(len(meshX[itr1])):
                Z[itr1][itr2] = stats.multivariate_normal.pdf([meshX[itr1][itr2], meshY[itr1][itr2]], mean=means[j],
                                                              cov=covs[j])
        CS = plt.contour(meshX, meshY, Z, numLine, colors='k')
        plt.clabel(CS, inline=1, fontsize=10)
    #     plt.axis('off')  # LJH
    plt.show()


def plot_dpgmm_xy_tr_example(wbm_id, wbm_obj, contour=True, figsize=(18, 3), save=True, save_format='svg'):
    fig, axs = plt.subplots(1, 6, figsize=figsize)
    dpgmm_xy = wbm_class.DPGMM_VI(wbm_id=wbm_id, wbm_obj=wbm_obj, coordinate='xy')
    dpgmm_tr = wbm_class.DPGMM_VI(wbm_id=wbm_id, wbm_obj=wbm_obj, coordinate='polar')
    dpgmm_xy.fit_dpgmm()
    dpgmm_tr.fit_dpgmm()

    # axs 0 : origianl wbm imshow
    axs[0].imshow(wbm_obj.data_with_nan[wbm_id].reshape(wbm_obj.map_shape), aspect='auto',
                  interpolation='none')
    # axs 1 : defective chips in cartesian coordinates
    axs[1].scatter(dpgmm_xy.data[:, 0], dpgmm_xy.data[:, 1], marker='.', c='b')
    # axs 2 : dpgmm result in cartesian coordinates
    axs[2].scatter(dpgmm_xy.data[:, 0], dpgmm_xy.data[:, 1], marker='.', c=dpgmm_xy.idxClusterAssignment,
                   cmap='rainbow')
    # axs 3 : defective chips in polar coordinates
    axs[3].scatter(dpgmm_tr.data[:, 0], dpgmm_tr.data[:, 1], marker='.', c='b')
    # axs 4 : dpgmm result in polar coordinates
    axs[4].scatter(dpgmm_tr.data[:, 0], dpgmm_tr.data[:, 1], marker='.', c=dpgmm_tr.idxClusterAssignment,
                   cmap='rainbow')
    # axs 5 : dpgmm result in cartesian coordinates
    axs[5].scatter(dpgmm_xy.data[:, 0], dpgmm_xy.data[:, 1], marker='.', c=dpgmm_tr.idxClusterAssignment,
                   cmap='rainbow')

    axs[1].set_xlim(wbm_obj.min_x - wbm_obj.offset_x, wbm_obj.max_x + wbm_obj.offset_x)
    axs[1].set_ylim(wbm_obj.min_y - wbm_obj.offset_y, wbm_obj.max_y + wbm_obj.offset_y)
    axs[2].set_xlim(wbm_obj.min_x - wbm_obj.offset_x, wbm_obj.max_x + wbm_obj.offset_x)
    axs[2].set_ylim(wbm_obj.min_y - wbm_obj.offset_y, wbm_obj.max_y + wbm_obj.offset_y)
    axs[5].set_xlim(wbm_obj.min_x - wbm_obj.offset_x, wbm_obj.max_x + wbm_obj.offset_x)
    axs[5].set_ylim(wbm_obj.min_y - wbm_obj.offset_y, wbm_obj.max_y + wbm_obj.offset_y)

    axs[3].set_xlim(wbm_obj.min_t - wbm_obj.offset_t, wbm_obj.max_t + wbm_obj.offset_t)
    axs[3].set_ylim(wbm_obj.min_r - wbm_obj.offset_r, wbm_obj.max_r + wbm_obj.offset_r)
    axs[4].set_xlim(wbm_obj.min_t - wbm_obj.offset_t, wbm_obj.max_t + wbm_obj.offset_t)
    axs[4].set_ylim(wbm_obj.min_r - wbm_obj.offset_r, wbm_obj.max_r + wbm_obj.offset_r)

    for i in range(6):
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
        axs[i].set_facecolor('#a9a9a9')
    for i, v in enumerate(['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
        axs[i].set_title(v)

    if contour:
        for clst, cnt in enumerate(dpgmm_xy.cntClusterAssignment):
            if cnt > 2:
                mu = dpgmm_xy.paramClusterMu[clst]
                cov = dpgmm_xy.paramClusterSigma[clst]
                gridX = np.arange(wbm_obj.min_x - wbm_obj.offset_x, wbm_obj.max_x + wbm_obj.offset_x,
                                  (wbm_obj.max_x + wbm_obj.offset_x) / 100)
                gridY = np.arange(wbm_obj.min_y - wbm_obj.offset_y, wbm_obj.max_y + wbm_obj.offset_y,
                                  (wbm_obj.max_y + wbm_obj.offset_y) / 100)
                meshX, meshY = np.meshgrid(gridX, gridY)

                Z = np.zeros(shape=(len(gridY), len(gridX)), dtype=float)
                for itr1 in range(len(meshX)):
                    for itr2 in range(len(meshX[itr1])):
                        Z[itr1][itr2] = stats.multivariate_normal.pdf([meshX[itr1][itr2],
                                                                       meshY[itr1][itr2]],
                                                                      mean=mu, cov=cov)
                axs[2].contour(meshX, meshY, Z, 1, colors='k', linewidths=1)

        for clst, cnt in enumerate(dpgmm_tr.cntClusterAssignment):
            if cnt > 2:
                mu = dpgmm_tr.paramClusterMu[clst]
                cov = dpgmm_tr.paramClusterSigma[clst]
                gridX = np.arange(wbm_obj.min_t - wbm_obj.offset_t, wbm_obj.max_t + wbm_obj.offset_t,
                                  (wbm_obj.max_t + wbm_obj.offset_t) / 100)
                gridY = np.arange(wbm_obj.min_r - wbm_obj.offset_r, wbm_obj.max_r + wbm_obj.offset_r,
                                  (wbm_obj.max_r + wbm_obj.offset_r) / 100)
                meshX, meshY = np.meshgrid(gridX, gridY)

                Z = np.zeros(shape=(len(gridY), len(gridX)), dtype=float)
                for itr1 in range(len(meshX)):
                    for itr2 in range(len(meshX[itr1])):
                        Z[itr1][itr2] = stats.multivariate_normal.pdf([meshX[itr1][itr2],
                                                                       meshY[itr1][itr2]],
                                                                      mean=mu, cov=cov)
                axs[4].contour(meshX, meshY, Z, 1, colors='k', linewidths=1)
    if save:
        fname = wbm_obj.save_folder_figures + f'plot_dpgmm_xy_tr_example_wf_{wbm_id}.{save_format}'
        fig.savefig(fname)
    plt.show()


def sampleFromDistribution(dist):
    draw = np.random.rand()
    for itr in range(len(dist) - 1):
        if draw < dist[itr]:
            return itr
    return len(dist) - 1


def get_overlap_idx(data):
    data_dist = squareform(pdist(data))
    data_len = len(data_dist)
    if (data_dist.flatten() == 0).sum() == data_len:
        return np.ones(data_len) == 1
    else:
        overlap_replace_value = np.max(data_dist) + 100

        for i in range(len(data_dist)):
            data_dist[i, i] = 1
        for i in range(len(data_dist)):
            for j in range(len(data_dist)):
                if data_dist[i, j] == 0:
                    data_dist[i, j] = overlap_replace_value
        same_arr = []
        triu = np.triu(data_dist)
        for i in range(len(data_dist)):
            for j in range(len(data_dist)):
                if triu[i, j] == overlap_replace_value:
                    same_arr.append([i, j])
        overlap_idx_arr = np.vstack(same_arr).T[1]
        overlap_idx_arr_bool = np.ones(len(data_dist)) == 1
        overlap_idx_arr_bool[overlap_idx_arr] = False
        return overlap_idx_arr_bool
