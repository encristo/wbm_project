import math
import json

from wbm_util_func import *
from PIL import Image
from collections import Counter
from tqdm import *
from sklearn.mixture import BayesianGaussianMixture
from scipy.cluster.hierarchy import cut_tree, dendrogram
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean, squareform, cdist, pdist, cityblock
from sklearn.metrics import confusion_matrix, auc
from sklearn.covariance import empirical_covariance
from numpy.random import multivariate_normal as mvn
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

color_list_gray_red = ["lightgray", "red"]
cmap_gray_red = ListedColormap(color_list_gray_red)
cmap_gray_red_seg = LinearSegmentedColormap.from_list("mycmap", color_list_gray_red)


class WM811K_DATASET:
    def __init__(self, verbose=False):
        data_dir = 'D:/ML/other_works/WaPIRL-main/data/wm811k/labeled/train'
        self.map_type_list = os.listdir(data_dir)
        self.glob_shape_valid_list = []
        self.glob_shape_list = []
        self.glob_valid_list = []
        self.glob_map_type_list = []
        self.glob_map_label_list = []
        self.glob_fail_rate_list = []
        self.map_type_dict = {}

        for i, map_type in enumerate(self.map_type_list):
            png_list = os.listdir(os.path.join(data_dir, map_type))
            shape_list = []

            for file in tqdm(png_list, desc=f'{map_type}'):
                png_path = os.path.join(data_dir, map_type, file)
                image = np.array(Image.open(png_path))
                image_valid = (image != 0).sum()
                self.glob_fail_rate_list.append(np.round((image == 255).sum() / image_valid, 4))
                self.glob_shape_list.append(image.shape)
                self.glob_valid_list.append(image_valid)
                self.glob_map_type_list.append(map_type)
                self.glob_map_label_list.append(i)
                shape_valid = (image.shape[0], image.shape[1], image_valid)
                shape_list.append(shape_valid)
                self.glob_shape_valid_list.append(shape_valid)

                if verbose:
                    print(f'{map_type} / {file} / {shape_valid}')
            self.map_type_dict[map_type] = Counter(shape_list).most_common()

        self.glob_shape_counter = Counter(self.glob_shape_valid_list).most_common()

        self.map_type_list = np.array(self.map_type_list)
        self.glob_shape_valid_list = np.array(self.glob_shape_valid_list)
        self.glob_shape_list = np.array(self.glob_shape_list)
        self.glob_valid_list = np.array(self.glob_valid_list)
        self.glob_map_type_list = np.array(self.glob_map_type_list)
        self.glob_map_label_list = np.array(self.glob_map_label_list)
        self.glob_fail_rate_list = np.array(self.glob_fail_rate_list)

        self.glob_label_count_list = np.zeros_like(self.glob_map_label_list)
        for i, v in enumerate(self.glob_shape_counter):
            shape_valid_mask = ((self.glob_shape_valid_list == v[0]).sum(axis=1) == 3).flatten()
            local_label_list = self.glob_map_label_list[shape_valid_mask]
            for label in np.unique(local_label_list):
                label_mask = self.glob_map_label_list == label
                shape_valid_label_mask = shape_valid_mask * label_mask == 1
                label_count = shape_valid_label_mask.sum()
                self.glob_label_count_list[shape_valid_label_mask] = label_count


class WM811K:
    def __init__(self, wm811k_dataset_obj, map_shape, n_valid, label_count_lim=10, map_type_exclude='none',
                 verbose=True):
        data_dir = 'D:/ML/other_works/WaPIRL-main/data/wm811k/labeled/train'
        self.map_shape = map_shape
        self.n_valid = n_valid
        self.label_count_lim = label_count_lim
        self.map_type_list = wm811k_dataset_obj.map_type_list  # center, donut, edge-loc, edge-ring, loc, near-full, none, random, scratch 9개
        self.seq = np.arange(len(wm811k_dataset_obj.glob_fail_rate_list))
        self.ex_mask = np.ones((len(wm811k_dataset_obj.glob_fail_rate_list)))
        for ex_map_type in map_type_exclude:
            ex_label = np.argmax(wm811k_dataset_obj.map_type_list == ex_map_type)
            boolean_arr = wm811k_dataset_obj.glob_map_label_list != ex_label
            self.ex_mask = self.ex_mask * boolean_arr
        self.ex_mask *= ((wm811k_dataset_obj.glob_shape_list == map_shape).sum(axis=1) == 2)
        self.ex_mask *= wm811k_dataset_obj.glob_valid_list == n_valid
        self.ex_mask *= wm811k_dataset_obj.glob_label_count_list > label_count_lim
        self.ex_mask = self.ex_mask == 1
        self.seq_mask = self.seq[self.ex_mask]

        self.data = []
        glob_counter = 0
        match_counter = 0
        match_seq = self.seq_mask[match_counter]

        for i, map_type in enumerate(self.map_type_list):
            png_list = os.listdir(os.path.join(data_dir, map_type))

            for file in png_list:
                if glob_counter == match_seq:
                    png_path = os.path.join(data_dir, map_type, file)
                    image = np.array(Image.open(png_path))
                    self.data.append(image.flatten())
                    if verbose:
                        print(
                            f'map_type:{map_type}, glob_counter:{glob_counter}, match_counter:{match_counter}, match_seq:{self.seq_mask[match_counter]}')
                    if match_counter != len(self.seq_mask) - 1:
                        match_counter += 1
                        match_seq = self.seq_mask[match_counter]
                glob_counter += 1

        self.data = np.array(self.data)
        self.data[self.data == 127] = 1
        self.data[self.data == 255] = 2
        self.data_len = len(self.data)
        self.fail_rate_list = wm811k_dataset_obj.glob_fail_rate_list[self.seq_mask]
        self.label_list = wm811k_dataset_obj.glob_map_label_list[self.seq_mask]

        self.label_count_dict = {k: (self.label_list == k).sum() for k in range(len(self.map_type_list))}

    def plot_failrate_boxplot(self):
        fail_rate_arr = [self.fail_rate_list[self.label_list == i] for i in np.unique(self.label_list)]
        map_type_label = self.map_type_list[np.unique(self.label_list)]
        plt.figure(figsize=(8, 5))
        plt.boxplot(fail_rate_arr, labels=map_type_label)
        plt.ylim(0, 1)
        plt.show()

    def plot_sample_9_imgs(self):
        fig, axs = plt.subplots(1, 9, figsize=(20, 2))
        fig.suptitle(self.map_shape, y=1.1)
        for i in range(9):
            axs[i].axis('off')
        for i, label_count in enumerate(self.label_count_dict.values()):
            axs[i].set_title(self.map_type_list[i])
            if label_count != 0:
                sample_id = int(np.random.choice(np.arange(self.data_len)[self.label_list == i], 1))
                axs[i].imshow(self.data[sample_id].reshape(self.map_shape), aspect='auto', interpolation='none')

        plt.show()

    def plot_avg_map(self):
        fig, axs = plt.subplots(1, 9, figsize=(20, 2))
        fig.suptitle(self.map_shape, y=1.1)
        for i, label_count in enumerate(self.label_count_dict.values()):
            axs[i].set_title(self.map_type_list[i])
            axs[i].axis('off')
            if label_count != 0:
                avg_map = np.array(self.data[self.label_list == i], dtype='float')
                avg_map[avg_map == 0] = np.nan
                avg_map[avg_map == 1] = 0
                avg_map[avg_map == 2] = 1
                axs[i].imshow(avg_map.mean(axis=0).reshape(self.map_shape), aspect='auto', interpolation='none')
        plt.show()

    def plot_sample_9xn_imgs(self):
        n_cols = self.label_count_lim - 1
        fig, axs = plt.subplots(9, n_cols, figsize=(20, 20))
        for i in range(9):
            for j in range(n_cols):
                axs[i, j].axis('off')
        for i, label_count in enumerate(self.label_count_dict.values()):
            axs[i, 0].set_title(self.map_type_list[i])
            if label_count != 0:
                sample_id_arr = np.random.choice(np.arange(self.data_len)[self.label_list == i], n_cols, replace=False)
                for j, sample_id in enumerate(sample_id_arr):
                    axs[i][j].imshow(self.data[sample_id].reshape(self.map_shape), aspect='auto', interpolation='none')
        plt.show()

    def plot_piechart(self):
        fig, axs = plt.subplots(figsize=(6, 6))
        axs.pie(self.label_count_dict.values(), labels=self.map_type_list, autopct='%1.1f%%', shadow=True,
                startangle=90)
        axs.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()


class DPGMM_MC:

    def __init__(self, wbm_id, wbm_obj, coordinate='polar'):
        self.wbm_id = wbm_id
        self.wbm_obj = wbm_obj
        self.target_label = self.wbm_obj.label_list[self.wbm_id]
        if coordinate == 'polar':
            self.data = self.wbm_obj.get_polar_points(self.wbm_id)
        else:
            self.data = self.wbm_obj.get_xy_points(self.wbm_id)
        self.data_xy = self.wbm_obj.get_xy_points(self.wbm_id)

        self.cntData = len(self.data)
        self.max_t = self.wbm_obj.max_t
        self.max_r = self.wbm_obj.max_r
        self.max_x = self.wbm_obj.max_x
        self.max_y = self.wbm_obj.max_y
        self.min_t = self.wbm_obj.min_t
        self.min_r = self.wbm_obj.min_r
        self.min_x = self.wbm_obj.min_x
        self.min_y = self.wbm_obj.min_y
        self.offset_t = self.wbm_obj.offset_t
        self.offset_r = self.wbm_obj.offset_r
        self.offset_x = self.wbm_obj.offset_x
        self.offset_y = self.wbm_obj.offset_y

        self.cntClusters_list_over_iter = []
        self.setClusterAssignment = []  # 각 cluster 의 assign 된 data 의 index 를 저장
        self.paramClusterSigma = []  # 각 cluster 의 sigma parameter 를 저장
        self.paramClusterMu = []  # 각 cluster 의 mu parameter 를 저장
        self.cntClusterAssignment = []  # 각 cluster 에 assign 된 data 개수를 counting

    def cluster(self,
                gamma=0.001,
                itr=10,
                cov_1=1,
                cov_2=20,
                visualize=False,
                printOut=False,
                tqdm_on=False,
                save_plot=False,
                truncate=sys.maxsize):
        self.save_folder_dpgmm_mc = make_sub_folder(self.wbm_obj.save_folder_figures,
                                                    'dpgmm_mc', f'gamma_{gamma}', f'wbm_id_{self.wbm_id}')
        self.gamma = gamma  # gamma 값을 저장
        self.cntClusters = 0  # cluster 의 개수
        self.idxClusterAssignment = np.zeros(self.cntData, dtype=int) - 1  # 각 data 의 assign 된 cluster index 를 저장 및 초기화

        if tqdm_on:
            itr_range = trange(itr)
        else:
            itr_range = range(itr)

        for i in itr_range:  # while sampling iteration
            for idx in range(self.cntData):  # while each data instance in the dataset
                # instance 의 현재 assignment 지워주기, 맨처음 iteration 에서는 건너 뜀. (cluster idx = -1)
                if self.idxClusterAssignment[idx] != -1:
                    self.cntClusterAssignment[self.idxClusterAssignment[idx]] = \
                        self.cntClusterAssignment[self.idxClusterAssignment[idx]] - 1
                    self.setClusterAssignment[self.idxClusterAssignment[idx]].remove(idx)
                    if self.cntClusterAssignment[self.idxClusterAssignment[idx]] == 0:  # 빈 cluster 일 경우 cluster 제거
                        self.removeEmptyCluster(self.idxClusterAssignment[idx])
                    self.idxClusterAssignment[idx] = -1

                # prior 값 계산
                normalize = 0.0
                prior = []
                for itrCluster in range(self.cntClusters):  # 현재 cluster 갯수만큼 iteration 진행
                    prior.append(self.cntClusterAssignment[itrCluster])
                    normalize += self.cntClusterAssignment[itrCluster]
                if self.cntClusters < truncate:  # cluster 개수의 상계(upper limit)를 초과하지 않은 경우
                    prior.append(self.gamma)
                    normalize += self.gamma
                for itrCluster in range(len(prior)):
                    prior[itrCluster] /= normalize

                # posterior 값 계산 과정. posterior = prior * likelihood
                instance = self.data[idx]
                posterior = []
                for itrCluster in range(self.cntClusters):
                    likelihood = stats.multivariate_normal(self.paramClusterMu[itrCluster],
                                                           self.paramClusterSigma[itrCluster]).pdf(instance)

                    posterior.append(prior[itrCluster] * likelihood)
                if self.cntClusters < truncate:
                    posterior.append(prior[len(prior) - 1] * 1.0)
                normalize = 0.0
                for itrCluster in range(len(posterior)):
                    normalize = normalize + posterior[itrCluster]
                for itrCluster in range(len(posterior)):
                    posterior[itrCluster] = posterior[itrCluster] / normalize  # 위에서 계산한 값으로 normalizing
                idxSampledCluster = sampleFromDistribution(posterior)

                # parameter mu, sigma 업데이트
                if idxSampledCluster != self.cntClusters:  # 현재 instance 가 기존 cluster 로 assign 되었을 때
                    self.idxClusterAssignment[idx] = int(idxSampledCluster)
                    self.setClusterAssignment[idxSampledCluster].append(idx)
                    self.cntClusterAssignment[idxSampledCluster] = self.cntClusterAssignment[idxSampledCluster] + 1
                    dataComponent = np.ndarray(
                        shape=(len(self.setClusterAssignment[idxSampledCluster]), len(self.data[0])),
                        dtype=np.float32)
                    for idxComponentSample in range(len(self.setClusterAssignment[idxSampledCluster])):
                        dataComponentInstance = self.data[
                            self.setClusterAssignment[idxSampledCluster][idxComponentSample]]
                        for idxDimension in range(len(dataComponentInstance)):
                            dataComponent[idxComponentSample][idxDimension] = dataComponentInstance[idxDimension]
                    self.paramClusterMu[idxSampledCluster] = np.mean(dataComponent, axis=0).tolist()
                    self.paramClusterSigma[idxSampledCluster] = (
                            np.cov(dataComponent.T) + np.identity(len(instance)) * cov_1 /
                            self.cntClusterAssignment[idxSampledCluster]).tolist()
                else:  # 현재 instance 가 새 cluster 로 assign 되었을 때
                    self.idxClusterAssignment[idx] = int(idxSampledCluster)
                    self.cntClusters = self.cntClusters + 1
                    self.setClusterAssignment.append([idx])
                    self.cntClusterAssignment.append(1)
                    self.paramClusterMu.append([])
                    self.paramClusterSigma.append([])
                    self.paramClusterMu[idxSampledCluster] = instance.tolist()
                    self.paramClusterSigma[idxSampledCluster] = (np.identity(len(instance)) * cov_2).tolist()

            self.cntClusters_list_over_iter.append(self.cntClusters)
            # visualization
            if printOut:
                print('#####################################################')
                print('Iteration ', i + 1)
                print('#####################################################')
                self.printOut()
            if visualize:
                self.plot_iter(itr=i, save=save_plot)
                # plotPoints(self.data, self.idxClusterAssignment, self.paramClusterMu, self.paramClusterSigma, numLine=1)
        self.paramClusterMu = np.vstack(self.paramClusterMu).reshape(-1, 2)
        self.paramClusterSigma = np.vstack(self.paramClusterSigma).reshape((-1, 2, 2))

    # 빈 클러스터를 제거하는 과정
    def removeEmptyCluster(self, idxEmptyCluster):
        idxEndCluster = self.cntClusters - 1
        for itrClusterSample in range(len(self.setClusterAssignment[idxEndCluster])):
            self.idxClusterAssignment[self.setClusterAssignment[idxEndCluster][itrClusterSample]] = idxEmptyCluster
        self.setClusterAssignment[idxEmptyCluster] = self.setClusterAssignment[idxEndCluster]
        self.cntClusterAssignment[idxEmptyCluster] = self.cntClusterAssignment[idxEndCluster]
        self.paramClusterMu[idxEmptyCluster] = self.paramClusterMu[idxEndCluster]
        self.paramClusterSigma[idxEmptyCluster] = self.paramClusterSigma[idxEndCluster]

        self.setClusterAssignment.pop(idxEndCluster)
        self.cntClusterAssignment.pop(idxEndCluster)
        self.paramClusterSigma.pop(idxEndCluster)
        self.paramClusterMu.pop(idxEndCluster)
        self.cntClusters = self.cntClusters - 1

    def printOut(self):
        func = lambda x: round(x, 2)
        mu = [list(map(func, i)) for i in self.paramClusterMu]
        sigma = [[list(map(func, i)) for i in j] for j in self.paramClusterSigma]
        # print ("Data : ",self.data)
        print("Cluster Assignment : ", self.idxClusterAssignment)
        print("Cluster Set : ", self.setClusterAssignment)
        print("Cluster Assignment Count : ", self.cntClusterAssignment)
        print("Cluster Num : ", self.cntClusters)
        print("Cluster Mu : ", mu)
        print("Cluster Sigma : ", sigma)
        print("Cluster list over iter : ", self.cntClusters_list_over_iter)

    def plot(self, contour=True, title=True, save=False, show=True, save_format='pdf', figsize=(18, 3)):
        fig, axs = plt.subplots(1, 6, figsize=figsize)
        axs[0].imshow(self.wbm_obj.data_with_nan[self.wbm_id].reshape(self.wbm_obj.map_shape), aspect='auto',
                      interpolation='none')
        axs[1].scatter(self.data_xy[:, 0], self.data_xy[:, 1], marker='.', c='b')
        axs[2].scatter(self.data[:, 0], self.data[:, 1], marker='.', c='b')
        axs[3].scatter(self.data[:, 0], self.data[:, 1], c=self.idxClusterAssignment, marker='.', cmap='rainbow')
        axs[3].scatter(self.paramClusterMu.T[0], self.paramClusterMu.T[1], c='w', marker='*')
        axs[4].scatter(self.data_xy[:, 0], self.data_xy[:, 1], c=self.idxClusterAssignment, marker='.', cmap='rainbow')

        axs[1].set_xlim(self.min_x - self.wbm_obj.offset_x, self.max_x + self.wbm_obj.offset_x)
        axs[1].set_ylim(self.min_y - self.wbm_obj.offset_y, self.max_y + self.wbm_obj.offset_y)
        axs[4].set_xlim(self.min_x - self.wbm_obj.offset_x, self.max_x + self.wbm_obj.offset_x)
        axs[4].set_ylim(self.min_y - self.wbm_obj.offset_y, self.max_y + self.wbm_obj.offset_y)

        axs[2].set_xlim(self.min_t - self.wbm_obj.offset_t, self.max_t + self.wbm_obj.offset_t)
        axs[2].set_ylim(self.min_r - self.wbm_obj.offset_r, self.max_r + self.wbm_obj.offset_r)
        axs[3].set_xlim(self.min_t - self.wbm_obj.offset_t, self.max_t + self.wbm_obj.offset_t)
        axs[3].set_ylim(self.min_r - self.wbm_obj.offset_r, self.max_r + self.wbm_obj.offset_r)

        axs[5].plot(self.cntClusters_list_over_iter, 'bo-')

        if title:
            axs[0].set_title(f'{self.wbm_obj.label_name_org[self.target_label]}')
            axs[5].set_title(f'last cntClusters{self.cntClusters_list_over_iter[-1]}')

        for i in range(5):
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
            axs[i].set_facecolor('#a9a9a9')

        if contour:
            for clst, cnt in enumerate(self.cntClusterAssignment):
                if cnt > 2:
                    mu = self.paramClusterMu[clst]
                    cov = self.paramClusterSigma[clst]
                    gridX = np.arange(0 - self.offset_t, self.max_t + self.offset_t, (self.max_t + self.offset_t) / 100)
                    gridY = np.arange(0 - self.offset_r, self.max_r + self.offset_r, (self.max_r + self.offset_r) / 100)
                    meshX, meshY = np.meshgrid(gridX, gridY)

                    Z = np.zeros(shape=(len(gridY), len(gridX)), dtype=float)
                    for itr1 in range(len(meshX)):
                        for itr2 in range(len(meshX[itr1])):
                            Z[itr1][itr2] = stats.multivariate_normal.pdf([meshX[itr1][itr2],
                                                                           meshY[itr1][itr2]],
                                                                          mean=mu, cov=cov)
                    axs[3].contour(meshX, meshY, Z, 1, colors='k', linewidths=1)
        if save:
            fname = self.wbm_obj.save_folder_figures + f'dpgmm_mc_plot_{self.wbm_id}.{save_format}'
            fig.savefig(fname)
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_iter(self, itr, contour=True, save=False, save_format='png', figsize=(9, 3)):
        para_Mu = np.vstack(self.paramClusterMu).reshape(-1, 2)
        para_Sigma = np.vstack(self.paramClusterSigma).reshape((-1, 2, 2))
        fig, axs = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f'wbm id={self.wbm_id}, gamma={self.gamma}, # of iter={itr}, cntClusters={self.cntClusters}')
        axs[0].scatter(self.data[:, 0], self.data[:, 1], c=self.idxClusterAssignment, marker='.', cmap='rainbow')
        axs[0].scatter(para_Mu.T[0], para_Mu.T[1], c='w', marker='*')
        axs[0].set_xlim(self.min_t - self.wbm_obj.offset_t, self.max_t + self.wbm_obj.offset_t)
        axs[0].set_ylim(self.min_r - self.wbm_obj.offset_r, self.max_r + self.wbm_obj.offset_r)

        axs[1].scatter(self.data_xy[:, 0], self.data_xy[:, 1], c=self.idxClusterAssignment, marker='.', cmap='rainbow')
        axs[1].set_xlim(self.min_x - self.wbm_obj.offset_x, self.max_x + self.wbm_obj.offset_x)
        axs[1].set_ylim(self.min_y - self.wbm_obj.offset_y, self.max_y + self.wbm_obj.offset_y)

        for i in range(2):
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
            axs[i].set_facecolor('#a9a9a9')

        axs[2].plot(self.cntClusters_list_over_iter, 'bo-')

        if contour:
            for clst, cnt in enumerate(self.cntClusterAssignment):
                if cnt > 2:
                    mu = para_Mu[clst]
                    cov = para_Sigma[clst]
                    gridX = np.arange(0 - self.offset_t, self.max_t + self.offset_t, (self.max_t + self.offset_t) / 100)
                    gridY = np.arange(0 - self.offset_r, self.max_r + self.offset_r, (self.max_r + self.offset_r) / 100)
                    meshX, meshY = np.meshgrid(gridX, gridY)

                    Z = np.zeros(shape=(len(gridY), len(gridX)), dtype=float)
                    for itr1 in range(len(meshX)):
                        for itr2 in range(len(meshX[itr1])):
                            Z[itr1][itr2] = stats.multivariate_normal.pdf([meshX[itr1][itr2],
                                                                           meshY[itr1][itr2]],
                                                                          mean=mu, cov=cov)
                    axs[0].contour(meshX, meshY, Z, 1, colors='k', linewidths=1)
        if save:
            fname = self.save_folder_dpgmm_mc + f'dpgmm_mc_plot_iter_{self.wbm_id}_iter_{itr}.{save_format}'
            fig.savefig(fname)
        plt.show()

    def plot_polar_alone(self, contour=False, figsize=(4, 4), save=True, save_format='pdf'):

        fig, axs = plt.subplots(figsize=figsize)
        axs.scatter(self.data[:, 0], self.data[:, 1], c=self.idxClusterAssignment, marker='.', cmap='rainbow')
        axs.scatter(self.paramClusterMu.T[0], self.paramClusterMu.T[1], c='w', marker='*')
        axs.set_xlim(self.min_t - self.offset_t, self.max_t + self.offset_t)
        axs.set_ylim(self.min_r - self.offset_r, self.max_r + self.offset_r)

        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        axs.set_facecolor('#a9a9a9')

        if contour:
            for clst, cnt in enumerate(self.cntClusterAssignment):
                if cnt > 2:
                    mu = self.paramClusterMu[clst]
                    cov = self.paramClusterSigma[clst]
                    gridX = np.arange(0 - self.offset_t, self.max_t + self.offset_t, (self.max_t + self.offset_t) / 100)
                    gridY = np.arange(0 - self.offset_r, self.max_r + self.offset_r, (self.max_r + self.offset_r) / 100)
                    meshX, meshY = np.meshgrid(gridX, gridY)

                    Z = np.zeros(shape=(len(gridY), len(gridX)), dtype=float)
                    for itr1 in range(len(meshX)):
                        for itr2 in range(len(meshX[itr1])):
                            Z[itr1][itr2] = stats.multivariate_normal.pdf([meshX[itr1][itr2],
                                                                           meshY[itr1][itr2]],
                                                                          mean=mu, cov=cov)
                    axs.contour(meshX, meshY, Z, 1, colors='k', linewidths=1)
        if save:
            fname = self.wbm_obj.save_folder_figures + f'dpgmm_result_polar_alone{self.wbm_id}.{save_format}'
            fig.savefig(fname)
        plt.show()
        return axs

    def plot_xy_alone(self, figsize=(4, 4)):

        fig, axs = plt.subplots(figsize=figsize)

        axs.scatter(self.data_xy[:, 0], self.data_xy[:, 1], c=self.idxClusterAssignment, marker='.', cmap='rainbow')
        axs.set_xlim(self.min_x - self.offset_x, self.max_x + self.offset_x)
        axs.set_ylim(self.min_y - self.offset_y, self.max_y + self.offset_y)

        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        axs.set_facecolor('#a9a9a9')

        plt.show()

    def plot_polar_and_xy(self, figsize=(4, 4), contour=True):

        fig, axs = plt.subplots(figsize=figsize)

        axs.scatter(self.data_xy[:, 0], self.data_xy[:, 1], c=self.idxClusterAssignment, marker='.', cmap='rainbow')
        axs.set_xlim(self.min_x - self.offset_x, self.max_x + self.offset_x)
        axs.set_ylim(self.min_y - self.offset_y, self.max_y + self.offset_y)

        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        axs.set_facecolor('#a9a9a9')

        fig, axs = plt.subplots(figsize=figsize)
        axs.scatter(self.data[:, 0], self.data[:, 1], c=self.idxClusterAssignment, marker='.', cmap='rainbow')
        axs.scatter(self.paramClusterMu.T[0], self.paramClusterMu.T[1], c='w', marker='*')
        axs.set_xlim(self.min_t - self.offset_t, self.max_t + self.offset_t)
        axs.set_ylim(self.min_r - self.offset_r, self.max_r + self.offset_r)

        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        axs.set_facecolor('#a9a9a9')

        if contour:
            for clst, cnt in enumerate(self.cntClusterAssignment):
                if cnt > 2:
                    mu = self.paramClusterMu[clst]
                    cov = self.paramClusterSigma[clst]
                    gridX = np.arange(0 - self.offset_t, self.max_t + self.offset_t, (self.max_t + self.offset_t) / 100)
                    gridY = np.arange(0 - self.offset_r, self.max_r + self.offset_r, (self.max_r + self.offset_r) / 100)
                    meshX, meshY = np.meshgrid(gridX, gridY)

                    Z = np.zeros(shape=(len(gridY), len(gridX)), dtype=float)
                    for itr1 in range(len(meshX)):
                        for itr2 in range(len(meshX[itr1])):
                            Z[itr1][itr2] = stats.multivariate_normal.pdf([meshX[itr1][itr2],
                                                                           meshY[itr1][itr2]],
                                                                          mean=mu, cov=cov)
                    axs.contour(meshX, meshY, Z, 1, colors='k', linewidths=1)


class DPGMM_VI:
    def __init__(self, wbm_id, wbm_obj, coordinate='polar'):
        self.wbm_id = wbm_id
        self.wbm_obj = wbm_obj
        self.target_label = self.wbm_obj.label_list[self.wbm_id]

        if coordinate == 'polar':
            self.data = self.wbm_obj.get_polar_points(self.wbm_id)
        else:
            self.data = self.wbm_obj.get_xy_points(self.wbm_id)
        self.data_xy = self.wbm_obj.get_xy_points(self.wbm_id)

        self.cntData = len(self.data)
        self.max_t = self.wbm_obj.max_t
        self.max_r = self.wbm_obj.max_r
        self.max_x = self.wbm_obj.max_x
        self.max_y = self.wbm_obj.max_y
        self.min_t = self.wbm_obj.min_t
        self.min_r = self.wbm_obj.min_r
        self.min_x = self.wbm_obj.min_x
        self.min_y = self.wbm_obj.min_y
        self.offset_t = self.wbm_obj.offset_t
        self.offset_r = self.wbm_obj.offset_r
        self.offset_x = self.wbm_obj.offset_x
        self.offset_y = self.wbm_obj.offset_y

        if self.cntData >= 10:
            self.n_components = 10
        elif self.cntData <= 3:
            self.n_components = 1
        else:
            self.n_components = self.cntData

    def fit_dpgmm(self,
                  covariance_type='full',  # default 'full'
                  tol=1e-3,  # default 1e-3
                  reg_covar=1e-6,  # default 1e-6
                  max_iter=500,  # default 100
                  n_init=5,  # default 1
                  init_params='random',  # default 'kmeans'
                  weight_concentration_prior_type='dirichlet_process',  # default 'dirichlet_process'
                  weight_concentration_prior=0.1,  # gamma
                  mean_precision_prior=0.01,  # default 1, must be greater than 0.
                  covariance_prior=None,  # default None
                  random_state=3,  # default None
                  verbose=False,
                  verbose_interval=10):

        self.dpgmm = BayesianGaussianMixture(n_components=self.n_components,
                                             covariance_type=covariance_type,
                                             tol=tol,
                                             reg_covar=reg_covar,
                                             max_iter=max_iter,
                                             n_init=n_init,
                                             init_params=init_params,
                                             weight_concentration_prior_type=weight_concentration_prior_type,
                                             weight_concentration_prior=weight_concentration_prior,
                                             mean_precision_prior=mean_precision_prior,
                                             covariance_prior=covariance_prior,
                                             random_state=random_state,
                                             verbose=verbose,
                                             verbose_interval=verbose_interval)

        self.idxClusterAssignment = self.dpgmm.fit_predict(self.data)
        self.idxClusters = np.unique(self.idxClusterAssignment)
        self.cntClusters = len(self.idxClusters)
        self.paramClusterMu = self.dpgmm.means_[self.idxClusters]
        self.paramClusterSigma = self.dpgmm.covariances_[self.idxClusters]
        self.cntClusterAssignment = []

        for i in range(self.cntClusters):
            selected_cluster = self.idxClusters[i]
            self.cntClusterAssignment.append(sum(self.idxClusterAssignment == selected_cluster))

    def plot(self, contour=False, title=True, save=False, save_format='pdf', figsize=(15, 3)):
        fig, axs = plt.subplots(1, 5, figsize=figsize)
        axs[0].imshow(self.wbm_obj.data_with_nan[self.wbm_id].reshape(self.wbm_obj.map_shape), aspect='auto',
                      interpolation='none')
        axs[1].scatter(self.data_xy[:, 0], self.data_xy[:, 1], marker='.', c='b')
        axs[2].scatter(self.data[:, 0], self.data[:, 1], marker='.', c='b')
        axs[3].scatter(self.data[:, 0], self.data[:, 1], c=self.idxClusterAssignment, marker='.', cmap='rainbow')
        axs[3].scatter(self.paramClusterMu.T[0], self.paramClusterMu.T[1], c='w', marker='*')
        axs[4].scatter(self.data_xy[:, 0], self.data_xy[:, 1], c=self.idxClusterAssignment, marker='.', cmap='rainbow')

        axs[1].set_xlim(self.min_x - self.wbm_obj.offset_x, self.max_x + self.wbm_obj.offset_x)
        axs[1].set_ylim(self.min_y - self.wbm_obj.offset_y, self.max_y + self.wbm_obj.offset_y)
        axs[4].set_xlim(self.min_x - self.wbm_obj.offset_x, self.max_x + self.wbm_obj.offset_x)
        axs[4].set_ylim(self.min_y - self.wbm_obj.offset_y, self.max_y + self.wbm_obj.offset_y)

        axs[2].set_xlim(self.min_t - self.wbm_obj.offset_t, self.max_t + self.wbm_obj.offset_t)
        axs[2].set_ylim(self.min_r - self.wbm_obj.offset_r, self.max_r + self.wbm_obj.offset_r)
        axs[3].set_xlim(self.min_t - self.wbm_obj.offset_t, self.max_t + self.wbm_obj.offset_t)
        axs[3].set_ylim(self.min_r - self.wbm_obj.offset_r, self.max_r + self.wbm_obj.offset_r)

        if title:
            axs[0].set_title(f'{self.wbm_obj.label_name_org[self.target_label]}')

        for i in range(5):
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
            axs[i].set_facecolor('#a9a9a9')

        if contour:
            for clst, cnt in enumerate(self.cntClusterAssignment):
                if cnt > 2:
                    mu = self.paramClusterMu[clst]
                    cov = self.paramClusterSigma[clst]
                    gridX = np.arange(0 - self.offset_t, self.max_t + self.offset_t, (self.max_t + self.offset_t) / 100)
                    gridY = np.arange(0 - self.offset_r, self.max_r + self.offset_r, (self.max_r + self.offset_r) / 100)
                    meshX, meshY = np.meshgrid(gridX, gridY)

                    Z = np.zeros(shape=(len(gridY), len(gridX)), dtype=float)
                    for itr1 in range(len(meshX)):
                        for itr2 in range(len(meshX[itr1])):
                            Z[itr1][itr2] = stats.multivariate_normal.pdf([meshX[itr1][itr2],
                                                                           meshY[itr1][itr2]],
                                                                          mean=mu, cov=cov)
                    axs[3].contour(meshX, meshY, Z, 1, colors='k', linewidths=1)
        if save:
            fname = self.wbm_obj.save_folder_figures + f'dpgmm_result_plot_{self.wbm_id}.{save_format}'
            fig.savefig(fname)
        plt.show()

    def plot_polar_alone(self, contour=False, figsize=(4, 4), save=True, save_format='pdf'):

        fig, axs = plt.subplots(figsize=figsize)
        axs.scatter(self.data[:, 0], self.data[:, 1], c=self.idxClusterAssignment, marker='.', cmap='rainbow')
        axs.scatter(self.paramClusterMu.T[0], self.paramClusterMu.T[1], c='w', marker='*')
        axs.set_xlim(self.min_t - self.offset_t, self.max_t + self.offset_t)
        axs.set_ylim(self.min_r - self.offset_r, self.max_r + self.offset_r)

        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        axs.set_facecolor('#a9a9a9')

        if contour:
            for clst, cnt in enumerate(self.cntClusterAssignment):
                if cnt > 2:
                    mu = self.paramClusterMu[clst]
                    cov = self.paramClusterSigma[clst]
                    gridX = np.arange(0 - self.offset_t, self.max_t + self.offset_t, (self.max_t + self.offset_t) / 100)
                    gridY = np.arange(0 - self.offset_r, self.max_r + self.offset_r, (self.max_r + self.offset_r) / 100)
                    meshX, meshY = np.meshgrid(gridX, gridY)

                    Z = np.zeros(shape=(len(gridY), len(gridX)), dtype=float)
                    for itr1 in range(len(meshX)):
                        for itr2 in range(len(meshX[itr1])):
                            Z[itr1][itr2] = stats.multivariate_normal.pdf([meshX[itr1][itr2],
                                                                           meshY[itr1][itr2]],
                                                                          mean=mu, cov=cov)
                    axs.contour(meshX, meshY, Z, 1, colors='k', linewidths=1)
        if save:
            fname = self.wbm_obj.save_folder_figures + f'dpgmm_result_polar_alone{self.wbm_id}.{save_format}'
            fig.savefig(fname)
        plt.show()
        return axs

    def plot_xy_alone(self, figsize=(4, 4)):

        fig, axs = plt.subplots(figsize=figsize)

        axs.scatter(self.data_xy[:, 0], self.data_xy[:, 1], c=self.idxClusterAssignment, marker='.', cmap='rainbow')
        axs.set_xlim(self.min_x - self.offset_x, self.max_x + self.offset_x)
        axs.set_ylim(self.min_y - self.offset_y, self.max_y + self.offset_y)

        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        axs.set_facecolor('#a9a9a9')

        plt.show()

    def plot_polar_and_xy(self, figsize=(4, 4), contour=True):

        fig, axs = plt.subplots(figsize=figsize)

        axs.scatter(self.data_xy[:, 0], self.data_xy[:, 1], c=self.idxClusterAssignment, marker='.', cmap='rainbow')
        axs.set_xlim(self.min_x - self.offset_x, self.max_x + self.offset_x)
        axs.set_ylim(self.min_y - self.offset_y, self.max_y + self.offset_y)

        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        axs.set_facecolor('#a9a9a9')

        fig, axs = plt.subplots(figsize=figsize)
        axs.scatter(self.data[:, 0], self.data[:, 1], c=self.idxClusterAssignment, marker='.', cmap='rainbow')
        axs.scatter(self.paramClusterMu.T[0], self.paramClusterMu.T[1], c='w', marker='*')
        axs.set_xlim(self.min_t - self.offset_t, self.max_t + self.offset_t)
        axs.set_ylim(self.min_r - self.offset_r, self.max_r + self.offset_r)

        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        axs.set_facecolor('#a9a9a9')

        if contour:
            for clst, cnt in enumerate(self.cntClusterAssignment):
                if cnt > 2:
                    mu = self.paramClusterMu[clst]
                    cov = self.paramClusterSigma[clst]
                    gridX = np.arange(0 - self.offset_t, self.max_t + self.offset_t, (self.max_t + self.offset_t) / 100)
                    gridY = np.arange(0 - self.offset_r, self.max_r + self.offset_r, (self.max_r + self.offset_r) / 100)
                    meshX, meshY = np.meshgrid(gridX, gridY)

                    Z = np.zeros(shape=(len(gridY), len(gridX)), dtype=float)
                    for itr1 in range(len(meshX)):
                        for itr2 in range(len(meshX[itr1])):
                            Z[itr1][itr2] = stats.multivariate_normal.pdf([meshX[itr1][itr2],
                                                                           meshY[itr1][itr2]],
                                                                          mean=mu, cov=cov)
                    axs.contour(meshX, meshY, Z, 1, colors='k', linewidths=1)
        plt.show()


class WBM:
    def __init__(self, dataset_012, label_list, label_name, map_shape, fail_rate_limit=(0, 1), norm_factor=False):
        self.overlap_idx = get_overlap_idx(dataset_012)
        self.data_len = len(dataset_012)
        self.map_shape = map_shape
        self.map_len = map_shape[0] * map_shape[1]
        self.n_valid = sum(dataset_012[0] != 0)
        self.n_label = len(np.unique(label_list))
        self.label_name_org = label_name
        self.label_cnt_dict_org = {i: (label_list == i).sum() for i in np.unique(label_list)}
        self.fail_rate_list = (dataset_012 == 2).sum(axis=1).flatten() / self.n_valid
        self.fail_rate_mask_seq_min = self.fail_rate_list >= fail_rate_limit[0]
        self.fail_rate_mask_seq_max = self.fail_rate_list <= fail_rate_limit[1]
        self.mask = self.fail_rate_mask_seq_min * self.fail_rate_mask_seq_max * self.overlap_idx

        self.data = dataset_012[self.mask]
        self.data_len = len(self.data)
        self.label_list = label_list[self.mask]
        self.label_name = label_name[np.unique(self.label_list)]
        self.unique_label = list(set(self.label_list))
        self.label_cnt_dict = {i: (self.label_list == i).sum() for i in np.unique(self.label_list)}
        self.fail_rate_list_limited = self.fail_rate_list[self.mask]

        self.sample_zero_one = np.array(dataset_012[0].reshape(self.map_shape))
        self.sample_zero_one[self.sample_zero_one == 2] = 1
        self.epsilon = 1e-8
        self.target_wf_list = []
        for label in np.unique(self.label_list):
            self.target_wf_list.append(int(np.arange(self.data_len)[self.label_list == label][0]))
        self.n_target_wf = len(self.target_wf_list)

        self.data_with_nan = np.empty_like(self.data, dtype=np.float)
        self.data_with_nan[self.data == 0] = np.nan
        self.data_with_nan[self.data == 1] = 0
        self.data_with_nan[self.data == 2] = 1
        self.data_without_nan = self.data_with_nan[~np.isnan(self.data_with_nan)].reshape(self.data_len, -1)

        temp = np.flipud(self.sample_zero_one)
        x, y = np.meshgrid(np.arange(temp.shape[1]), np.arange(temp.shape[0]))
        self.x = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x.flatten().reshape(-1, 1))
        self.y = np.flip(MinMaxScaler(feature_range=(-1, 1)).fit_transform(y.flatten().reshape(-1, 1)))
        self.xy = np.hstack((self.x, self.y))
        r = cdist(self.xy, np.array([0, 0]).reshape((1, -1)))
        t_temp = self.x / (r + self.epsilon)
        t_temp = np.arccos(t_temp) * 180 / math.pi
        t = np.empty_like(t_temp)
        yidx = self.y >= 0
        t[yidx] = t_temp[yidx]
        t[~yidx] = 360 - t_temp[~yidx]
        self.t = MinMaxScaler(feature_range=(0, 1)).fit_transform(t.flatten().reshape(-1, 1))
        self.r = MinMaxScaler(feature_range=(0, 1)).fit_transform(r.flatten().reshape(-1, 1))
        self.tr = np.hstack((self.t, self.r))

        if norm_factor:
            self.t *= norm_factor
            max_r = self.r[self.sample_zero_one.flatten() != 0].max()
            self.r *= (norm_factor / max_r)
            self.tr = np.hstack((self.t, self.r))
            self.x *= norm_factor
            self.y *= norm_factor
            self.xy = np.hstack((self.x, self.y))

        self.max_x = self.x[self.sample_zero_one.flatten() != 0].max()
        self.max_y = self.y[self.sample_zero_one.flatten() != 0].max()
        self.max_t = self.t[self.sample_zero_one.flatten() != 0].max()
        self.max_r = self.r[self.sample_zero_one.flatten() != 0].max()
        self.min_x = self.x[self.sample_zero_one.flatten() != 0].min()
        self.min_y = self.y[self.sample_zero_one.flatten() != 0].min()
        self.min_t = self.t[self.sample_zero_one.flatten() != 0].min()
        self.min_r = self.r[self.sample_zero_one.flatten() != 0].min()

        self.offset_x = (self.max_x - self.min_x) * 0.05
        self.offset_y = (self.max_y - self.min_y) * 0.05
        self.offset_t = (self.max_t - self.min_t) * 0.05
        self.offset_r = (self.max_r - self.min_r) * 0.05

        self.tr_valid = self.tr[self.sample_zero_one.flatten() != 0]
        self.xy_valid = self.xy[self.sample_zero_one.flatten() != 0]

        self.euc_dist_from_center_valid = cdist(self.xy_valid, np.array([0, 0]).reshape(1, -1))
        self.euc_dist_max = max(self.euc_dist_from_center_valid)
        self.euc_dist_mtx = cdist(self.xy_valid, self.xy_valid)

        self.pad_map = np.pad(np.ones(self.map_shape), pad_width=1)
        self.pad_map_shape = self.pad_map.shape
        self.pad_map_unpad_idx = self.pad_map.flatten()
        self.pad_map_unpad_idx = np.arange(len(self.pad_map_unpad_idx))[
            self.pad_map_unpad_idx == 1]  # data_len 만큼의 vector
        self.pad_map_row_col = np.empty((self.pad_map_shape[0] * self.pad_map_shape[1], 3), dtype=np.int)
        counter = 0
        for row in range(self.pad_map_shape[0]):
            for col in range(self.pad_map_shape[1]):
                self.pad_map_row_col[counter, 0] = counter
                self.pad_map_row_col[counter, 1] = row
                self.pad_map_row_col[counter, 2] = col
                counter += 1

        self.sim_mtx_dict = {'euc': squareform(pdist(self.data_without_nan))}

        self.map_shape_zero_padded = (map_shape[0] + 2, map_shape[1] + 2)
        self.map_len_zero_padded = (map_shape[0] + 2) * (map_shape[1] + 2)
        self.data_zero_padded = np.zeros((self.data_len, self.map_len_zero_padded))
        for i in range(self.data_len):
            self.data_zero_padded[i] = np.pad(self.data[i].reshape(self.map_shape), pad_width=1).flatten()
        self.sample_zero_one_padded = np.array(self.data_zero_padded[0].reshape(self.map_shape_zero_padded))
        self.sample_zero_one_padded[self.sample_zero_one_padded == 2] = 1

        temp = np.flipud(self.sample_zero_one_padded)
        x, y = np.meshgrid(np.arange(temp.shape[1]), np.arange(temp.shape[0]))
        self.x_pad = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x.flatten().reshape(-1, 1))
        self.y_pad = np.flip(MinMaxScaler(feature_range=(-1, 1)).fit_transform(y.flatten().reshape(-1, 1)))
        self.xy_pad = np.hstack((self.x_pad, self.y_pad))
        self.r_pad = cdist(self.xy_pad, np.array([0, 0]).reshape((1, -1)))

        map_pad = self.sample_zero_one_padded
        map_pad_edge_top = np.zeros_like(map_pad)
        for i in range(map_pad.shape[0] - 1):
            map_pad_edge_top[i] = np.logical_or(map_pad[i], map_pad[i + 1])
        map_pad = np.flipud(map_pad)
        map_pad_edge_bot = np.zeros_like(map_pad)
        for i in range(map_pad.shape[0] - 1):
            map_pad_edge_bot[i] = np.logical_or(map_pad[i], map_pad[i + 1])
        map_pad_edge_bot = np.flipud(map_pad_edge_bot)
        map_pad_edge = np.logical_or(map_pad_edge_bot, map_pad_edge_top)
        map_pad = np.transpose(map_pad_edge)
        map_pad_edge_tran1 = np.zeros_like(map_pad)
        for i in range(map_pad.shape[0] - 1):
            map_pad_edge_tran1[i] = np.logical_or(map_pad[i], map_pad[i + 1])
        map_pad_edge_tran1 = np.transpose(map_pad_edge_tran1)
        map_pad = np.flipud(np.transpose(map_pad_edge))
        map_pad_edge_tran2 = np.zeros_like(map_pad)
        for i in range(map_pad.shape[0] - 1):
            map_pad_edge_tran2[i] = np.logical_or(map_pad[i], map_pad[i + 1])
        map_pad_edge_tran2 = np.fliplr(np.transpose(map_pad_edge_tran2))
        self.map_pad_edge = np.logical_or(map_pad_edge_tran1, map_pad_edge_tran2) * 1
        self.xy_pad_valid = self.xy_pad[self.map_pad_edge.flatten() == 1]
        self.n_valid_pad = self.map_pad_edge.sum()
        self.total_fail = (self.data == 2).sum()

        self.save_folder_results = make_sub_folder('results',
                                                   f'wbm_{self.map_shape}_{self.n_valid}_{self.data_len}_{self.total_fail}')
        self.save_folder_wmhd_values = make_sub_folder(self.save_folder_results, 'wmhd_values')
        self.save_folder_figures = make_sub_folder(self.save_folder_results, 'figures')
        self.save_folder_scores = make_sub_folder(self.save_folder_results, 'scores')
        self.save_folder_runtime = make_sub_folder(self.save_folder_results, 'runtime')

    def get_target_wf_group_list(self, n_groups=100, save=True):
        fname = self.save_folder_results + f'target_wf_group_list_{n_groups}ea.csv'
        if os.path.isfile(fname):
            self.target_wf_group_list = np.loadtxt(fname, dtype='int', delimiter=',').tolist()
            self.n_groups = len(self.target_wf_group_list)
            print('target wf group list file exist, data have been loaded.')
        else:
            self.n_groups = n_groups
            self.target_wf_group_list = np.zeros((len(self.unique_label), n_groups), dtype='int')
            for i, v in enumerate(self.unique_label):
                self.target_wf_group_list[i] = np.random.choice(np.arange(self.data_len)[self.label_list == v],
                                                                n_groups,
                                                                replace=True)

            self.target_wf_group_list = self.target_wf_group_list.T.tolist()
            if save:
                print('target wf group list file does not exist, data have been saved.')
                np.savetxt(fname, np.array(self.target_wf_group_list), delimiter=',', fmt='%1.8f')
            else:
                print('target wf group list file does not exist. data is not saved')

    def plot_sample_imshow(self, wbm_id):
        fig, axs = plt.subplots(figsize=(3, 3))
        axs.imshow(self.data_with_nan[wbm_id].reshape(self.map_shape), aspect='auto', interpolation='none',
                   cmap=ListedColormap(["lightgray", "red"]))
        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        plt.show()
        return axs

    def plot_sample_all_class(self, rand=False, save=False, save_format='pdf'):
        fig, axs = plt.subplots(self.n_label, 10, figsize=(20, self.n_label * 2))
        for row, label in enumerate(self.label_cnt_dict):
            sample_idx = np.arange(self.data_len)[self.label_list == label]
            if rand:
                sample_idx = np.random.choice(sample_idx, 10, replace=False)
            else:
                sample_idx = sample_idx[:10]
            for col, wbm_id in enumerate(sample_idx):
                axs[row][col].imshow(self.data_with_nan[wbm_id].reshape(self.map_shape),
                                     aspect='auto', interpolation='none', cmap='binary')
                axs[row][col].set_facecolor('gray')
                axs[row][col].get_xaxis().set_visible(False)
                axs[row][col].get_yaxis().set_visible(False)
        if save:
            fname = self.save_folder_figures + f'plot_random_sample_all_class.{save_format}'
            fig.savefig(fname)

        plt.show()

    def plot_multiple_sample_imshow(self, id_list, title=True):
        n_args = len(id_list)
        fig, axs = plt.subplots(1, n_args, figsize=(n_args * 3, 3))
        for i, wbm_id in enumerate(id_list):
            axs[i].imshow(self.data_with_nan[wbm_id].reshape(self.map_shape), aspect='auto', interpolation='none',
                          cmap=ListedColormap(["lightgray", "red"]))
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
            if title:
                axs[i].set_title(wbm_id)
        plt.show()

    def plot_sample_xy(self, wbm_id, save=True, save_format='pdf'):
        fig, axs = plt.subplots(figsize=(4, 4))
        idx = self.data[wbm_id] != 0
        axs.scatter(self.xy.T[0][idx], self.xy.T[1][idx], c=self.data[wbm_id][idx], marker='.',
                    cmap='binary', edgecolors='k', linewidths=0.5)
        axs.set_facecolor('lightgray')
        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        if save:
            fname = self.save_folder_figures + f'sample_cartesian_WBM_{wbm_id}.{save_format}'
            fig.savefig(fname)
        plt.show()

    def plot_sample_tr(self, wbm_id, save=True, save_format='pdf'):
        fig, axs = plt.subplots(figsize=(4, 4))
        idx = self.data[wbm_id] != 0
        axs.scatter(self.tr.T[0][idx], self.tr.T[1][idx], c=self.data[wbm_id][idx], marker='.',
                    cmap='binary', edgecolors='k', linewidths=0.5)
        axs.set_facecolor('lightgray')
        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        if save:
            fname = self.save_folder_figures + f'sample_polar_WBM_{wbm_id}.{save_format}'
            fig.savefig(fname)
        plt.show()

    def plot_sample_xy_tr(self, wbm_id, save=True, save_format='pdf'):
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        idx = self.data[wbm_id] != 0

        axs[0].scatter(self.xy.T[0][idx], self.xy.T[1][idx], c=self.data[wbm_id][idx], marker='.',
                       cmap='binary', edgecolors='k', linewidths=1)
        # axs[0].set_facecolor('lightgray')
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)

        axs[1].scatter(self.tr.T[0][idx], self.tr.T[1][idx], c=self.data[wbm_id][idx], marker='.',
                       cmap='binary', edgecolors='k', linewidths=1)
        # axs[1].set_facecolor('lightgray')
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)

        if save:
            fname = self.save_folder_figures + f'sample_cartesian_and_polar_WBM_{wbm_id}.{save_format}'
            fig.savefig(fname)
        plt.show()

    def plot_class_sample_imshow(self, title=True, save=True, save_format='pdf'):
        fig, axs = plt.subplots(1, self.n_label, figsize=(self.n_label * 3, 3))
        for i, wbm_id in enumerate(self.target_wf_list):
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
            axs[i].set_facecolor('gray')
            axs[i].imshow(self.data_with_nan[wbm_id].reshape(self.map_shape), aspect='auto', interpolation='none',
                          cmap='binary')
            if title:
                axs[i].set_title(f'{self.label_name_org[self.label_list[wbm_id]]}, ID : {wbm_id}')
        if save:
            fname = self.save_folder_figures + f'plot_class_sample_imshow.{save_format}'
            fig.savefig(fname)
        plt.show()

    def plot_class_average_map(self, title=True, save=True, save_format='pdf'):
        fig, axs = plt.subplots(1, self.n_label, figsize=(self.n_label * 3, 3))
        if self.n_label == 1:
            unique_label = int(np.unique(self.label_list))
            axs.get_xaxis().set_visible(False)
            axs.get_yaxis().set_visible(False)
            axs.set_facecolor('gray')
            avg_map = (self.data_with_nan[self.label_list == unique_label]).mean(axis=0)
            axs.imshow(avg_map.reshape(self.map_shape), aspect='auto', interpolation='none', cmap='binary')
            if title:
                axs.set_title(f'{self.label_name[0]}, {(self.label_list == unique_label).sum()}ea')
        else:
            for i, v in enumerate(np.unique(self.label_list)):
                axs[i].get_xaxis().set_visible(False)
                axs[i].get_yaxis().set_visible(False)
                axs[i].set_facecolor('gray')
                avg_map = (self.data_with_nan[self.label_list == v]).mean(axis=0)
                axs[i].imshow(avg_map.reshape(self.map_shape), aspect='auto', interpolation='none', cmap='binary')
                if title:
                    axs[i].set_title(f'{self.label_name[i]}, {(self.label_list == v).sum()}ea')
        if save:
            fname = self.save_folder_figures + f'plot_class_average_map.{save_format}'
            fig.savefig(fname)
        plt.show()

    def plot_class_all_samples(self, n_cols=10, save=False, save_format='pdf'):
        for class_seq, (class_id, n_samples) in enumerate(self.label_cnt_dict.items()):
            class_name = self.label_name[class_seq]
            sample_idx = np.arange(self.data_len)[self.label_list == class_id]
            n_rows = int(np.ceil(n_samples / n_cols))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
            for i, ax in enumerate(axs.flat[:n_samples]):
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.imshow(self.data_with_nan[sample_idx[i]].reshape(self.map_shape),
                          aspect='auto',
                          interpolation='none',
                          cmap=cmap_gray_red)
                ax.set_title(class_name, fontsize='xx-large')
            if save:
                fname = self.save_folder_figures + f'class_all_samples_{class_id}_{class_name}.{save_format}'
                fig.savefig(fname)
            plt.show()

    def get_polar_points(self, wbm_id):
        return self.tr[self.data[wbm_id] == 2]

    def get_xy_points(self, wbm_id):
        return self.xy[self.data[wbm_id] == 2]

    def mountain(self, wbm_id, m=1):
        """

        :param wbm_id: wafer id
        :param m: int > 0
        :return: mountain_val[defect_idx], mountain_val
        """
        defect_idx = self.data_without_nan[wbm_id] == 1  # n_valid 기준 index
        xy = self.get_xy_points(wbm_id)  # coordinate of defective dies (n_defect, 2)
        XY = self.xy[~np.isnan(self.data_with_nan[0])]  # coordinate of all dies (n_valid, 2)
        beta = 1 / (self.euc_dist_from_center_valid[self.data_without_nan[wbm_id] == 1].sum() / self.n_valid)

        mountain_val = np.exp(cdist(XY, xy) * beta * m * (-1)).sum(axis=1)
        mountain_val_defect_only = mountain_val[defect_idx]
        return mountain_val_defect_only, mountain_val

    def mountain_for_kingmove(self, wbm_id, m=1):
        xy = self.xy_pad[self.data_zero_padded[wbm_id] == 2]
        XY = self.xy_pad_valid
        beta = 1 / (self.euc_dist_from_center_valid[self.data_without_nan[wbm_id] == 1].sum() / self.n_valid)

        mountain_pad_valid = np.exp(cdist(XY, xy) * beta * m * (-1)).sum(axis=1)
        mountain_pad_mtx = np.zeros(self.map_len_zero_padded)
        mountain_pad_mtx[self.map_pad_edge.flatten() == 1] = mountain_pad_valid
        mountain_pad_mtx = mountain_pad_mtx.reshape(self.map_shape_zero_padded)
        return mountain_pad_mtx

    def get_mountain_kingmove_mtx(self, m):
        self.wmhd_para_m = m
        self.mountain_kingmove_mtx = np.empty((self.data_len,
                                               self.map_shape_zero_padded[0],
                                               self.map_shape_zero_padded[1]))
        time_start = time.time()
        for wbm_id in range(self.data_len):
            self.mountain_kingmove_mtx[wbm_id] = self.mountain_for_kingmove(wbm_id, m=m)
        return time.time() - time_start

    def wmhd_sim_calc(self, target_id, compared_id, dist_type, s):
        if dist_type == 'cityblock':
            dist_func = cityblock
        else:
            dist_func = euclidean

        xy_target = self.get_xy_points(target_id)
        wbm_bool_target = np.array(self.data_without_nan[target_id], dtype='bool')
        wbm_bool_compare = np.array(self.data_without_nan[compared_id], dtype='bool')
        n_target = len(xy_target)
        out_idx_arr = np.array(np.ones(len(xy_target)), dtype='bool')
        if s:
            s_out = self.euc_dist_max * s
            out_idx_arr = self.euc_dist_mtx[wbm_bool_target].T[wbm_bool_compare].T.min(axis=1) < s_out
        else:
            s_out = 0
        xy_target = xy_target[out_idx_arr == True]
        n_out = n_target - len(xy_target)
        n_target = n_target - n_out

        h = 0  # wmhd initialization
        Ma_mtx = self.mountain_kingmove_mtx[target_id]
        Mb_mtx = self.mountain_kingmove_mtx[compared_id]
        target_row_col = np.argwhere(self.data_zero_padded[target_id].reshape(self.map_shape_zero_padded) == 2)
        compared_row_col = np.argwhere(self.data_zero_padded[compared_id].reshape(self.map_shape_zero_padded) == 2)

        neighbor_idx = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]])
        for a_row_col in target_row_col:
            w = 0
            dist_min_arr = []
            a_xy = self.xy_pad[
                np.arange(self.map_len_zero_padded).reshape(self.map_shape_zero_padded)[tuple(a_row_col)]]
            for b_row_col in compared_row_col:
                b_xy = self.xy_pad[
                    np.arange(self.map_len_zero_padded).reshape(self.map_shape_zero_padded)[tuple(b_row_col)]]
                for neighbor in neighbor_idx:
                    Ma_neighbor = Ma_mtx[a_row_col[0] + neighbor[0]][a_row_col[1] + neighbor[1]]
                    Mb_neighbor = Mb_mtx[b_row_col[0] + neighbor[0]][b_row_col[1] + neighbor[1]]
                    w += np.abs(Ma_neighbor - Mb_neighbor) / max(Ma_neighbor, Mb_neighbor)
                dist_min_arr.append(w * dist_func(a_xy, b_xy))
            h += min(np.array(dist_min_arr))
        h /= (n_target + self.epsilon)

        sim_val = h + (n_out * s_out)
        return sim_val

    def wmhd_sim_calc_all(self, target_id, dist_type, s, wmhd_tqdm=True):
        wmhd_sim_res = []
        if wmhd_tqdm:
            pbar = tqdm(range(self.data_len))
            for compared_id in pbar:
                if target_id == compared_id:
                    wmhd_sim_res.append(int(-1))
                else:
                    wmhd_sim_res.append(
                        self.wmhd_sim_calc(target_id, compared_id, dist_type=dist_type, s=s))
                pbar.set_description(
                    f'm:{self.wmhd_para_m:.2f} s:{s:.3f} target wf : {target_id:4}, compared_wf : {compared_id:4}')
        else:
            for compared_id in range(self.data_len):
                if target_id == compared_id:
                    wmhd_sim_res.append(int(-1))
                else:
                    wmhd_sim_res.append(
                        self.wmhd_sim_calc(target_id, compared_id, dist_type=dist_type, s=s))

        return np.array(wmhd_sim_res).flatten()


class CG:
    def __init__(self,
                 wbm_obj,
                 res_linkage,
                 mean_arr,
                 cov_arr,
                 points_in_cww_list,
                 cov_type,
                 norm_likelihood,
                 n_cg):

        self.epsilon = 1e-8
        self.weight_vector_arr = np.zeros((wbm_obj.data_len, n_cg))
        self.avg_mean_vec_list = []
        self.avg_cov_mtx_list = []
        self.n_members_list = []
        self.overall_points_in_cg_list = []
        self.cluster_info = cut_tree(res_linkage, n_cg).reshape(-1, )

        n_elements = len(res_linkage) + 1

        if cov_type == 'average':
            for i in range(n_cg):
                self.n_members_list.append((self.cluster_info == i).sum())
                self.avg_mean_vec_list.append(
                    mean_arr[list(np.arange(n_elements)[(self.cluster_info == i).reshape(-1, )])].mean(axis=0))
                self.avg_cov_mtx_list.append(
                    cov_arr[list(np.arange(n_elements)[(self.cluster_info == i).reshape(-1, )])].mean(axis=0))

            self.avg_mean_vec_list = np.array(self.avg_mean_vec_list)
            self.avg_cov_mtx_list = np.array(self.avg_cov_mtx_list)

        elif cov_type == 'real':
            for i in range(n_cg):
                self.n_members_list.append((self.cluster_info == i).sum())
                overall_points_in_cg = [points_in_cww_list[cww] for cww, v in enumerate(self.cluster_info) if v == i]
                overall_points_in_cg = np.vstack(overall_points_in_cg)
                self.avg_mean_vec_list.append(overall_points_in_cg.mean(axis=0))
                self.avg_cov_mtx_list.append(empirical_covariance(overall_points_in_cg))

        else:  # cov_type == 'sample'
            for i in range(n_cg):
                selected_cg_idx = self.cluster_info == i
                n_CwW_in_cg = selected_cg_idx.sum()
                self.n_members_list.append(n_CwW_in_cg)
                mean_list_temp = mean_arr[selected_cg_idx].tolist()
                cov_list_temp = cov_arr[selected_cg_idx].tolist()
                sample_points_in_cg = [mvn(mean_list_temp[j], cov_list_temp[j], size=100) for j in range(n_CwW_in_cg)]
                sample_points_in_cg = np.vstack(sample_points_in_cg)
                self.avg_mean_vec_list.append(sample_points_in_cg.mean(axis=0))
                self.avg_cov_mtx_list.append(empirical_covariance(sample_points_in_cg))

            self.avg_mean_vec_list = np.array(self.avg_mean_vec_list)
            self.avg_cov_mtx_list = np.array(self.avg_cov_mtx_list)

        # LIKEIHOOD MTX, WEIGHT VECTOR
        if norm_likelihood:
            for wbm_id in range(wbm_obj.data_len):
                data = wbm_obj.get_polar_points(wbm_id)
                data_len = len(data)
                likelihood_mtx = np.zeros((data_len, n_cg))
                for i in range(n_cg):
                    likelihood_mtx[:, i] = multivariate_normal.pdf(data,
                                                                   self.avg_mean_vec_list[i],
                                                                   self.avg_cov_mtx_list[i])

                likelihood_mtx /= likelihood_mtx.sum(axis=1).reshape(-1, 1)
                self.weight_vector_arr[wbm_id] = likelihood_mtx.mean(axis=0)
                self.weight_vector_arr[wbm_id] += (np.random.rand(n_cg) * self.epsilon)
                self.weight_vector_arr[wbm_id] /= self.weight_vector_arr[wbm_id].sum()
        else:
            for wbm_id in range(wbm_obj.data_len):
                data = wbm_obj.get_polar_points(wbm_id)
                data_len = len(data)
                likelihood_mtx = np.zeros((data_len, n_cg))
                for i in range(n_cg):
                    likelihood_mtx[:, i] = multivariate_normal.pdf(data,
                                                                   self.avg_mean_vec_list[i],
                                                                   self.avg_cov_mtx_list[i])

                self.weight_vector_arr[wbm_id] = likelihood_mtx.mean(axis=0)
                # self.weight_vector_arr[wbm_id] += (np.random.rand(n_cg) * self.epsilon)
                self.weight_vector_arr[wbm_id] /= self.weight_vector_arr[wbm_id].sum()

        # LIKELIHOOD FOR ALL POSITIONS AND MAKE IT AS 'SUM TO ONE'
        data = wbm_obj.tr_valid
        self.likelihood_all = np.zeros((n_cg, len(data)))
        self.likelihood_all_sum_to_one = np.zeros((n_cg, len(data)))

        for i in range(n_cg):
            self.likelihood_all[i] = multivariate_normal.pdf(data,
                                                             self.avg_mean_vec_list[i],
                                                             self.avg_cov_mtx_list[i])
            self.likelihood_all_sum_to_one[i] = self.likelihood_all[i] / self.likelihood_all[i].sum()

        # REPRODUCED MAP
        self.re_map = self.weight_vector_arr @ self.likelihood_all_sum_to_one
        self.re_map_scaled = np.zeros((wbm_obj.data_len, wbm_obj.n_valid))

        for i in range(wbm_obj.data_len):
            self.re_map_scaled[i] = MinMaxScaler().fit_transform(self.re_map[i].reshape(-1, 1)).flatten()

        # MSE LOSS : BETWEEN ORIGINAL WBM vs. RESCALED REPRODUCED MAP
        self.loss_raw = wbm_obj.data_without_nan - self.re_map_scaled
        self.loss_raw = self.loss_raw ** 2
        self.loss = self.loss_raw.mean(axis=1)
        self.loss_mean = self.loss.mean()


class MODEL:
    def __init__(self, wbm_obj):

        self.wbm_obj = wbm_obj
        self.sim_mtx_dict = {}
        self.dict_rank = {'rank': {'EUC': {'none': {}}, 'JSD': {}, 'WMHD': {}},
                          'value': {'EUC': {'none': {}}, 'JSD': {}, 'WMHD': {}}}
        self.dict_score = {'cor': {}, 'pre': {}, 'acc': {}, 'tpr': {}, 'fpr': {}, 'auc': {},
                           'pre_avg': {}, 'acc_avg': {}, 'auc_avg': {}}
        for score_key in self.dict_score.keys():
            self.dict_score[score_key].update({'EUC': {}, 'JSD': {}, 'WMHD': {}})
        self.runtime_dict = {}
        self.runtime_dict.update({'EUC': {'none': {}}})
        self.runtime_dict.update({'WMHD': {}})
        self.fname_runtime_dict = self.wbm_obj.save_folder_runtime + f'runtime_dict.json'

        if not os.path.isfile(self.fname_runtime_dict):
            with open(self.fname_runtime_dict, 'w') as fw:
                fw.write(json.dumps({'0_init': get_now_str()}, indent=4, sort_keys=True))

    def get_dpgmm(self, infer_method='vi', load=True, save=True):
        self.dpgmm_infer_method = infer_method
        self.fname_dpgmm_list = f'dpgmm_list_{self.dpgmm_infer_method}'
        # 기존에 만든 파일이 있을 경우
        if load & os.path.isfile(self.wbm_obj.save_folder_results + self.fname_dpgmm_list):
            self.load_check_dpgmm = True
            infile = open(self.wbm_obj.save_folder_results + self.fname_dpgmm_list, 'rb')
            self.dpgmm_list = pickle.load(infile, encoding='latin1')
        # 파일이 없을 경우, 새로 만들고 시간을 기록한다.
        else:
            self.load_check_dpgmm = False
            time_dpgmm_start = time.time()
            self.dpgmm_list = []
            if self.dpgmm_infer_method == 'vi':
                for wbm_id in tqdm(range(self.wbm_obj.data_len), desc='DPGMM...'):
                    dpgmm = DPGMM_VI(wbm_id, self.wbm_obj)
                    dpgmm.fit_dpgmm()
                    self.dpgmm_list.append(dpgmm)
            else:
                for wbm_id in tqdm(range(self.wbm_obj.data_len), desc='DPGMM...'):
                    dpgmm = DPGMM_MC(wbm_id, self.wbm_obj)
                    dpgmm.cluster()
                    self.dpgmm_list.append(dpgmm)
            time_dpgmm_end = time.time()
            time_dpgmm = np.round(time_dpgmm_end - time_dpgmm_start, 3)
            self.runtime_dict['dpgmm'] = {self.dpgmm_infer_method: time_dpgmm}
            if save:
                save_list(self.dpgmm_list, self.wbm_obj.save_folder_results + self.fname_dpgmm_list)

    def get_skldm(self, linkage_method='complete', min_defects=2, load=True, save=True):
        self.linkage_method = linkage_method
        self.cntClusters_list = []
        self.mean_list = []
        self.cov_list = []
        self.points_in_cww_list = []

        for i in range(len(self.dpgmm_list)):
            dpgmm = self.dpgmm_list[i]
            cww_id = np.unique(dpgmm.idxClusterAssignment)
            for j, n_defects in enumerate(dpgmm.cntClusterAssignment):
                if n_defects > min_defects:
                    self.cntClusters_list.append(n_defects)
                    self.mean_list.append(dpgmm.paramClusterMu[j])
                    self.cov_list.append(dpgmm.paramClusterSigma[j])
                    self.points_in_cww_list.append(dpgmm.data[dpgmm.idxClusterAssignment == cww_id[j]])

        self.cntClusters_arr = np.array(self.cntClusters_list)
        self.mean_arr = np.array(self.mean_list)
        self.cov_arr = np.array(self.cov_list)

        self.fname_skldm = f'skldm_{self.dpgmm_infer_method}'
        if load & os.path.isfile(self.wbm_obj.save_folder_results + self.fname_skldm):
            self.load_check_skldm = True
            infile = open(self.wbm_obj.save_folder_results + self.fname_skldm, 'rb')
            self.skldm = pickle.load(infile, encoding='latin1')

        else:
            self.load_check_skldm = False
            time_skldm_start = time.time()
            self.skldm = get_SKLDM(self.mean_list, self.cov_list)
            time_skldm_end = time.time()
            time_skldm = np.round(time_skldm_end - time_skldm_start, 3)
            self.runtime_dict['skldm'] = {self.dpgmm_infer_method: time_skldm}
            if save:
                save_list(self.skldm, self.wbm_obj.save_folder_results + self.fname_skldm)

        for i in range(len(self.skldm)):
            self.skldm[i, i] = 0
        self.skldm[self.skldm < 0] = 0
        self.ordered_dist_mat, self.res_order, self.res_linkage = compute_serial_matrix(self.skldm,
                                                                                        linkage_method=linkage_method,
                                                                                        optimal=False)

    def get_cg(self, n_cg=28, cov_type='real', norm_likelihood=True):
        self.n_cg = n_cg
        self.cov_type = cov_type
        time_cg_start = time.time()
        self.cg = CG(self.wbm_obj,
                     self.res_linkage,
                     self.mean_arr,
                     self.cov_arr,
                     self.points_in_cww_list,
                     cov_type=cov_type,
                     norm_likelihood=norm_likelihood,
                     n_cg=self.n_cg)
        time_cg_end = time.time()
        time_cg = np.round(time_cg_end - time_cg_start, 3)
        self.runtime_dict['cg'] = {self.dpgmm_infer_method: {n_cg: time_cg}}
        self.runtime_dict['JSD'] = {n_cg: {}}
        self.dict_rank['value']['JSD'].update({n_cg: {}})

    def get_cg_list(self, max_n_cg=60, cov_type='real', norm_likelihood=True, plot=True):
        self.cg_list = []
        time_cg_list_start = time.time()
        for i in trange(1, max_n_cg + 1):
            self.cg_list.append(CG(self.wbm_obj,
                                   self.res_linkage,
                                   self.mean_arr,
                                   self.cov_arr,
                                   self.points_in_cww_list,
                                   cov_type=cov_type,
                                   norm_likelihood=norm_likelihood,
                                   n_cg=i))
        time_cg_list_end = time.time()
        self.cg_loss_arr = np.vstack([cg.loss.flatten() for cg in self.cg_list])
        self.cg_loss_arr_mean = self.cg_loss_arr.mean(axis=1).flatten()
        time_cg_list = np.round(time_cg_list_end - time_cg_list_start, 3)
        self.runtime_dict['cg_list'] = {self.dpgmm_infer_method: {max_n_cg: time_cg_list}}
        self.min_mse_cg = np.argsort(self.cg_loss_arr_mean)[0]+1
        if plot:
            xticks = np.arange(1, max_n_cg + 1)
            fig, axs = plt.subplots(figsize=(15, 4))
            axs.plot(xticks, self.cg_loss_arr_mean, 'bo-')
            axs.set_xticks(xticks)
            axs.grid()
            plt.show()
        return self.cg_loss_arr_mean, self.min_mse_cg

    def plot_skldm(self, vmax=100, save=True, save_format='pdf'):
        """
        plot_skldm(self, vmax=100, save=True, save_format='pdf')
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].imshow(self.skldm, aspect='auto', interpolation='none', vmax=vmax, cmap='binary')
        axs[1].imshow(self.ordered_dist_mat, aspect='auto', interpolation='none', vmax=vmax, cmap='binary')
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)
        axs[1].get_xaxis().set_visible(False)
        axs[1].get_yaxis().set_visible(False)

        if save:
            fname = self.wbm_obj.save_folder_figures + f'plot_skldm_{self.linkage_method}.{save_format}'
            fig.savefig(fname)
        plt.show()

    def plot_skldm_dendrogram(self, save=True, save_format='pdf'):
        plt.figure(figsize=(6, 6))
        plt.xlabel('CwW index (labels have been skipped)')
        plt.ylabel('Symmetric KL Divergence')
        dendrogram(
            self.res_linkage,
            leaf_rotation=90.,  # rotates the x axis labels
            leaf_font_size=8.,  # font size for the x axis labels
            no_labels=True
        )
        if save:
            fname = self.wbm_obj.save_folder_figures + f'plot_skldm_{self.linkage_method}.{save_format}'
            plt.savefig(fname)
        plt.show()

    def plot_sim_rank_result(self, TWL_id, method, para, layout='series', save=False):
        TF_color_dict = {True: 'b', False: 'r'}
        if type(TWL_id) == int:
            TWL_id_str = str(self.wbm_obj.target_wf_group_list[TWL_id])
            prec_avg = self.dict_score['pre_avg'][method][para][TWL_id_str].round(3)
            folder_name = f'{TWL_id_str}_TWLid_{TWL_id}_{method}_{para}_{prec_avg}'
            save_folder = make_sub_folder(self.wbm_obj.save_folder_figures, folder_name)
            TWL = self.wbm_obj.target_wf_group_list[TWL_id]

        else:
            TWL_id_str = str(TWL_id)
            prec_avg = self.dict_score['pre_avg'][method][para][TWL_id_str].round(3)
            folder_name = f'{TWL_id_str}_TWLid_none_{method}_{para}_{prec_avg}'
            save_folder = make_sub_folder(self.wbm_obj.save_folder_figures, folder_name)
            TWL = TWL_id
        # layout : series
        if layout == 'series':
            for target_seq, target_id in enumerate(TWL):
                cor = self.dict_score['cor'][method][para][str(TWL)][target_id]
                n_img = min(len(cor), 15)
                ranked_id_arr = self.dict_rank['rank'][method][para][str(TWL)][target_id][:n_img]
                map_type_arr = np.array(
                    [self.wbm_obj.label_name_org[i] for i in self.wbm_obj.label_list[ranked_id_arr]])
                fig, axs = plt.subplots(1, 15, figsize=(15 * 3, 3))

                for i, ax in enumerate(axs.flat[:n_img]):
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.imshow(self.wbm_obj.data_with_nan[ranked_id_arr[i]].reshape(self.wbm_obj.map_shape),
                              aspect='auto',
                              interpolation='none',
                              cmap=cmap_gray_red)
                    ax.set_title(f'{cor[i]} \n {map_type_arr[i]}', fontsize='xx-large', c=TF_color_dict[cor[i]])
                axs[0].set_title(f'Target \n {map_type_arr[0]}', fontsize='xx-large', c='k')
                if n_img < 15:
                    for ax in axs.flat[n_img:]:
                        ax.axis('off')
                if save:
                    fig.savefig(save_folder + f'{layout}_target_id_{target_id}.pdf', bbox_inches='tight')
                plt.show()

        # layout : square
        elif layout == 'square':
            for target_seq, target_id in enumerate(TWL):
                cor = self.dict_score['cor'][method][para][str(TWL)][target_id]
                n_img = min(len(cor), 15)
                ranked_id_arr = self.dict_rank['rank'][method][para][str(TWL)][target_id][:n_img]
                map_type_arr = np.array(
                    [self.wbm_obj.label_name_org[i] for i in self.wbm_obj.label_list[ranked_id_arr]])
                fig, axs = plt.subplots(3, 5, figsize=(5 * 3, 3 * 3))

                for i, ax in enumerate(axs.flat[:n_img]):
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                    ax.imshow(self.wbm_obj.data_with_nan[ranked_id_arr[i]].reshape(self.wbm_obj.map_shape),
                              aspect='auto',
                              interpolation='none',
                              cmap=cmap_gray_red)
                    ax.set_title(f'{cor[i]} \n {map_type_arr[i]}', fontsize='xx-large', c=TF_color_dict[cor[i]])
                axs[0].set_title(f'Target \n {map_type_arr[0]}', fontsize='xx-large', c='k')
                if n_img < 15:
                    for ax in axs.flat[n_img:]:
                        ax.axis('off')
                if save:
                    fig.savefig(save_folder + f'{layout}_target_id_{target_id}.pdf', bbox_inches='tignt')
                plt.show()

    def update_sim_mtx_dict_euclidean(self):
        time_start = time.time()
        self.sim_mtx_dict['EUC'] = squareform(pdist(self.wbm_obj.data_without_nan))
        time_end = time.time()
        self.runtime_dict['update_sim_mtx_dict_euclidean'] = time_end - time_start

    def update_sim_mtx_dict_JSD(self):
        time_start = time.time()
        self.sim_mtx_dict['JSD'] = get_JSD_cat_mtx(self.cg.weight_vector_arr)
        time_end = time.time()
        self.runtime_dict['update_sim_mtx_dict_JSD'] = time_end - time_start

    def plot_cg_mean_cov(self, contour=True, save=True, save_format='pdf'):

        """
        plot_cg_mean_cov(self, contour=True):
        """

        gridX = np.arange(self.wbm_obj.min_t - self.wbm_obj.offset_t,
                          self.wbm_obj.max_t + self.wbm_obj.offset_t,
                          (self.wbm_obj.max_t + self.wbm_obj.offset_t) / 100)
        gridY = np.arange(self.wbm_obj.min_r - self.wbm_obj.offset_r,
                          self.wbm_obj.max_r + self.wbm_obj.offset_r,
                          (self.wbm_obj.max_r + self.wbm_obj.offset_r) / 100)
        meshX, meshY = np.meshgrid(gridX, gridY)
        Z = np.zeros(shape=(len(gridY), len(gridX)), dtype=float)
        fig, axs = plt.subplots(1, self.n_cg, figsize=(self.n_cg * 3, 3))
        for i in range(self.n_cg):
            selected_mean_arr = self.mean_arr[self.cg.cluster_info == i]
            selected_cov_arr = self.cov_arr[self.cg.cluster_info == i]
            axs[i].scatter(selected_mean_arr.T[0], selected_mean_arr.T[1], c='b', marker='*')
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
            axs[i].set_xlim(self.wbm_obj.min_t - self.wbm_obj.offset_t,
                            self.wbm_obj.max_t + self.wbm_obj.offset_t)
            axs[i].set_ylim(self.wbm_obj.min_r - self.wbm_obj.offset_r,
                            self.wbm_obj.max_r + self.wbm_obj.offset_r)

            if contour:
                for j in trange(len(selected_mean_arr)):
                    mu = selected_mean_arr[j]
                    cov = selected_cov_arr[j]
                    for itr1 in range(len(meshX)):
                        for itr2 in range(len(meshX[itr1])):
                            Z[itr1][itr2] = stats.multivariate_normal.pdf([meshX[itr1][itr2], meshY[itr1][itr2]],
                                                                          mean=mu, cov=cov)
                    axs[i].contour(meshX, meshY, Z, 1, colors='k', linewidths=0.5)
        if save:
            fname = self.wbm_obj.save_folder_figures + f'plot_cg_mean_cov_{self.n_cg}.{save_format}'
            fig.savefig(fname, bbox_inches='tight')
        plt.show()

    def plot_cg_contour_avg_mean_cov(self, save=True, save_format='pdf'):

        """
        plot_cg_contour_avg_mean_cov(self):
        """
        gridX = np.arange(self.wbm_obj.min_t - self.wbm_obj.offset_t,
                          self.wbm_obj.max_t + self.wbm_obj.offset_t,
                          (self.wbm_obj.max_t + self.wbm_obj.offset_t) / 100)
        gridY = np.arange(self.wbm_obj.min_r - self.wbm_obj.offset_r,
                          self.wbm_obj.max_r + self.wbm_obj.offset_r,
                          (self.wbm_obj.max_r + self.wbm_obj.offset_r) / 100)
        meshX, meshY = np.meshgrid(gridX, gridY)
        Z = np.zeros(shape=(len(gridY), len(gridX)), dtype=float)
        fig, axs = plt.subplots(1, self.n_cg, figsize=(self.n_cg * 3, 3))
        for i in range(self.n_cg):
            selected_mean_arr = self.cg.avg_mean_vec_list[i]
            axs[i].scatter(selected_mean_arr.T[0], selected_mean_arr.T[1], c='b', marker='.')
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
            axs[i].set_xlim(self.wbm_obj.min_t - self.wbm_obj.offset_t,
                            self.wbm_obj.max_t + self.wbm_obj.offset_t)
            axs[i].set_ylim(self.wbm_obj.min_r - self.wbm_obj.offset_r,
                            self.wbm_obj.max_r + self.wbm_obj.offset_r)

            mu = self.cg.avg_mean_vec_list[i]
            cov = self.cg.avg_cov_mtx_list[i]
            for itr1 in range(len(meshX)):
                for itr2 in range(len(meshX[itr1])):
                    Z[itr1][itr2] = stats.multivariate_normal.pdf([meshX[itr1][itr2], meshY[itr1][itr2]],
                                                                  mean=mu, cov=cov)
            axs[i].contour(meshX, meshY, Z, 1, colors='k', linewidths=0.5)
        if save:
            fname = self.wbm_obj.save_folder_figures + f'plot_cg_contour_avg_mean_cov_{self.n_cg}.{save_format}'
            fig.savefig(fname)
        plt.show()

    def plot_cg_contourf_avg_mean_cov(self, save=True, save_format='pdf'):

        """
        plot_cg_contourf_avg_mean_cov(self):
        """

        gridX = np.arange(self.wbm_obj.min_t - self.wbm_obj.offset_t,
                          self.wbm_obj.max_t + self.wbm_obj.offset_t,
                          (self.wbm_obj.max_t + self.wbm_obj.offset_t) / 100)
        gridY = np.arange(self.wbm_obj.min_r - self.wbm_obj.offset_r,
                          self.wbm_obj.max_r + self.wbm_obj.offset_r,
                          (self.wbm_obj.max_r + self.wbm_obj.offset_r) / 100)
        meshX, meshY = np.meshgrid(gridX, gridY)
        pos = np.dstack((meshX, meshY))

        fig, axs = plt.subplots(1, self.n_cg, figsize=(self.n_cg * 3, 3))
        for i in range(self.n_cg):
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
            axs[i].set_xlim(self.wbm_obj.min_t - self.wbm_obj.offset_t,
                            self.wbm_obj.max_t + self.wbm_obj.offset_t)
            axs[i].set_ylim(self.wbm_obj.min_r - self.wbm_obj.offset_r,
                            self.wbm_obj.max_r + self.wbm_obj.offset_r)
            rv = multivariate_normal(self.cg.avg_mean_vec_list[i], self.cg.avg_cov_mtx_list[i])
            axs[i].contourf(meshX, meshY, rv.pdf(pos), cmap='binary')
        if save:
            fname = self.wbm_obj.save_folder_figures + f'plot_cg_contourf_avg_mean_cov_{self.n_cg}.{save_format}'
            fig.savefig(fname)
        plt.show()

    def plot_cg_scatter_means(self, save=True, save_format='pdf'):

        """
        plot_cg_scatter_means(self):
        """

        fig, axs = plt.subplots(1, self.n_cg, figsize=(self.n_cg * 3, 3), sharex=True, sharey=True)
        for i in range(self.n_cg):
            selected_mean_arr = self.mean_arr[self.cg.cluster_info == i]
            axs[i].scatter(selected_mean_arr.T[0], selected_mean_arr.T[1], c='b', marker='.')
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
            axs[i].set_xlim(self.wbm_obj.min_t - self.wbm_obj.offset_t,
                            self.wbm_obj.max_t + self.wbm_obj.offset_t)
            axs[i].set_ylim(self.wbm_obj.min_r - self.wbm_obj.offset_r,
                            self.wbm_obj.max_r + self.wbm_obj.offset_r)
        if save:
            fname = self.wbm_obj.save_folder_figures + f'plot_cg_scatter_means_{self.n_cg}.{save_format}'
            fig.savefig(fname)
        plt.show()

    def plot_cg_scatter_norm_likelihood(self, map_type='polar', title=False, save=True, save_format='pdf'):

        """
        plot_cg_scatter_norm_likelihood(self, map_type='polar', title=False):
        """

        if map_type == 'polar':
            x = self.wbm_obj.tr_valid[:, 0]
            y = self.wbm_obj.tr_valid[:, 1]
        else:
            x = self.wbm_obj.xy_valid[:, 0]
            y = self.wbm_obj.xy_valid[:, 1]

        fig, axs = plt.subplots(1, self.n_cg, figsize=(self.n_cg * 3, 3))
        for i in range(self.n_cg):
            axs[i].scatter(x, y, c=self.cg.likelihood_all_sum_to_one[i], marker='.', cmap='Reds')
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
            axs[i].set_facecolor('gray')
            if title:
                axs[i].set_title(('CG_' + str(i + 1)))
        if save:
            fname = self.wbm_obj.save_folder_figures + f'plot_cg_scatter_norm_likelihood_{self.n_cg}.{save_format}'
            fig.savefig(fname)
        plt.show()

    def plot_cg_contourf_avg_dist_to_wbm(self, save=True, save_format='pdf'):

        """
        plot_cg_contourf_avg_dist_to_wbm(self):
        """

        fig, axs = plt.subplots(1, self.n_cg, figsize=(self.n_cg * 3, 3))
        for i in range(self.n_cg):
            pdf = multivariate_normal.pdf(self.wbm_obj.tr,
                                          mean=self.cg.avg_mean_vec_list[i], cov=self.cg.avg_cov_mtx_list[i])
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
            axs[i].imshow(pdf.reshape(self.wbm_obj.map_shape), aspect='auto', interpolation='none', cmap='binary')
        if save:
            fname = self.wbm_obj.save_folder_figures + f'plot_cg_contourf_avg_dist_to_wbm_{self.n_cg}.{save_format}'
            fig.savefig(fname)
        plt.show()

    def plot_cg_weight_vector(self, *args, save=False, save_format='pdf'):

        """
        plot_cg_weight_vector(self, *args):
        """

        fig, axs = plt.subplots(figsize=(5, 4))
        xticks = np.arange(1, self.n_cg + 1)
        for i, wbm_id in enumerate(args):
            if i == 0:
                line_color = 'ro-'
            else:
                line_color = 'bo-'
            axs.plot(xticks, self.cg.weight_vector_arr[wbm_id], line_color)
        axs.set_xticks(xticks)
        axs.set_xlabel('cluster group')
        axs.set_ylabel('weight')
        axs.set_ylim(-0.05, 1.05)
        axs.grid()
        if save:
            fname = self.wbm_obj.save_folder_figures + f'plot_cg_weight_vector_{self.n_cg}.{save_format}'
            fig.savefig(fname)
        plt.show()

    def plot_org_rep_map_comparison(self, wbm_id_list, save=False, save_format='pdf'):

        if type(wbm_id_list) == int:
            fig, axs = plt.subplots(1, 4, figsize=(12, 3))
            axs[0].scatter(self.wbm_obj.xy_valid.T[0], self.wbm_obj.xy_valid.T[1],
                           c=self.wbm_obj.data_without_nan[wbm_id_list], cmap=cmap_gray_red, marker='.')
            axs[1].scatter(self.wbm_obj.xy_valid.T[0], self.wbm_obj.xy_valid.T[1],
                           c=self.cg.re_map_scaled[wbm_id_list], cmap='Reds', marker='.')
            axs[2].scatter(self.wbm_obj.tr_valid.T[0], self.wbm_obj.tr_valid.T[1],
                           c=self.wbm_obj.data_without_nan[wbm_id_list], cmap=cmap_gray_red, marker='.')
            axs[3].scatter(self.wbm_obj.tr_valid.T[0], self.wbm_obj.tr_valid.T[1],
                           c=self.cg.re_map_scaled[wbm_id_list], cmap='Reds', marker='.')
            axs[0].set_title(wbm_id_list)
            for ax in axs.flat:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()
            if save:
                fname = self.wbm_obj.save_folder_figures + f'org_rep_map_comparison_{str(wbm_id_list)}.{save_format}'
                fig.savefig(fname, bbox_inches='tight')

        elif type(wbm_id_list) == list or type(wbm_id_list) == np.ndarray:
            n_col = len(wbm_id_list)
            fig, axs = plt.subplots(4, n_col, figsize=(n_col * 3, 12))
            for i, wbm_id in enumerate(wbm_id_list):
                axs[0][i].scatter(self.wbm_obj.xy_valid.T[0], self.wbm_obj.xy_valid.T[1],
                                  c=self.wbm_obj.data_without_nan[wbm_id], cmap=cmap_gray_red, marker='.')
                axs[1][i].scatter(self.wbm_obj.xy_valid.T[0], self.wbm_obj.xy_valid.T[1],
                                  c=self.cg.re_map_scaled[wbm_id], cmap='Reds', marker='.')
                axs[2][i].scatter(self.wbm_obj.tr_valid.T[0], self.wbm_obj.tr_valid.T[1],
                                  c=self.wbm_obj.data_without_nan[wbm_id], cmap=cmap_gray_red, marker='.')
                axs[3][i].scatter(self.wbm_obj.tr_valid.T[0], self.wbm_obj.tr_valid.T[1],
                                  c=self.cg.re_map_scaled[wbm_id], cmap='Reds', marker='.')
                axs[0][i].set_title(wbm_id)
            for ax in axs.flat:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
            plt.show()
            if save:
                fname = self.wbm_obj.save_folder_figures + f'org_rep_map_comparison_{str(wbm_id_list)}.{save_format}'
                fig.savefig(fname, bbox_inches='tight')

    def calc_sim_res_score(self, sim_rank, rank_interval):
        true_label_sim_rank_sorted = self.wbm_obj.label_list[sim_rank]
        target_label = int(true_label_sim_rank_sorted[0])
        target_label_count = self.wbm_obj.label_cnt_dict[target_label]

        tpr = []
        fpr = []
        for end_rank in range(0, self.wbm_obj.data_len, rank_interval):
            pred_boolean = np.zeros(self.wbm_obj.data_len)
            pred_boolean[:end_rank + 1] = 1
            pred_boolean = pred_boolean == 1
            pred_boolean = pred_boolean[1:]
            true_boolean = (true_label_sim_rank_sorted == target_label)[1:]
            confusion_mtx = confusion_matrix(true_boolean, pred_boolean, labels=[True, False])
            TN = confusion_mtx[0, 0]
            FP = confusion_mtx[0, 1]
            FN = confusion_mtx[1, 0]
            TP = confusion_mtx[1, 1]
            if TP + FN == 0:
                sensitivity = 0
            else:
                sensitivity = TP / (TP + FN)
            if FP + TN == 0:
                specificity = 0
            else:
                specificity = TN / (FP + TN)
            tpr.append(sensitivity)
            fpr.append(1 - specificity)

        bool_arr = true_label_sim_rank_sorted == target_label
        pred_arr = np.zeros_like(bool_arr)
        pred_arr[:target_label_count] = 1

        pre = bool_arr[1: target_label_count].mean().round(4)
        acc = (~np.logical_xor(bool_arr[1:], pred_arr[1:])).sum() / (self.wbm_obj.data_len - 1)
        return pre, acc.round(4), tpr, fpr, auc(fpr, tpr).round(4), bool_arr[:target_label_count].tolist()

    def update_dict_sim_val_JSD(self, target_wf_list):
        key_n_cg = self.n_cg
        n_target_wf = len(target_wf_list)
        key_wf_list = str(target_wf_list)

        self.dict_rank['value']['JSD'][key_n_cg].update({key_wf_list: {}})
        self.runtime_dict['JSD'][key_n_cg].update({key_wf_list: {}})

        time_JSD_vector = 0
        for target_wf in target_wf_list:
            JSD_vector, time_elapsed = self.calc_JSD_vector(target_wf)
            time_JSD_vector += time_elapsed
            self.dict_rank['value']['JSD'][key_n_cg][key_wf_list].update({target_wf: JSD_vector})
        time_JSD_vector_avg = time_JSD_vector / n_target_wf
        self.runtime_dict['JSD'][key_n_cg][key_wf_list].update({'JSD_vec_total': time_JSD_vector})
        self.runtime_dict['JSD'][key_n_cg][key_wf_list].update({'JSD_vec_avg': time_JSD_vector_avg})

        if not (self.load_check_dpgmm and self.load_check_skldm):
            time_JSD = np.round(self.runtime_dict['dpgmm'][self.dpgmm_infer_method]
                                + self.runtime_dict['skldm'][self.dpgmm_infer_method]
                                + self.runtime_dict['cg'][self.dpgmm_infer_method][self.n_cg]
                                + self.runtime_dict['JSD'][key_n_cg][key_wf_list]['JSD_vec_total'],
                                3)
            self.runtime_dict['JSD'][key_n_cg][key_wf_list].update({'JSD_total': time_JSD})

    def update_dict_sim_val_EUC(self, target_wf_list):
        n_target_wf = len(target_wf_list)
        key_wf_list = str(target_wf_list)
        self.dict_rank['value']['EUC']['none'].update({key_wf_list: {}})
        self.runtime_dict['EUC']['none'].update({key_wf_list: {}})

        time_EUC_vector = 0
        for target_wf in target_wf_list:
            EUC_vector, time_elapsed = self.calc_EUC_vector(target_wf)
            time_EUC_vector += time_elapsed
            self.dict_rank['value']['EUC']['none'][key_wf_list].update({target_wf: EUC_vector})
        time_EUC_vector_avg = time_EUC_vector / n_target_wf
        self.runtime_dict['EUC']['none'][key_wf_list].update({'EUC_vec_total': time_EUC_vector})
        self.runtime_dict['EUC']['none'][key_wf_list].update({'EUC_vec_avg': time_EUC_vector_avg})

    def set_para_wmhd(self, dist_type='cityblock', m=0.1, s=0.1):
        self.dist_type = dist_type
        self.wmhd_m = m
        self.wmhd_s = s
        self.wmhd_para = f'{dist_type}_m_{m:.3f}_s_{s:.3f}'
        self.dict_rank['value']['WMHD'].update({self.wmhd_para: {}})
        self.runtime_dict['WMHD'].update({self.wmhd_para: {}})
        time_mountain_func = self.wbm_obj.get_mountain_kingmove_mtx(m=m)
        self.runtime_dict['WMHD'][self.wmhd_para].update({'mountain_func': time_mountain_func})

    def update_dict_sim_val_WMHD(self, target_wf_list, wmhd_tqdm=True):
        target_wf_list_id = str(target_wf_list)[1:-1].replace(', ', '_')
        fname_wmhd_sim_val = f'{self.wmhd_para}_id_{target_wf_list_id}.csv'

        key_wf_list = str(target_wf_list)
        self.dict_rank['value']['WMHD'][self.wmhd_para].update({key_wf_list: {}})

        time_wmhd_total = 0
        if os.path.isfile(self.wbm_obj.save_folder_wmhd_values + fname_wmhd_sim_val):
            self.load_check_wmhd = True
            for i, target_wf in enumerate(target_wf_list):
                wmhd_sim_val_arr = np.loadtxt(self.wbm_obj.save_folder_wmhd_values + fname_wmhd_sim_val, delimiter=',')
                self.dict_rank['value']['WMHD'][self.wmhd_para][key_wf_list].update({target_wf: wmhd_sim_val_arr[i]})
        else:
            self.load_check_wmhd = False
            wmhd_sim_val_arr = np.empty((len(target_wf_list), self.wbm_obj.data_len))
            for i, target_wf in enumerate(target_wf_list):
                time_start_for_wf = time.time()
                wmhd_sim_val = self.wbm_obj.wmhd_sim_calc_all(target_wf,
                                                              dist_type=self.dist_type, s=self.wmhd_s,
                                                              wmhd_tqdm=wmhd_tqdm)
                time_end_for_wf = time.time()
                time_wmhd_for_wf = np.round(time_end_for_wf - time_start_for_wf, 3)
                self.runtime_dict['WMHD'] = {self.wmhd_para: {target_wf: time_wmhd_for_wf}}
                self.dict_rank['value']['WMHD'][self.wmhd_para][key_wf_list].update({target_wf: wmhd_sim_val})
                wmhd_sim_val_arr[i] = wmhd_sim_val.flatten()
                time_wmhd_total += time_wmhd_for_wf

            self.runtime_dict['WMHD'] = {self.wmhd_para: {'total_time': time_wmhd_total}}
            np.savetxt(self.wbm_obj.save_folder_wmhd_values + fname_wmhd_sim_val, wmhd_sim_val_arr,
                       delimiter=',', fmt='%1.8f')

    def update_dict_sim_score(self, rank_interval=1):

        for sim_method in self.dict_rank['value']:
            if not len(self.dict_rank['value'][sim_method]) == 0:
                for para, key_wf_list_dict in self.dict_rank['value'][sim_method].items():
                    self.dict_rank['rank'][sim_method][para] = {}
                    for score_key in self.dict_score:
                        self.dict_score[score_key][sim_method].update({para: {}})
                    for key_wf_list, val_dict in tqdm(key_wf_list_dict.items()):
                        self.dict_rank['rank'][sim_method][para][key_wf_list] = {}
                        for score_key in self.dict_score:
                            self.dict_score[score_key][sim_method][para].update({key_wf_list: {}})

                        n_target_wf = len(self.dict_rank['value'][sim_method][para][key_wf_list])

                        self.dict_score['cor'][sim_method][para][key_wf_list] = {}
                        self.dict_score['pre'][sim_method][para][key_wf_list] = []
                        self.dict_score['acc'][sim_method][para][key_wf_list] = []
                        self.dict_score['tpr'][sim_method][para][key_wf_list] = {}
                        self.dict_score['fpr'][sim_method][para][key_wf_list] = {}
                        self.dict_score['auc'][sim_method][para][key_wf_list] = []

                        pre_avg = 0
                        acc_avg = 0
                        auc_avg = 0
                        for target_wf, sim_val in val_dict.items():
                            sim_rank = np.argsort(sim_val)
                            self.dict_rank['rank'][sim_method][para][key_wf_list][target_wf] = sim_rank
                            pre, acc, tpr, fpr, auc, bool_arr = self.calc_sim_res_score(sim_rank, rank_interval)
                            pre_avg += pre
                            acc_avg += acc
                            auc_avg += auc
                            self.dict_score['cor'][sim_method][para][key_wf_list][target_wf] = bool_arr
                            self.dict_score['tpr'][sim_method][para][key_wf_list][target_wf] = tpr
                            self.dict_score['fpr'][sim_method][para][key_wf_list][target_wf] = fpr
                            self.dict_score['pre'][sim_method][para][key_wf_list].append(pre)
                            self.dict_score['acc'][sim_method][para][key_wf_list].append(acc)
                            self.dict_score['auc'][sim_method][para][key_wf_list].append(auc)

                        self.dict_score['pre_avg'][sim_method][para][key_wf_list] = pre_avg / n_target_wf
                        self.dict_score['acc_avg'][sim_method][para][key_wf_list] = acc_avg / n_target_wf
                        self.dict_score['auc_avg'][sim_method][para][key_wf_list] = auc_avg / n_target_wf

        self.xticks = np.array([i for i in range(rank_interval, self.wbm_obj.data_len, rank_interval)])
        self.dict_score.update({'xticks': self.xticks.tolist()})

    def export_sim_score_dict_to_json(self):
        export_name = self.wbm_obj.save_folder_score + f'sim_score_{get_now_str()}.json'
        with open(export_name, 'w') as fw:
            fw.write(json.dumps(self.dict_score, indent=4, sort_keys=True))

    def export_runtime_dict_to_json(self):
        now_str = get_now_str()
        with open(self.fname_runtime_dict, 'r') as jsonFile:
            runtime_dict_temp = json.load(jsonFile)
        runtime_dict_temp[now_str] = self.runtime_dict
        with open(self.fname_runtime_dict, 'w') as fw:
            fw.write(json.dumps(runtime_dict_temp, indent=4, sort_keys=True))

    def get_best_worst_rank(self, method_list, para_list):
        self.dict_best_worst_rank = {'pre_avg': {'JSD': {},
                                                 'EUC': {},
                                                 'WMHD': {}},
                                     'acc_avg': {'JSD': {},
                                                 'EUC': {},
                                                 'WMHD': {}},
                                     'auc_avg': {'JSD': {},
                                                 'EUC': {},
                                                 'WMHD': {}}}
        for metric in self.dict_best_worst_rank.keys():
            for method, para in zip(method_list, para_list):
                self.dict_best_worst_rank[metric][method].update({para: {}})
                rank_arr = np.argsort(np.vstack([self.dict_score[metric][method][para][str(TWL)] for TWL in
                                                 self.wbm_obj.target_wf_group_list]).flatten())
                self.dict_best_worst_rank[metric][method][para]['best'] = rank_arr[-1]
                self.dict_best_worst_rank[metric][method][para]['worst'] = rank_arr[0]

    def get_score_summary(self, method_list, para_list, target_wf_group_list):
        TWGL = target_wf_group_list
        dict_score = self.dict_score
        n_class = len(self.wbm_obj.target_wf_list)
        n_method = len(method_list)
        self.dict_summary = {'pre': {'avg': np.empty((n_method, n_class + 1)),
                                     'std': np.empty((n_method, n_class + 1))},
                             'acc': {'avg': np.empty((n_method, n_class + 1)),
                                     'std': np.empty((n_method, n_class + 1))},
                             'auc': {'avg': np.empty((n_method, n_class + 1)),
                                     'std': np.empty((n_method, n_class + 1))}}
        for i, (method, para) in enumerate(zip(method_list, para_list)):
            self.dict_summary['pre']['avg'][i, :n_class] = np.vstack(
                [dict_score['pre'][method][para][str(TWL)] for TWL in TWGL]).mean(axis=0)
            self.dict_summary['acc']['avg'][i, :n_class] = np.vstack(
                [dict_score['acc'][method][para][str(TWL)] for TWL in TWGL]).mean(axis=0)
            self.dict_summary['auc']['avg'][i, :n_class] = np.vstack(
                [dict_score['auc'][method][para][str(TWL)] for TWL in TWGL]).mean(axis=0)
            self.dict_summary['pre']['std'][i, :n_class] = np.vstack(
                [dict_score['pre'][method][para][str(TWL)] for TWL in TWGL]).std(axis=0)
            self.dict_summary['acc']['std'][i, :n_class] = np.vstack(
                [dict_score['acc'][method][para][str(TWL)] for TWL in TWGL]).std(axis=0)
            self.dict_summary['auc']['std'][i, :n_class] = np.vstack(
                [dict_score['auc'][method][para][str(TWL)] for TWL in TWGL]).std(axis=0)

            self.dict_summary['pre']['avg'][i, n_class] = np.vstack(
                [dict_score['pre_avg'][method][para][str(TWL)] for TWL in TWGL]).mean()
            self.dict_summary['acc']['avg'][i, n_class] = np.vstack(
                [dict_score['acc_avg'][method][para][str(TWL)] for TWL in TWGL]).mean()
            self.dict_summary['auc']['avg'][i, n_class] = np.vstack(
                [dict_score['auc_avg'][method][para][str(TWL)] for TWL in TWGL]).mean()
            self.dict_summary['pre']['std'][i, n_class] = np.vstack(
                [dict_score['pre_avg'][method][para][str(TWL)] for TWL in TWGL]).std()
            self.dict_summary['acc']['std'][i, n_class] = np.vstack(
                [dict_score['acc_avg'][method][para][str(TWL)] for TWL in TWGL]).std()
            self.dict_summary['auc']['std'][i, n_class] = np.vstack(
                [dict_score['auc_avg'][method][para][str(TWL)] for TWL in TWGL]).std()

        score_summary_arr = np.empty((n_class + 1, n_method*6))
        for i, metric in enumerate(self.dict_summary.keys()):
            for j, (method, para) in enumerate(zip(method_list, para_list)):
                column_seq = (i*n_method + j)*2
                score_summary_arr[:, column_seq] = self.dict_summary[metric]['avg'][j]
                score_summary_arr[:, column_seq+1] = self.dict_summary[metric]['std'][j]

        return score_summary_arr

    def calc_JSD_vector(self, target_id):
        time_start = time.time()
        JSD_vector = np.zeros(self.wbm_obj.data_len)
        p = self.cg.weight_vector_arr[target_id]
        for comp_id in range(self.wbm_obj.data_len):
            q = self.cg.weight_vector_arr[comp_id]
            if target_id == comp_id:
                JSD_vector[comp_id] = 0
            else:
                JSD_vector[comp_id] = JSD(p, q)
        time_end = time.time()
        time_elapsed = time_end - time_start
        return JSD_vector, time_elapsed

    def calc_EUC_vector(self, target_id):
        time_start = time.time()
        EUC_vector = np.zeros(self.wbm_obj.data_len)
        p = self.wbm_obj.data_without_nan[target_id]
        for comp_id in range(self.wbm_obj.data_len):
            q = self.wbm_obj.data_without_nan[comp_id]
            if target_id == comp_id:
                EUC_vector[comp_id] = 0
            else:
                EUC_vector[comp_id] = euclidean(p, q)
        time_end = time.time()
        time_elapsed = time_end - time_start
        return EUC_vector, time_elapsed
