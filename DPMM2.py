import sys
import random
import numpy as np
import scipy.stats as stats

from tqdm import *
from wbm_util_func import plotPoints


class DPMM:

    def __init__(self):
        self.cntClusters_list_over_iter = []
        self.setClusterAssignment = []  # 각 cluster 의 assign 된 data 의 index 를 저장
        self.paramClusterSigma = []  # 각 cluster 의 sigma parameter 를 저장
        self.paramClusterMu = []  # 각 cluster 의 mu parameter 를 저장
        self.cntClusterAssignment = []  # 각 cluster 에 assign 된 data 개수를 counting

    def cluster(self, data, gamma=0.001, itr=10,
                visualize=False, printOut=False, tqdm_on=False, truncate=sys.maxsize):
        self.data = data  # data 를 저장
        self.gamma = gamma  # gamma 값을 저장
        self.cntClusters = 0  # cluster 의 개수
        self.idxClusterAssignment = np.ndarray(shape=(len(data)), dtype=int)  # 각 data 의 assign 된 cluster index 를 저장

        # 초기 cluster(table) 지정
        for idx in range(len(data)):
            self.idxClusterAssignment[idx] = -1

        if tqdm_on:
            itr_range = trange(itr)
        else:
            itr_range = range(itr)

        for i in itr_range:  # while sampling iteration
            for idx in range(len(data)):  # while each data instance in the dataset
                # instance 의 현재 assignment 지워주기
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
                    normalize = normalize + self.cntClusterAssignment[itrCluster]
                if self.cntClusters < truncate:  # cluster 개수의 상계(upper limit)를 초과하지 않은 경우
                    prior.append(self.gamma)
                    normalize = normalize + self.gamma
                for itrCluster in range(len(prior)):
                    prior[itrCluster] = prior[itrCluster] / normalize

                # posterior 값 계산 과정. posterior = prior * likelihood
                instance = data[idx]
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
                idxSampledCluster = self.sampleFromDistribution(posterior)

                # parameter mu, sigma 업데이트
                if idxSampledCluster != self.cntClusters:  # 현재 instance 가 기존 cluster 로 assign 되었을 때
                    self.idxClusterAssignment[idx] = int(idxSampledCluster)
                    self.setClusterAssignment[idxSampledCluster].append(idx)
                    self.cntClusterAssignment[idxSampledCluster] = self.cntClusterAssignment[idxSampledCluster] + 1
                    dataComponent = np.ndarray(shape=(len(self.setClusterAssignment[idxSampledCluster]), len(data[0])),
                                               dtype=np.float32)
                    for idxComponentSample in range(len(self.setClusterAssignment[idxSampledCluster])):
                        dataComponentInstance = data[self.setClusterAssignment[idxSampledCluster][idxComponentSample]]
                        for idxDimension in range(len(dataComponentInstance)):
                            dataComponent[idxComponentSample][idxDimension] = dataComponentInstance[idxDimension]
                    self.paramClusterMu[idxSampledCluster] = np.mean(dataComponent, axis=0).tolist()
                    self.paramClusterSigma[idxSampledCluster] = (
                            np.cov(dataComponent.T) + np.identity(len(instance)) * 1.0 / self.cntClusterAssignment[
                        idxSampledCluster]).tolist()
                else:  # 현재 instance 가 새 cluster 로 assign 되었을 때
                    self.idxClusterAssignment[idx] = int(idxSampledCluster)
                    self.cntClusters = self.cntClusters + 1
                    self.setClusterAssignment.append([idx])
                    self.cntClusterAssignment.append(1)
                    self.paramClusterMu.append([])
                    self.paramClusterSigma.append([])
                    self.paramClusterMu[idxSampledCluster] = instance.tolist()
                    self.paramClusterSigma[idxSampledCluster] = (np.identity(len(instance)) * 10.0).tolist()

            self.cntClusters_list_over_iter.append(self.cntClusters)
            # visualization
            if printOut:
                print('#####################################################')
                print('Iteration ', i + 1)
                print('#####################################################')
                self.printOut()
            if visualize:
                plotPoints(data, self.idxClusterAssignment, self.paramClusterMu, self.paramClusterSigma, numLine=1)

    def sampleFromDistribution(self, dist):
        draw = random.uniform(0, 1)
        for itr in range(len(dist) - 1):
            if draw < dist[itr]:
                return itr
        return len(dist) - 1

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

    def plot(self, data):
        plotPoints(data, self.idxClusterAssignment, self.paramClusterMu, self.paramClusterSigma, numLine=1)
