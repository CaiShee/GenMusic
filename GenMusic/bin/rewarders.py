import torch
import numpy as np
import warnings
from torch import nn
import glob
import os
from dtw import dtw
# from .classifiers import classifier, nn_classifier


class rewarder():
    def __init__(self) -> None:
        ...

    def reward(self, env: "torch.Tensor|np.ndarray") -> torch.Tensor:
        ...


class demo_rewarder(rewarder):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def reward(env: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(env) == 0:
            warnings.warn("The length of env must > 0!")
            env = torch.rand(10, 2)

        output = torch.softmax(env, dim=env.dim()-1)
        output = torch.square(output)
        output = torch.sum(output, dim=env.dim()-1)
        output = torch.reshape(output, (-1, 1))
        return output


# class classify_rewarder(rewarder):
#     def __init__(self, clf: classifier) -> None:
#         super().__init__()
#         self.clf = clf
#         self.reward_item = 0

#     def reward(self, env: torch.Tensor) -> torch.Tensor:
#         """ dim of env < 3

#         Args:
#             env (torch.Tensor): _description_

#         Returns:
#             torch.Tensor: _description_
#         """

#         if len(env) == 0:
#             raise RuntimeError("The length of env must > 0!")

#         classification = self.clf.classify(env)
#         classification_dim = classification.dim()

#         if classification_dim > 2:
#             raise RuntimeError("classification_dim must < 3 !")

#         output = classification[(slice(None),) *
#                                 classification_dim + (self.reward_item,)]
#         output = torch.reshape(output, (-1, 1))

#         return output

#     def set_item_class(self, item_idx: int) -> None:
#         self.reward_item = item_idx


class nngrader_rewarder(rewarder):
    def __init__(self, grader: nn.Module) -> None:
        self.dev = torch.device('cpu')
        self.grader = grader.to(self.dev)

    def reward(self, env: torch.Tensor) -> torch.Tensor:
        b = len(env)
        scores = self.grader(env)
        loss = torch.square(scores-self.tgt)
        loss = torch.mean(loss, dim=1)
        loss = torch.reshape(loss, (b, -1))
        return loss

    def set_tgt(self, tgt: torch.Tensor) -> None:
        self.tgt = tgt


def get_nngrader_rewarder(nn_pth: str) -> nngrader_rewarder:
    nn_pretrained = torch.load(nn_pth)
    return nngrader_rewarder(nn_pretrained)


class sad_test_rewarder(rewarder):
    def __init__(self) -> None:
        super().__init__()
        self.pitchs = [0, 2, 4, 5, 7, 9]
        self.rhythms = [1, 0]

    def reward(self, env: torch.Tensor) -> torch.Tensor:
        audio_num = round(env.shape[1]/2)
        audio_pitch = env[:, :audio_num]
        audio_rhythm = env[:, audio_num:]

        rwd_score = torch.zeros((audio_pitch.shape[0], 1))
        for pitch in self.pitchs:
            tmp = torch.round(audio_pitch-pitch)
            tmp = torch.where(tmp == 0, 1, 0)
            tmp = torch.sum(tmp, dim=1)
            tmp = torch.reshape(tmp, (-1, 1))
            rwd_score += tmp
        for rhy in self.rhythms:
            tmp = torch.round(audio_rhythm-rhy)
            tmp = torch.where(tmp == 0, 0.5, 0)
            tmp = torch.sum(tmp, dim=1)
            tmp = torch.reshape(tmp, (-1, 1))
            rwd_score += tmp

        return rwd_score


class test_dtw_midi_rewarder(rewarder):
    def __init__(self, happy_path: str, sad_path: str, covInv_path: str) -> None:
        self.happy_templates = [np.load(h) for h in glob.glob(
            os.path.join(happy_path, "*"))]
        self.sad_templates = [np.load(s) for s in glob.glob(
            os.path.join(sad_path, "*"))]
        self.covInv = np.load(covInv_path)
        self.m_ds = lambda x, y: np.sqrt(
            np.dot(np.dot((x-y), self.covInv), (x-y).T))

    def reward(self, envs: np.ndarray) -> np.ndarray:
        rwd = np.zeros((len(envs), 1))
        l = envs.shape[1]//4
        for i in range(len(envs)):
            env = np.zeros((4, l))
            env[0, :] = envs[i, :l]
            env[1, :] = envs[i, l:2*l]
            env[2, :] = envs[i, 2*l:3*l]
            env[3, :] = envs[i, 3*l:]
            env = env.T

            min_dis = 1e9
            for tmp in self.sad_templates:
                d, _, _, _ = dtw(tmp, env, dist=self.m_ds)
                if d < min_dis:
                    min_dis = d
            rwd[i] = min_dis
        return rwd

    def set_tgt(self, tgt: str) -> None:
        self.tgt = tgt

    @staticmethod
    def dtw(dist_matrix: np.ndarray) -> float:
        # 获取距离矩阵的大小
        n, m = dist_matrix.shape

        # 初始化累积距离矩阵
        acc_dist_matrix = np.zeros((n, m))
        acc_dist_matrix[0, 0] = dist_matrix[0, 0]

        # 计算第一行和第一列的累积距离
        acc_dist_matrix[1:, 0] = np.cumsum(dist_matrix[1:, 0])
        acc_dist_matrix[0, 1:] = np.cumsum(dist_matrix[0, 1:])

        # 计算剩余单元格的累积距离
        for i in range(1, n):
            for j in range(1, m):
                acc_dist_matrix[i, j] = min(
                    acc_dist_matrix[i-1, j],    # 插入
                    acc_dist_matrix[i, j-1],    # 替换
                    acc_dist_matrix[i-1, j-1]   # 删除
                ) + dist_matrix[i, j]

        # 返回DTW距离（右下角的累积距离）
        return acc_dist_matrix[-1, -1]
