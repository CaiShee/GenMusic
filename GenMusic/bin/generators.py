import torch
import geatpy as ea
from typing import Callable
from .rewarders import rewarder

from .rewarders import *


class ga_problem(ea.Problem):

    def __init__(self):
        ...

    def aimFunc(self, pop: ea.Population):
        self.func(self, pop)

    def link_func(self, func: Callable[['ga_problem', ea.Population], None]):
        self.func = func

    def set_params(self, name: str, M: int, maxormins: "list[int]", Dim: int,
                   varTypes: "list[int]", lb: "list[int]", ub: "list[int]",
                   lbin: "list[int]" = None, ubin: "list[int]" = None):
        """设置参数

        Args:
            name (str): 初始化name（函数名称，可以随意设置）
            M (int): 初始化M（目标维数）
            maxormins (list[int]): 初始化maxormins（目标最小最大化标记列表，1：最小化该目标; -1：最大化该目标）
            Dim (int): 初始化Dim（决策变量维数）
            varTypes (list[int]): 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的; 1表示是离散的）
            lb (list[int]): 决策变量下界
            ub (list[int]): 决策变量上界
            lbin (list[int], optional): 决策变量下边界（0表示不包含该变量的下边界，1表示包含） Defaults to None.
            ubin (list[int], optional): 决策变量上边界 Defaults to None.
        """
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)


class ga_generator():
    def __init__(self, rwd: rewarder) -> None:
        self.rewarder = rwd
        self.ga_pbl = ga_problem()
        self.best_tensor = None

    def set_problem_params(self, maxormins: "list[int]", Dim: int,
                           varTypes: "list[int]", lb: "list[int]", ub: "list[int]",
                           name: str = "problem", M: int = 1, lbin: "list[int]" = None, ubin: "list[int]" = None):
        """设置 problem 参数

        Args:
            name (str): 初始化name（函数名称，可以随意设置）
            M (int): 初始化M（目标维数）
            maxormins (list[int]): 初始化maxormins（目标最小最大化标记列表，1：最小化该目标; -1：最大化该目标）
            Dim (int): 初始化Dim（决策变量维数）
            varTypes (list[int]): 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的; 1表示是离散的）
            lb (list[int]): 决策变量下界
            ub (list[int]): 决策变量上界
            lbin (list[int], optional): 决策变量下边界（0表示不包含该变量的下边界，1表示包含） Defaults to None.
            ubin (list[int], optional): 决策变量上边界 Defaults to None.
        """
        self.ga_pbl.set_params(name, M, maxormins, Dim,
                               varTypes, lb, ub, lbin, ubin)

    def set_problem_aimFunc(self, func: Callable[['ga_problem', ea.Population], None]):
        self.ga_pbl.link_func(func)

    def set_population_params(self, Encoding: str = 'BG', nind: int = 40):
        """设置种群参数

        Args:
            Encoding (str, optional): 编码方式. Defaults to 'BG'.
            nind (int, optional): 种群规模. Defaults to 40.
        """

        self.Field = ea.crtfld(Encoding, self.ga_pbl.varTypes,
                               self.ga_pbl.ranges, self.ga_pbl.borders)
        self.population = ea.Population(Encoding, self.Field, nind)

    def set_algorithm_params(self, maxgen: int = 25, logTras: int = 0,
                             verbose: bool = False, drawing: int = 0):
        """设置算法参数

        Args:
            maxgen (int, optional): 最大进化代数. Defaults to 25.
            logTras (int, optional): 设置每隔多少代记录日志，若设置成0则表示不记录日志. Defaults to 0.
            verbose (bool, optional): 设置是否打印输出日志信息. Defaults to False.
            drawing (int, optional): 设置绘图方式（0：不绘图; 1：绘制结果图; 2：绘制目标空间过程动画; 3：绘制决策空间过程动画. Defaults to 0.
        """
        self.myAlgorithm = ea.soea_SEGA_templet(
            self.ga_pbl, self.population)
        self.myAlgorithm.MAXGEN = maxgen
        self.myAlgorithm.logTras = logTras
        self.myAlgorithm.verbose = verbose
        self.myAlgorithm.drawing = drawing

    def generate_ori(self, sv_path: str = "", show_log: bool = False):
        """_summary_

        Args:
            sv_path (str, optional): _description_. Defaults to "".
            show_log (bool, optional): _description_. Defaults to False.
        """
        [self.BestIndi, self.last_population] = self.myAlgorithm.run()
        if sv_path != "":
            self.BestIndi.save(sv_path)
        if show_log:
            if self.BestIndi.sizes != 0:
                print('最优的目标函数值为：%s' % self.BestIndi.ObjV[0][0])
                print('最优的序列为：')
                for i in range(self.BestIndi.Phen.shape[1]):
                    print(self.BestIndi.Phen[0, i])
            else:
                print('没找到可行解。')
        if self.BestIndi.sizes != 0:
            l = list()
            for i in range(self.BestIndi.Phen.shape[1]):
                l.append(self.BestIndi.Phen[0, i])
        self.best_tensor = torch.tensor(l)
        self.best_tensor = self.best_tensor.float()

    def aimFun_1(self, plb: 'ga_problem', pop: ea.Population):
        x = pop.Phen
        x = torch.from_numpy(x)
        x = x.float()
        y = self.rewarder.reward(x)
        y = y.detach().numpy()
        pop.ObjV = y
