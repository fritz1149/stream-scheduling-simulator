import typing
import networkx as nx
import numpy as np
import math
import copy
import json
from graph import ExecutionGraph, Vertex
from .dynamic_calculator import *
from topo import Domain, Scenario, Node
from .result import SchedulingResult, SchedulingResultStatus
from .scheduler import RandomScheduler, Scheduler, SourcedGraph
from sko.GA import GA
from functools import partial

class DynamicScheduler(Scheduler):
    def __init__(self, scenario: Scenario) -> None:
        super().__init__(scenario)
        
    def schedule(self, g: ExecutionGraph) -> SchedulingResult:
        return self.schedule_multiple([g])
        
    def schedule_multiple(
        self, graph_list: typing.List[ExecutionGraph]
    ) -> typing.List[SchedulingResult]:
        # vertices是算子列表，nodes是网络节点列表
        get_lower_bound(graph_list, self.scenario)
        return None
    
# 计算特定任务（网络环境、流式计算图）的下界，对目标函数的下界函数采取线性函数的手段
def lower_bound(graph: ExecutionGraph, 
    scenario: Scenario
    ) -> float:
    
    return 0.0

def schedule_main(graph: ExecutionGraph, 
    scenario: Scenario
    ) -> None:
    # 超参数直接在下面改吧
    lb = lower_bound(graph, scenario)
    ga = GA(func=partial(gap, graph, scenario, lb),
            n_dim=len(graph.nodes), size_pop=50, max_iter=800, prob_mut=0.001, 
            lb=[-1, -1], ub=[1, 1], precision=1e-7)
    map_list = ga.run()
    
# 计算特定部署方案的目标值
def f(graph: ExecutionGraph, 
    scenario: Scenario,
    map_list: typing.List[int]
    ) -> float:
    return 0.0

def gap(graph: ExecutionGraph, 
    scenario: Scenario,
    lb: float,
    map_list: typing.List[int],
    ) -> float:
    x = f(graph, scenario, map_list)
    return (x - lb) / lb