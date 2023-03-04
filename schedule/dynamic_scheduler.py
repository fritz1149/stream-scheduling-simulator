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
        self.gap1 = 0.1
        self.gap2 = 0.3
        
    def schedule(self, g: ExecutionGraph) -> SchedulingResult:
        return self.schedule_multiple([g])
        
    def schedule_multiple(
        self, graph_list: typing.List[ExecutionGraph]
    ) -> typing.List[SchedulingResult]:
        # vertices是算子列表，nodes是网络节点列表
        ga(graph_list, self.scenario)
        return None

# 遗传算法
def ga(graph_list: typing.List[ExecutionGraph],
    scenario: Scenario
    ):
    data = get_lower_bound(graph_list, scenario)
    lb = data["lb"]
    n_ops = 0
    for graph in graph_list:
        n_ops += len(graph.g.nodes)
    n_net_nodes = len(scenario.topo.g.nodes)
    # 约束
    flow_node_restr = data["flow_node_restr"] 
    def post_restr(x: typing.List):
        ret = 1
        for i, pos in enumerate(x):
            ret = ret & flow_node_restr[i][pos]
        return ret
    mips_positive = data["mips_positive"]
    def node_mips_positive(x: typing.List):
        ret = 1
        for pos in x:
            ret = ret & mips_positive[pos]
        return ret
    
    constraint_eq =  [
        post_restr, mips_positive, 
    ]
    constraint_ueq = [
        
    ]
    # 超参数直接在下面改吧
    ga = GA(func=partial(gap, graph_list, scenario, lb),
            n_dim=n_ops, size_pop=50, max_iter=800, prob_mut=0.001, 
            lb=[0 for _ in range(n_ops)], ub=[n_net_nodes - 1 for _ in range(n_ops)], precision=[1 for _ in range(n_ops)])
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