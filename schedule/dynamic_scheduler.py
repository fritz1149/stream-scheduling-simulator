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
        data = dict()
        ga(graph_list, self.scenario, data)
        return None

# 遗传算法
def ga(graph_list: typing.List[ExecutionGraph],
    scenario: Scenario,
    data: dict
    ):
    lb = get_lower_bound(graph_list, scenario, data)
    n_ops = 0
    for graph in graph_list:
        n_ops += len(graph.g.nodes)
    net_node_num = data["net_node_num"]
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
    flow_endpoint = data["flow_endpoint"]
    edge_of = data["net_node_edge_index"]
    def no_edge_to_edge(x: typing.List):
        ret = 1
        for u, v in flow_endpoint:
            ret = ret & (edge_of[u] == edge_of[v] or edge_of[v] == -1)
        return ret
    slot = data["slot"]
    def slot_capacity(x: typing.List):
        ret = 1
        occupied = [0 for _ in range(net_node_num)]
        for pos in x:
            occupied[pos] = occupied[pos] + 1
        for i in range(net_node_num):
            ret = ret & (occupied[i] <= slot[i])
        return ret
    eq =  [
        no_edge_to_edge, post_restr, slot_capacity, mips_positive,
    ]
    # 超参数直接在下面改吧
    ga = GA(func=partial(gap, graph_list, scenario, lb, data),
            n_dim=n_ops, size_pop=50, max_iter=800, prob_mut=0.001, 
            lb=[0 for _ in range(n_ops)], ub=[net_node_num - 1 for _ in range(n_ops)], precision=[1 for _ in range(n_ops)],
            constraint_eq=eq)
    map_list = ga.run()
    
def branch_and_bound(graph_list: typing.List[ExecutionGraph],
    scenario: Scenario
    ):
    a = 1