import typing
import networkx as nx
import numpy as np
import math
import copy
import time
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
    # lb = get_lower_bound(graph_list, scenario, data)
    # with open("main_data.json", "w") as f:
    #     f.write(json.dumps(data))
    # return 
    lb = 0
    with open("main_data.json", "r") as f:
        data = json.loads(f.read())
        lb = data["lb"]
        
    n_ops = data["flow_node_num"]
    net_node_num = data["net_node_num"]
    # 约束
    flow_node_restr = data["flow_node_restr"] 
    def post_restr(x: typing.List):
        fault = 0
        for i, pos in enumerate(x):
            fault = fault + 1 - flow_node_restr[i][int(pos)]
        return fault
    mips_positive = data["mips_positive"]
    def node_mips_positive(x: typing.List):
        fault = 0
        for pos in x:
            fault = fault + 1 - mips_positive[int(pos)]
        return fault
    flow_endpoint = data["flow_endpoint"]
    edge_of = data["net_node_edge_index"]
    def no_edge_to_edge(x: typing.List):
        fault = 0
        for u, v in flow_endpoint:
            a = int(x[u])
            b = int(x[v])
            fault = fault + 1 - (edge_of[a] == edge_of[b] or edge_of[b] == -1)
        return fault
    slot = data["slot"]
    def slot_capacity(x: typing.List):
        fault = 0
        occupied = [0 for _ in range(net_node_num)]
        for pos in x:
            occupied[int(pos)] = occupied[int(pos)] + 1
        for i in range(net_node_num):
            fault = fault + 1 - (occupied[i] <= slot[i])
        return fault
    eq =  [
        no_edge_to_edge, post_restr, slot_capacity, node_mips_positive,
    ]
    # 超参数直接在下面改吧
    ga = GA(func=partial(gap, lb, data),
            n_dim=n_ops, size_pop=50, max_iter=800, prob_mut=0.001, 
            lb=[0 for _ in range(n_ops)], ub=[net_node_num - 1 for _ in range(n_ops)], precision=[1 for _ in range(n_ops)],
            constraint_eq=eq)
    t1 = time.time()
    map_list = ga.run()
    t2 = time.time()
    print('程序运行时间:%s毫秒' % ((t2 - t1)*1000))
    
    map_list = map_list[0].astype(np.int32).tolist()
    goal = obj(map_list, data, True)
    print("f:", map_list)
    print("obj: %f, lb: %f, gap: %f"%(goal, lb, (goal - lb) / lb))
    
def branch_and_bound(graph_list: typing.List[ExecutionGraph],
    scenario: Scenario
    ):
    a = 1