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
from sko.tools import set_run_mode
from pyscipopt import Model, quicksum
from functools import partial
from itertools import product

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
        debug = True
        if debug:
            with open("main_data.json", "r") as f:
                load_data = json.loads(f.read())
            data.update(load_data)
        else:
            parse_data(graph_list, self.scenario, data)
            get_lower_bound(graph_list, self.scenario, data)
            with open("main_data.json", "w") as f:
                f.write(json.dumps(data))
        lb = data["lb"]
        # ga(graph_list, self.scenario, data)
        bnb(graph_list, self.scenario, data)
        return None

# 遗传算法
def ga(graph_list: typing.List[ExecutionGraph],
    scenario: Scenario,
    data: dict
    ):
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
    ga_func = partial(gap, lb, data)
    # set_run_mode(ga_func, "cached")
    ga = GA(func=ga_func,
            n_dim=n_ops, size_pop=50, max_iter=800, prob_mut=0.001, 
            lb=[0 for _ in range(n_ops)], ub=[net_node_num - 1 for _ in range(n_ops)], precision=[1 for _ in range(n_ops)],
            constraint_eq=eq)
    t1 = time.time()
    res = ga.run()
    t2 = time.time()
    print('程序运行时间:%s毫秒' % ((t2 - t1)*1000))
    
    map_list = res[0]
    goal = obj(map_list, data, True)
    # print("f:", map_list.astype(np.int32).tolist())
    print("ga_obj:", res[1])
    print("no_edge_to_edge:", no_edge_to_edge(map_list))
    print("post_restr:", post_restr(map_list))
    print("slot_capacity:", slot_capacity(map_list))
    print("node_mips_positive:", node_mips_positive(map_list))
    print("\nobj: %f, lb: %f, gap: %f"%(goal, lb, (goal - lb) / lb))
    
def bnb(graph_list: typing.List[ExecutionGraph],
    scenario: Scenario,
    data: dict
    ):
    model = Model("schedule-main")  # model name is optional
    flow_node_num = data["flow_node_num"]
    flow_edge_num = data["flow_edge_num"]
    net_node_num = data["net_node_num"]
    net_edge_num = data["net_edge_num"]
    net_path_num = data["net_path_num"]
    # 决策变量
    f = {}
    for flow_node in range(flow_node_num):
        for net_node in range(net_node_num):
            f[flow_node, net_node] = model.addVar(vtype="BINARY", 
                                                  name="f(%d,%d)"%(flow_node,net_node))
    # 流式边与网络路径映射相关中间变量
    flow_edge_as_net_path = {}
    flow_edge_as_net_path_origin = {}
    flow_edge_as_net_path_dest = {}
    for flow_edge in range(flow_edge_num):
        for net_path in range(net_path_num):
            flow_edge_as_net_path[flow_edge, net_path] = model.addVar(vtype="BINARY",
                                    name="flow_edge_as_net_path(%d,%d)"%(flow_edge,net_path))
            flow_edge_as_net_path_origin[flow_edge, net_path] = model.addVar(vtype="BINARY",
                                    name="flow_edge_as_net_path_origin(%d,%d)"%(flow_edge,net_path))
            flow_edge_as_net_path_dest[flow_edge, net_path] = model.addVar(vtype="BINARY",
                                    name="flow_edge_as_net_path_dest(%d,%d)"%(flow_edge,net_path))
    net_edge_in_flow_edge = {}
    for net_edge in range(net_edge_num):
        for flow_edge in range(flow_edge_num):
            net_edge_in_flow_edge[net_edge, flow_edge] = model.addVar(vtype="BINARY",
                                    name="net_edge_in_flow_edge(%d,%d)"%(net_edge,flow_edge))
    out_flow_edge = data["out_flow_edge"]
    in_flow_edge = data["in_flow_edge"]
    net_path_origin = data["net_path_origin"]
    net_path_dest = data["net_path_dest"]
    net_edge_in_path = data["net_edge_in_path"]
    for flow_edge in range(flow_edge_num):
        for net_path in range(net_path_num):
            model.addCons(quicksum(
                out_flow_edge[flow_node, flow_edge] * f[flow_node, net_node] * net_path_origin[net_node, net_path]
                for flow_node, net_node in product(range(flow_node_num, net_node_num))
            ) == flow_edge_as_net_path_origin[flow_edge, net_path],
                          name="flow_edge_as_net_path_origin(%d,%d)"%(flow_edge,net_path))
            model.addCons(quicksum(
                in_flow_edge[flow_node, flow_edge] * f[flow_node, net_node] * net_path_dest[net_node, net_path]
                for flow_node, net_node in product(range(flow_node_num, net_node_num))
            ) == flow_edge_as_net_path_dest[flow_edge, net_path],
                          name="flow_edge_as_net_path_dest(%d,%d)"%(flow_edge,net_path))
            model.addConsAnd([flow_edge_as_net_path_origin[flow_edge, net_path],
                              flow_edge_as_net_path_dest[flow_edge, net_path]],
                             flow_edge_as_net_path[flow_edge, net_path])
    for net_edge in range(net_edge_num):
        for flow_edge in range(flow_edge_num):
            model.addCons(quicksum(net_edge_in_path[net_edge][net_path] * flow_edge_as_net_path[flow_edge, net_path] for net_path in range(net_path_num))
                          == net_edge_in_flow_edge[net_edge, flow_edge],
                          name="net_edge_in_flow_edge(%d,%d)"%(net_edge,flow_edge))
    # 延迟相关中间变量
    comp_lat = {}
    op_lat = {}
    mi = data["mi"]
    mips = data["mips"]
    mips_reciprocal = copy.deepcopy(data["mips_reciprocal"])
    for i in range(net_node_num):
        mips_reciprocal[i] = float(mips_reciprocal[i])
    for flow_node in range(flow_node_num):
        comp_lat[flow_node] = model.addVar(vtype="CONTINUOUS", name="comp_lat(%d)"%flow_node)
        op_lat[flow_node] = model.addVar(vtype="CONTINUOUS", name="op_lat(%d)"%flow_node)
        model.addCons(quicksum(mi[flow_node] * f[flow_node, net_node] * mips_reciprocal[net_node] for net_node in range(net_node_num))
                      == comp_lat[flow_node],
                      name="comp_lat(%d)"%flow_node)
    intr_lat = {}
    tran_lat = {}
    net_edge_intr_lat = data["net_edge_intr_lat"]
    bandwidth = data["bandwidth"]
    tuple_size = data["tuple_size"]
    for flow_edge in range(flow_edge_num):
        intr_lat[flow_edge] = model.addVar(vtype="CONTINUOUS", name="intr_lat(%d)"%flow_edge)
        tran_lat[flow_edge] = model.addVar(vtype="CONTINUOUS", name="tran_lat(%d)"%flow_edge)
        model.addCons(quicksum(net_edge_in_flow_edge[net_edge, flow_edge] * net_edge_intr_lat[net_edge] for net_edge in range(net_edge_num))
                      == intr_lat[flow_edge],
                      name="intr_lat(%d)"%flow_edge)
        model.addCons(quicksum(net_edge_in_flow_edge[net_edge, flow_edge] * tuple_size[flow_edge] / bandwidth[net_edge] for net_edge in range(net_edge_num))
                      == tran_lat[flow_edge],
                      name="tran_lat(%d)"%flow_edge)
    flow = data["flow"]
    in_flow_sum_reciprocal = copy.deepcopy(data["in_flow_sum_reciprocal"])
    for i in range(flow_node_num):
        in_flow_sum_reciprocal[i] = float(in_flow_sum_reciprocal[i])
    op_in_edge = data["op_in_edge"]
    flow_endpoint = data["flow_endpoint"]
    for flow_node in range(flow_node_num):
        model.addCons(comp_lat[flow_node] + quicksum(flow[flow_edge] * (tran_lat[flow_edge] + intr_lat[flow_edge] +
                    quicksum(op_lat[edge_origin] for edge_origin, _ in flow_endpoint)) for flow_edge in op_in_edge[flow_node])
                    * in_flow_sum_reciprocal[flow_node] == op_lat[flow_node],
                    name="op_lat(%d)"%flow_node)
    lat = model.addVar(vtype="CONTINUOUS", name="lat")
    sinks = data["sinks"]
    in_flow_sum = data["in_flow_sum"]
    sink_in_flow_sum = data["sink_in_flow_sum"]
    model.addCons(quicksum(op_lat[sink] * in_flow_sum[sink] for sink in sinks) / sink_in_flow_sum == lat,
                  name="lat")
    # 云边流量相关中间变量
    flow_edge_cross = {}
    for flow_edge in range(flow_edge_num):
        flow_edge_cross[flow_edge] = model.addVar(vtype="BINARY", name="flow_edge_cross(%d)"%flow_edge)
    flow_node_in_cloud = {}
    net_node_in_cloud = data["net_node_in_cloud"]
    for flow_node in range(flow_node_num):
        flow_node_in_cloud[flow_node] = model.addVar(vtype="BINARY", name="flow_node_in_cloud(%d)"%flow_node)
        model.addCons(quicksum(f[flow_node, net_node] * net_node_in_cloud[net_node] for net_node in range(net_node_num))
                      == flow_node_in_cloud[flow_node],
                      name="flow_node_in_cloud(%d)"%flow_node)
    flow_incidence = data["flow_incidence"]
    for flow_edge in range(flow_edge_num):
        model.addCons(-1 * quicksum(flow_incidence[flow_node][flow_edge] * flow_node_in_cloud[flow_node] for flow_node in range(flow_node_num))
                      == flow_edge_cross[flow_edge],
                      name="flow_edge_cross(%d)"%flow_edge)
    flow_cross = model.addVar(vtype="CONTINUOUS", name="flow_cross")
    model.addCons(quicksum(flow_edge_cross[flow_edge] * flow[flow_edge] for flow_edge in range(flow_edge_num))
                  == flow_cross,
                  name="flow_cross")
    # 不允许边边通信、从云到边的通信
    edge_domain_num = data["edge_domain_num"]
    net_node_in_edge = data["net_node_in_edge"]
    for flow_edge in range(flow_edge_num):
        for edge_domain in range(edge_domain_num):
            model.addCons(quicksum(flow_incidence[flow_node][flow_edge] * f[flow_node, net_node] * net_node_in_edge[net_node][edge_domain] for flow_node, net_node in product(range(flow_node_num), range(net_node_num)))
                          <= 1,
                          name="no_edge_to_edge_1(%d,%d)"%(flow_edge, edge_domain))
            model.addCons(quicksum(flow_incidence[flow_node][flow_edge] * f[flow_node, net_node] * net_node_in_edge[net_node][edge_domain] for flow_node, net_node in product(range(flow_node_num), range(net_node_num)))
                          >= 0,
                          name="no_edge_to_edge_2(%d,%d)"%(flow_edge, edge_domain))
    # 特定算子部署到特定节点集的限制
    flow_node_restr = data["flow_node_restr"]
    for flow_node in range(flow_node_num):
        for net_node in range(net_node_num):
            model.addCons(f[flow_node, net_node] * flow_node_restr[flow_node][net_node] == f[flow_node, net_node],
                          "pos_restr(%d,%d)"%(flow_node,net_node))
    # 计算节点插槽数量限制
    slot = data["slot"]
    for net_node in range(net_node_num):
        model.addCons(quicksum(f[flow_node, net_node] for flow_node in range(flow_node_num)) <= slot[net_node],
                      "slot_capacity(%d)"%net_node)
    # 一个算子只能部署到一个计算节点
    for flow_node in range(flow_node_num):
        model.addCons(quicksum(f[flow_node, net_node] for net_node in range(net_node_num)) == 1,
                      "1op_1node(%d)"%flow_node)
    # 算子只能部署到mips非0的节点上
    mips_positive = data["mips_positive"]
    for flow_node in range(flow_node_num):
        model.addCons(quicksum(f[flow_node, net_node] * mips_positive[net_node] for net_node in range(net_node_num)) >= 1,
                      "f_mips_positive(%d)"%flow_node)
    # 目标
    flow_min = data["flow_min"]
    lat_min = data["lat_min"]
    model.setObjective(flow_cross/flow_min + lat/lat_min, sense="maximize")
    model.optimize() 
    sol = model.getBestSol()
    print("f:")
    for flow_node in range(flow_node_num):
        for net_node in range(net_node_num):
            print(int(sol[f[flow_node, net_node]]), end="")
        print()