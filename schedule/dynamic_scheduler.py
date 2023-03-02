import typing
import networkx as nx
import numpy as np
import math
import copy
import json
from graph import ExecutionGraph, Vertex
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
        data_for_solver(graph_list, self.scenario)
        return None

def gen_data_file(f: typing.TextIO, data: typing.Dict):
    f.write("data;\n")
    def write_variable(name: str, data):
        f.write("param %s := %s;\n"%(name, str(data)))
    def write_array(name: str, data: typing.List):
        f.write("param %s :="%name)
        for i, x in enumerate(data):
            f.write("\n\t%d %s"%(i, str(x)))
        f.write(";\n")
    def write_matrix(name: str, data: typing.List):
        f.write("param %s : "%name)
        b = len(data[0])
        for i in range(b):
            f.write("%d "%i)
        f.write(":=")
        for i, l in enumerate(data):
            f.write("\n\t%d "%i)
            for x in l:
                f.write("%s "%str(x))
        f.write(";\n")
    for name, content in data.items():
        if type(content) != list:
            write_variable(name, content)
        elif type(content[0]) != list:
            write_array(name, content)
        else:
            write_matrix(name, content)
        
    
def data_for_solver(graph_list: typing.List[ExecutionGraph], 
    scenario: Scenario):
    # 流式计算图相关
    
    graph_merge = nx.DiGraph()
    for graph in graph_list:
        graph_merge = nx.disjoint_union(graph_merge, graph.g)
    flow_node_num = len(graph_merge.nodes)
    flow_edge_num = len(graph_merge.edges)
    flow_incidence = nx.incidence_matrix(G=graph_merge, oriented=True).toarray().astype(np.int32).tolist()
    mi = [e[1] for e in graph_merge.nodes.data("mi")]
    flow = [e[2]["unit_size"] * e[2]["per_second"] for e in graph_merge.edges.data()]
    flow_node_is_sink = [1 if v[1] == "sink" else 0 for v in graph_merge.nodes.data("type")]
    in_flow_edge = copy.deepcopy(flow_incidence)
    for row in in_flow_edge:
        for i in range(len(row)):
            if row[i] == -1:
                row[i] = 1
            else:
                row[i] = 0
    flow_edge_s = [e[0] for e in graph_merge.edges]
    # 实际网络相关
    net = scenario.topo.g
    net_node_num = len(net.nodes)
    net_edge_num = len(net.edges)
    edge_index = dict()
    for i, edge in enumerate(net.edges):
        u, v = edge
        edge_index[edge] = i
        edge_index[(v, u)] = i
    node_uuid = [None] * net_node_num
    node_index = dict()
    for i, uuid in enumerate(net.nodes):
        node_uuid[i] = uuid
        node_index[uuid] = i
    # u和v之间的path，下标为u*net_node_num + v
    net_path_num = net_node_num * net_node_num
    edge_domain_num = len(scenario.domains)
    flow_node_restr = [[0 for i in range(net_node_num)] for j in range(flow_node_num)]
    for i, node in enumerate(graph_merge.nodes.data("domain_constraint")):
        if (h := node[1].get("host")) != None:
            flow_node_restr[i][node_index[h]] = 1
    net_incidence = nx.incidence_matrix(G=net, oriented=True).toarray().astype(np.int32).tolist()
    
    net_path_incidence = [[0 for i in range(net_path_num)] for j in range(net_node_num)]
    for u in range(0, net_node_num):
        for v in range(0, net_node_num):
            path = u * net_node_num + v
            net_path_incidence[u][path] = 1
            net_path_incidence[v][path] = -1
            
    net_edge_in_path = [[0 for i in range(net_path_num)] for j in range(net_edge_num)]
    for u in range(0, net_node_num):
        for v in range(0, net_node_num):
            if u == v:
                continue
            path = u * net_node_num + v
            p = nx.all_simple_paths(net, source=node_uuid[u], target=node_uuid[v])
            # print(u, v, node_uuid[u], node_uuid[v], p)
            p = list(map(nx.utils.pairwise, p))[0]
            for edge in list(p):
                net_edge_in_path[edge_index[edge]][path] = 1
    bandwidth = [e[2] for e in net.edges.data("bd")]
    net_edge_intr_lat = [e[2] for e in net.edges.data("delay")]
    mips = [e[1] for e in net.nodes.data("mips")]
    cores = [e[1] for e in net.nodes.data("cores")]
    slot = [e[1] for e in net.nodes.data("slots")]
    
    net_node_in_edge = [[0 for i in range(edge_domain_num)] for j in range(net_node_num)]
    net_node_in_cloud = [0] * net_node_num
    for i, domain in enumerate(scenario.domains):
        if domain.type == "cloud":
            for node in domain.topo.g.nodes:
                net_node_in_cloud[node_index[node]] = 1
        else:
            for node in domain.topo.g.nodes:
                net_node_in_edge[node_index[node]][i] = 1
    # 求所有流式计算边最小割的和
    flow_min = 0
    for graph in graph_list:
        nodes = graph.g.nodes.data("type")
        sources = [node[0] for node in nodes if node[1] == "source"]
        sinks = [node[0] for node in nodes if node[1] == "sink"]
        super_source = "super_source"
        super_sink = "super_sink"
        for source in sources:
            graph.g.add_edge(super_source, source, flow = math.inf)
        for sink in sinks:
            graph.g.add_edge(sink, super_sink, flow = math.inf)
        flow_min = flow_min + nx.minimum_cut_value(flowG=graph.g, _s=super_source, _t=super_sink, capacity="flow")
        graph.g.remove_node(super_source)
        graph.g.remove_node(super_sink)
    # 求网络链路中最大/小延迟的链路的延迟
    lat_min = math.inf
    lat_max = 0
    for graph in graph_list:
        nodes = graph.g.nodes.data("type")
        sources = [node[0] for node in nodes if node[1] == "source"]
        sinks = [node[0] for node in nodes if node[1] == "sink"]
        # 求出网络链路中不算算子计算时间的延迟最大/最小的路径
        # 再加上对流式路径上所有算子计算时间的较大/较小估计
        # 就得到了对网络链路中最大/最小延迟的估计
    
    param_list = [
        "flow_node_num", "flow_edge_num", "flow_incidence", "mi", "flow", "flow_node_is_sink",
        "in_flow_edge", "flow_edge_s", "net_node_num", "net_edge_num", "net_path_num", "edge_domain_num",
        "flow_node_restr", "net_incidence", "net_path_incidence", "net_edge_in_path", "bandwidth", 
        "net_edge_intr_lat", "mips", "cores", "slot", "net_node_in_edge", "net_node_in_cloud",
        "flow_min"
    ]
    ret = dict()
    items = locals()
    for param in param_list:
        # print(param, type(items[param]))5\
        ret[param] = items[param]
    with open("../glpk/data", "w") as f:
        gen_data_file(f, ret)
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