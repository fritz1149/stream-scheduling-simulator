import typing
import networkx as nx
import copy
from graph import ExecutionGraph, Vertex
from topo import Domain, Scenario, Node
from .result import SchedulingResult, SchedulingResultStatus
from .scheduler import RandomScheduler, Scheduler, SourcedGraph
from sko.GA import GA
from functools import partial

class DynamicScheduler(Scheduler):
    def __init__(self, scenario: Scenario) -> None:
        super.__init__(scenario)
        
    def schedule(self, g: ExecutionGraph) -> SchedulingResult:
        return self.schedule_multiple([g])
        
    def schedule_multiple(
        self, graph_list: typing.List[ExecutionGraph]
    ) -> typing.List[SchedulingResult]:
        # vertices是算子列表，nodes是网络节点列表
        data_for_solver()
        return None
def data_for_solver(graph_list: typing.List[ExecutionGraph], 
    scenario: Scenario):
    # 流式计算图相关
    graph_merge = nx.DiGraph()
    for graph in graph_list:
        graph_merge = nx.disjoint_union(graph_merge, graph.g)
    flow_node_num = len(graph_merge.nodes)
    flow_edge_num = len(graph_merge.edges)
    flow_incidence = nx.incidence_matrix(G=graph_merge, oriented=True)
    flow = [e[2]["unit_size"] * e[2]["per_second"] for e in graph_merge.edges.data()]
    flow_node_is_sink = [v[1] == "sink" for v in graph_merge.nodes.data("type")]
    in_flow_edge = copy.deepcopy(flow_incidence)
    for row in in_flow_edge:
        for x in row:
            if x == -1:
                x = 1
            else:
                x = 0
    flow_edge_s = [e[0] for e in graph_merge.edges.data]
    # 实际网络相关
    net = scenario.topo.g
    net_node_num = len(net.nodes)
    net_edge_num = len(net.edges)
    edge_index = dict()
    for i, edge in net.edges:
        edge_index[edge] = i
    node_uuid = [None] * net_node_num
    node_index = dict()
    for i, uuid in net.nodes:
        node_uuid[i] = uuid
        node_index[uuid] = i
    # u和v之间的path，下标为u*net_node_num + v
    net_path_num = net_node_num * net_node_num
    edge_domain_num = len(scenario.domains)
    flow_node_restr = [[0] * flow_node_num] * net_node_num
    for i, node in graph_merge.nodes.data("domain_constraint"):
        if (h := node[1].get("host")) != None:
            flow_node_restr[i][node_index[h]] = 1
    net_incidence = nx.incidence_matrix(G=net, oriented=True)
    
    net_path_incidence = [[0] * net_node_num] * net_node_num
    for u in range(0, net_node_num):
        for v in range(0, net_node_num):
            path = u * net_node_num + v
            net_path_incidence[u][path] = 1
            net_path_incidence[v][path] = -1
            
    net_edge_in_path = [[0] * net_path_num] * net_edge_num
    for u in range(0, net_node_num):
        for v in range(0, net_node_num):
            path = u * net_node_num + v
            p = nx.all_simple_paths(net, source=node_uuid[u], target=node_uuid[v])
            p = list(map(nx.utils.pairwise, p))[0]
            for edge in list(p):
                net_edge_in_path[edge_index[edge]][path] = 1
    bandwidth = [e[2] for e in net.edges.data("bd")]
    net_edge_intr_lat = [e[2] for e in net.edges.data("delay")]
    mips = [e[1] for e in net.nodes.data("mips")]
    cores = [e[1] for e in net.nodes.data("cores")]
    slot = [e[1] for e in net.nodes.data("slots")]
    
    net_node_in_edge = [[0] * edge_domain_num] * net_node_num
    net_node_in_cloud = [0] * net_node_num
    for i, domain in enumerate(scenario.domains):
        if domain.type == "cloud":
            for node in domain.topo.g.nodes:
                net_node_in_cloud[node_index[node]] = 1
        else:
            for node in domain.topo.g.nodes:
                net_node_in_edge[node_index[node]][i] = 1
    with
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