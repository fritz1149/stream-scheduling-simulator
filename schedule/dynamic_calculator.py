import typing
import networkx as nx
import math
import copy
import numpy as np
import subprocess
from graph import ExecutionGraph
from topo import Scenario

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

def get_flow_min(graph_list: typing.List[ExecutionGraph]) -> float:
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
    return flow_min

def get_flow_max(graph_list: typing.List[ExecutionGraph]) -> float:
    flow_max = 0
    for graph in graph_list:
        # print(graph.g.edges)
        node_num = len(graph.g.nodes)
        edge_num = len(graph.g.edges)
        incidence = nx.incidence_matrix(G=graph.g).toarray().astype(np.int32).tolist()
        endpoint = [[-1, -1] for _ in range(edge_num)]
        for edge in range(edge_num):
            for node in range(node_num):
                if incidence[node][edge] == 1:
                    if endpoint[edge][0] == -1:
                        endpoint[edge][0] = node
                    else:
                        endpoint[edge][1] = node
        flow = [e[2] for e in graph.g.edges.data("flow")]
        nodes = graph.g.nodes.data("type")
        sources = [i for i, node in enumerate(nodes) if node[1] == "source"]
        sinks = [i for i, node in enumerate(nodes) if node[1] == "sink"]
        source_num = len(sources)
        sink_num = len(sinks)
        param_list = [
            "node_num", "edge_num", "endpoint", "flow", "source_num", "sink_num",
            "sources", "sinks"
        ]
        ret = dict()
        items = locals()
        for param in param_list:
            ret[param] = items[param]
        with open("../glpk/data/flow_max_data", "w") as f:
            gen_data_file(f, ret)
        result = subprocess.run(
            ["glpsol", "--model", "../glpk/model/max_cut.mod", "--data", "../glpk/data/flow_max_data"],
            capture_output=True
        )
        output = result.stdout.decode("ASCII")
        # print(output)
        cut = output.split("\n")[-3].strip()
        flow_max = flow_max + float(cut)
    return flow_max
    
def get_lat_min(graph_list: typing.List[ExecutionGraph], 
    scenario: Scenario) -> float:
    return 0
    
def get_lat_max(graph_list: typing.List[ExecutionGraph], 
    scenario: Scenario) -> float:
    return 0
    
def get_lower_bound(graph_list: typing.List[ExecutionGraph], 
    scenario: Scenario):
    # 流式计算图相关
    graph_merge = nx.DiGraph()
    for graph in graph_list:
        graph_merge = nx.disjoint_union(graph_merge, graph.g)
    flow_node_num = len(graph_merge.nodes)
    flow_edge_num = len(graph_merge.edges)
    flow_incidence = nx.incidence_matrix(G=graph_merge, oriented=True).toarray().astype(np.int32).tolist()
    mi = [e[1] for e in graph_merge.nodes.data("mi")]
    flow = [e[2] for e in graph_merge.edges.data("flow")]
    flow_node_is_sink = [1 if v[1] == "sink" else 0 for v in graph_merge.nodes.data("type")]
    tuple_size = [e[2] for e in graph_merge.edges.data("unit_size")]
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
    # flow_min = get_flow_min(graph_list)
    # flow_max = get_flow_max(graph_list)
    # lat_min = get_lat_min(graph_list, scenario)
    # lat_max = get_lat_max(graph_list, scenario)
    
    param_list = [
        "flow_node_num", "flow_edge_num", "flow_incidence", "mi", "flow", "flow_node_is_sink", "tuple_size",
        "in_flow_edge", "flow_edge_s", "net_node_num", "net_edge_num", "net_path_num", "edge_domain_num",
        "flow_node_restr", "net_incidence", "net_path_incidence", "net_edge_in_path", "bandwidth", 
        "net_edge_intr_lat", "mips", "cores", "slot", "net_node_in_edge", "net_node_in_cloud",
        # "flow_min", "flow_max"
    ]
    ret = dict()
    items = locals()
    for param in param_list:
        ret[param] = items[param]
    with open("../glpk/data/lower_bound_data", "w") as f:
        gen_data_file(f, ret)