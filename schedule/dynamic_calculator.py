import typing
import networkx as nx
import math
import copy
import numpy as np
import subprocess
import json
from graph import ExecutionGraph
from topo import Scenario

def gen_data_file(f: typing.TextIO, data: typing.Dict, param_list: typing.List):
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
    for name in param_list:
        content = data.get(name)
        if type(content) != list:
            write_variable(name, content)
        elif type(content[0]) != list:
            write_array(name, content)
        else:
            write_matrix(name, content)
    f.write("end;\n")

def parse_data(graph_list: typing.List[ExecutionGraph], 
    scenario: Scenario, data:dict):
    graph_merge = nx.DiGraph()
    for graph in graph_list:
        graph_merge = nx.disjoint_union(graph_merge, graph.g)
    flow_node_num = len(graph_merge.nodes)
    flow_edge_num = len(graph_merge.edges)
    flow_incidence = nx.incidence_matrix(G=graph_merge, oriented=True).toarray().astype(np.int32).tolist()
    for row in flow_incidence:
        for i in range(len(row)):
            row[i] = -row[i]
    mi = [e[1] for e in graph_merge.nodes.data("mi")]
    flow = [e[2] for e in graph_merge.edges.data("flow")]
    urate = [e[2] for e in graph_merge.edges.data("per_second")]
    flow_node_is_sink = [1 if v[1] == "sink" else 0 for v in graph_merge.nodes.data("type")]
    tuple_size = [e[2] for e in graph_merge.edges.data("unit_size")]
    in_flow_edge = [[0 for _ in range(flow_edge_num)] for _ in range(flow_node_num)]
    out_flow_edge = [[0 for _ in range(flow_edge_num)] for _ in range(flow_node_num)]
    for i in range(flow_node_num):
        for j in range(flow_edge_num):
            if flow_incidence[i][j] == 0:
                continue
            elif flow_incidence[i][j] == -1:
                in_flow_edge[i][j] = 1
            else:
                out_flow_edge[i][j] = 1
    flow_edge_s = [e[0] for e in graph_merge.edges]
    in_urate_sum = [0 for _ in range(flow_node_num)]
    in_urate_sum_reciprocal = [0 for _ in range(flow_node_num)]
    for node in range(flow_node_num):
        for edge in range(flow_edge_num):
            in_urate_sum[node] += urate[edge] * in_flow_edge[node][edge]
    sink_in_urate_sum = 0
    # for node in range(flow_node_num):
    #     print("%d %d\n"%(flow_node_is_sink[node], in_urate_sum_reciprocal[node]))
    for node in range(flow_node_num):
        if flow_node_is_sink[node] == 1:
            sink_in_urate_sum = sink_in_urate_sum + in_urate_sum[node]
        if in_urate_sum[node] != 0:
            in_urate_sum_reciprocal[node] = 1 / in_urate_sum[node]
        in_urate_sum_reciprocal[node] = np.format_float_positional(in_urate_sum_reciprocal[node], trim='-')
    print("sink_in_urate_sum: {}".format(sink_in_urate_sum))
    
    in_flow_sum = [0 for _ in range(flow_node_num)]
    in_flow_sum_reciprocal = [0 for _ in range(flow_node_num)]
    for node in range(flow_node_num):
        for edge in range(flow_edge_num):
            in_flow_sum[node] += flow[edge] * in_flow_edge[node][edge]
    sink_in_flow_sum = 0
    # for node in range(flow_node_num):
    #     print("%d %d\n"%(flow_node_is_sink[node], in_flow_sum_reciprocal[node]))
    for node in range(flow_node_num):
        if flow_node_is_sink[node] == 1:
            sink_in_flow_sum = sink_in_flow_sum + in_flow_sum[node]
        if in_flow_sum[node] != 0:
            in_flow_sum_reciprocal[node] = 1 / in_flow_sum[node]
        in_flow_sum_reciprocal[node] = np.format_float_positional(in_flow_sum_reciprocal[node], trim='-')
    print("sink_in_flow_sum: {}".format(sink_in_flow_sum))
    
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
    flow_node_restr = [[0 for _ in range(net_node_num)] for _ in range(flow_node_num)]
    for i, node in enumerate(graph_merge.nodes.data("domain_constraint")):
        if (h := node[1].get("host")) != None:
            flow_node_restr[i][node_index[h]] = 1
        else:
            for j in range(net_node_num):
                flow_node_restr[i][j] = 1
    net_incidence = nx.incidence_matrix(G=net, oriented=True).toarray().astype(np.int32).tolist()
    
    net_path_origin = [[0 for i in range(net_path_num)] for j in range(net_node_num)]
    net_path_dest = [[0 for i in range(net_path_num)] for j in range(net_node_num)]
    for u in range(0, net_node_num):
        for v in range(0, net_node_num):
            path = u * net_node_num + v
            net_path_origin[u][path] = 1
            net_path_dest[v][path] = 1
            
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
    mips_reciprocal = [e[1] for e in net.nodes.data("mips")]
    mips_positive = []
    for i, mip in enumerate(mips_reciprocal):
        if mip != 0:
            mips_reciprocal[i] = 1 / mip
            mips_positive.append(1)
        else:
            mips_positive.append(0)
        mips_reciprocal[i] = np.format_float_positional(mips_reciprocal[i], trim='-')
    cores = [e[1] for e in net.nodes.data("cores")]
    cores_reciprocal = [0 for _ in range(net_node_num)]
    for i, core in enumerate(cores):
        if core != 0:
            cores_reciprocal[i] = 1 / core
    slot = [e[1] for e in net.nodes.data("slots")]
    
    net_node_in_edge = [[0 for i in range(edge_domain_num)] for j in range(net_node_num)]
    net_node_edge_index = [-1 for _ in range(net_node_num)]
    net_node_in_cloud = [0] * net_node_num
    for i, domain in enumerate(scenario.domains):
        if domain.type == "cloud":
            for node in domain.topo.g.nodes:
                net_node_in_cloud[node_index[node]] = 1
        else:
            for node in domain.topo.g.nodes:
                net_node_in_edge[node_index[node]][i] = 1
                net_node_edge_index[node_index[node]] = i
    
    flow_endpoint = [[-1, -1] for _ in range(flow_edge_num)]
    for edge in range(flow_edge_num):
        for node in range(flow_node_num):
            if flow_incidence[node][edge] == 0:
                continue
            elif flow_incidence[node][edge] == 1:
                flow_endpoint[edge][0] = node
            else:
                flow_endpoint[edge][1] = node       
    
    path_endpoint = [[-1, -1] for _ in range(net_path_num)]
    for u in range(net_node_num):
        for v in range(net_node_num):
            path = u *net_node_num + v
            path_endpoint[path][0] = u
            path_endpoint[path][1] = v
    net_edge_of_path = [[] for _ in range(net_path_num)]
    for edge in range(net_edge_num):
        for path in range(net_path_num):
            if net_edge_in_path[edge][path] == 1:
                net_edge_of_path[path].append(edge)    
    
    nodes = graph_merge.nodes.data("type")
    sources = [i for i, node in enumerate(nodes) if node[1] == "source"]
    sinks = [i for i, node in enumerate(nodes) if node[1] == "sink"]
    op_in_edge = [[] for _ in range(flow_node_num)]
    op_out_edge = [[] for _ in range(flow_node_num)]
    op_indegree = [0 for _ in range(flow_node_num)]
    for node in range(flow_node_num):
        for edge in range(flow_edge_num):
            if flow_incidence[node][edge] == 0:
                continue
            elif flow_incidence[node][edge] == -1:
                op_in_edge[node].append(edge)
                op_indegree[node] = op_indegree[node] + 1
            else:
                op_out_edge[node].append(edge)
    mips = [e[1] for e in net.nodes.data("mips")]
    
    param_list = [
        "flow_node_num", "flow_edge_num", "flow_incidence", "mi", "flow", "urate", "flow_node_is_sink", "tuple_size",
        "in_flow_edge", "out_flow_edge", "flow_edge_s", "net_node_num", "net_edge_num", "net_path_num", "edge_domain_num",
        "flow_node_restr", "net_incidence", "net_path_origin", "net_path_dest", "net_edge_in_path", "bandwidth", 
        "net_edge_intr_lat", "mips_reciprocal", "mips_positive", "cores", "slot", "net_node_in_edge", "net_node_in_cloud",
        "flow_endpoint", "path_endpoint", "net_edge_of_path", "sources", "sinks", "sink_in_urate_sum", "op_in_edge", "op_out_edge",
        "in_urate_sum", "in_urate_sum_reciprocal", "in_flow_sum", "in_flow_sum_reciprocal", "op_indegree", "mips", "cores_reciprocal"
    ]
    items = locals()
    for name in param_list:
        data[name] = items[name]

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
    print("flow_min %f"%flow_min)
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
                        break
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
        data = dict()
        items = locals()
        for param in param_list:
            data[param] = items[param]
        with open("../glpk/data/flow_max_data", "w") as f:
            gen_data_file(f, data)
        result = subprocess.run(
            ["glpsol", "--model", "../glpk/model/max_cut.mod", "--data", "../glpk/data/flow_max_data", "--tmlim", "30"],
            capture_output=True
        )
        output = result.stdout.decode("ASCII")
        # print(output)
        cut = output.split("\n")[-3].strip()
        flow_max = flow_max + float(cut)
    print("flow_max %f"%flow_max)
    return flow_max
    
def get_lat_min() -> float:
    result = subprocess.run(
            ["glpsol", "--model", "../glpk/model/lat_min.mod", "--data", "../glpk/data/lat_bound_data", "--tmlim", "30"],
            capture_output=True
        )
    output = result.stdout.decode("ASCII")
    lat = output.split("\n")[-3].strip()
    print("lat_min %s"%lat)
    return float(lat)
    
def get_lat_max() -> float:
    result = subprocess.run(
            ["glpsol", "--model", "../glpk/model/lat_max.mod", "--data", "../glpk/data/lat_bound_data", "--tmlim", "30"],
            capture_output=True
        )
    output = result.stdout.decode("ASCII")
    lat = output.split("\n")[-3].strip()
    print("lat_max %s"%lat)
    return float(lat)
    
def get_lower_bound(graph_list: typing.List[ExecutionGraph], 
    scenario: Scenario, data:dict):
    # 流式计算图相关
    # graph_0 = graph_list[0]
    # print(graph_0.g.nodes.data("type"))
    # print(graph_0.g.edges)
    # print(nx.incidence_matrix(G=graph_0.g, oriented=True).toarray().astype(np.int32).tolist())
    
    param_list = [
        "flow_node_num", "flow_edge_num", "flow_incidence", "mi", "flow", "urate", "flow_node_is_sink", "tuple_size",
        "in_flow_edge", "out_flow_edge", "in_urate_sum_reciprocal", "flow_edge_s", "net_node_num", "net_edge_num", "net_path_num", "edge_domain_num",
        "flow_node_restr", "net_incidence", "net_path_origin", "net_path_dest", "net_edge_in_path", "bandwidth", 
        "net_edge_intr_lat", "mips_reciprocal", "mips_positive", "cores", "slot", "net_node_in_edge", "net_node_in_cloud",
    ]
    with open("../glpk/data/lat_bound_data", "w") as f:
        gen_data_file(f, data, param_list)

    flow_min = get_flow_min(graph_list)
    # flow_max = get_flow_max(graph_list)
    lat_min = get_lat_min()
    # lat_max = get_lat_max(graph_list, scenario) 
    data["flow_min"] = flow_min
    data["lat_min"] = lat_min

    # param_list.extend(["flow_min", "lat_min"])
    # with open("../glpk/data/lower_bound_data", "w") as f:
    #     gen_data_file(f, data, param_list)
    # result = subprocess.run(
    #         ["glpsol", "--model", "../glpk/model/lower_bound.mod", "--data", "../glpk/data/lower_bound_data", "--tmlim", "30"],
    #         capture_output=True
    #     )
    # output = result.stdout.decode("ASCII")
    # lb = output.split("\n")[-3].strip()
    # lb = float(lb)
    # print("lower_bound: %s"%lb)
    # data["lb"] = lb
    # return lb

    return 0

# 计算特定部署方案的目标值
def obj(f: typing.List,
    data: dict,
    debug: bool = False 
    ) -> float:
    flow_node_num = data["flow_node_num"]
    net_node_num = data["net_node_num"]
    mi = data["mi"]
    mips = data["mips"]
    cores = data["cores"]
    occupied = [0 for _ in range(net_node_num)]
    if type(f) != list:
        f = f.astype(np.int32).tolist()
    # for pos in f:
    #     print(pos, mips[pos], end=",")
    # print()
    for pos in f:
        occupied[pos] = occupied[pos] + 1
    for i, x in enumerate(occupied):
        if x < cores[i]:
            occupied[i] = cores[i]
    comp_lat = [0 for _ in range(flow_node_num)]
    penalty_lat = 100000
    for i in range(flow_node_num):
        if mips[f[i]] == 0:
            comp_lat[i] = penalty_lat
        else:
            comp_lat[i] = (mi[i] * occupied[f[i]]) / (cores[f[i]] * mips[f[i]])
    
    flow_edge_num = data["flow_edge_num"]
    net_path_num = data["net_path_num"]
    flow_endpoint = data["flow_endpoint"]
    path_endpoint = data["path_endpoint"]
    net_edge_of_path = data["net_edge_of_path"]
    flow_edge_as_net_path = [-1 for _ in range(flow_edge_num)]
    net_edge_intr_lat = data["net_edge_intr_lat"]
    bandwidth = data["bandwidth"]
    tuple_size = data["tuple_size"]
    for edge in range(flow_edge_num):
        for path in range(net_path_num):
            if f[flow_endpoint[edge][0]] == path_endpoint[path][0] and f[flow_endpoint[edge][1]] == path_endpoint[path][1]:
                flow_edge_as_net_path[edge] = path
                break
    intr_lat = [0 for _ in range(flow_edge_num)]
    tran_lat = [0 for _ in range(flow_edge_num)]
    for edge in range(flow_edge_num):
        path = flow_edge_as_net_path[edge]
        for net_edge in net_edge_of_path[path]:
            intr_lat[edge] = intr_lat[edge] + net_edge_intr_lat[net_edge]
            # 下面bandwidth处存在问题，此处为真实计算，应该用分配的值
            tran_lat[edge] = tran_lat[edge] + tuple_size[edge] / bandwidth[net_edge]
    op_lat = [0 for _ in range(flow_node_num)]
    sources = data["sources"]
    flow = data["flow"]
    flow_edge_s = data["flow_edge_s"]
    op_in_edge = data["op_in_edge"]
    op_out_edge = data["op_out_edge"]
    # if debug:
    #     print("flow_endpoint: {}".format(flow_endpoint))
    #     print("op_in_edge: {}".format(op_in_edge))
    #     print("op_out_edge: {}".format(op_out_edge))
    in_urate_sum = data["in_urate_sum"]
    op_indegree = copy.deepcopy(data["op_indegree"])
    queue = copy.deepcopy(sources)
    head = 0
    tail = len(sources)
    while head < tail:
        node = queue[head]
        if in_urate_sum[node] != 0:
            for edge in op_in_edge[node]:
                origin = flow_endpoint[edge][0]
                op_lat[node] = op_lat[node] + flow[edge] * (op_lat[origin] + tran_lat[edge] + intr_lat[edge])
            op_lat[node] = op_lat[node] / in_urate_sum[node] + comp_lat[node]
        else:
            op_lat[node] = comp_lat[node]
        # if debug:
        #     print("node: {}, opt_lat: {}".format(node, op_lat[node]))
        for edge in op_out_edge[node]:
            dest = flow_endpoint[edge][1]
            op_indegree[dest] = op_indegree[dest] - 1
            # if debug:
            #     print("edge:{}, dest:{}, indegree:{}".format(edge, dest, op_indegree[dest]))
            if op_indegree[dest] == 0:
                queue.append(dest)
                tail = tail + 1
        head = head + 1
    
    lat = 0
    sinks = data["sinks"]
    sink_in_urate_sum = data["sink_in_urate_sum"]
    for node in sinks:
        # if debug:
        #     print("node: {}, op_lat: {}, in_urate_sum: {}".format(node, op_lat[node], in_urate_sum[node]))
        lat = lat + op_lat[node] * in_urate_sum[node]
    lat = lat / sink_in_urate_sum
    
    flow_ = 0
    net_node_in_cloud = data["net_node_in_cloud"]
    for edge in range(flow_edge_num):
        u = flow_endpoint[edge][0]
        v = flow_endpoint[edge][1]
        if net_node_in_cloud[f[u]] != net_node_in_cloud[f[v]]:
            flow_ = flow_ + flow[edge]
    
    flow_min = data["flow_min"]
    lat_min = data["lat_min"]
    obj = lat / lat_min + flow_ / flow_min
    if debug:
    #     print("comp_lat:", comp_lat)
    #     print("tran_lat:", tran_lat)
    #     print("intr_lat:", intr_lat)
        print("flow_min:", flow_min, "lat_min:", lat_min)
        print("flow:", flow_, "lat:", lat)
        print("obj:", obj)
    return obj

def gap(lb: float,
    data: dict,
    f: typing.List,
    ) -> float:
    x = obj(f, data)
    return (x - lb) / lb

if __name__ == "__main__":
    print("hello")