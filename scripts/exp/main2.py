import sys

sys.path.insert(0, "../..")

import graph
import logging
import math
import networkx as nx
import schedule as sch
import random
import topo
import uuid
import yaml


def gen_uuid():
    return str(uuid.uuid4())[:8]


def run():
    sc = topo.Scenario.from_dict(
        yaml.load(open("../../samples/1e6h.yaml", "r").read(), Loader=yaml.Loader)
    )
    # print(sc.topo.g.nodes)
    def unit_size_cb(r: int):
        return 10000 * math.pow(10, random.randint(0, 1))

    def gen_graphs(graph_count, source_selector_dict):
        source_selector = graph.MultiDomainSourceSelector(source_selector_dict)
        gen_args_list = [
                {
                "total_rank": random.randint(3, 7),
                "max_node_per_rank": random.randint(1, 3),
                "max_predecessors": random.randint(1, 2),
                "mi_cb": lambda: 1,
                "memory_cb": lambda: int(2e8),
                "unit_size_cb": unit_size_cb,
                "unit_rate_cb": lambda: random.randint(10, 20),
                "source_hosts": source_selector,
                "sink_hosts": ["cloud1"],
                "sources_num": random.randint(1, 3),
            }
            for _ in range(graph_count)
        ]
        return [
            graph.MultiSourceGraphGenerator("g" + str(idx), **gen_args).gen_dag_graph()
            for idx, gen_args in enumerate(gen_args_list)
        ]
    graph_count = 20
    source_selector_dict = {"edge0": {'e0rasp1': 10, 'e0rasp2': 10, 'e0rasp3': 10},
                    "edge1": {'e1rasp1': 10, 'e1rasp2': 10, 'e1rasp3': 10}}
    
    graph_list = gen_graphs(graph_count, source_selector_dict)
    with open("../../cases/a.yaml", "w") as f:
        graph.ExecutionGraph.save_all(graph_list, f)
    # f = open("../../cases/a.yaml")
    # graph_list = graph.ExecutionGraph.load_all(f)
    # f.close()
    
    
    # print(graph_list)
    for g in graph_list:
        print(g.uuid, ":")
        print([(n[0], n[1]["color"], n[1]["type"]) for n in g.g.nodes.data()])
        print([(n[0], n[1]["domain_constraint"].get("host"))for n in g.g.nodes.data()])
        # print(g.g.nodes.data("color"))
        # print(g.g.nodes.data("type"))
        print(g.g.edges)
        
    sc.topo.clear_occupied()
    flow_scheduler = sch.MultiEdgeFlowScheduler(sc)
    flow_calculator = sch.LatencyCalculator(sc.topo)
    flow_result_list = flow_scheduler.schedule_multiple(graph_list)
    for g, result in zip(graph_list, flow_result_list):
        assert result is not None
        flow_calculator.add_scheduled_graph(g, result)
        print()
    flow_latency, flow_bp, _ = flow_calculator.compute_latency()
    print(flow_latency, flow_bp)


if __name__ == "__main__":
    run()
