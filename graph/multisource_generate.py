from collections import defaultdict
import random
import typing

from .execution_graph import ExecutionGraph, Vertex
from .generate import SourceSelector, GraphGenerator

class MultiDomainSourceSelector():
    def __init__(self, sources_dict: typing.Dict[str, typing.Dict[str, int]]):
        self.source_selectors = dict()
        self.total_slots = 0
        for domain, sources in sources_dict.items():
            source_selector = SourceSelector(sources)
            self.source_selectors[domain] = source_selector
            self.total_slots += source_selector.total_slots
        
    def select(self) -> str:
        idx = random.randint(0, self.total_slots - 1)
        for domain in random.sample(self.source_selectors.keys(), len(self.source_selectors)):
            if idx < self.source_selectors[domain].total_slots:
                source = self.source_selectors[domain].select()
                self.total_slots -= 1
                return domain, source
            idx -= self.source_selectors[domain].total_slots
        assert False


class MultiSourceGraphGenerator(GraphGenerator):
    def gen_dag_graph(self) -> ExecutionGraph:
        total_rank = self.gen_args["total_rank"]
        max_node_per_rank = self.gen_args["max_node_per_rank"]
        max_predecessors = self.gen_args["max_predecessors"]
        # 与多源算子有关的参数
        sources_num = self.gen_args["sources_num"]

        g = ExecutionGraph(self.name)

        node_rank_map: typing.Dict[int, int] = dict()
        ranked_nodes = [[i for i in range(sources_num)]]
        node_seq_num = sources_num
        for i in range(sources_num):
            node_rank_map[i] = 0
        edges = []
        node_successor_cnt = defaultdict(int)
        for rank in range(1, total_rank):
            cur_node_cnt = random.randint(1, max_node_per_rank)
            cur_nodes = [node_seq_num + i for i in range(cur_node_cnt)]
            node_seq_num += cur_node_cnt
            for node in cur_nodes:
                node_rank_map[node] = rank
                pre_cnt = random.randint(1, max_predecessors)
                for pre_node in self.dag_select_predecessors(
                    ranked_nodes, pre_cnt, node_successor_cnt
                ):
                    edges.append((pre_node, node))
                    node_successor_cnt[pre_node] += 1
                    # print("{} ---> {}".format(pre_node, node))
            ranked_nodes.append(cur_nodes)

        node_cnt = node_seq_num
        node_out_degree: typing.Dict[int, int] = defaultdict(int)
        for e in edges:
            node_out_degree[e[0]] += 1

        node_vertex_map: typing.Dict[int, Vertex] = dict()
        
        single_sources = []
        for i in range(sources_num):
            if node_out_degree[i] == 0:
                single_sources.append(i)
            color, source = self.gen_args["source_hosts"].select()
            node_vertex_map[i] = Vertex.from_spec(
                "{}-v{}".format(self.name, i),
                "source",
                {"host": source},
                0,
                0,
                self.gen_args["mi_cb"](),
                self.gen_args["memory_cb"](),
                color
            )
        for node in range(sources_num, node_cnt):
            if node_out_degree[node] == 0:
                node_type = "sink"
                labels = {"host": random.choice(self.gen_args["sink_hosts"])}
                color = "cloud"
                for source in single_sources:
                    edges.append((source, node))
                single_sources = []
            else:
                node_type = "operator"
                labels = {}
                color = None
            node_vertex_map[node] = Vertex.from_spec(
                "{}-v{}".format(self.name, node),
                node_type,
                labels,
                0,
                0,
                self.gen_args["mi_cb"](),
                self.gen_args["memory_cb"](),
                color
            )
        for v in node_vertex_map.values():
            g.add_vertex(v)

        for e in edges:
            g.connect(
                node_vertex_map[e[0]],
                node_vertex_map[e[1]],
                self.gen_args["unit_size_cb"](node_rank_map[e[1]]),
                self.gen_args["unit_rate_cb"](),
            )
        
        g.color_graph()
        return g