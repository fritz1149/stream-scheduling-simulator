from collections import defaultdict
import typing

from graph import ExecutionGraph
from topo import Topology
from utils import avg, get_logger

from .result import SchedulingResult


class ScheduledGraph(typing.NamedTuple):
    graph: ExecutionGraph
    result: SchedulingResult


class LatencyCalculator:
    graph_list: typing.List[ScheduledGraph]

    def __init__(self, topo: Topology) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.topo = topo
        self.graph_list = []

    def add_scheduled_graph(
        self, graph: ExecutionGraph, result: SchedulingResult
    ) -> None:
        if not result.check_complete(graph):
            print("incomplete scheduling")
            return
        self.graph_list.append(ScheduledGraph(graph, result))
        for u, v, d in graph.get_edges():
            self.topo.occupy_link(
                result.get_scheduled_node(u),
                result.get_scheduled_node(v),
                d["unit_size"] * d["per_second"],
            )
        # this has been executed in scheduler
        # for v in graph.get_vertices():
        #     self.topo.occupy_node(result.get_scheduled_node(v.uuid))

    def compute_latency(
        self,
    ) -> typing.Tuple[typing.Dict[str, float], typing.Dict[str, float]]:
        latency = dict()
        bp_rate = dict()
        cross_bd = 0
        for g in self.graph_list:
            lat, bp, bd = self.topological_graph_latency(g)
            latency[g.graph.uuid] = lat
            bp_rate[g.graph.uuid] = bp / len(g.graph.get_edges())
            cross_bd += bd
        # print("cloud-edge bd:", cross_bd)
        return latency, bp_rate, cross_bd

    def topological_graph_latency(
        self, g: ScheduledGraph
    ) -> typing.Tuple[int, int, int]:
        cross_bd = 0
        back_pressure_acc = 0
        latency_dict = {}
        last_vid = None
        for v in g.graph.topological_order():
            up_vertices = g.graph.get_up_vertices(v.uuid)
            node_lat = [latency_dict[u.uuid] for u in up_vertices]
            intri_lat = [
                self.topo.get_n2n_intrinsic_latency(
                    g.result.get_scheduled_node(u.uuid),
                    g.result.get_scheduled_node(v.uuid),
                )
                for u in up_vertices
            ]
            trans_lat = [
                self.topo.get_n2n_transmission_latency(
                    g.result.get_scheduled_node(u.uuid),
                    g.result.get_scheduled_node(v.uuid),
                    g.graph.get_edge(u.uuid, v.uuid)["unit_size"],
                    g.graph.get_edge(u.uuid, v.uuid)["unit_size"]
                    * g.graph.get_edge(u.uuid, v.uuid)["per_second"],
                )
                for u in up_vertices
            ]
            # print("intr_lat:")
            # for lat in intri_lat:
            #     print(lat, end=" ")
            # print()
            # print("trans_lat:")
            # for lat in trans_lat:
            #     print(lat, end=" ")
            # print()
            # NOTE cross cloud-edge bandwidth usage
            for u in up_vertices:
                node_u = g.result.get_scheduled_node(u.uuid)
                node_v = g.result.get_scheduled_node(v.uuid)
                if node_u.startswith("rasp") and node_v.startswith("cloud"):
                    cross_bd += (
                        g.graph.get_edge(u.uuid, v.uuid)["unit_size"]
                        * g.graph.get_edge(u.uuid, v.uuid)["per_second"]
                    )
            # NOTE check if back-pressure exist
            for u, lat in zip(up_vertices, trans_lat):
                if lat == 0:
                    continue
                real_freq = 1000 / lat
                expected_freq = g.graph.get_edge(u.uuid, v.uuid)["per_second"]
                if real_freq < expected_freq:
                    #     self.logger.info(
                    #         "bp from %s to %s: expected %.2fHz, got %.2fHz",
                    #         u.uuid,
                    #         v.uuid,
                    #         expected_freq,
                    #         real_freq,
                    #     )
                    back_pressure_acc += (expected_freq - real_freq) / expected_freq

            # TODO: configurable latency aggregation
            weighted_sum = 0
            total_weight = 0
            for u, lat1, lat2, lat3 in zip(up_vertices, node_lat, intri_lat, trans_lat):
                weight = g.graph.get_edge(u.uuid, v.uuid)["per_second"]
                total_weight += weight
                weighted_sum += weight * (lat1 + lat2 + lat3)
            # up_latency = avg(*[sum(i) for i in zip(node_lat, intri_lat, trans_lat)])
            if len(up_vertices) == 0:
                up_latency = 0
            else:
                up_latency = weighted_sum / total_weight
            latency_dict[v.uuid] = up_latency + self.topo.get_computation_latency(
                g.result.get_scheduled_node(v.uuid), g.graph.get_vertex(v.uuid).mi
            )
            # print(v.uuid, g.result.get_scheduled_node(v.uuid), latency_dict[v.uuid])
            # print(g.graph.get_vertex(v.uuid).mi, self.topo.g.nodes[g.result.get_scheduled_node(v.uuid)]["mips"])
            # print(up_latency)
            # self.logger.info(
            #     "vertex %s: %s, com %d",
            #     v.uuid,
            #     str(
            #         list(
            #             zip(
            #                 [i.uuid for i in up_vertices],
            #                 node_lat,
            #                 intri_lat,
            #                 trans_lat,
            #             )
            #         )
            #     ),
            #     latency_dict[v.uuid] - up_latency,
            # )
            last_vid = v.uuid

        assert last_vid is not None
        avg_e2e_lat = avg(*[latency_dict[v.uuid] for v in g.graph.get_sinks()])
        return avg_e2e_lat, back_pressure_acc, cross_bd
