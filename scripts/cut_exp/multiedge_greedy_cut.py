import logging
import typing
from collections import defaultdict

import coloredlogs
import networkx as nx
from graph import ExecutionGraph
from topo import Scenario

from .flow_cut import SourcedGraph, extract_edge_domain
from .multiedge_flow_cut import domains_of_sources

logger = logging.getLogger(__name__)
coloredlogs.install(level="debug", logger=logger)


def multiedge_greedy_cut(
    scenario: Scenario, graph_list: typing.List[ExecutionGraph]
) -> typing.List[typing.Tuple[typing.Set[str], typing.Set[str]]]:
    s_cuts = [set() for _ in graph_list]
    t_cuts = [set() for _ in graph_list]
    graph_cut_results = [None for _ in graph_list]

    sourced_graphs: typing.List[SourcedGraph] = [
        SourcedGraph(idx, g) for idx, g in enumerate(graph_list)
    ]

    edge_domain_map: typing.Dict[str, typing.List[SourcedGraph]] = defaultdict(list)
    for sg in sourced_graphs:
        edge_domain_list = domains_of_sources(scenario, sg.g)
        assert len(edge_domain_list) > 0
        for edge_domain in edge_domain_list:
            sg_cut = SourcedGraph(sg.idx, sg.g.color_sub_graph(edge_domain.name))
            edge_domain_map[edge_domain.name].append(sg_cut)

    for domain_name, sg_list in edge_domain_map.items():
        edge_domain = scenario.find_domain(domain_name)
        assert edge_domain is not None

        big_g = nx.DiGraph()
        s_cut_list = [set() for _ in sg_list]
        for sg_idx, sg in enumerate(sg_list):
            for v in sg.g.get_vertices():
                big_g.add_node(v.uuid, in_bd=0, sg_idx=sg_idx)
            for v in sg.g.get_sources():
                s_cut_list[sg_idx].add(v.uuid)
            for u, v, data in sg.g.get_edges():
                big_g.add_edge(u, v)
                big_g.nodes[v]["in_bd"] += data["unit_size"] * data["per_second"]
            for v in sg.g.get_sources():
                big_g.remove_node(v.uuid)
            for v in sg.g.get_sinks_and_cloud_vertices():
                big_g.remove_node(v.uuid)

        free_slots = sum(
            [n.slots - n.occupied for n in edge_domain.topo.get_nodes()]
        ) - sum([len(sg.g.get_sources()) for sg in sg_list])
        while free_slots > 0:
            free_slots -= 1

            ingress_nodes = [
                (node, big_g.nodes[node]["in_bd"])
                for node, in_degree in big_g.in_degree
                if in_degree == 0
            ]
            if len(ingress_nodes) == 0:
                break
            max_node = max(ingress_nodes, key=lambda i: i[1])[0]
            s_cut_list[big_g.nodes[max_node]["sg_idx"]].add(max_node)
            big_g.remove_node(max_node)

        for sg, s_cut in zip(sg_list, s_cut_list):
            t_cut = set([v.uuid for v in sg.g.get_vertices()]) - s_cut
            s_cuts[sg.idx] |= s_cut
            t_cuts[sg.idx] |= t_cut
    
    for i in range(len(graph_list)):
        graph_cut_results[i] = (s_cuts[i], t_cuts[i])

    return graph_cut_results


def multiedge_greedy_cut2(
    scenario: Scenario, graph_list: typing.List[ExecutionGraph]
) -> typing.List[typing.Tuple[typing.Set[str], typing.Set[str]]]:
    s_cuts = [set() for _ in graph_list]
    t_cuts = [set() for _ in graph_list]
    graph_cut_results = [None for _ in graph_list]

    sourced_graphs: typing.List[SourcedGraph] = [
        SourcedGraph(idx, g) for idx, g in enumerate(graph_list)
    ]

    edge_domain_map: typing.Dict[str, typing.List[SourcedGraph]] = defaultdict(list)
    for sg in sourced_graphs:
        edge_domain_list = domains_of_sources(scenario, sg.g)
        assert len(edge_domain_list) > 0
        for edge_domain in edge_domain_list:
            sg_cut = SourcedGraph(sg.idx, sg.g.color_sub_graph(edge_domain.name))
            edge_domain_map[edge_domain.name].append(sg_cut)

    for domain_name, sg_list in edge_domain_map.items():
        edge_domain = scenario.find_domain(domain_name)
        assert edge_domain is not None

        big_g = nx.DiGraph()
        s_cut_list = [set() for _ in sg_list]
        for sg_idx, sg in enumerate(sg_list):
            for v in sg.g.get_vertices():
                big_g.add_node(v.uuid, in_bd=0, out_bd=0, sg_idx=sg_idx)
            for v in sg.g.get_sources():
                s_cut_list[sg_idx].add(v.uuid)
            for u, v, data in sg.g.get_edges():
                big_g.add_edge(u, v)
                big_g.nodes[v]["in_bd"] += data["unit_size"] * data["per_second"]
                big_g.nodes[u]["out_bd"] += data["unit_size"] * data["per_second"]
            for v in sg.g.get_sources():
                big_g.remove_node(v.uuid)
            for v in sg.g.get_sinks_and_cloud_vertices():
                big_g.remove_node(v.uuid)

        free_slots = sum(
            [n.slots - n.occupied for n in edge_domain.topo.get_nodes()]
        ) - sum([len(sg.g.get_sources()) for sg in sg_list])
        while free_slots > 0:
            free_slots -= 1

            ingress_nodes = [
                (node, big_g.nodes[node]["in_bd"] - big_g.nodes[node]["out_bd"])
                for node, in_degree in big_g.in_degree
                if in_degree == 0
            ]
            if len(ingress_nodes) == 0:
                break
            max_node, max_diff = max(ingress_nodes, key=lambda i: i[1])
            # logger.info("%s %f", max_node, max_diff)
            if max_diff < 0:
                break
            s_cut_list[big_g.nodes[max_node]["sg_idx"]].add(max_node)
            big_g.remove_node(max_node)

        for sg, s_cut in zip(sg_list, s_cut_list):
            t_cut = set([v.uuid for v in sg.g.get_vertices()]) - s_cut
            s_cuts[sg.idx] |= s_cut
            t_cuts[sg.idx] |= t_cut

    for i in range(len(graph_list)):
        graph_cut_results[i] = (s_cuts[i], t_cuts[i])

    return graph_cut_results
