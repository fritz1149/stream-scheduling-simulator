import logging
import os
import sys
import typing
from collections import defaultdict

from algo import min_cut2
from graph import ExecutionGraph
from topo import Domain, Scenario
from utils import gen_uuid, grouped_exactly_one_full_binpack
from .flow_cut import SourcedGraph, CutOption, logger, cross_bd

def multiedge_flow_cut(
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

        graph_cut_options: typing.List[typing.List[CutOption]] = [
            sorted(gen_cut_options(sg.g), key=lambda o: o.flow, reverse=False)
            for sg in sg_list
        ]
        # for sg, options in zip(sg_list, graph_cut_options):
        #     print("graph", sg.g.uuid)
        #     for op in options:
        #         print(str(op))
        if len([None for options in graph_cut_options if len(options) == 0]) > 0:
            logger.error("no option provided")
            continue

        free_slots = sum([n.slots - n.occupied for n in edge_domain.topo.get_nodes()])
        if (
            sum(
                [
                    len(options[0].s_cut)
                    for options in graph_cut_options
                    if len(options) > 0
                ]
            )
            <= free_slots
        ):
            # logger.info("graph_cutting: best")
            s_cut_list: typing.List[typing.Set[str]] = [
                options[0].s_cut for options in graph_cut_options
            ]
            t_cut_list: typing.List[typing.Set[str]] = [
                options[0].t_cut for options in graph_cut_options
            ]
        else:
            # logger.info("graph_cutting: binpack")
            groups = [
                [(len(option.s_cut), option.flow) for option in options]
                for options in graph_cut_options
            ]
            solution = grouped_exactly_one_full_binpack(free_slots, groups)
            s_cut_list: typing.List[typing.Set[str]] = [
                options[s_idx].s_cut
                for options, s_idx in zip(graph_cut_options, solution)
            ]
            t_cut_list: typing.List[typing.Set[str]] = [
                options[s_idx].t_cut
                for options, s_idx in zip(graph_cut_options, solution)
            ]

        for sg, s_cut, t_cut in zip(sg_list, s_cut_list, t_cut_list):
            s_cuts[sg.idx] |= s_cut
            t_cuts[sg.idx] |= t_cut
            
    for i in range(len(graph_list)):
        graph_cut_results[i] = (s_cuts[i], t_cuts[i])
        
    return graph_cut_results

def domains_of_sources(scenario: Scenario, g: ExecutionGraph) -> typing.List[Domain]:
    domain_set = set()
    for s in g.get_sources():
        for d in scenario.get_edge_domains():
            if d.find_host(s.domain_constraint["host"]) is not None:
                domain_set.add(d.name)
    domain_list = []
    for domain_name in domain_set:
        domain_list.append(scenario.find_domain(domain_name))
    return domain_list

def gen_cut_options(g: ExecutionGraph) -> typing.List[CutOption]:
    options: typing.List[CutOption] = []
    # s_cut, t_cut = min_cut(g)
    s_cut, t_cut = min_cut2(g, True)
    # print("first cut:")
    # print(s_cut)
    # print(t_cut)
    flow = cross_bd(g, s_cut, t_cut)
    options.append(CutOption(s_cut, t_cut, flow))

    while len(s_cut) > 1:
        sub_graph = g.sub_graph(s_cut, gen_uuid())
        if sub_graph.contain_only_sources():
            break
        # s_cut, _ = min_cut(sub_graph)
        s_cut, _ = min_cut2(sub_graph, True)
        t_cut = set([v.uuid for v in g.get_vertices()]) - s_cut
        flow = cross_bd(g, s_cut, t_cut)
        options.append(CutOption(s_cut, t_cut, flow))
    
    return options