import random
import typing
from collections import defaultdict

from graph import ExecutionGraph
from utils import gen_uuid

from .result import SchedulingResult, SchedulingResultStatus
from .scheduler import RandomScheduler,  Scheduler, SourcedGraph
from .all_cloud_scheduler import AllCloudScheduler

class MultiEdgeAllCloudScheduler(AllCloudScheduler):
    def schedule_multiple(self, graph_list: typing.List[ExecutionGraph]) -> typing.List[SchedulingResult]:
        failed = [False for _ in graph_list]
        results = [[] for _ in graph_list]
        t_graphs = [[] for _ in graph_list]

        for idx, g in enumerate(graph_list):
            if len(g.get_sources()) == 0:
                results[idx] = RandomScheduler(self.scenario).schedule(
                    g, random.choice(self.scenario.get_cloud_domains()).topo
                )

        sourced_graphs: typing.List[SourcedGraph] = [
            SourcedGraph(idx, g)
            for idx, g in enumerate(graph_list)
            if g.get_sources() != 0
        ]

        edge_domain_map: typing.Dict[str, typing.List[SourcedGraph]] = defaultdict(list)
        for sg in sourced_graphs:
            edge_domain_list = self.domains_of_sources(sg.g)
            for edge_domain in edge_domain_list:
                sg_cut = SourcedGraph(sg.idx, sg.g.color_sub_graph(edge_domain.name))
                edge_domain_map[edge_domain.name].append(sg_cut)

        for domain_name, sg_list in edge_domain_map.items():
            edge_domain = self.scenario.find_domain(domain_name)
            assert edge_domain is not None

            not_failed_sg_list = []
            for sg in sg_list:
                if not failed[sg.idx]:
                    not_failed_sg_list.append(sg)
            sg_list = not_failed_sg_list
            if len(sg_list) == 0:
                continue

            s_graph_list = []
            t_graph_list = []
            for sg in sg_list:
                s_cut = set([v.uuid for v in sg.g.get_sources()])
                t_cut = set([v.uuid for v in sg.g.get_sinks()]).union(
                    set([v.uuid for v in sg.g.get_operators()])
                )
                s_graph_list.append(sg.g.sub_graph(s_cut, sg.g.uuid + "-s"))
                t_graph_list.append(sg.g.sub_graph(t_cut, sg.g.uuid + "-t"))
            
            s_result_list = RandomScheduler(self.scenario).schedule_multiple(
                s_graph_list, edge_domain.topo
            )
            
            for sg, s_result in zip(sg_list, s_result_list):
                results[sg.idx].append(s_result)
                if s_result.status == SchedulingResultStatus.FAILED:
                    failed[sg.idx] = True
                
            for sg, t_graph in zip(sg_list, t_graph_list):
                t_graphs[sg.idx].append(t_graph)
        
        valid_t_graphs = []
        valid_t_idx = []
        for i, t_graph_list in enumerate(t_graphs):
            if len(t_graph_list) == 0 or failed[i]:
                continue
            t_graph = ExecutionGraph.merge(t_graph_list, gen_uuid())
            valid_t_graphs.append(t_graph)
            valid_t_idx.append(i)    
        
        t_result_list = RandomScheduler(self.scenario).schedule_multiple(
                valid_t_graphs, random.choice(self.scenario.get_cloud_domains()).topo
            )
        
        for i, t_result in zip(valid_t_idx, t_result_list):
            results[i].append(t_result)
        for i in range(len(results)):
            results[i] = SchedulingResult.merge(*results[i])
        for i, result in enumerate(results):
            if result.status == SchedulingResultStatus.FAILED:
                print(graph_list[i].uuid, "failed")
        
        return results