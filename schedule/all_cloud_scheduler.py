import random
import typing
from collections import defaultdict

from graph import ExecutionGraph

from .result import SchedulingResult, SchedulingResultStatus
from .scheduler import RandomScheduler, Scheduler, SourcedGraph


class AllCloudScheduler(Scheduler):
    def schedule(self, g: ExecutionGraph) -> SchedulingResult:
        return self.schedule_multiple([g])

    def schedule_multiple(
        self, graph_list: typing.List[ExecutionGraph]
    ) -> typing.List[SchedulingResult]:
        results = [None for _ in graph_list]

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
            edge_domain = self.if_source_in_single_domain(sg.g)
            if edge_domain is None:
                results[sg.idx] = SchedulingResult.failed(
                    "sources not in single domain"
                )
                continue
            edge_domain_map[edge_domain.name].append(sg)

        for domain_name, sg_list in edge_domain_map.items():
            edge_domain = self.scenario.find_domain(domain_name)
            assert edge_domain is not None

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
            t_result_list = RandomScheduler(self.scenario).schedule_multiple(
                t_graph_list, random.choice(self.scenario.get_cloud_domains()).topo
            )
            for sg, s_result, t_result in zip(sg_list, s_result_list, t_result_list):
                if s_result.status == SchedulingResultStatus.FAILED:
                    results[sg.idx] = s_result
                    continue
                if t_result.status == SchedulingResultStatus.FAILED:
                    results[sg.idx] = t_result
                    continue
                results[sg.idx] = SchedulingResult.merge(s_result, t_result)

        return results
