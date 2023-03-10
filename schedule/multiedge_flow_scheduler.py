import random
import typing
from collections import defaultdict

from algo import min_cut, min_cut2, cross_bd
from graph import ExecutionGraph
from topo import Domain, Scenario
from utils import gen_uuid, grouped_exactly_one_nonfull_binpack

from .flow_provisioner import TopologicalProvisioner
from .provision import Provisioner
from .result import SchedulingResult, SchedulingResultStatus
from .scheduler import RandomScheduler, Scheduler, SourcedGraph
from .flow_scheduler import FlowScheduler

class MultiEdgeFlowScheduler(FlowScheduler):
    def schedule_multiple(
        self, graph_list: typing.List[ExecutionGraph]
    ) -> typing.List[SchedulingResult]:
        failed = [False for _ in graph_list]
        results = [[] for _ in graph_list]
        t_graphs = [[] for _ in graph_list]

        # NOTE schedule non-contrained graphs to cloud
        for idx, g in enumerate(graph_list):
            if len(g.get_sources()) == 0:
                results[idx] = RandomScheduler(self.scenario).schedule(
                    g, random.choice(self.scenario.get_cloud_domains()).topo
                )

        # NOTE continue algorithm for contrained graphs
        sourced_graphs: typing.List[SourcedGraph] = [
            SourcedGraph(idx, g)
            for idx, g in enumerate(graph_list)
            if g.get_sources() != 0
        ]

        # NOTE group graphs by edge domains
        # NOTE 处理源算子在多个边缘域的情况
        edge_domain_map: typing.Dict[str, typing.List[SourcedGraph]] = defaultdict(list)
        for sg in sourced_graphs:
            edge_domain_list = self.domains_of_sources(sg.g)
            for edge_domain in edge_domain_list:
                sg_cut = SourcedGraph(sg.idx, sg.g.color_sub_graph(edge_domain.name))
                edge_domain_map[edge_domain.name].append(sg_cut)


        # NOTE for each edge domain
        for domain_name, sg_list in edge_domain_map.items():
                
            edge_domain = self.scenario.find_domain(domain_name)
            assert edge_domain is not None
            
            not_failed_sg_list = []
            for sg in sg_list:
                if not failed[sg.idx]:
                    not_failed_sg_list.append(sg)
            sg_list = not_failed_sg_list
            
            assert len(sg_list) > 0
            # print(domain_name, ":")
            # for sg in sg_list:
            #     print(sg.g)
                
            if not self.if_source_fit([sg.g for sg in sg_list], edge_domain):
                for sg in sg_list:
                    results[sg.idx].append(SchedulingResult.failed(
                        "insufficient resource for sources"
                    ))
                    print(domain_name, "边缘资源不足")
                    failed[sg.idx] = True
                continue

            try:
                s_graph_list, t_graph_list = self.cloud_edge_cutting(
                    sg_list, edge_domain, True
                )
            except RuntimeError as e:
                self.logger.error(e)
                continue
            
            # for s_graph, t_graph in zip(s_graph_list, t_graph_list):
            #     print("s_graph, t_graph:")
            #     print(s_graph.g.nodes)
            #     print(t_graph.g.nodes)
            
            # self.logger.info(
            #     "s_graph_list: %s", [g.number_of_vertices() for g in s_graph_list]
            # )
            # self.logger.info(
            #     "t_graph_list: %s", [g.number_of_vertices() for g in t_graph_list]
            # )
            s_result_list = self.get_provisioner(domain_name).schedule_multiple(
                s_graph_list
            )
            # self.logger.info("s_result_list: %s", s_result_list)
            for sg, s_result in zip(sg_list, s_result_list):
                results[sg.idx].append(s_result)
            
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
        
        t_result_list = self.get_provisioner(
            random.choice(self.scenario.get_cloud_domains()).name
        ).schedule_multiple(valid_t_graphs)
        
        for i, t_result in zip(valid_t_idx, t_result_list):
            results[i].append(t_result)
        for i in range(len(results)):
            results[i] = SchedulingResult.merge(*results[i])
        for i, is_failed in enumerate(failed):
            if is_failed:
                print(graph_list[i].uuid, "failed")
        # for result in results:
        #     print(result.assign_map)
        # print(result_s)
        return results