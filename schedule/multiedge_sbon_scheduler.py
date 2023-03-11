import random
import typing
from collections import defaultdict

from graph import ExecutionGraph
import topo
from vivaldi import vivaldi_compute, constrained_balance, create_coordinate_class

from .result import SchedulingResult, SchedulingResultStatus
from .scheduler import RandomScheduler, Scheduler, SourcedGraph
from .sbon_scheduler import SBONScheduler, Coord3D, PickItem

class MultiEdgeSbonScheduler(SBONScheduler):
    def schedule_multiple(
        self, graph_list: typing.List[ExecutionGraph]
    ) -> typing.List[SchedulingResult]:
        results = [SchedulingResult() for _ in graph_list]

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

        cloud_domain = random.choice(self.scenario.get_cloud_domains())
        cloud_default_host = random.choice(list(cloud_domain.host_lookup_table.keys()))
        # print("cloud_default_host:", cloud_default_host)

        for domain_name, sg_list in edge_domain_map.items():
            edge_domain = self.scenario.find_domain(domain_name)
            assert edge_domain is not None

            op_pick_list = dict()
            for sg in sg_list:
                op_coords = {
                    v.uuid: Coord3D.random_unit_vector() for v in sg.g.get_vertices()
                }
                movable = dict()
                for v in sg.g.get_vertices():
                    movable[v.uuid] = len(v.domain_constraint) == 0 and v.color != "cloud"
                    if not movable[v.uuid]:
                        assert (
                            v.color == "cloud" or
                            self.node_coords.get(v.domain_constraint["host"], None)
                            is not None
                        )
                        op_coords[v.uuid] = self.node_coords[
                            v.domain_constraint["host"]
                        ] if len(v.domain_constraint) > 0 else self.node_coords[cloud_default_host]
                op_result = constrained_balance(sg.g, op_coords, movable, 0.001, 2000)
                for v in sg.g.get_sources():
                    host = edge_domain.find_host(v.domain_constraint["host"])
                    assert host is not None
                    results[sg.idx].assign(host.node.uuid, v.uuid)
                    host.node.occupy(1)
                for v in sg.g.get_sinks_and_cloud_vertices():
                    host = cloud_domain.find_host(v.domain_constraint["host"]) if len(v.domain_constraint) > 0 \
                        else cloud_domain.find_host(cloud_default_host)
                    assert host is not None
                    results[sg.idx].assign(host.node.uuid, v.uuid)
                    host.node.occupy(1)
                for v in sg.g.get_operators_colored_not_cloud():
                    op_pick_list[v.uuid] = PickItem(op_result[v.uuid])

            big_result = SchedulingResult()
            while True:
                while True:
                    updated = False
                    for domain in [edge_domain, cloud_domain]:
                        for host in domain.topo.get_hosts():
                            if host.slots <= host.occupied or len(op_pick_list) == 0:
                                continue
                            assert self.node_coords.get(host.uuid, None) is not None
                            self_coord = self.node_coords[host.uuid]
                            min_dist = 1e10
                            min_op = None
                            for op_id, op_item in op_pick_list.items():
                                dist = abs(op_item.coord - self_coord)
                                if dist < min_dist:
                                    min_dist = dist
                                    min_op = op_id
                                if op_item.min_dist is None or dist < op_item.min_dist:
                                    op_item.min_dist = dist
                            assert min_dist < 1e10 and min_op is not None
                            big_result.assign(host.uuid, min_op)
                            host.occupy(1)
                            op_pick_list.pop(min_op)
                            updated = True
                    if not updated:
                        break
                if len(op_pick_list) == 0:
                    break
                for _, op_item in op_pick_list:
                    op_item.min_dist = None

            for sg in sg_list:
                results[sg.idx] = SchedulingResult.merge(
                    results[sg.idx],
                    big_result.extract(set([v.uuid for v in sg.g.get_operators_colored_not_cloud()])),
                )

        return results