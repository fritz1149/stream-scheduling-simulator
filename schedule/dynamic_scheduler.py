import typing
from graph import ExecutionGraph, Vertex
from topo import Domain, Scenario, Node
from .result import SchedulingResult, SchedulingResultStatus
from .scheduler import RandomScheduler, Scheduler, SourcedGraph
from sko.GA import GA
from functools import partial

class DynamicScheduler(Scheduler):
    def __init__(self, scenario: Scenario) -> None:
        super.__init__(scenario)
        
    def schedule(self, g: ExecutionGraph) -> SchedulingResult:
        return self.schedule_multiple([g])
        
    def schedule_multiple(
        self, graph_list: typing.List[ExecutionGraph]
    ) -> typing.List[SchedulingResult]:
        # vertices是算子列表，nodes是网络节点列表
        vertice_number_list = []
        vertices = []
        for g in graph_list:
            vertices_tmp = g.get_vertices()
            vertice_number_list.append(len(vertices_tmp))
            vertices.extend(vertices_tmp)
        nodes = self.scenario.topo.get_nodes()
        
        return None

# 计算特定任务（网络环境、流式计算图）的下界，对目标函数的下界函数采取线性函数的手段
def lower_bound(vertices: typing.List[Vertex], 
    nodes: typing.List[Node]
    ) -> float:
    
    return 0.0

def schedule_main(vertices: typing.List[Vertex], 
    nodes: typing.List[Node]
    ) -> None:
    # 超参数直接在下面改吧
    ga = GA(func=partial(gap, vertices, nodes, lb=lower_bound(vertices, nodes)),
            n_dim=len(vertices), size_pop=50, max_iter=800, prob_mut=0.001, 
            lb=[-1, -1], ub=[1, 1], precision=1e-7)
    map_list = ga.run()
    
# 计算特定部署方案的目标值
def f(vertices: typing.List[Vertex], 
    nodes: typing.List[Node],
    map_list: typing.List[int]
    ) -> float:
    return 0.0

def gap(vertices: typing.List[Vertex], 
    nodes: typing.List[Node],
    lb: float,
    map_list: typing.List[int],
    ) -> float:
    x = f(vertices, nodes, map_list)
    return (x - lb) / lb