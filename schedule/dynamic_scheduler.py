import typing
from graph import ExecutionGraph
from topo import Domain, Scenario
from .result import SchedulingResult, SchedulingResultStatus
from .scheduler import RandomScheduler, Scheduler, SourcedGraph
from sko.GA import GA

class DynamicScheduler(Scheduler):
    def __init__(self, scenario: Scenario) -> None:
        super.__init__(scenario)
        
    def schedule(self, g: ExecutionGraph) -> SchedulingResult:
        return self.schedule_multiple([g])
        
    def schedule_multiple(
        self, graph_list: typing.List[ExecutionGraph]
    ) -> typing.List[SchedulingResult]:
        
        return None

def gap_calculate():
    None