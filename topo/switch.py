import uuid

from topo.host import Host
from topo.topology import Node, Topology


class Switch:
    def __init__(self, name: str, bd: int, delay: int) -> None:
        self.name = name
        self.bd = bd
        self.delay = delay
        # self.node = Node.from_spec(str(uuid.uuid4())[:8], "switch", 0, 0, 0, 0, 0, {})
        self.node = Node.from_spec(name, "switch", 0, 0, 0, 0, 0, {})

    def connect_host(self, topo: Topology, host: Host) -> None:
        topo.connect(self.node, host.node, str(uuid.uuid4())[:8], self.bd, self.delay)

    def replace_node(self, node: Node) -> None:
        self.node = node

    @classmethod
    def from_dict(cls, name, data):
        return cls(name, int(data["bd"] * 1e6), int(data["delay"]))
