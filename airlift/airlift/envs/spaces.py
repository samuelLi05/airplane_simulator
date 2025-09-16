import typing

import networkx as nx
import numpy as np

import gym
from gym import Space
from gym.spaces.utils import flatdim, flatten

from airlift.utils.seeds import generate_seed

# General note: it appears that recent versions of OpenAI gym are introducing similar space types (e.g., Sequence and Graph).
# We may use these in the future.

class List(Space[typing.List]):
    def __init__(self, space: Space, maxsize: int, dtype=np.int64, seed=None):
        self.space = space
        self.maxsize = maxsize

        super().__init__(None, dtype, seed)

    def seed(self, seed: typing.Optional[int] = None) -> list:
        seeds = super().seed(seed)
        seeds.append(self.space.seed(generate_seed(self.np_random)))

        return seeds # seeds is not the same shape as the set. Does that matter?

    def sample(self) -> typing.Set:
        # Number of possible values returned by the space's sample function
        n = flatdim(self.space)

        # Determine how large the set should be.
        # Essentially we are iterating through each possible entry, and determine if it should be included or not.from
        # The probability that the element is not included is the same as the probably that it takes any
        # given value in the space.
        nentries = self.np_random.binomial(self.maxsize, n/(n+1))

        # Generate each entry by sampling from the space
        return [self.space.sample() for _ in range(nentries)]

    def contains(self, x) -> bool:
        if isinstance(x, (list, np.ndarray)):
            x = list(x)  # Promote list and ndarray to tuple for contains check
        return (
            isinstance(x, list)
            and len(x) <= self.maxsize
            and all(self.space.contains(part) for part in x)
        )

    def __repr__(self):
        return f"List({self.space},{self.maxsize})"

    def __len__(self):
        return self.maxsize


@flatdim.register(List)
def _flatdim_list(space: List) -> int:
    return space.maxsize * flatdim(space.space)

@flatten.register(List)
# Entries in the ndarray which are unoccupied will be filled with nan's
def _flatten_list(space: List, x) -> np.ndarray:
    # Build a full-sized ndarray filled with nan's
    a = np.empty((flatdim(space),))
    a[:] = np.nan

    # Fill in the occupied entries
    actualsize = len(x) * flatdim(space.space)
    if actualsize > 0:
        a[0:actualsize] = np.concatenate([flatten(space.space, x_part) for x_part in x])

    return a


class NamedTuple(gym.spaces.Tuple):
    def __init__(
        self, namedtuple: typing.Type[typing.NamedTuple], spaces: typing.Dict[str, Space], seed: typing.Optional[typing.Union[int, typing.List[int]]] = None
    ):
        # Verify that namedtuple is a NamedTuple class.
        assert hasattr(namedtuple, '_fields'), "namedtuple must be a NamedTuple"

        assert frozenset(spaces.keys()) == frozenset(namedtuple._fields), "Keys in spaces must match field names in namedtuple"

        self.namedtuple = namedtuple

        # Assumes field names are returned in proper order
        spacestuple = (spaces[k] for k in namedtuple._fields)
        super().__init__(spacestuple, seed)

    def sample(self) -> tuple:
        return self.namedtuple(space.sample() for space in self.spaces)

    def contains(self, x) -> bool:
        if isinstance(x, (list, np.ndarray)):
            x = self.namedtuple(x)  # Promote list and ndarray to tuple for contains check

        if not isinstance(x, self.namedtuple):
            return False
        else:
            return super().contains(x)

    def __repr__(self) -> str:
        return self.namedtuple.__name__ + "(" + ", ".join([str(self.namedtuple._fields[i]) + ":" + str(s) for i, s in enumerate(self.spaces)]) + ")"

    def __eq__(self, other) -> bool:
        return isinstance(other, self.namedtuple) and self.spaces == other.spaces




class DiGraph(Space[nx.DiGraph]):
    def __init__(self, nnodes: int, node_attributes: typing.Container[str], edge_attributes: typing.Container[str], dtype=np.int64, seed=None):
        self.nnodes = nnodes
        self.node_attributes = node_attributes
        self.edge_attributes = edge_attributes

        # Note the shape is too big. We really need a separate a shape for the edges and nodes.
        super().__init__((nnodes, nnodes, len(node_attributes) + len(edge_attributes)), dtype, seed)

    def contains(self, x) -> bool:
        return type(x) == nx.DiGraph and len(x.nodes) < self.nnodes # Note: it's ok to have fewer nodes in the graph than specified by the space

    def __repr__(self):
        return f"DiGraph({self.nnodes}, {self.node_attributes}, {self.edge_attributes})"


@flatdim.register(DiGraph)
def _flatdim_digraph(space: DiGraph) -> int:
    # return len(space.attributes) * space.nnodes**2
    raise NotImplementedError()

@flatten.register(DiGraph)
# Entries in the ndarray which are unoccupied will be filled with nan's
def _flatten_digraph(space: DiGraph, x) -> np.ndarray:
    # return np.concatenate([nx.to_numpy_array(x, weight=a).flatten() for a in space.attributes])
    raise NotImplementedError()

