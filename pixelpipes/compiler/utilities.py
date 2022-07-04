
from functools import reduce as _reduce
import typing
import inspect

from intbitset import intbitset

from . import CompilerException
from ..graph import Node, Reference, ValidationException

from .. import types

def toposort(data):
    """Dependencies are expressed as a dictionary whose keys are items
and whose values are a set of dependent items. Output is a list of
sets in topological order. The first set consists of items with no
dependences, each subsequent set consists of items that depend upon
items in the preceeding sets.
"""
    # Special case empty input.
    if len(data) == 0:
        return

    # Copy the input so as to leave it unmodified.
    data = data.copy()

    # Ignore self dependencies.
    for k, v in data.items():
        v.discard(k)
    # Find all items that don't depend on anything.
    extra_items_in_deps = _reduce(set.union, data.values()) - set(data.keys())
    # Add empty dependences where needed.
    data.update({item : set() for item in extra_items_in_deps})
    while True:
        ordered = set(item for item, dep in data.items() if len(dep) == 0)
        if not ordered:
            break
        yield ordered
        data = {item: (dep - ordered) for item, dep in data.items() if item not in ordered}
    if len(data) != 0:
        raise CompilerException('Cyclic dependency detected among nodes: {}'.format(', '.join(repr(x) for x in data.keys())))

def merge(i, j):
    """ Combine two minterms. """
    if i[1] != j[1]:
        return None
    y = i[0] ^ j[0]
    if not (y & (~y + 1)) == y:
        return None
    return (i[0] & j[0], i[1] | y)


def _insert_minterm(minterms, new):
    (pos2, neg2) = tuple(new)

    for i, (pos1, neg1) in enumerate(minterms):
        used1 = pos1 | neg1 # Terms used in clause 1
        used2 = pos2 | neg2 # Terms used in clause 2
        common = used1 & used2 # All terms used in both clauses

        if used1 == common and used2 == common: # Same variables used
            if pos1 == pos2: # Same clause
                return minterms
            change = pos1 ^ pos2
            if len(change) == 1: # We can remove a single independent variable
                del minterms[i]
                new = (pos1 - change, neg1 - change)
                return _insert_minterm(minterms, new)
        elif pos1 == (pos2 & pos1) and neg1 == (neg2 & neg1):
            return minterms # Reduce to clause 1, already merged
        elif pos2 == (pos2 & pos1) and neg2 == (neg2 & neg1):
            del minterms[i]
            return _insert_minterm(minterms, (pos2, neg2)) # Reduce to clause 2
        # Clause not the same, move to next one

    # Not merged, add to list
    return minterms + [new]

class BranchSet(object):

    def __init__(self, variables):
        self._variables = variables
        self._branches = []

    def add(self, **variables):
        pos = intbitset([self._variables[x] for x, v in variables.items() if x in self._variables and v is not None and v])
        neg = intbitset([self._variables[x] for x, v in variables.items() if x in self._variables and v is not None and not v])

        self._branches = _insert_minterm(self._branches, (pos, neg))

    def condition(self):
        return self._branches

    def text(self, minterms = None):
        if minterms is None:
            minterms = self._branches

        def parentheses(glue, array):
            if not array:
                return "True"
            if len(array) > 1:
                return ''.join(['(', glue.join(array), ')'])
            else:
                return glue.join(array)

        or_terms = []
        for (pos, neg) in minterms:
            and_terms = []
            for name, j in self._variables.items():
                if j in pos:
                    and_terms.append(name)
                elif j in neg:
                    and_terms.append('(NOT %s)' % name)
            or_terms.append(parentheses(' AND ', and_terms))
        return parentheses(' OR ', or_terms)

    def used(self, minterms = None):
        if minterms is None:
            minterms = self._branches
        vars = set()
        for (pos, neg) in minterms:
            and_terms = []
            for name, j in self._variables.items():
                if j in pos:
                    vars.add(name)
                elif j in neg:
                    vars.add(name)
        return vars

    def function(self, minterms = None):
        if minterms is None:
            minterms = self._branches

        variables = []
        structure = []
        for (pos, neg) in minterms:
            and_terms = []
            for name, j in self._variables.items():
                if j in pos:
                    and_terms.append(False)
                    variables.append(name)
                elif j in neg:
                    and_terms.append(True)
                    variables.append(name)
            if and_terms:
                structure.append(and_terms)
        if structure:
            return structure, variables
        else:
            return None, []
