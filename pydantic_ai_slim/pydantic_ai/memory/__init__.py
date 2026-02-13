"""Observational memory for Pydantic AI agents.

Provides a three-agent compression system (Observer + Reflector + main agent)
that enables unlimited conversation length by compressing message history into
structured observations.
"""

from ._observational import ObservationalMemory
from ._storage import InMemoryOMStorage, OMStorage
from ._types import OMContext, OMRecord

__all__ = [
    'ObservationalMemory',
    'OMStorage',
    'OMRecord',
    'OMContext',
    'InMemoryOMStorage',
]
