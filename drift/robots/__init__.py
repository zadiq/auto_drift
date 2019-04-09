"""
Different implementations of simulated robots and interface with real robots
"""
from drift.robots.eth import ETHModel
from drift.commons import Animation  # For backward compatibility with my notebooks

__all__ = [
    'ETHModel',
    'Animation'
]
