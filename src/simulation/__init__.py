"""
Log Simulation Module
====================

Generate realistic OpenStack log streams for testing and demo.
"""

from .log_simulator import LogSimulator
from .scenarios import IncidentScenario

__all__ = ['LogSimulator', 'IncidentScenario']
