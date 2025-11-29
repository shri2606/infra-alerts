"""
Incident Scenarios
==================

Define and generate realistic incident scenarios for demonstration.
"""

import random
from typing import List, Dict
from .log_simulator import LogSimulator


class IncidentScenario:
    """
    Generate incident scenarios by injecting anomalies into log streams.
    """

    def __init__(self):
        """Initialize the scenario engine."""
        self.simulator = LogSimulator()

    def memory_spike_incident(self, duration_seconds: int = 60) -> List[Dict]:
        """
        Generate a memory spike incident scenario.

        Pattern: Normal baseline -> Sudden spike to 2560+ MB -> Sustained high usage

        Args:
            duration_seconds: Duration of the incident

        Returns:
            List of log events representing the incident
        """
        events = []
        events_per_interval = 3

        # Phase 1: Normal baseline (first 20%)
        baseline_duration = int(duration_seconds * 0.2)
        for _ in range(baseline_duration * events_per_interval):
            events.append(self.simulator.generate_resource_event(
                used_ram=random.choice([512, 1024, 2048]),
                total_ram=64172
            ))

        # Phase 2: Spike begins (next 20%)
        spike_start = int(duration_seconds * 0.2)
        for _ in range(spike_start * events_per_interval):
            events.append(self.simulator.generate_resource_event(
                used_ram=random.choice([2560, 3072]),
                total_ram=64172
            ))

        # Phase 3: Sustained high usage (remaining 60%)
        sustained_duration = int(duration_seconds * 0.6)
        for _ in range(sustained_duration * events_per_interval):
            events.append(self.simulator.generate_resource_event(
                used_ram=random.choice([2560, 3072, 4096]),
                total_ram=64172
            ))

        return events

    def api_degradation_incident(self, duration_seconds: int = 60) -> List[Dict]:
        """
        Generate an API degradation incident scenario.

        Pattern: Normal response times -> Slow responses -> Some errors

        Args:
            duration_seconds: Duration of the incident

        Returns:
            List of log events representing the incident
        """
        events = []
        events_per_second = 5

        # Phase 1: Normal (first 30%)
        normal_duration = int(duration_seconds * 0.3)
        for _ in range(normal_duration * events_per_second):
            events.append(self.simulator.generate_api_request(
                method='GET',
                status=200,
                response_time=random.uniform(0.20, 0.35)
            ))

        # Phase 2: Degraded performance (next 40%)
        degraded_duration = int(duration_seconds * 0.4)
        for _ in range(degraded_duration * events_per_second):
            events.append(self.simulator.generate_api_request(
                method='GET',
                status=200,
                response_time=random.uniform(0.50, 0.80)  # Slow responses
            ))

        # Phase 3: Errors start appearing (remaining 30%)
        error_duration = int(duration_seconds * 0.3)
        for _ in range(error_duration * events_per_second):
            # Mix of slow responses and errors
            if random.random() < 0.3:  # 30% error rate
                status = random.choice([404, 500, 503])
            else:
                status = 200

            events.append(self.simulator.generate_api_request(
                method='GET',
                status=status,
                response_time=random.uniform(0.50, 1.0)
            ))

        return events

    def vm_lifecycle_incident(self, duration_seconds: int = 60) -> List[Dict]:
        """
        Generate a VM lifecycle incident scenario.

        Pattern: Normal lifecycle -> Duplicate resume events -> Incomplete sequences

        Args:
            duration_seconds: Duration of the incident

        Returns:
            List of log events representing the incident
        """
        events = []
        instance_id = self.simulator._generate_uuid()

        # Normal lifecycle
        events.append(self.simulator.generate_vm_lifecycle_event('started', instance_id))
        events.append(self.simulator.generate_vm_lifecycle_event('paused', instance_id))

        # Anomaly: Duplicate resume events (should only happen once)
        events.append(self.simulator.generate_vm_lifecycle_event('resumed', instance_id))
        events.append(self.simulator.generate_vm_lifecycle_event('resumed', instance_id))
        events.append(self.simulator.generate_vm_lifecycle_event('resumed', instance_id))

        # Incomplete termination (no proper cleanup)
        events.append(self.simulator.generate_vm_lifecycle_event('terminated', instance_id))

        return events

    def system_warnings_incident(self, duration_seconds: int = 60) -> List[Dict]:
        """
        Generate a system warnings incident scenario.

        Pattern: Normal operations -> Warning messages appear

        Args:
            duration_seconds: Duration of the incident

        Returns:
            List of log events representing the incident
        """
        events = []
        events_per_interval = 2

        # Generate warning events
        for _ in range(duration_seconds * events_per_interval):
            events.append(self.simulator.generate_warning_event(
                message=random.choice([
                    "Unknown base file",
                    "Failed to load base file",
                    "Missing configuration"
                ])
            ))

        return events

    def mixed_incident(self, duration_seconds: int = 120) -> List[Dict]:
        """
        Generate a complex incident with multiple issues.

        Combines memory spike, API degradation, and warnings.

        Args:
            duration_seconds: Duration of the incident

        Returns:
            List of log events representing the incident
        """
        events = []

        # Divide time into phases
        phase_duration = duration_seconds // 3

        # Phase 1: Memory spike
        events.extend(self.memory_spike_incident(phase_duration))

        # Phase 2: API degradation
        events.extend(self.api_degradation_incident(phase_duration))

        # Phase 3: System warnings
        events.extend(self.system_warnings_incident(phase_duration))

        # Shuffle to make it more realistic
        random.shuffle(events)

        return events

    def generate_demo_scenario(self, total_duration: int = 300) -> List[Dict]:
        """
        Generate a complete demo scenario.

        Timeline:
        - 0-60s: Normal operations
        - 60-120s: Memory spike incident
        - 120-180s: Return to normal
        - 180-240s: API degradation incident
        - 240-300s: Return to normal

        Args:
            total_duration: Total duration in seconds (default: 5 minutes)

        Returns:
            List of log events for the entire demo
        """
        events = []

        # Minute 1: Normal baseline
        events.extend(self.simulator.generate_normal_stream(duration_seconds=60))

        # Minute 2: Memory spike incident
        events.extend(self.memory_spike_incident(duration_seconds=60))

        # Minute 3: Return to normal
        events.extend(self.simulator.generate_normal_stream(duration_seconds=60))

        # Minute 4: API degradation incident
        events.extend(self.api_degradation_incident(duration_seconds=60))

        # Minute 5: Return to normal
        events.extend(self.simulator.generate_normal_stream(duration_seconds=60))

        return events
