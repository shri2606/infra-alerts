"""
Log Simulator
=============

Generate realistic OpenStack log events based on actual dataset patterns.
"""

import pandas as pd
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


class LogSimulator:
    """
    Generate realistic OpenStack log events.

    Uses patterns extracted from actual OpenStack logs to generate
    realistic event streams for testing and demonstration.
    """

    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize the log simulator.

        Args:
            dataset_path: Path to OpenStack CSV dataset (optional)
        """
        if dataset_path is None:
            dataset_path = "data/raw/OpenStack_2k.log_structured.csv"

        self.dataset_path = Path(dataset_path)
        self.df = None
        self.patterns = {}

        # Load and analyze dataset if it exists
        if self.dataset_path.exists():
            self._load_dataset()
            self._extract_patterns()

    def _load_dataset(self):
        """Load the dataset from CSV."""
        self.df = pd.read_csv(self.dataset_path)

    def _extract_patterns(self):
        """Extract log patterns from the dataset."""
        if self.df is None:
            return

        # Extract common patterns
        self.patterns = {
            'api_get': self.df[self.df['EventId'] == 'E25'].copy(),
            'api_post': self.df[self.df['EventId'] == 'E26'].copy(),
            'vm_started': self.df[self.df['EventId'] == 'E22'].copy(),
            'vm_paused': self.df[self.df['EventId'] == 'E20'].copy(),
            'vm_resumed': self.df[self.df['EventId'] == 'E21'].copy(),
            'vm_terminated': self.df[self.df['EventId'] == 'E11'].copy(),
            'resource_tracking': self.df[self.df['EventId'] == 'E32'].copy(),
        }

    def generate_api_request(self,
                            method: str = 'GET',
                            status: int = 200,
                            response_time: float = 0.25) -> Dict:
        """
        Generate an API request log event.

        Args:
            method: HTTP method (GET, POST, DELETE)
            status: HTTP status code (200, 404, 500, etc.)
            response_time: Response time in seconds

        Returns:
            Dictionary with log event fields
        """
        project_id = self._generate_uuid()
        server_id = self._generate_uuid()
        req_id = f"req-{self._generate_uuid()}"

        if method == 'GET':
            content = f'10.11.10.1 "GET /v2/{project_id}/servers/detail HTTP/1.1" status: {status} len: 1893 time: {response_time:.7f}'
            event_id = 'E25'
        elif method == 'POST':
            content = f'10.11.10.1 "POST /v2/{project_id}/os-server-external-events HTTP/1.1" status: {status} len: 380 time: {response_time:.7f}'
            event_id = 'E26'
        else:  # DELETE
            content = f'10.11.10.1 "DELETE /v2/{project_id}/servers/{server_id} HTTP/1.1" status: {status} len: 203 time: {response_time:.7f}'
            event_id = 'E24'

        return {
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Time': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'Level': 'INFO',
            'Component': 'nova.osapi_compute.wsgi.server',
            'ADDR': req_id,
            'Content': content,
            'EventId': event_id
        }

    def generate_vm_lifecycle_event(self, event_type: str, instance_id: Optional[str] = None) -> Dict:
        """
        Generate a VM lifecycle event.

        Args:
            event_type: Type of event (started, paused, resumed, terminated)
            instance_id: VM instance UUID (generated if not provided)

        Returns:
            Dictionary with log event fields
        """
        if instance_id is None:
            instance_id = self._generate_uuid()

        req_id = f"req-{self._generate_uuid()}"

        event_map = {
            'started': ('VM Started (Lifecycle Event)', 'E22'),
            'paused': ('VM Paused (Lifecycle Event)', 'E20'),
            'resumed': ('VM Resumed (Lifecycle Event)', 'E21'),
            'terminated': ('Terminating instance', 'E11'),
        }

        message, event_id = event_map.get(event_type, ('Unknown event', 'E1'))

        return {
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Time': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'Level': 'INFO',
            'Component': 'nova.compute.manager',
            'ADDR': req_id,
            'Content': f'[instance: {instance_id}] {message}',
            'EventId': event_id
        }

    def generate_resource_event(self, used_ram: int = 2560, total_ram: int = 64172) -> Dict:
        """
        Generate a resource tracking event.

        Args:
            used_ram: Used RAM in MB
            total_ram: Total RAM in MB

        Returns:
            Dictionary with log event fields
        """
        req_id = f"req-{self._generate_uuid()}"
        node_name = "cp-1.slowvm1.tcloud-pg0.utah.cloudlab.us"

        content = (f'Final resource view: name={node_name} '
                  f'phys_ram={total_ram}MB used_ram={used_ram}MB '
                  f'phys_disk=15GB used_disk=20GB total_vcpus=16 used_vcpus=1 pci_stats=[]')

        return {
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Time': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'Level': 'INFO',
            'Component': 'nova.compute.resource_tracker',
            'ADDR': req_id,
            'Content': content,
            'EventId': 'E32'
        }

    def generate_warning_event(self, message: str = "Unknown base file") -> Dict:
        """
        Generate a warning event.

        Args:
            message: Warning message

        Returns:
            Dictionary with log event fields
        """
        req_id = f"req-{self._generate_uuid()}"
        base_file = f"/var/lib/nova/instances/_base/{self._generate_hash()}"

        return {
            'Date': datetime.now().strftime('%Y-%m-%d'),
            'Time': datetime.now().strftime('%H:%M:%S.%f')[:-3],
            'Level': 'WARNING',
            'Component': 'nova.virt.libvirt.imagecache',
            'ADDR': req_id,
            'Content': f'{message}: {base_file}',
            'EventId': 'E29'
        }

    def _generate_uuid(self) -> str:
        """Generate a random UUID."""
        return str(uuid.uuid4())

    def _generate_hash(self) -> str:
        """Generate a random hash for file names."""
        return ''.join(random.choices('abcdef0123456789', k=40))

    def generate_normal_stream(self, duration_seconds: int = 60, events_per_second: int = 5) -> List[Dict]:
        """
        Generate a stream of normal log events.

        Args:
            duration_seconds: Duration of the stream in seconds
            events_per_second: Number of events per second

        Returns:
            List of log event dictionaries
        """
        events = []
        total_events = duration_seconds * events_per_second

        for _ in range(total_events):
            # 70% API requests, 20% VM lifecycle, 10% resource tracking
            rand = random.random()

            if rand < 0.7:
                # Normal API request
                event = self.generate_api_request(
                    method=random.choice(['GET', 'GET', 'GET', 'POST']),
                    status=random.choice([200, 200, 200, 202]),
                    response_time=random.uniform(0.20, 0.35)
                )
            elif rand < 0.9:
                # VM lifecycle
                event = self.generate_vm_lifecycle_event(
                    event_type=random.choice(['started', 'paused', 'resumed'])
                )
            else:
                # Resource tracking
                event = self.generate_resource_event(
                    used_ram=random.choice([512, 1024, 2048]),
                    total_ram=64172
                )

            events.append(event)

        return events

    def to_dataframe(self, events: List[Dict]) -> pd.DataFrame:
        """
        Convert list of events to DataFrame.

        Args:
            events: List of event dictionaries

        Returns:
            DataFrame with log events
        """
        return pd.DataFrame(events)

    def save_to_csv(self, events: List[Dict], output_path: str):
        """
        Save events to CSV file.

        Args:
            events: List of event dictionaries
            output_path: Path to output CSV file
        """
        df = self.to_dataframe(events)
        df.to_csv(output_path, index=False)
