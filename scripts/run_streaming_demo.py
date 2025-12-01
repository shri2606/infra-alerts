#!/usr/bin/env python3
"""
Streaming Demo
==============

Real-time anomaly detection demo with log simulation.

Usage:
    python scripts/run_streaming_demo.py --scenario demo
    python scripts/run_streaming_demo.py --scenario memory_spike --duration 60
    python scripts/run_streaming_demo.py --scenario normal --duration 120
"""

import os
# Enable MPS fallback for Apple Silicon compatibility
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import time
import logging
from datetime import datetime

from src.simulation.log_simulator import LogSimulator
from src.simulation.scenarios import IncidentScenario
from src.inference.streaming_predictor import StreamingAnomalyPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StreamingDemo:
    """Orchestrate streaming demo."""

    def __init__(self, window_size=50, stride=10):
        """Initialize demo components."""
        self.predictor = StreamingAnomalyPredictor(
            window_size=window_size,
            stride=stride
        )

        # Set prediction callback
        self.predictor.set_prediction_callback(self.on_prediction)

        # Statistics
        self.total_anomalies_detected = 0
        self.total_predictions = 0

    def on_prediction(self, result: dict):
        """
        Callback when prediction is made.

        Args:
            result: Prediction result dictionary
        """
        self.total_predictions += 1

        logger.info("=" * 80)
        logger.info(f"PREDICTION #{result['prediction_id']}")
        logger.info("=" * 80)
        logger.info(f"Events processed: {result['total_events_processed']}")
        logger.info(f"Window size: {result['window_size']}")
        logger.info(f"Anomalies detected: {result['num_anomalies']}")
        logger.info(f"Anomaly rate: {result['anomaly_rate']:.1f}%")

        if result['anomaly_indices']:
            self.total_anomalies_detected += result['num_anomalies']
            logger.info(f"Anomaly positions in window: {result['anomaly_indices']}")
            scores_list = [f"{result['scores'][i]:.3f}" for i in result['anomaly_indices']]
            logger.info(f"Anomaly scores: {scores_list}")

            # Show sample anomalous events
            logger.info("\nSample anomalous events:")
            for idx in result['anomaly_indices'][:3]:
                event = result['events'][idx]
                logger.info(f"  Event {idx}: {event['Time']} - {event['Level']} - {event['Content'][:60]}...")
        else:
            logger.info("No anomalies detected in this window")

        logger.info("")

    def run_scenario(self, scenario_type: str, duration: int = 60, event_rate: int = 5):
        """
        Run streaming demo with specified scenario.

        Args:
            scenario_type: Type of scenario (normal, memory_spike, etc.)
            duration: Duration in seconds
            event_rate: Events per second
        """
        logger.info("=" * 80)
        logger.info("STREAMING ANOMALY DETECTION DEMO")
        logger.info("=" * 80)
        logger.info(f"Scenario: {scenario_type}")
        logger.info(f"Duration: {duration} seconds")
        logger.info(f"Event rate: {event_rate} events/sec")
        logger.info(f"Window size: {self.predictor.window_size}")
        logger.info(f"Stride: {self.predictor.stride}")
        logger.info("=" * 80)
        logger.info("")

        # Generate events
        events = self.generate_events(scenario_type, duration, event_rate)

        logger.info(f"Generated {len(events)} events total")
        logger.info("")

        # Stream events
        self.stream_events(events, event_rate)

        # Print summary
        self.print_summary()

    def generate_events(self, scenario_type: str, duration: int, event_rate: int):
        """Generate events for the scenario."""
        logger.info(f"Generating {scenario_type} scenario...")

        if scenario_type == 'demo':
            scenario = IncidentScenario()
            return scenario.generate_demo_scenario(total_duration=300)

        elif scenario_type == 'normal':
            simulator = LogSimulator()
            return simulator.generate_normal_stream(
                duration_seconds=duration,
                events_per_second=event_rate
            )

        else:
            scenario = IncidentScenario()
            scenario_map = {
                'memory_spike': scenario.memory_spike_incident,
                'api_degradation': scenario.api_degradation_incident,
                'vm_lifecycle': scenario.vm_lifecycle_incident,
                'system_warnings': scenario.system_warnings_incident,
                'mixed': scenario.mixed_incident,
            }

            if scenario_type in scenario_map:
                return scenario_map[scenario_type](duration_seconds=duration)
            else:
                raise ValueError(f"Unknown scenario: {scenario_type}")

    def stream_events(self, events, event_rate: int):
        """
        Stream events to predictor with realistic timing.

        Args:
            events: List of events to stream
            event_rate: Events per second
        """
        delay_per_event = 1.0 / event_rate

        logger.info("Starting event stream...")
        logger.info("")

        for i, event in enumerate(events, 1):
            # Process event
            self.predictor.process_event(event)

            # Show warmup progress
            if self.predictor.is_warming_up():
                if i % 10 == 0:
                    progress = self.predictor.get_warmup_progress()
                    logger.info(f"Warming up... {progress:.0f}% "
                              f"({len(self.predictor.buffer)}/{self.predictor.window_size} events)")

            # Simulate realistic timing
            time.sleep(delay_per_event)

        logger.info("")
        logger.info("Event stream completed")
        logger.info("")

    def print_summary(self):
        """Print demo summary."""
        status = self.predictor.get_status()

        logger.info("=" * 80)
        logger.info("DEMO SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total events processed: {status['total_events']}")
        logger.info(f"Total predictions made: {self.total_predictions}")
        logger.info(f"Total anomalies detected: {self.total_anomalies_detected}")
        logger.info("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Streaming Anomaly Detection Demo')

    parser.add_argument(
        '--scenario',
        type=str,
        default='demo',
        choices=['normal', 'memory_spike', 'api_degradation', 'vm_lifecycle',
                'system_warnings', 'mixed', 'demo'],
        help='Type of scenario to run'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Duration in seconds (ignored for demo scenario)'
    )

    parser.add_argument(
        '--event-rate',
        type=int,
        default=5,
        help='Events per second'
    )

    parser.add_argument(
        '--window-size',
        type=int,
        default=50,
        help='Size of sliding window'
    )

    parser.add_argument(
        '--stride',
        type=int,
        default=10,
        help='Stride between predictions'
    )

    args = parser.parse_args()

    # Create and run demo
    demo = StreamingDemo(window_size=args.window_size, stride=args.stride)

    try:
        if args.scenario == 'demo':
            demo.run_scenario('demo', duration=300, event_rate=args.event_rate)
        else:
            demo.run_scenario(args.scenario, duration=args.duration, event_rate=args.event_rate)

    except KeyboardInterrupt:
        logger.info("\nDemo interrupted by user")
        demo.print_summary()


if __name__ == "__main__":
    main()
