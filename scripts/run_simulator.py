#!/usr/bin/env python3
"""
Log Simulator Runner
====================

Generate realistic OpenStack log streams for testing and demo.

Usage:
    python scripts/run_simulator.py --scenario demo --output logs/simulated_logs.csv
    python scripts/run_simulator.py --scenario memory_spike --duration 60
    python scripts/run_simulator.py --scenario normal --duration 120
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import logging
from datetime import datetime

from src.simulation.log_simulator import LogSimulator
from src.simulation.scenarios import IncidentScenario

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_normal_scenario(duration: int, output_path: str):
    """Generate normal log stream."""
    logger.info(f"Generating normal log stream ({duration} seconds)...")

    simulator = LogSimulator()
    events = simulator.generate_normal_stream(duration_seconds=duration)

    logger.info(f"Generated {len(events)} events")

    if output_path:
        simulator.save_to_csv(events, output_path)
        logger.info(f"Saved to: {output_path}")

    return events


def run_incident_scenario(scenario_type: str, duration: int, output_path: str):
    """Generate incident scenario."""
    logger.info(f"Generating {scenario_type} incident ({duration} seconds)...")

    scenario_engine = IncidentScenario()

    scenario_map = {
        'memory_spike': scenario_engine.memory_spike_incident,
        'api_degradation': scenario_engine.api_degradation_incident,
        'vm_lifecycle': scenario_engine.vm_lifecycle_incident,
        'system_warnings': scenario_engine.system_warnings_incident,
        'mixed': scenario_engine.mixed_incident,
    }

    if scenario_type not in scenario_map:
        logger.error(f"Unknown scenario: {scenario_type}")
        logger.info(f"Available scenarios: {list(scenario_map.keys())}")
        return None

    events = scenario_map[scenario_type](duration_seconds=duration)

    logger.info(f"Generated {len(events)} events")

    if output_path:
        scenario_engine.simulator.save_to_csv(events, output_path)
        logger.info(f"Saved to: {output_path}")

    return events


def run_demo_scenario(output_path: str):
    """Generate complete 5-minute demo scenario."""
    logger.info("Generating 5-minute demo scenario...")
    logger.info("Timeline:")
    logger.info("  0-60s:   Normal operations")
    logger.info("  60-120s: Memory spike incident")
    logger.info("  120-180s: Return to normal")
    logger.info("  180-240s: API degradation incident")
    logger.info("  240-300s: Return to normal")

    scenario_engine = IncidentScenario()
    events = scenario_engine.generate_demo_scenario(total_duration=300)

    logger.info(f"Generated {len(events)} total events")

    if output_path:
        scenario_engine.simulator.save_to_csv(events, output_path)
        logger.info(f"Saved to: {output_path}")

    return events


def preview_events(events, num_events: int = 5):
    """Print preview of generated events."""
    logger.info(f"\nPreview (first {num_events} events):")
    logger.info("-" * 80)

    for i, event in enumerate(events[:num_events], 1):
        logger.info(f"Event {i}:")
        logger.info(f"  Time: {event['Time']}")
        logger.info(f"  Level: {event['Level']}")
        logger.info(f"  Component: {event['Component']}")
        logger.info(f"  Content: {event['Content'][:80]}...")
        logger.info("")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='OpenStack Log Simulator')

    parser.add_argument(
        '--scenario',
        type=str,
        default='demo',
        choices=['normal', 'memory_spike', 'api_degradation', 'vm_lifecycle',
                'system_warnings', 'mixed', 'demo'],
        help='Type of scenario to generate'
    )

    parser.add_argument(
        '--duration',
        type=int,
        default=60,
        help='Duration in seconds (ignored for demo scenario)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (optional)'
    )

    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show preview of generated events'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("OPENSTACK LOG SIMULATOR")
    logger.info("="*80)

    # Generate events based on scenario
    if args.scenario == 'demo':
        events = run_demo_scenario(args.output)
    elif args.scenario == 'normal':
        events = run_normal_scenario(args.duration, args.output)
    else:
        events = run_incident_scenario(args.scenario, args.duration, args.output)

    if events and args.preview:
        preview_events(events)

    logger.info("\nSimulation complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nSimulation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error during simulation: {e}", exc_info=True)
        sys.exit(1)
