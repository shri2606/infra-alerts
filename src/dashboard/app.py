"""
Simple Dashboard for Real-Time Infrastructure Monitoring
"""

import streamlit as st
import pandas as pd
import time
import re
from datetime import datetime
from collections import deque
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.inference.streaming_predictor import StreamingAnomalyPredictor
from src.simulation.scenarios import IncidentScenario


st.set_page_config(
    page_title="CloudInfraAI - Real-Time Monitoring",
    page_icon="🔍",
    layout="wide"
)


def initialize_session_state():
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'events_history' not in st.session_state:
        st.session_state.events_history = deque(maxlen=500)
    if 'total_anomalies' not in st.session_state:
        st.session_state.total_anomalies = 0
    if 'start_time' not in st.session_state:
        st.session_state.start_time = None
    if 'scenario' not in st.session_state:
        st.session_state.scenario = None
    if 'anomaly_breakdown' not in st.session_state:
        st.session_state.anomaly_breakdown = {'healthy': 0, 'warning': 0, 'critical': 0}
    if 'anomaly_categories' not in st.session_state:
        st.session_state.anomaly_categories = {
            'memory_spike': 0,
            'slow_api': 0,
            'http_error': 0
        }


def get_health_status(anomaly_rate):
    if anomaly_rate < 5:
        return "HEALTHY", "green"
    elif anomaly_rate < 15:
        return "WARNING", "orange"
    else:
        return "CRITICAL", "red"


def process_prediction_result(result):
    st.session_state.total_anomalies += result['num_anomalies']

    # Classify each anomaly by severity and category
    for idx in result['anomaly_indices']:
        score = result['scores'][idx]
        event = result['events'][idx]
        content = event.get('Content', '')

        # Severity classification
        if score < 0.73:
            st.session_state.anomaly_breakdown['healthy'] += 1
        elif score < 0.77:
            st.session_state.anomaly_breakdown['warning'] += 1
        else:
            st.session_state.anomaly_breakdown['critical'] += 1

        # Category classification based on content
        if 'used_ram' in content and ('2560' in content or '3072' in content or '4096' in content):
            st.session_state.anomaly_categories['memory_spike'] += 1
        elif 'time:' in content:
            # Check for slow API (time > 0.5s)
            time_match = re.search(r'time:\s*(\d+\.\d+)', content)
            if time_match and float(time_match.group(1)) >= 0.5:
                st.session_state.anomaly_categories['slow_api'] += 1

        if 'status:' in content:
            # Check for HTTP errors
            status_match = re.search(r'status:\s*(\d+)', content)
            if status_match and int(status_match.group(1)) >= 400:
                st.session_state.anomaly_categories['http_error'] += 1


def simulation_loop(scenario_func, duration, placeholder):
    scenario = IncidentScenario()
    events = scenario_func(duration)

    predictor = st.session_state.predictor
    event_rate = 5
    delay = 1.0 / event_rate

    for event in events:
        if not st.session_state.is_running:
            break

        st.session_state.events_history.append(event)
        result = predictor.process_event(event)

        if result is not None:
            process_prediction_result(result)

        with placeholder.container():
            render_dashboard_content()

        time.sleep(delay)


def render_dashboard_content():
    if st.session_state.predictor is None:
        return

    status = st.session_state.predictor.get_status()

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Events", status['total_events'])

    with col2:
        st.metric("Predictions Made", status['predictions_made'])

    with col3:
        anomaly_rate = (st.session_state.total_anomalies / max(status['total_events'], 1)) * 100
        st.metric("Anomalies Detected", st.session_state.total_anomalies,
                 f"{anomaly_rate:.1f}%")

    with col4:
        uptime = (datetime.now() - st.session_state.start_time).seconds if st.session_state.start_time else 0
        st.metric("Uptime", f"{uptime}s")

    # Warmup
    if status['is_warming_up']:
        st.info(f"Warming up buffer... {status['warmup_progress']:.0f}% ({status['buffer_size']}/{status['window_size']} events)")
        st.progress(status['warmup_progress'] / 100)
        return

    # Anomaly Breakdown
    st.markdown("---")
    st.subheader("Anomaly Breakdown")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Low Severity",
            value=st.session_state.anomaly_breakdown['healthy'],
            help="Anomaly score < 0.75"
        )

    with col2:
        st.metric(
            label="Medium Severity",
            value=st.session_state.anomaly_breakdown['warning'],
            help="Anomaly score 0.73 - 0.77"
        )

    with col3:
        st.metric(
            label="High Severity",
            value=st.session_state.anomaly_breakdown['critical'],
            help="Anomaly score > 0.77"
        )

    # Anomaly Categories
    st.markdown("---")
    st.subheader("Anomaly Categories")

    col1, col2, col3 = st.columns(3)

    with col1:
        count = st.session_state.anomaly_categories['memory_spike']
        color = "red" if count > 10 else ("orange" if count > 5 else "green")
        st.markdown(f"**Memory Spike**: :{color}[{count}]")

    with col2:
        count = st.session_state.anomaly_categories['slow_api']
        color = "red" if count > 10 else ("orange" if count > 5 else "green")
        st.markdown(f"**Slow API**: :{color}[{count}]")

    with col3:
        count = st.session_state.anomaly_categories['http_error']
        color = "red" if count > 10 else ("orange" if count > 5 else "green")
        st.markdown(f"**HTTP Error**: :{color}[{count}]")

    # Recent Events
    st.markdown("---")
    st.subheader("Recent Events")

    if st.session_state.events_history:
        recent = list(st.session_state.events_history)[-15:]
        df = pd.DataFrame([
            {
                'Level': e.get('Level', 'N/A'),
                'Component': e.get('Component', 'N/A')[:30],
                'Content': e.get('Content', 'N/A')[-80:]  # Show last 80 chars (includes RAM values)
            }
            for e in recent
        ])
        st.dataframe(df, use_container_width=True, height=400)
    else:
        st.info("No events received yet.")


def main():
    initialize_session_state()

    st.title("CloudInfraAI - Real-Time Infrastructure Monitoring")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        scenario_options = {
            "Demo (5 min - 2 incidents)": ("demo", 300),
            "Normal Operations (60s)": ("normal", 60),
            "Memory Spike (60s)": ("memory_spike", 60),
            "API Degradation (60s)": ("api_degradation", 60),
        }

        selected = st.selectbox(
            "Select Scenario",
            options=list(scenario_options.keys()),
            disabled=st.session_state.is_running
        )

        scenario_key, duration = scenario_options[selected]

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Start", disabled=st.session_state.is_running, use_container_width=True):
                st.session_state.predictor = StreamingAnomalyPredictor(
                    window_size=50,
                    stride=10
                )
                st.session_state.is_running = True
                st.session_state.start_time = datetime.now()
                st.session_state.scenario = scenario_key
                st.success("Started")

        with col2:
            if st.button("Stop", disabled=not st.session_state.is_running, use_container_width=True):
                st.session_state.is_running = False
                st.warning("Stopped")

        if st.button("Reset", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.rerun()

        st.markdown("---")
        st.markdown("**Settings (Fixed)**")
        st.text("Window Size: 50")
        st.text("Stride: 10")
        st.text("Event Rate: 5/sec")

    # Main content
    dashboard_placeholder = st.empty()

    if st.session_state.is_running and st.session_state.predictor is not None:
        scenario = IncidentScenario()

        scenario_map = {
            "demo": scenario.generate_demo_scenario,
            "normal": lambda d: scenario.simulator.generate_normal_stream(d),
            "memory_spike": scenario.memory_spike_incident,
            "api_degradation": scenario.api_degradation_incident,
        }

        scenario_func = scenario_map[st.session_state.scenario]

        try:
            simulation_loop(scenario_func, duration, dashboard_placeholder)
        except Exception as e:
            st.error(f"Error during simulation: {str(e)}")
            st.session_state.is_running = False
    else:
        with dashboard_placeholder.container():
            if st.session_state.predictor is None:
                st.info("Select a scenario from the sidebar and click Start to begin monitoring.")

                st.markdown("---")
                st.subheader("What This Dashboard Does")
                st.markdown("""
                - Monitors OpenStack infrastructure logs in real-time
                - Detects anomalies using AI (Transformer model)
                - Shows system health status
                - Displays recent events as they arrive

                **Model Performance:**
                - Accuracy: 97%
                - Precision: 72%
                - Recall: 81%
                - F1-Score: 76%
                """)
            else:
                render_dashboard_content()


if __name__ == "__main__":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    main()
