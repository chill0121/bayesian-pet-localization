"""
Streamlit Dashboard for Bayesian Pet Localization

Displays real-time position, historical heatmaps, and system status.
"""

import os
import time

import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

INFERENCE_API_URL = os.getenv("INFERENCE_API_URL", "http://localhost:8000")
REFRESH_INTERVAL = 1  # seconds

# -----------------------------------------------------------------------------
# API Functions
# -----------------------------------------------------------------------------

def get_position():
    """Fetch current position from inference API."""
    try:
        response = requests.get(f"{INFERENCE_API_URL}/position", timeout=2)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None


def get_rssi():
    """Fetch raw RSSI readings from inference API."""
    try:
        response = requests.get(f"{INFERENCE_API_URL}/rssi", timeout=2)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return {}


def get_health():
    """Fetch system health status."""
    try:
        response = requests.get(f"{INFERENCE_API_URL}/health", timeout=2)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return {"status": "offline", "mqtt_connected": False, "anchors_active": 0}


def get_floorplan():
    """Fetch floor plan data with anchor coordinates."""
    try:
        response = requests.get(f"{INFERENCE_API_URL}/floorplan", timeout=2)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None


# -----------------------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------------------

def create_floor_plan(floor_data: dict, position: dict = None, rssi_data: dict = None):
    """Create a floor plan visualization with rooms, anchors, and position marker."""
    floor_num = floor_data["floor"]
    width = floor_data.get("width_ft", 20)
    height = floor_data.get("height_ft", 20)
    name = floor_data.get("name", f"Floor {floor_num}")
    
    fig = go.Figure()
    
    # Draw room outlines with labels
    for room in floor_data.get("rooms", []):
        b = room["bounds"]
        if isinstance(b, dict):
            xs = [b["x1"], b["x2"], b["x2"], b["x1"], b["x1"]]
            ys = [b["y1"], b["y1"], b["y2"], b["y2"], b["y1"]]
        else:
            xs = [pt[0] for pt in b] + [b[0][0]]
            ys = [pt[1] for pt in b] + [b[0][1]]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="rgba(100, 100, 100, 0.8)", width=1.5),
            fill="toself", fillcolor="rgba(230, 240, 250, 0.3)",
            showlegend=False, hoverinfo="skip",
        ))
        # Room label at center
        cx = sum(xs[:-1]) / (len(xs) - 1)
        cy = sum(ys[:-1]) / (len(ys) - 1)
        fig.add_annotation(
            x=cx, y=cy,
            text=room.get("label", room["name"]),
            showarrow=False,
            font=dict(size=9, color="rgba(80, 80, 80, 0.7)"),
        )
    
    # Draw anchor positions
    anchor_x, anchor_y, anchor_labels = [], [], []
    for anchor in floor_data.get("anchors", []):
        pos = anchor.get("position", [anchor.get("x", 0), anchor.get("y", 0)])
        anchor_x.append(pos[0])
        anchor_y.append(pos[1])
        
        # Show RSSI value next to anchor if available
        rssi_val = ""
        if rssi_data and anchor["id"] in rssi_data:
            r = rssi_data[anchor["id"]]
            rssi_val = f"<br>RSSI: {r.get('rssi', r) if isinstance(r, dict) else r:.0f}"
        anchor_labels.append(f"{anchor['id']}{rssi_val}")
    
    if anchor_x:
        fig.add_trace(go.Scatter(
            x=anchor_x, y=anchor_y,
            mode="markers+text",
            marker=dict(size=10, color="dodgerblue", symbol="diamond"),
            text=[a["id"] for a in floor_data.get("anchors", [])],
            textposition="bottom center",
            textfont=dict(size=8),
            customdata=anchor_labels,
            hovertemplate="%{customdata}<extra></extra>",
            name="Anchors",
            showlegend=False,
        ))
    
    # Add position marker if the pet is on THIS floor
    if position and position.get("floor") == floor_num:
        x = position.get("x", width / 2)
        y = position.get("y", height / 2)
        confidence = position.get("confidence", 0.5)
        
        # Confidence circle (larger = less confident)
        radius = 2 * (1 - confidence) + 0.5
        fig.add_shape(
            type="circle",
            x0=x - radius, y0=y - radius,
            x1=x + radius, y1=y + radius,
            fillcolor="rgba(255, 80, 80, 0.2)",
            line=dict(color="red", width=1),
        )
        
        # Position marker
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=16, color="red", symbol="circle",
                       line=dict(width=2, color="darkred")),
            text=["🐕"],
            textposition="top center",
            textfont=dict(size=14),
            name="Pet",
            showlegend=False,
            hovertemplate=f"Pet @ ({x:.1f}, {y:.1f})<br>{position.get('location_label', '')}<extra></extra>",
        ))
    
    fig.update_layout(
        title=dict(text=name, font=dict(size=14)),
        xaxis=dict(range=[-1, width + 1], title="ft", dtick=5),
        yaxis=dict(range=[-1, height + 1], title="ft", scaleanchor="x", dtick=5),
        showlegend=False,
        height=max(350, int(height / width * 400) + 50),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    
    return fig


def create_rssi_bar_chart(rssi_data: dict):
    """Create bar chart of RSSI values from each anchor."""
    if not rssi_data:
        return None
    
    anchors = list(rssi_data.keys())
    values = [rssi_data[a].get("rssi", -100) if isinstance(rssi_data[a], dict) else rssi_data[a] 
              for a in anchors]
    
    fig = px.bar(
        x=anchors,
        y=values,
        labels={"x": "Anchor", "y": "RSSI (dBm)"},
        title="RSSI by Anchor",
        color=values,
        color_continuous_scale="RdYlGn",
    )
    
    fig.update_layout(
        yaxis=dict(range=[-100, -30]),
        height=300,
    )
    
    return fig


# -----------------------------------------------------------------------------
# Streamlit App
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Pet Localization Dashboard",
    page_icon="🐕",
    layout="wide",
)

st.title("Bayesian Pet Localization")
st.markdown("Real-time indoor positioning system")

# Sidebar - System Status
with st.sidebar:
    st.header("System Status")
    
    health = get_health()
    
    status_color = "green" if health["status"] == "ok" else "red"
    st.markdown(f"**API:** :{status_color}[{health['status']}]")
    
    mqtt_color = "green" if health["mqtt_connected"] else "red"
    st.markdown(f"**MQTT:** :{mqtt_color}[{'Connected' if health['mqtt_connected'] else 'Disconnected'}]")
    
    st.markdown(f"**Active Anchors:** {health['anchors_active']}")
    
    st.divider()
    
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    if auto_refresh:
        refresh_rate = st.slider("Refresh rate (sec)", 0.5, 5.0, 1.0, 0.5)

# Main content
position = get_position()
rssi = get_rssi()
floorplan = get_floorplan()

# Current position info
col1, col2, col3, col4 = st.columns(4)

if position:
    col1.metric("Location", position.get("location_label", "Unknown"))
    col2.metric("Floor", position.get("floor", "-"))
    col3.metric("Confidence", f"{position.get('confidence', 0) * 100:.0f}%")
    col4.metric("Activity", position.get("activity", "Unknown"))
else:
    col1.metric("Location", "No data")
    col2.metric("Floor", "-")
    col3.metric("Confidence", "-")
    col4.metric("Activity", "-")

st.divider()

# Floor plans
st.subheader("Floor Plans")

if floorplan and "floorplan" in floorplan and "floors" in floorplan["floorplan"]:
    floors = floorplan["floorplan"]["floors"]
    floor_cols = st.columns(len(floors))
    
    for i, floor_data in enumerate(floors):
        with floor_cols[i]:
            fig = create_floor_plan(floor_data, position, rssi)
            st.plotly_chart(fig, width="stretch")
else:
    # Fallback: simple empty floor plans
    floor_cols = st.columns(3)
    fallback_floors = [
        {"floor": 1, "name": "First Floor", "width_ft": 14, "height_ft": 12, "rooms": [], "anchors": []},
        {"floor": 2, "name": "Second Floor", "width_ft": 30, "height_ft": 15, "rooms": [], "anchors": []},
        {"floor": 3, "name": "Third Floor", "width_ft": 14.5, "height_ft": 20, "rooms": [], "anchors": []},
    ]
    for i, floor_data in enumerate(fallback_floors):
        with floor_cols[i]:
            fig = create_floor_plan(floor_data, position, rssi)
            st.plotly_chart(fig, width="stretch")

st.divider()

# RSSI readings
st.subheader("Anchor Signal Strength")

if rssi:
    rssi_chart = create_rssi_bar_chart(rssi)
    if rssi_chart:
        st.plotly_chart(rssi_chart, width="stretch")
    
    with st.expander("Raw RSSI Data"):
        st.json(rssi)
else:
    st.info("No RSSI data available. Waiting for beacon signal...")

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
