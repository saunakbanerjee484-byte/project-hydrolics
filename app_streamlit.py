"""
app_streamlit.py
=================
Interactive web dashboard built on Streamlit + Plotly, sharing the exact
same physics engine as main.py / gui_tkinter.py.

Run with:  streamlit run app_streamlit.py
"""
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from open_channel_flow.geometry import Trapezoidal, GeometryError
from open_channel_flow.core import OpenChannelFlow, ConvergenceError
from open_channel_flow.energy import energy_curve, specific_energy
from open_channel_flow.gvf import solve_profile, stitch_uniform_flow
from open_channel_flow.jump import locate_jump
from open_channel_flow.io_utils import build_station_table

st.set_page_config(page_title="Open Channel Flow Analysis", layout="wide")
st.title("Open Channel Flow -- Interactive GVF Analysis")

with st.sidebar:
    st.header("Channel & Flow Inputs")
    B = st.slider("Bottom width B (m)", 0.5, 20.0, 4.0, 0.1)
    z = st.slider("Side slope z (H:V)", 0.0, 4.0, 1.5, 0.1)
    S0 = st.number_input("Bed slope S0 (m/m)", 0.0001, 0.05, 0.001, format="%.4f")
    Q = st.slider("Discharge Q (m^3/s)", 1.0, 200.0, 20.0, 1.0)
    n = st.slider("Manning's n", 0.010, 0.060, 0.025, 0.001)
    st.header("GVF Boundary")
    y_start = st.slider("Boundary depth y_start (m)", 0.1, 10.0, 3.0, 0.1)
    L = st.number_input("Reach length L (m)", 10.0, 20000.0, 1000.0)
    dx = st.number_input("Step size dx (m)", 1.0, 100.0, 10.0)

try:
    section = Trapezoidal(B=B, z=z)
    ocf = OpenChannelFlow(section=section, Q=Q, n=n, S0=S0)
    yn, yc = ocf.normal_depth(), ocf.critical_depth()
except (GeometryError, ConvergenceError) as e:
    st.error(str(e))
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Normal depth yn", f"{yn:.3f} m")
col2.metric("Critical depth yc", f"{yc:.3f} m")
col3.metric("Froude @ yn", f"{ocf.froude(yn):.3f}")
col4.metric("Classification", ocf.classify_channel())

gvf_result = solve_profile(ocf, y_start, L, dx, yn=yn, yc=yc)
jump = locate_jump(gvf_result, ocf, yn, yc)

import dataclasses
x_full, y_full = stitch_uniform_flow(gvf_result, yn, jump)
gvf_result = dataclasses.replace(gvf_result, x=x_full, y=y_full)

st.subheader(f"Curve type: {gvf_result.curve_type}")
st.caption(gvf_result.regime)

y_curve, E_curve, e_min_point = energy_curve(Q, ocf.g, section, yn, yc)

c1, c2 = st.columns(2)
with c1:
    fig_e = go.Figure()
    fig_e.add_trace(go.Scatter(x=E_curve, y=y_curve, mode="lines", name="E-y curve"))
    fig_e.add_trace(go.Scatter(x=[e_min_point[0]], y=[yc], mode="markers",
                                marker=dict(color="red", size=10), name=f"Critical yc={yc:.3f}"))
    E_at_yn = specific_energy(yn, Q, ocf.g, section)
    fig_e.add_trace(go.Scatter(x=[E_at_yn], y=[yn], mode="markers",
                                marker=dict(color="orange", size=10), name=f"Normal yn={yn:.3f}"))
    fig_e.update_layout(title="Specific Energy Diagram", xaxis_title="E (m)", yaxis_title="y (m)")
    st.plotly_chart(fig_e, use_container_width=True)

with c2:
    x_full = np.linspace(0, L, 200)
    bed = S0 * x_full
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=x_full, y=bed, mode="lines", name="Bed", line=dict(color="saddlebrown")))
    fig_p.add_trace(go.Scatter(x=x_full, y=bed + yn, mode="lines", name="Normal depth line",
                                line=dict(color="black", dash="dot")))
    fig_p.add_trace(go.Scatter(x=x_full, y=bed + yc, mode="lines", name="Critical depth line",
                                line=dict(color="black", dash="dash")))
    fig_p.add_trace(go.Scatter(x=gvf_result.x, y=S0 * gvf_result.x + gvf_result.y, mode="lines",
                                name="Water surface", line=dict(color="royalblue", width=3)))
    if jump:
        fig_p.add_vline(x=jump["x_jump"], line=dict(color="crimson", dash="dashdot"),
                         annotation_text="Hydraulic jump")
    fig_p.update_layout(title="Water Surface Profile", xaxis_title="x (m)", yaxis_title="Elevation (m)")
    st.plotly_chart(fig_p, use_container_width=True)

if jump:
    st.warning(f"Hydraulic jump at x = {jump['x_jump']:.1f} m -- "
               f"y1 = {jump['y1']:.3f} m -> y2 = {jump['y2']:.3f} m "
               f"(energy dissipated: {jump['delta_E']:.3f} m)")

rows = build_station_table(ocf, gvf_result)
df = pd.DataFrame(rows, columns=["x_m", "y_m", "V_mps", "Fr", "Sf", "E_m"])
st.subheader("Station-by-station results")
st.dataframe(df, use_container_width=True, height=300)
st.download_button("Download CSV", df.to_csv(index=False), file_name="gvf_results.csv")
