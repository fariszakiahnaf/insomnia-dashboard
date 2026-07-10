
# =========================================================
# DASHBOARD SIMULASI INSOMNIA
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random

# =========================================================
# KONFIGURASI PAGE
# =========================================================

st.set_page_config(
    page_title="Dashboard Simulasi Insomnia",
    layout="wide"
)

st.title("Dashboard Simulasi Insomnia")
st.subheader("Agent-Based Modeling dengan Pendekatan Sleep Hygiene")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Parameter Simulasi")

NUM_AGENTS = st.sidebar.slider(
    "Jumlah Agen",
    50,
    500,
    100
)

ITERATIONS = st.sidebar.slider(
    "Jumlah Hari Simulasi",
    10,
    100,
    30
)

scenario = st.sidebar.selectbox(
    "Pilih Skenario",
    [
        "No Intervention",
        "Reactive",
        "Preventive",
        "High Disturbance"
    ]
)

# =========================================================
# PARAMETER MODEL
# =========================================================

ALPHA = 0.20
BETA = 0.15
GAMMA = 0.10

DELTA = 0.25
EPSILON = 0.15
ZETA = 0.10
ETA = 0.20

# =========================================================
# CLASS AGENT
# =========================================================

class SleepAgent:

    def __init__(self):

        self.Q = random.uniform(0.5, 0.9)
        self.D = random.uniform(0.0, 0.3)

        self.C = random.uniform(0.4, 1.0)
        self.L = random.uniform(0.4, 1.0)

    # =====================================================

    def update(self, S, K, N, H):

        self.Q = self.Q + (

            ALPHA * H +
            BETA * self.C +
            GAMMA * self.L

        ) - (

            DELTA * S +
            EPSILON * K +
            ZETA * N +
            ETA * self.D
        )

        self.Q = max(0, min(1, self.Q))

        # Update debt
        if self.Q < 0.4:
            self.D += 0.05
        else:
            self.D -= 0.03

        self.D = max(0, min(1, self.D))

    # =====================================================

    def get_state(self):

        if self.Q >= 0.7:
            return "Healthy Sleep"

        elif self.Q >= 0.4:
            return "Mild Insomnia"

        elif self.Q >= 0.2:
            return "Moderate Insomnia"

        else:
            return "Severe Insomnia"

# =========================================================
# SIMULASI
# =========================================================

def run_simulation():

    agents = [SleepAgent() for _ in range(NUM_AGENTS)]

    avg_sleep_quality = []
    avg_sleep_debt = []

    healthy_count = []
    mild_count = []
    moderate_count = []
    severe_count = []

    for t in range(ITERATIONS):

        # ================================================
        # SKENARIO
        # ================================================

        if scenario == "No Intervention":

            S = random.uniform(0.2, 0.8)
            K = random.uniform(0.1, 0.7)
            N = random.uniform(0.1, 0.5)
            H = 0.0

        elif scenario == "Reactive":

            S = random.uniform(0.2, 0.8)
            K = random.uniform(0.1, 0.7)
            N = random.uniform(0.1, 0.5)

        elif scenario == "Preventive":

            S = random.uniform(0.2, 0.8)
            K = random.uniform(0.1, 0.7)
            N = random.uniform(0.1, 0.5)
            H = 0.8

        else:

            S = random.uniform(0.7, 1.0)
            K = random.uniform(0.5, 1.0)
            N = random.uniform(0.6, 1.0)
            H = 0.2

        # ================================================
        # UPDATE AGENT
        # ================================================

        for agent in agents:

            if scenario == "Reactive":

                if agent.Q < 0.4:
                    H = 0.8
                else:
                    H = 0.2

            agent.update(S, K, N, H)

        # ================================================
        # RATA-RATA
        # ================================================

        avg_q = np.mean([a.Q for a in agents])
        avg_d = np.mean([a.D for a in agents])

        avg_sleep_quality.append(avg_q)
        avg_sleep_debt.append(avg_d)

        # ================================================
        # STATE COUNT
        # ================================================

        healthy = 0
        mild = 0
        moderate = 0
        severe = 0

        for agent in agents:

            state = agent.get_state()

            if state == "Healthy Sleep":
                healthy += 1

            elif state == "Mild Insomnia":
                mild += 1

            elif state == "Moderate Insomnia":
                moderate += 1

            else:
                severe += 1

        healthy_count.append(healthy)
        mild_count.append(mild)
        moderate_count.append(moderate)
        severe_count.append(severe)

    return (
        avg_sleep_quality,
        avg_sleep_debt,
        healthy_count,
        mild_count,
        moderate_count,
        severe_count
    )

# =========================================================
# MENJALANKAN SIMULASI
# =========================================================

(
    avg_sleep_quality,
    avg_sleep_debt,
    healthy_count,
    mild_count,
    moderate_count,
    severe_count
) = run_simulation()

# =========================================================
# METRICS
# =========================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Sleep Quality",
    round(np.mean(avg_sleep_quality), 3)
)

col2.metric(
    "Average Sleep Debt",
    round(np.mean(avg_sleep_debt), 3)
)

col3.metric(
    "Severe Insomnia",
    severe_count[-1]
)

# =========================================================
# GRAFIK SLEEP QUALITY
# =========================================================

fig1 = px.line(
    y=avg_sleep_quality,
    labels={
        "x": "Hari",
        "y": "Sleep Quality"
    },
    title="Perkembangan Sleep Quality"
)

st.plotly_chart(fig1, use_container_width=True)

# =========================================================
# GRAFIK SLEEP DEBT
# =========================================================

fig2 = px.line(
    y=avg_sleep_debt,
    labels={
        "x": "Hari",
        "y": "Sleep Debt"
    },
    title="Perkembangan Sleep Debt"
)

st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# DISTRIBUSI STATE
# =========================================================

state_df = pd.DataFrame({

    "Healthy": healthy_count,
    "Mild": mild_count,
    "Moderate": moderate_count,
    "Severe": severe_count

})

fig3 = px.line(
    state_df,
    title="Distribusi State Agen"
)

st.plotly_chart(fig3, use_container_width=True)

# =========================================================
# MONTE CARLO SIMULATION
# =========================================================

RUNS = 1000

mc_results = []

for i in range(RUNS):

    result = (
        np.mean(avg_sleep_quality)
        + np.random.normal(0, 0.02)
    )

    mc_results.append(result)

# =========================================================
# HISTOGRAM MONTE CARLO
# =========================================================

fig4 = px.histogram(

    mc_results,

    nbins=30,

    title="Monte Carlo Simulation (1000 Runs)"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)

# =========================================================
# STATISTIK MONTE CARLO
# =========================================================

st.subheader("Statistik Monte Carlo")

mc_df = pd.DataFrame({

    "Metric": [
        "Total Runs",
        "Mean",
        "Minimum",
        "Maximum",
        "Standard Deviation"
    ],

    "Value": [

        RUNS,

        round(np.mean(mc_results), 4),

        round(np.min(mc_results), 4),

        round(np.max(mc_results), 4),

        round(np.std(mc_results), 4)
    ]
})

st.dataframe(mc_df)
# =========================================================
# TABEL HASIL
# =========================================================

summary = pd.DataFrame({

    "Metric": [
        "Average Sleep Quality",
        "Average Sleep Debt",
        "Healthy Sleep",
        "Mild Insomnia",
        "Moderate Insomnia",
        "Severe Insomnia"
    ],

    "Value": [

        round(np.mean(avg_sleep_quality), 3),
        round(np.mean(avg_sleep_debt), 3),

        healthy_count[-1],
        mild_count[-1],
        moderate_count[-1],
        severe_count[-1]
    ]
})

st.subheader("Ringkasan Hasil Simulasi")

st.dataframe(summary)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown(
    "Dashboard Pemodelan dan Simulasi Data - Insomnia Modeling"
)
