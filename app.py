
# =========================================================
# DASHBOARD SIMULASI INSOMNIA
# (model disamakan dengan Minggu 12: trait/state terpisah,
# homeostasis, dan jaringan sosial antar-agent)
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

NUM_AGENTS = st.sidebar.slider("Jumlah Agen", 50, 500, 100)

ITERATIONS = st.sidebar.slider("Jumlah Hari Simulasi", 10, 100, 45)

K_NEIGHBORS = st.sidebar.slider("Ukuran Lingkaran Sosial (Tetangga)", 2, 10, 5)

social_interaction = st.sidebar.checkbox("Aktifkan Interaksi Sosial", value=True)

scenario = st.sidebar.selectbox(
    "Pilih Skenario",
    ["No Intervention", "Reactive", "Preventive", "High Disturbance"]
)

# =========================================================
# PARAMETER MODEL
# =========================================================

ALPHA = 0.018
BETA  = 0.010
GAMMA = 0.007

DELTA   = 0.020
EPSILON = 0.011
ZETA    = 0.007
ETA     = 0.012

Q_EQ   = 0.55
K_HOME = 0.035

W_SOCIAL_H = 0.35
W_SOCIAL_S = 0.25

# =========================================================
# CLASS AGENT
# =========================================================

class SleepAgent:

    def __init__(self, agent_id):

        self.agent_id = agent_id

        # TRAIT (tetap)
        self.C = random.uniform(0.4, 1.0)
        self.L = random.uniform(0.4, 1.0)

        # STATE (dinamis)
        self.Q = max(0, min(1, Q_EQ + random.uniform(-0.1, 0.1)))
        self.D = random.uniform(0.0, 0.2)
        self.H = random.uniform(0.2, 0.5)
        self.S = random.uniform(0.2, 0.6)

    def get_state(self):

        if self.Q >= 0.7:
            return "Healthy Sleep"
        elif self.Q >= 0.4:
            return "Mild Insomnia"
        elif self.Q >= 0.2:
            return "Moderate Insomnia"
        else:
            return "Severe Insomnia"


def get_scenario_inputs(agent, scenario):

    if scenario == "No Intervention":
        H_target = 0.05
        caffeine = random.uniform(0.2, 0.6)
        noise = random.uniform(0.1, 0.4)
        S_own = random.uniform(0.3, 0.7)

    elif scenario == "Reactive":
        if agent.Q < 0.4:
            H_target = 0.75
        elif agent.Q < 0.6:
            H_target = 0.45
        else:
            H_target = 0.15
        caffeine = random.uniform(0.2, 0.6)
        noise = random.uniform(0.1, 0.4)
        S_own = random.uniform(0.3, 0.7)

    elif scenario == "Preventive":
        H_target = 0.75
        caffeine = random.uniform(0.1, 0.4)
        noise = random.uniform(0.1, 0.3)
        S_own = random.uniform(0.2, 0.5)

    else:
        H_target = 0.15
        caffeine = random.uniform(0.4, 0.8)
        noise = random.uniform(0.4, 0.8)
        S_own = random.uniform(0.5, 0.9)

    return H_target, caffeine, noise, S_own


def compute_next_state(agent, scenario, neighbor_agents, social_interaction):

    H_target, caffeine, noise, S_own = get_scenario_inputs(agent, scenario)

    if social_interaction and len(neighbor_agents) > 0:
        peer_H = np.mean([nb.H for nb in neighbor_agents])
        peer_S = np.mean([nb.S for nb in neighbor_agents])
        H_day = (1 - W_SOCIAL_H) * H_target + W_SOCIAL_H * peer_H
        S_day = (1 - W_SOCIAL_S) * S_own + W_SOCIAL_S * peer_S
    else:
        H_day = H_target
        S_day = S_own

    H_day = max(0, min(1, H_day))
    S_day = max(0, min(1, S_day))

    positive = ALPHA * H_day + BETA * agent.C + GAMMA * agent.L
    negative = DELTA * S_day + EPSILON * caffeine + ZETA * noise + ETA * agent.D
    homeostasis = K_HOME * (Q_EQ - agent.Q)

    Q_new = max(0, min(1, agent.Q + homeostasis + positive - negative))

    if Q_new < 0.2:
        D_new = agent.D + 0.020
    elif Q_new < 0.4:
        D_new = agent.D + 0.012
    elif Q_new < 0.7:
        D_new = agent.D + 0.003
    else:
        D_new = agent.D - 0.015

    D_new = max(0, min(1, D_new))

    return Q_new, D_new, H_day, S_day


# =========================================================
# SIMULASI
# =========================================================

def run_simulation():

    agents = [SleepAgent(i + 1) for i in range(NUM_AGENTS)]
    agent_by_id = {a.agent_id: a for a in agents}

    neighbor_map = {
        a.agent_id: random.sample(
            [o.agent_id for o in agents if o.agent_id != a.agent_id],
            min(K_NEIGHBORS, NUM_AGENTS - 1)
        )
        for a in agents
    }

    avg_sleep_quality = []
    avg_sleep_debt = []

    healthy_count = []
    mild_count = []
    moderate_count = []
    severe_count = []

    for t in range(ITERATIONS):

        computed = []

        for agent in agents:

            neighbor_agents = [agent_by_id[nid] for nid in neighbor_map[agent.agent_id]]

            computed.append(
                compute_next_state(agent, scenario, neighbor_agents, social_interaction)
            )

        for agent, (Q_new, D_new, H_new, S_new) in zip(agents, computed):
            agent.Q, agent.D, agent.H, agent.S = Q_new, D_new, H_new, S_new

        avg_sleep_quality.append(np.mean([a.Q for a in agents]))
        avg_sleep_debt.append(np.mean([a.D for a in agents]))

        healthy = mild = moderate = severe = 0

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
        avg_sleep_quality, avg_sleep_debt,
        healthy_count, mild_count, moderate_count, severe_count
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

col1.metric("Average Sleep Quality", round(np.mean(avg_sleep_quality), 3))
col2.metric("Average Sleep Debt", round(np.mean(avg_sleep_debt), 3))
col3.metric("Severe Insomnia", severe_count[-1])

# =========================================================
# GRAFIK SLEEP QUALITY
# =========================================================

fig1 = px.line(
    y=avg_sleep_quality,
    labels={"x": "Hari", "y": "Sleep Quality"},
    title="Perkembangan Sleep Quality"
)

st.plotly_chart(fig1, use_container_width=True)

# =========================================================
# GRAFIK SLEEP DEBT
# =========================================================

fig2 = px.line(
    y=avg_sleep_debt,
    labels={"x": "Hari", "y": "Sleep Debt"},
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

fig3 = px.line(state_df, title="Distribusi State Agen")

st.plotly_chart(fig3, use_container_width=True)

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

st.caption(
    "Interaksi sosial: " + ("Aktif" if social_interaction else "Nonaktif")
    + f" | Lingkaran sosial: {K_NEIGHBORS} tetangga per agent"
)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown("Dashboard Pemodelan dan Simulasi Data - Insomnia Modeling")
