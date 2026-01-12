# ================================
# ACO.py
# Employee Shift Scheduling (09-17 & 14-22)
# Modified: best_schedule from Pareto front + convergence + marked Pareto point
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import os
import time
import matplotlib.pyplot as plt

# ================================
# CONFIG
# ================================
st.title("🐜 ACO Employee Shift Scheduling (09-17 & 14-22)")

n_departments = 6
n_days = 7
n_periods = 28
SHIFT_LENGTH = 14
REST_PROB = 0.25

# ================================
# LOAD DEMAND
# ================================
DEMAND = np.zeros((n_departments, n_days, n_periods), dtype=int)
folder_path = "./Demand/"

st.sidebar.header("📥 Demand Files")
for dept in range(n_departments):
    file_path = os.path.join(folder_path, f"Dept{dept+1}.xlsx")
    if not os.path.exists(file_path):
        st.sidebar.error(f"❌ Dept{dept+1}.xlsx not found")
        continue

    df = pd.read_excel(file_path, header=None)
    df_subset = df.iloc[1:1+n_days, 1:1+n_periods]
    df_subset = df_subset.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    DEMAND[dept] = df_subset.values
    st.sidebar.success(f"✅ Dept{dept+1} loaded")

# ================================
# HELPER FUNCTIONS
# ================================
def longest_consecutive_ones(arr):
    max_len = curr = 0
    for v in arr:
        if v == 1:
            curr += 1
            max_len = max(max_len, curr)
        else:
            curr = 0
    return max_len

def pareto_filter(points):
    pareto = []
    for p in points:
        if not any((q[0] <= p[0] and q[1] <= p[1]) and q != p for q in points):
            pareto.append(p)
    return pareto

# ================================
# FITNESS FUNCTION
# ================================
def fitness(schedule, demand, max_hours):
    penalty = 0
    n_departments, days, periods, employees = schedule.shape

    for dept in range(n_departments):
        for d in range(days):
            for t in range(periods):
                assigned = np.sum(schedule[dept, d, t, :])
                required = demand[dept, d, t]
                if assigned < required:
                    penalty += (required - assigned) * 200

        for e in range(employees):
            total_hours = np.sum(schedule[:, :, :, e])
            if total_hours > max_hours:
                penalty += (total_hours - max_hours) * 150

            days_worked = np.sum(np.sum(schedule[:, :, :, e], axis=2) > 0)
            if days_worked < (n_days - 1):
                penalty += 300

        for d in range(days):
            for e in range(employees):
                daily = schedule[dept, d, :, e]
                worked = np.sum(daily)
                if worked > 0 and worked != SHIFT_LENGTH:
                    penalty += 1000
                if worked == SHIFT_LENGTH and longest_consecutive_ones(daily) < SHIFT_LENGTH:
                    penalty += 2000

    return penalty

# ================================
# MULTI-OBJECTIVE
# ================================
def compute_objectives(schedule, demand, max_hours):
    total_shortage = 0
    workload_penalty = 0

    for dept in range(demand.shape[0]):
        for d in range(n_days):
            for t in range(n_periods):
                total_shortage += max(demand[dept, d, t] - np.sum(schedule[dept, d, t]), 0)

        for e in range(schedule.shape[3]):
            total_hours = np.sum(schedule[:, :, :, e])
            if total_hours > max_hours:
                workload_penalty += (total_hours - max_hours)

    return total_shortage, workload_penalty

# ================================
# OFF-DAY
# ================================
def generate_min_one_off_schedule(n_employees, n_days):
    off = np.zeros((n_employees, n_days), dtype=int)
    for e in range(n_employees):
        off[e, random.randint(0, n_days - 1)] = 1
    return off

# ================================
# ACO SCHEDULER
# ================================
def ACO_scheduler(demand, n_employees_per_dept, n_ants, n_iter,
                  alpha, evaporation, Q, max_hours, early_stop):

    pheromone = np.ones((n_departments, n_days, 2, max(n_employees_per_dept)))
    fitness_history = []
    pareto_raw = []
    pareto_schedules = []
    best_score_global = float("inf")
    best_schedule_global = None
    best_off_schedules_global = None
    no_improve = 0
    start_time = time.time()

    for it in range(n_iter):
        min_score_iter = float("inf")  # track minimum per iteration
        for _ in range(n_ants):
            schedule = np.zeros((n_departments, n_days, n_periods, max(n_employees_per_dept)))
            off_schedules = []

            for dept in range(n_departments):
                n_emp = n_employees_per_dept[dept]
                off = generate_min_one_off_schedule(n_emp, n_days)
                off_schedules.append(off)

                for d in range(n_days):
                    for e in range(n_emp):
                        if off[e, d] == 1 or random.random() < REST_PROB:
                            continue

                        tau_m = pheromone[dept, d, 0, e] ** alpha
                        tau_e = pheromone[dept, d, 1, e] ** alpha
                        p_m = tau_m / (tau_m + tau_e + 1e-6)

                        if random.random() < p_m:
                            schedule[dept, d, 0:SHIFT_LENGTH, e] = 1
                        else:
                            schedule[dept, d, 14:14+SHIFT_LENGTH, e] = 1

            score = fitness(schedule, demand, max_hours)
            s, w = compute_objectives(schedule, demand, max_hours)
            pareto_raw.append((s, w))
            pareto_schedules.append(schedule.copy())

            min_score_iter = min(min_score_iter, score)

            # Track global best (optional)
            if score < best_score_global:
                best_score_global = score
                best_schedule_global = schedule.copy()
                best_off_schedules_global = off_schedules.copy()
                no_improve = 0
            else:
                no_improve += 1

            pheromone *= (1 - evaporation)
            pheromone += Q / (1 + score)

        fitness_history.append(min_score_iter)

        if no_improve >= early_stop:
            break

    # ================================
    # Pareto front filtering
    # ================================
    pareto_filtered = pareto_filter(pareto_raw)
    filtered_schedules = [pareto_schedules[i] for i, p in enumerate(pareto_raw) if p in pareto_filtered]

    # ================================
    # Best schedule from Pareto (based on full fitness)
    # ================================
    best_score_from_pareto = float("inf")
    best_schedule_final = None
    best_index = None  # track which Pareto point
    for idx, sched in enumerate(filtered_schedules):
        score = fitness(sched, demand, max_hours)
        if score < best_score_from_pareto:
            best_score_from_pareto = score
            best_schedule_final = sched.copy()
            best_index = idx

    run_time = time.time() - start_time
    return best_schedule_final, best_score_from_pareto, fitness_history, pareto_filtered, run_time, best_off_schedules_global, best_index

# ================================
# STREAMLIT CONTROLS
# ================================
st.sidebar.header("⚙️ ACO Parameters")
n_ants = st.sidebar.slider("Ants", 5, 50, 20)
n_iter = st.sidebar.slider("Iterations", 10, 200, 50)
alpha = st.sidebar.slider("Alpha", 0.1, 5.0, 1.0)
evaporation = st.sidebar.slider("Evaporation", 0.01, 0.9, 0.3)
Q = st.sidebar.slider("Q", 1, 100, 50)
max_hours = st.sidebar.slider("Max Hours / Week", 20, 60, 40)
early_stop = st.sidebar.slider("Early Stop Iterations", 1, 50, 10)

st.sidebar.header("👥 Employees per Department")
n_employees_per_dept = [
    st.sidebar.number_input(f"Dept {i+1} Employees", 1, 50, 20)
    for i in range(n_departments)
]

# ================================
# RUN ACO
# ================================
if st.sidebar.button("🚀 Run ACO"):
    best_schedule, best_score, fitness_history, pareto_data, run_time, best_off_schedules, best_idx = \
        ACO_scheduler(DEMAND, n_employees_per_dept, n_ants, n_iter,
                      alpha, evaporation, Q, max_hours, early_stop)

    st.session_state.best_schedule = best_schedule
    st.session_state.best_off_schedules = best_off_schedules

    st.success(f"Best Fitness Score (from Pareto): {best_score:.2f}")
    st.info(f"Computation Time: {run_time:.2f} seconds")

    # ================================
    # Fitness Convergence
    # ================================
    st.subheader("📈 Fitness Convergence")
    fig, ax = plt.subplots()
    ax.plot(fitness_history, marker='o')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness per Iteration")
    st.pyplot(fig)

    # ================================
    # Pareto Front + Mark Selected Point
    # ================================
    st.subheader("🎯 Pareto Front (Shortage vs Workload)")
    p = np.array(pareto_data)
    fig, ax = plt.subplots()
    ax.scatter(p[:, 0], p[:, 1], alpha=0.6, label="Pareto points")
    if best_idx is not None:
        selected = p[best_idx]
        ax.scatter(selected[0], selected[1], color='red', s=100, label="Chosen Best Schedule")
    ax.set_xlabel("Total Shortage")
    ax.set_ylabel("Workload Penalty")
    ax.legend()
    st.pyplot(fig)
