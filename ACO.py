# ================================
# ACO.py
# Employee Shift Scheduling
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
ALLOWED_SHIFT_STARTS = [0, 14]

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
        dominated = False
        for q in points:
            if (q[0] <= p[0] and q[1] <= p[1]) and q != p:
                dominated = True
                break
        if not dominated:
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
                    penalty += (required - assigned) * 300

        for e in range(employees):
            total_hours = np.sum(schedule[:, :, :, e])
            if total_hours > max_hours:
                penalty += (total_hours - max_hours) * 200

            days_worked = np.sum(np.sum(schedule[:, :, :, e], axis=2) > 0)
            if days_worked != (n_days - 1):
                penalty += 500

        for d in range(days):
            for e in range(employees):
                daily = schedule[dept, d, :, e]
                worked = np.sum(daily)
                if worked > 0 and worked != SHIFT_LENGTH:
                    penalty += 1500
                if worked == SHIFT_LENGTH and longest_consecutive_ones(daily) < SHIFT_LENGTH:
                    penalty += 3000

    return penalty

# ================================
# MULTI OBJECTIVE
# ================================
def compute_objectives(schedule, demand, max_hours):
    total_shortage = 0
    workload_penalty = 0

    for dept in range(demand.shape[0]):
        for d in range(n_days):
            for t in range(n_periods):
                assigned = np.sum(schedule[dept, d, t, :])
                shortage = max(demand[dept, d, t] - assigned, 0)
                total_shortage += shortage

        for e in range(schedule.shape[3]):
            total_hours = np.sum(schedule[:, :, :, e])
            if total_hours > max_hours:
                workload_penalty += (total_hours - max_hours)

    return total_shortage, workload_penalty

# ================================
# OFF DAY
# ================================
def generate_min_one_off_schedule(n_employees, n_days):
    off = np.zeros((n_employees, n_days), dtype=int)
    for e in range(n_employees):
        off[e, random.randint(0, n_days-1)] = 1
    return off

# ================================
# ACO ALGORITHM (FIXED)
# ================================
def ACO_scheduler(demand, n_employees_per_dept, n_ants, n_iter,
                  alpha, evaporation, Q, max_hours, early_stop):

    pheromone = np.ones((n_departments, n_days, 2, max(n_employees_per_dept)))

    best_schedule = None
    best_score = float("inf")
    fitness_history = []
    pareto_data = []
    no_improve = 0

    start = time.time()

    for it in range(n_iter):
        for _ in range(n_ants):
            schedule = np.zeros((n_departments, n_days, n_periods, max(n_employees_per_dept)))
            off_all = []

            for dept in range(n_departments):
                n_emp = n_employees_per_dept[dept]
                off = generate_min_one_off_schedule(n_emp, n_days)
                off_all.append(off)

                for d in range(n_days):
                    for e in range(n_emp):
                        if off[e, d] == 1:
                            continue

                        tau_m = pheromone[dept, d, 0, e] ** alpha
                        tau_e = pheromone[dept, d, 1, e] ** alpha
                        p_m = tau_m / (tau_m + tau_e + 1e-6)

                        if random.random() < p_m:
                            schedule[dept, d, 0:SHIFT_LENGTH, e] = 1
                        else:
                            schedule[dept, d, 14:14+SHIFT_LENGTH, e] = 1

            score = fitness(schedule, demand, max_hours)
            shortage, workload = compute_objectives(schedule, demand, max_hours)
            pareto_data.append((shortage, workload))

            if score < best_score:
                best_score = score
                best_schedule = schedule.copy()
                best_off = off_all.copy()
                no_improve = 0
            else:
                no_improve += 1

            pheromone *= (1 - evaporation)
            pheromone += Q / (1 + score)

        fitness_history.append(best_score)
        if no_improve >= early_stop:
            break

    runtime = time.time() - start
    pareto_front = pareto_filter(pareto_data)

    return best_schedule, best_score, fitness_history, pareto_front, runtime, best_off

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
early_stop = st.sidebar.slider("Early Stop", 1, 50, 10)

st.sidebar.header("👥 Employees per Department")
n_employees_per_dept = [
    st.sidebar.number_input(f"Dept {i+1}", 1, 50, 20) for i in range(n_departments)
]

# ================================
# RUN ACO
# ================================
if st.sidebar.button("🚀 Run ACO"):
    best_schedule, best_score, fitness_history, pareto_data, run_time, best_off_schedules = \
        ACO_scheduler(DEMAND, n_employees_per_dept, n_ants, n_iter,
                      alpha, evaporation, Q, max_hours, early_stop)

    st.success(f"Best Fitness Score: {best_score:.2f}")
    st.info(f"Computation Time: {run_time:.2f}s")

    # Convergence
    st.subheader("📈 Fitness Convergence")
    fig, ax = plt.subplots()
    ax.plot(fitness_history, marker='o')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness Score")
    st.pyplot(fig)

    # Pareto
    st.subheader("🎯 Pareto Front")
    p = np.array(pareto_data)
    fig, ax = plt.subplots()
    ax.scatter(p[:, 0], p[:, 1], alpha=0.6)
    ax.set_xlabel("Total Shortage")
    ax.set_ylabel("Workload Penalty")
    st.pyplot(fig)

# ================================
# ⚠️ DISPLAY SCHEDULE & SUMMARY
# ================================
# 🔒 Bahagian ini TIDAK DIUBAH
# (kekalkan code asal awak)
