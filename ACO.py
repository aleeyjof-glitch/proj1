# ================================
# ACO.py
# Employee Shift Scheduling (09-17 & 14-22)
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
REST_PROB = 0.25   # 🔥 FIX UTAMA: benarkan employee rehat

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
# ACO SCHEDULER (FIXED)
# ================================
def ACO_scheduler(demand, n_employees_per_dept, n_ants, n_iter,
                  alpha, evaporation, Q, max_hours, early_stop):

    pheromone = np.ones((n_departments, n_days, 2, max(n_employees_per_dept)))
    best_schedule = None
    best_score = float("inf")
    fitness_history = []
    pareto_raw = []
    no_improve = 0
    start_time = time.time()

    for it in range(n_iter):
        for _ in range(n_ants):
            schedule = np.zeros((n_departments, n_days, n_periods, max(n_employees_per_dept)))
            off_schedules = []

            for dept in range(n_departments):
                n_emp = n_employees_per_dept[dept]
                off = generate_min_one_off_schedule(n_emp, n_days)
                off_schedules.append(off)

                for d in range(n_days):
                    for e in range(n_emp):
                        # 🔥 FIX UTAMA
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

            if score < best_score:
                best_score = score
                best_schedule = schedule.copy()
                best_off_schedules = off_schedules.copy()
                no_improve = 0
            else:
                no_improve += 1

            pheromone *= (1 - evaporation)
            pheromone += Q / (1 + score)

        fitness_history.append(best_score)
        if no_improve >= early_stop:
            break

    run_time = time.time() - start_time
    return best_schedule, best_score, fitness_history, pareto_filter(pareto_raw), run_time, best_off_schedules

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
    best_schedule, best_score, fitness_history, pareto_data, run_time, best_off_schedules = \
        ACO_scheduler(DEMAND, n_employees_per_dept, n_ants, n_iter,
                      alpha, evaporation, Q, max_hours, early_stop)

    st.session_state.best_schedule = best_schedule
    st.session_state.best_off_schedules = best_off_schedules

    st.success(f"Best Fitness Score: {best_score:.2f}")
    st.info(f"Computation Time: {run_time:.2f} seconds")

    # 📈 Convergence
    st.subheader("📈 Fitness Convergence")
    fig, ax = plt.subplots()
    ax.plot(fitness_history, marker='o')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Fitness Score")
    st.pyplot(fig)

    # 🎯 Pareto
    st.subheader("🎯 Pareto Front")
    p = np.array(pareto_data)
    fig, ax = plt.subplots()
    ax.scatter(p[:, 0], p[:, 1], alpha=0.6)
    ax.set_xlabel("Total Shortage")
    ax.set_ylabel("Workload Penalty")
    st.pyplot(fig)

# ================================
# DISPLAY SCHEDULE & SUMMARY (ASAL)
# ================================
if "best_schedule" in st.session_state:
    best_schedule = st.session_state.best_schedule
    best_off_schedules = st.session_state.best_off_schedules

    st.header("📋 Consolidated Staff Schedule per Department")

    shift_mapping = {
        "09:00-17:00": range(0, SHIFT_LENGTH),
        "14:00-22:00": range(14, 14+SHIFT_LENGTH)
    }

    summary_rows = []

    for dept in range(n_departments):
        n_employees = n_employees_per_dept[dept]
        employee_ids = [f"E{i+1}" for i in range(n_employees)]
        employee_off_schedule = best_off_schedules[dept]

        st.subheader(f"🏢 Department {dept+1}")
        rows = []
        total_shortage = 0
        heatmap_data = np.zeros((n_days, len(shift_mapping)))

        for d in range(n_days):
            for idx, (shift_label, period_range) in enumerate(shift_mapping.items()):
                assigned_emps = set()
                shortage_periods = {}
                shortage_total_shift = 0

                for t in period_range:
                    if t >= n_periods:
                        continue
                    assigned = [
                        employee_ids[e]
                        for e in range(n_employees)
                        if best_schedule[dept, d, t, e] == 1
                    ]
                    assigned_emps.update(assigned)
                    shortage = DEMAND[dept, d, t] - len(assigned)
                    if shortage > 0:
                        shortage_periods[f"P{t+1}"] = shortage
                        shortage_total_shift += shortage

                off_today = [
                    employee_ids[e]
                    for e in range(n_employees)
                    if employee_off_schedule[e, d] == 1
                ]

                total_shortage += shortage_total_shift
                heatmap_data[d, idx] = shortage_total_shift

                rows.append([
                    f"Day {d+1}",
                    shift_label,
                    ", ".join(sorted(assigned_emps)) if assigned_emps else "-",
                    ", ".join(off_today) if off_today else "-",
                    ", ".join([f"{k}({v})" for k, v in shortage_periods.items()]) if shortage_periods else "-"
                ])

        df_dept = pd.DataFrame(
            rows,
            columns=["Day", "Shift", "Employees Assigned", "Employee Off", "Shortage (People per Period)"]
        )

        def highlight_shortage(val):
            return "background-color: red; color: white" if val != "-" else ""

        st.dataframe(
            df_dept.style.applymap(highlight_shortage, subset=["Shortage (People per Period)"]),
            use_container_width=True
        )

        st.markdown(f"**Total Shortage for Department {dept+1}: {total_shortage} people**")
        summary_rows.append([f"Department {dept+1}", total_shortage])

        st.subheader(f"🌡️ Shortage Heatmap - Department {dept+1}")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(heatmap_data, cmap="Reds", aspect="auto")
        ax.set_xticks(range(len(shift_mapping)))
        ax.set_xticklabels(list(shift_mapping.keys()))
        ax.set_yticks(range(n_days))
        ax.set_yticklabels([f"Day {i+1}" for i in range(n_days)])
        for i in range(n_days):
            for j in range(len(shift_mapping)):
                ax.text(j, i, int(heatmap_data[i, j]), ha="center", va="center")
        st.pyplot(fig)

    st.header("📊 Summary of Total Shortage")
    df_summary = pd.DataFrame(summary_rows, columns=["Department", "Total Shortage (People)"])
    st.dataframe(df_summary, use_container_width=True)
