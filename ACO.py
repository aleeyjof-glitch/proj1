# ================================
# ACO.py
# Ant Colony Optimization for Employee Shift Scheduling
# Full Interactive Streamlit
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# ================================
# CONFIG
# ================================
st.title("🐜 ACO Employee Shift Scheduling")

# ================================
# PARAMETERS
# ================================
n_departments = 6
n_days = 7
n_periods = 28

DEMAND = np.zeros((n_departments, n_days, n_periods), dtype=int)

# ================================
# LOAD DEPARTMENT EXCEL FILES AUTOMATICALLY
# ================================
st.sidebar.header("Department Demand (Auto Load from folder)")

folder_path = "./Demand/"  # Folder demand file

for dept in range(n_departments):
    file_path = os.path.join(folder_path, f"Dept{dept+1}.xlsx") 
    if not os.path.exists(file_path):
        st.error(f"File {file_path} not found!")
        continue

    # Baca Excel tanpa header
    df = pd.read_excel(file_path, header=None)
    df = df.dropna(how='all', axis=0)
    df = df.dropna(how='all', axis=1)

    # Ambil n_days x n_periods sahaja
    df_subset = df.iloc[:n_days, :n_periods]
    df_subset = df_subset.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

    # Simpan ke numpy DEMAND
    DEMAND[dept, :, :] = df_subset.values

    # Buat Excel-style DataFrame preview
    df_display = pd.DataFrame(index=range(n_days+1), columns=range(n_periods+1), dtype=object)

    # Baris pertama: kosong di kiri atas, Period 1..28
    df_display.iloc[0,0] = ""
    df_display.iloc[0,1:] = np.arange(1, n_periods+1)

    # Kolum pertama: Day 1..7
    df_display.iloc[1:,0] = np.arange(1, n_days+1)

    # Demand values
    df_display.iloc[1:,1:] = df_subset.values

    st.write(f"Dept {dept+1} DEMAND preview:")
    st.dataframe(df_display)

# ================================
# FITNESS FUNCTION
# ================================
def fitness(schedule, demand, max_hours):
    penalty = 0
    n_departments, days, periods, employees = schedule.shape

    for dept in range(n_departments):
        # Hard constraint: meet demand
        for d in range(days):
            for t in range(periods):
                assigned = np.sum(schedule[dept, d, t, :])
                required = demand[dept, d, t]
                if assigned < required:
                    penalty += (required - assigned) * 1000

        # Hard constraint: max hours per employee
        for e in range(employees):
            total_hours = np.sum(schedule[dept, :, :, e])
            if total_hours > max_hours:
                penalty += (total_hours - max_hours) * 200

        # Soft constraint: fair workload
        workloads = [np.sum(schedule[dept, :, :, e]) for e in range(employees)]
        penalty += np.var(workloads) * 10

    return penalty

# ================================
# ACO ALGORITHM
# ================================
def ACO_scheduler(demand, n_employees, n_ants, n_iter, alpha, beta, evaporation, Q, max_hours):
    n_departments, days, periods = demand.shape
    pheromone = np.ones((n_departments, days, periods, n_employees))

    best_schedule = None
    best_score = float("inf")

    for _ in range(n_iter):
        all_solutions = []
        all_scores = []

        for ant in range(n_ants):
            schedule = np.zeros((n_departments, days, periods, n_employees))

            for dept in range(n_departments):
                for d in range(days):
                    for t in range(periods):
                        for e in range(n_employees):
                            prob = pheromone[dept, d, t, e] ** alpha
                            if random.random() < prob / (1 + prob):
                                schedule[dept, d, t, e] = 1

            score = fitness(schedule, demand, max_hours)
            all_solutions.append(schedule)
            all_scores.append(score)

            if score < best_score:
                best_score = score
                best_schedule = schedule.copy()

        # Pheromone update
        pheromone *= (1 - evaporation)
        for sol, score in zip(all_solutions, all_scores):
            pheromone += (Q / (1 + score)) * sol

    return best_schedule, best_score

# ================================
# STREAMLIT PARAMETERS
# ================================
st.sidebar.header("ACO Parameters")
n_employees = st.sidebar.slider("Number of Employees", 5, 50, 20)
n_ants = st.sidebar.slider("Number of Ants", 5, 50, 20)
n_iter = st.sidebar.slider("Iterations", 10, 200, 50)
alpha = st.sidebar.slider("Alpha (pheromone)", 0.1, 5.0, 1.0)
beta = st.sidebar.slider("Beta (heuristic)", 0.1, 5.0, 2.0)
evaporation = st.sidebar.slider("Evaporation Rate", 0.01, 0.9, 0.3)
Q = st.sidebar.slider("Q (deposit)", 1, 100, 50)
max_hours = st.sidebar.slider("Max Working Hours / Week", 20, 60, 40)

# ================================
# RUN BUTTON
# ================================
if "best_schedule" not in st.session_state:
    if st.sidebar.button("Run Scheduling ACO"):
        st.session_state.best_schedule, st.session_state.best_score = ACO_scheduler(
            DEMAND, n_employees, n_ants, n_iter, alpha, beta, evaporation, Q, max_hours
        )
        st.success(f"Best Fitness Score: {st.session_state.best_score:.2f}")

# ================================
# TABLE, SHORTAGE, WORKLOAD
# ================================
if "best_schedule" in st.session_state:
    best_schedule = st.session_state.best_schedule
    employee_ids = [f"E{i+1}" for i in range(n_employees)]

    st.subheader("📋 Staffing Tables per Department & Day")
    for dept in range(n_departments):
        st.markdown(f"## Department {dept+1}")
        staff_matrix = best_schedule[dept, :, :, :]

        for d in range(n_days):
            assigned_row = np.sum(staff_matrix[d, :, :], axis=1)
            required_row = DEMAND[dept, d, :].astype(int)
            shortage_row = np.maximum(0, required_row - assigned_row).astype(int)

            emp_rows = [
                ", ".join([employee_ids[e] for e in range(n_employees) if staff_matrix[d, t, e]==1]) or "-"
                for t in range(n_periods)
            ]

            # DataFrame with Excel-style first column = Day
            df_day = pd.DataFrame(
                [emp_rows, assigned_row, required_row, shortage_row],
                index=["Employees", "Assigned", "Required", "Shortage"]
            )
            df_day.insert(0, "Day", d+1)  # kolum pertama = Day number
            st.markdown(f"### Day {d+1}")
            st.dataframe(df_day)

        # Shortage summary
        st.subheader("⚠️ Shortage Summary per Department per Day")
        daily_shortage_summary = []
        for d in range(n_days):
            assigned_row = np.sum(staff_matrix[d, :, :], axis=1)
            required_row = DEMAND[dept, d, :].astype(int)
            shortage_row = np.maximum(0, required_row - assigned_row).astype(int)
            total_shortage_day = np.sum(shortage_row)
            daily_shortage_summary.append([d+1, *shortage_row, total_shortage_day])

        columns = ["Day"] + [f"P{i+1}" for i in range(n_periods)] + ["Total"]
        df_shortage_day = pd.DataFrame(daily_shortage_summary, columns=columns)
        st.dataframe(df_shortage_day)

        # Workload summary
        st.subheader("📊 Workload Summary")
        emp_workload = [np.sum(staff_matrix[:, :, e]) for e in range(n_employees)]
        df_workload = pd.DataFrame(
            [[employee_ids[e], emp_workload[e]] for e in range(n_employees)],
            columns=["Employee", "Total Assigned Periods"]
        )
        st.dataframe(df_workload)

# ================================
# HEATMAP
# ================================
if "best_schedule" in st.session_state:
    st.subheader("📈 Heatmap: Assigned Employees per Department")
    dept_choice = st.selectbox(
        "Select Department for Heatmap",
        [f"Dept {i+1}" for i in range(n_departments)]
    )
    dept_idx = int(dept_choice.split()[-1]) - 1
    staff_matrix = st.session_state.best_schedule[dept_idx, :, :, :]
    heatmap_data = np.sum(staff_matrix, axis=2)

    fig, ax = plt.subplots(figsize=(12,4))
    im = ax.imshow(heatmap_data, aspect='auto', cmap='viridis')

    ax.set_xticks(range(n_periods))
    ax.set_xticklabels([f"P{i+1}" for i in range(n_periods)], rotation=45)
    ax.set_yticks(range(n_days))
    ax.set_yticklabels([f"Day {i+1}" for i in range(n_days)])
    ax.set_xlabel("Time Periods")
    ax.set_ylabel("Days")
    ax.set_title(f"Department {dept_idx+1} Assigned Employees Heatmap")
    fig.colorbar(im, ax=ax, label="Number of Employees Assigned")
    st.pyplot(fig)
