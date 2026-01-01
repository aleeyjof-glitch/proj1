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

# ================================
# CONFIG
# ================================
DATA_FILE = "Store Size 6 - SS6-CV10-01.xlsx"  # Excel file

st.title("🐜 ACO Employee Shift Scheduling (Detailed)")

# ================================
# LOAD DATASET
# ================================
df = pd.read_excel(DATA_FILE, sheet_name=SHEET_NAME=None)
st.success(f"Dataset loaded from {DATA_FILE}")

# ================================
# CONVERT TO 3D NUMPY ARRAY: dept x day x period
# ================================
n_departments = 6
n_days = 7
n_periods = 28
DEMAND = np.zeros((n_departments, n_days, n_periods), dtype=int)

for dept in range(n_departments):
    dept_data = df.iloc[dept*n_days : (dept+1)*n_days, 0:n_periods].values
    DEMAND[dept, :, :] = dept_data.astype(int)

# ================================
# FITNESS FUNCTION
# ================================
def fitness(schedule, demand, max_hours):
    penalty = 0
    n_departments, days, periods, employees = schedule.shape

    # Hard Constraint: meet demand
    for dept in range(n_departments):
        for d in range(days):
            for t in range(periods):
                assigned = np.sum(schedule[dept, d, t, :])
                required = demand[dept, d, t]
                if assigned < required:
                    penalty += (required - assigned) * 1000

    # Hard Constraint: max hours per employee
    for dept in range(n_departments):
        for e in range(employees):
            total_hours = np.sum(schedule[dept, :, :, e])
            if total_hours > max_hours:
                penalty += (total_hours - max_hours) * 200

    # Soft Constraint: fair workload
    for dept in range(n_departments):
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

        # Evaporation
        pheromone *= (1 - evaporation)

        # Pheromone update
        for sol, score in zip(all_solutions, all_scores):
            pheromone += (Q / (1 + score)) * sol

    return best_schedule, best_score

# ================================
# STREAMLIT SIDEBAR PARAMETERS
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
if st.button("Run Scheduling ACO"):
    best_schedule, best_score = ACO_scheduler(
        DEMAND,
        n_employees,
        n_ants,
        n_iter,
        alpha,
        beta,
        evaporation,
        Q,
        max_hours
    )
    st.success(f"Best Fitness Score: {best_score:.2f}")

    # ================================
    # TABLE PER DEPARTMENT & DAY WITH EMPLOYEE IDs
    # ================================
    st.subheader("📋 Staffing Tables per Department & Day")
    employee_ids = [f"E{i+1}" for i in range(n_employees)]
    shortage_summary = []
    workload_summary = []

    for dept in range(DEMAND.shape[0]):
        st.markdown(f"## Department {dept+1}")
        staff_matrix = best_schedule[dept, :, :, :]  # dept x day x period x employee
        total_shortage = 0

        for d in range(DEMAND.shape[1]):
            rows = []
            assigned_row = np.sum(staff_matrix[d, :, :], axis=1)  # sum employees per period
            required_row = DEMAND[dept, d, :].astype(int)
            shortage_row = np.maximum(0, required_row - assigned_row).astype(int)
            total_shortage += np.sum(shortage_row)

            # Employee assignment per period
            emp_rows = []
            for t in range(DEMAND.shape[2]):
                emp_assigned = [employee_ids[e] for e in range(n_employees) if staff_matrix[dept, d, t, e]==1]
                emp_rows.append(", ".join(emp_assigned) if emp_assigned else "-")

            df_day = pd.DataFrame(
                [emp_rows, assigned_row, required_row, shortage_row],
                index=["Employees", "Assigned", "Required", "Shortage"],
                columns=[f"P{i+1}" for i in range(DEMAND.shape[2])]
            )
            st.markdown(f"### Day {d+1}")
            st.dataframe(df_day)

            # Add to shortage summary
            for t, s in enumerate(shortage_row):
                if s > 0:
                    shortage_summary.append([dept+1, d+1, t+1, s])

        # Workload summary per employee
        emp_workload = [np.sum(staff_matrix[dept, :, :, e]) for e in range(n_employees)]
        for e, w in enumerate(emp_workload):
            workload_summary.append([dept+1, employee_ids[e], w])

    # ================================
    # SHORTAGE SUMMARY TABLE
    # ================================
    st.subheader("⚠️ Shortage Summary")
    df_shortage = pd.DataFrame(shortage_summary, columns=["Department","Day","Period","Shortage"])
    st.dataframe(df_shortage)

    # ================================
    # WORKLOAD SUMMARY TABLE
    # ================================
    st.subheader("📊 Workload Summary")
    df_workload = pd.DataFrame(workload_summary, columns=["Department","Employee","Total Assigned Periods"])
    st.dataframe(df_workload)

    # ================================
    # HEATMAP PER DEPARTMENT
    # ================================
    st.subheader("📈 Heatmap: Assigned Employees per Department")
    dept_choice = st.selectbox("Select Department for Heatmap", [f"Dept {i+1}" for i in range(DEMAND.shape[0])])
    dept_idx = int(dept_choice.split()[-1]) - 1
    staff_matrix = np.sum(best_schedule[dept_idx, :, :, :], axis=2)

    fig, ax = plt.subplots(figsize=(12,4))
    im = ax.imshow(staff_matrix, aspect='auto', cmap='viridis')
    ax.set_xlabel("Time Period (1–28)")
    ax.set_ylabel("Day (1–7)")
    ax.set_title(f"Department {dept_idx+1} Assigned Employees Heatmap")
    plt.colorbar(im)
    st.pyplot(fig)
