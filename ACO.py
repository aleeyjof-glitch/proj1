# ================================
# ACO.py
# Employee Shift Scheduling
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

n_departments = 6
n_days = 7
n_periods = 28
SHIFT_LENGTH = 16  # 8 hours = 16 periods

# Allowed shift start (index)
ALLOWED_SHIFT_STARTS = [0, 12]  # P1 (8am) or P13 (2pm)

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

    # SKIP row 0 (period label) & column 0 (day label)
    df_subset = df.iloc[1:1+n_days, 1:1+n_periods]

    df_subset = (
        df_subset
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if df_subset.shape != (n_days, n_periods):
        st.sidebar.error(f"⚠️ Dept{dept+1} wrong shape {df_subset.shape}")
    else:
        DEMAND[dept] = df_subset.values
        st.sidebar.success(f"✅ Dept{dept+1} loaded")

# ================================
# HELPER FUNCTION
# ================================
def longest_consecutive_ones(arr):
    max_len = 0
    curr = 0
    for v in arr:
        if v == 1:
            curr += 1
            max_len = max(max_len, curr)
        else:
            curr = 0
    return max_len

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
                    penalty += (required - assigned) * 1000

        for e in range(employees):
            total_hours = np.sum(schedule[dept, :, :, e])
            if total_hours > max_hours:
                penalty += (total_hours - max_hours) * 200

        # ---- 8 HOURS CONSECUTIVE CONSTRAINT ----
        for d in range(days):
            for e in range(employees):
                daily = schedule[dept, d, :, e]
                worked = np.sum(daily)

                if worked > 0 and worked != SHIFT_LENGTH:
                    penalty += 3000

                if worked == SHIFT_LENGTH:
                    if longest_consecutive_ones(daily) < SHIFT_LENGTH:
                        penalty += 5000

    return penalty

# ================================
# ACO ALGORITHM
# ================================
def ACO_scheduler(demand, n_employees, n_ants, n_iter, alpha, evaporation, Q, max_hours):
    n_departments, days, periods = demand.shape
    pheromone = np.ones((n_departments, days, periods, n_employees))

    best_schedule = None
    best_score = float("inf")

    for _ in range(n_iter):
        solutions = []
        scores = []

        for _ in range(n_ants):
            schedule = np.zeros((n_departments, days, periods, n_employees))

            for dept in range(n_departments):
                for d in range(days):
                    for e in range(n_employees):
                        # Choose shift start
                        if random.random() < 0.7:
                            start = random.choice(ALLOWED_SHIFT_STARTS)
                            schedule[dept, d, start:start+SHIFT_LENGTH, e] = 1

            score = fitness(schedule, demand, max_hours)
            solutions.append(schedule)
            scores.append(score)

            if score < best_score:
                best_score = score
                best_schedule = schedule.copy()

        pheromone *= (1 - evaporation)
        for sol, sc in zip(solutions, scores):
            pheromone += (Q / (1 + sc)) * sol

    return best_schedule, best_score

# ================================
# STREAMLIT CONTROLS
# ================================
st.sidebar.header("⚙️ ACO Parameters")
n_employees = st.sidebar.slider("Employees", 5, 50, 20)
n_ants = st.sidebar.slider("Ants", 5, 50, 20)
n_iter = st.sidebar.slider("Iterations", 10, 200, 50)
alpha = st.sidebar.slider("Alpha", 0.1, 5.0, 1.0)
evaporation = st.sidebar.slider("Evaporation", 0.01, 0.9, 0.3)
Q = st.sidebar.slider("Q", 1, 100, 50)
max_hours = st.sidebar.slider("Max Hours / Week", 20, 60, 40)

# ================================
# RUN BUTTON
# ================================
if st.sidebar.button("🚀 Run ACO"):
    with st.spinner("Optimizing schedule..."):
        best_schedule, best_score = ACO_scheduler(
            DEMAND, n_employees, n_ants, n_iter,
            alpha, evaporation, Q, max_hours
        )
        st.session_state.best_schedule = best_schedule
        st.session_state.best_score = best_score
        st.success(f"Best Fitness Score: {best_score:.2f}")

# ================================
# DISPLAY FULL STAFFING TABLE
# ================================
if "best_schedule" in st.session_state:
    best_schedule = st.session_state.best_schedule
    employee_ids = [f"E{i+1}" for i in range(n_employees)]

    # Generate time labels
    time_labels = []
    hour = 8
    minute = 0
    for _ in range(n_periods):
        start = f"{hour:02d}:{minute:02d}"
        minute += 30
        if minute == 60:
            hour += 1
            minute = 0
        end = f"{hour:02d}:{minute:02d}"
        time_labels.append(f"{start}-{end}")

    st.header("📋 Staff Schedule per Department")

    for dept in range(n_departments):
        st.subheader(f"🏢 Department {dept+1}")
        staff_matrix = best_schedule[dept]

        for d in range(n_days):
            st.markdown(f"### 📅 Day {d+1}")

            rows = []
            for t in range(n_periods):
                assigned_emps = [
                    employee_ids[e]
                    for e in range(n_employees)
                    if staff_matrix[d, t, e] == 1
                ]

                assigned_count = len(assigned_emps)
                required = DEMAND[dept, d, t]
                shortage = max(0, required - assigned_count)

                rows.append([
                    f"P{t+1}",
                    time_labels[t],
                    ", ".join(assigned_emps) if assigned_emps else "-",
                    assigned_count,
                    required,
                    shortage
                ])

            df_day = pd.DataFrame(
                rows,
                columns=[
                    "Period",
                    "Time",
                    "Employees",
                    "Assigned",
                    "Required",
                    "Shortage"
                ]
            )

            st.dataframe(df_day, use_container_width=True)

