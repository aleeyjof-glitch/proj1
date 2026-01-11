# ================================
# ACO.py
# Employee Shift Scheduling (9-5 & 2-10)
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import os

# ================================
# CONFIG
# ================================
st.title("🐜 ACO Employee Shift Scheduling (Shift 09-17 & 14-22)")

n_departments = 6
n_days = 7
n_periods = 28
SHIFT_LENGTH = 16  # 8 hours = 16 periods
ALLOWED_SHIFT_STARTS = [0, 16]  # 0 = 09-17, 16 = 14-22

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

    if df_subset.shape != (n_days, n_periods):
        st.sidebar.error(f"⚠️ Dept{dept+1} wrong shape {df_subset.shape}")
    else:
        DEMAND[dept] = df_subset.values
        st.sidebar.success(f"✅ Dept{dept+1} loaded")

# ================================
# HELPER FUNCTIONS
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
            total_hours = np.sum(schedule[:, :, :, e])
            if total_hours > max_hours:
                penalty += (total_hours - max_hours) * 200

            # Pastikan 1 hari cuti
            days_worked = np.sum(np.sum(schedule[:, :, :, e], axis=2) > 0)
            if days_worked > (n_days - 1):
                penalty += 2000
            elif days_worked < (n_days - 1):
                penalty += 500

        # Pastikan shift 8h penuh dan consecutive
        for d in range(days):
            for e in range(employees):
                daily = schedule[:, d, :, e].sum(axis=0)
                worked = np.sum(daily)
                if worked > 0 and worked != SHIFT_LENGTH:
                    penalty += 3000
                if worked == SHIFT_LENGTH and longest_consecutive_ones(daily) < SHIFT_LENGTH:
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
                for d in range(n_days):
                    available_emps = list(range(n_employees))
                    random.shuffle(available_emps)
                    off_emp = available_emps.pop()  # 1 pekerja cuti hari ni

                    for e in available_emps:
                        if random.random() < 0.7:
                            start = random.choice(ALLOWED_SHIFT_STARTS)
                            schedule[dept, d, start:start+SHIFT_LENGTH, e] = 1

            score = fitness(schedule, demand, max_hours)
            solutions.append(schedule)
            scores.append(score)

            if score < best_score:
                best_score = score
                best_schedule = schedule.copy()

        # Update pheromone
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
# RUN ACO
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
# DISPLAY SCHEDULE & SHORTAGE (dengan Employee Off)
# ================================
if "best_schedule" in st.session_state:
    best_schedule = st.session_state.best_schedule
    employee_ids = [f"E{i+1}" for i in range(n_employees)]

    st.header("📋 Consolidated Staff Schedule per Department")
    st.subheader(f"🏢 Overall Fitness Score: {st.session_state.best_score:.2f}")

    shift_mapping = {
        "09:00-17:00": range(0, 16),
        "14:00-22:00": range(16, 32)
    }

    summary_rows = []

    for dept in range(n_departments):
        st.subheader(f"🏢 Department {dept+1}")
        rows = []
        total_shortage = 0

        # Track cuti setiap hari
        daily_off = []

        for d in range(n_days):
            # Dapatkan employee cuti hari ni
            available_emps = list(range(n_employees))
            random.shuffle(available_emps)
            off_emp_index = available_emps.pop()
            off_emp_name = employee_ids[off_emp_index]
            daily_off.append(off_emp_name)

            for shift_label, period_range in shift_mapping.items():
                assigned_emps = set()
                shortage_periods = {}

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

                shift_shortage_total = sum(shortage_periods.values())
                total_shortage += shift_shortage_total

                rows.append([
                    f"Day {d+1}",
                    shift_label,
                    ", ".join(sorted(assigned_emps)) if assigned_emps else "-",
                    ", ".join([f"{k}({v})" for k, v in shortage_periods.items()]) if shortage_periods else "-",
                    off_emp_name
                ])

        df_dept = pd.DataFrame(
            rows,
            columns=["Day", "Shift", "Employees Assigned", "Shortage (People per Period)", "Employee Off"]
        )

        def highlight_shortage(val):
            return "background-color: red; color: white" if val != "-" else ""

        st.dataframe(df_dept.style.applymap(highlight_shortage, subset=["Shortage (People per Period)"]), use_container_width=True)
        st.markdown(f"**Total Shortage for Department {dept+1}: {total_shortage} people**")

        summary_rows.append([f"Department {dept+1}", total_shortage])

    # Summary table
    st.header("📊 Summary of Total Shortage")
    df_summary = pd.DataFrame(summary_rows, columns=["Department", "Total Shortage (People)"])
    st.dataframe(df_summary, use_container_width=True)
True)
