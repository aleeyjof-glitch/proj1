# ================================
# ACO.py
# Employee Shift Scheduling (09-17 & 14-22) with:
# - Multi-objective (shortage & workload)
# - Fitness convergence
# - Heatmap
# - Pareto front
# - Minimum 1 off-day per employee per week (max 1)
# - Shift assignment consistent with off-day
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
SHIFT_LENGTH = 14  # 8 hours = 14 periods
ALLOWED_SHIFT_STARTS = [0, 14]  # 09-17 or 14-22

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
                daily = schedule[dept, d, :, e]
                worked = np.sum(daily)
                if worked > 0 and worked != SHIFT_LENGTH:
                    penalty += 3000
                if worked == SHIFT_LENGTH and longest_consecutive_ones(daily) < SHIFT_LENGTH:
                    penalty += 5000

    return penalty

# ================================
# MULTI-OBJECTIVE FUNCTION
# ================================
def compute_objectives(schedule, demand, max_hours):
    n_departments, days, periods, employees = schedule.shape
    total_shortage = 0
    workload_penalty = 0

    for dept in range(n_departments):
        for d in range(days):
            for t in range(periods):
                assigned_count = np.sum(schedule[dept, d, t, :])
                shortage = max(demand[dept, d, t] - assigned_count, 0)
                total_shortage += shortage

        for e in range(employees):
            total_hours = np.sum(schedule[:, :, :, e])
            if total_hours > max_hours:
                workload_penalty += (total_hours - max_hours)

    return total_shortage, workload_penalty

# ================================
# MINIMUM 1 OFF-DAY PER EMPLOYEE
# ================================
def generate_min_one_off_schedule(n_employees, n_days):
    """
    Setiap pekerja cuti **sekali sahaja** per minggu.
    Tidak boleh cuti lebih dari 1 kali.
    Beberapa pekerja boleh cuti pada hari sama.
    """
    employee_off_schedule = np.zeros((n_employees, n_days), dtype=int)

    for e in range(n_employees):
        off_day = random.randint(0, n_days - 1)
        employee_off_schedule[e, off_day] = 1

    return employee_off_schedule

# ================================
# ACO ALGORITHM
# ================================
def ACO_scheduler(demand, n_employees_per_dept, n_ants, n_iter, alpha, evaporation, Q, max_hours, early_stop=10):
    n_departments, days, periods = demand.shape
    max_employees = max(n_employees_per_dept)
    pheromone = np.ones((n_departments, days, periods, max_employees))

    best_schedule = None
    best_score = float("inf")
    no_improve_count = 0
    fitness_history = []
    pareto_data = []
    off_schedules_all = []

    start_time = time.time()

    for iter_num in range(n_iter):
        solutions = []
        scores = []

        for _ in range(n_ants):
            schedule = np.zeros((n_departments, days, periods, max_employees))
            off_schedules = []

            for dept in range(n_departments):
                n_employees = n_employees_per_dept[dept]

                # Generate min 1 off-day per employee
                employee_off_schedule = generate_min_one_off_schedule(n_employees, n_days)
                off_schedules.append(employee_off_schedule)

                # Assign shifts only if employee is not off
                for d in range(n_days):
                    available_emps = [e for e in range(n_employees) if employee_off_schedule[e, d] == 0]
                    random.shuffle(available_emps)
                    half = len(available_emps) // 2
                    shift1_emps = available_emps[:half]
                    shift2_emps = available_emps[half:]

                    for e in shift1_emps:
                        schedule[dept, d, 0:SHIFT_LENGTH, e] = 1
                    for e in shift2_emps:
                        schedule[dept, d, 14:14+SHIFT_LENGTH, e] = 1

            # Store off-day schedules for later display
            off_schedules_all.append(off_schedules)

            score = fitness(schedule, demand, max_hours)
            solutions.append(schedule)
            scores.append(score)

            # Multi-objective
            total_shortage, workload_penalty = compute_objectives(schedule, demand, max_hours)
            pareto_data.append((total_shortage, workload_penalty))

            if score < best_score:
                best_score = score
                best_schedule = schedule.copy()
                best_off_schedules = off_schedules.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

        iteration_best = min(scores)
        fitness_history.append(iteration_best)

        # Update pheromone
        pheromone *= (1 - evaporation)
        for sol, sc in zip(solutions, scores):
            pheromone += (Q / (1 + sc)) * sol

        if no_improve_count >= early_stop:
            print(f"Early stopping at iteration {iter_num+1}, best score: {best_score}")
            break

    run_time = time.time() - start_time
    return best_schedule, best_score, fitness_history, pareto_data, run_time, best_off_schedules

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
n_employees_per_dept = []
for dept in range(n_departments):
    n_emp = st.sidebar.number_input(
        f"Dept {dept+1} Employees", min_value=1, max_value=50, value=20, step=1
    )
    n_employees_per_dept.append(n_emp)

# ================================
# RUN ACO
# ================================
if st.sidebar.button("🚀 Run ACO"):
    with st.spinner("Optimizing schedule..."):
        best_schedule, best_score, fitness_history, pareto_data, run_time, best_off_schedules = ACO_scheduler(
            DEMAND,
            n_employees_per_dept,
            n_ants,
            n_iter,
            alpha,
            evaporation,
            Q,
            max_hours,
            early_stop
        )
        st.session_state.best_schedule = best_schedule
        st.session_state.best_score = best_score
        st.session_state.fitness_history = fitness_history
        st.session_state.pareto_data = pareto_data
        st.session_state.run_time = run_time
        st.session_state.best_off_schedules = best_off_schedules

        st.success(f"Best Fitness Score: {best_score:.2f}")
        st.info(f"Computation Time: {run_time:.2f} seconds")

        # Convergence Curve
        st.subheader("📈 Fitness Convergence")
        fig, ax = plt.subplots()
        ax.plot(fitness_history, marker='o')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Fitness Score")
        ax.set_title("ACO Convergence Curve")
        st.pyplot(fig)

        # Pareto Front
        st.subheader("🎯 Pareto Front: Shortage vs Workload Penalty")
        pareto_array = np.array(pareto_data)
        fig, ax = plt.subplots()
        ax.scatter(pareto_array[:, 0], pareto_array[:, 1], c='blue', alpha=0.6)
        ax.set_xlabel("Total Shortage")
        ax.set_ylabel("Workload Penalty (Hours over max)")
        ax.set_title("Pareto Front")
        st.pyplot(fig)

# ================================
# DISPLAY SCHEDULE & SHORTAGE
# ================================
if "best_schedule" in st.session_state:
    best_schedule = st.session_state.best_schedule
    best_off_schedules = st.session_state.best_off_schedules[0]  # Use first ant's off-schedule
    st.header("📋 Consolidated Staff Schedule per Department")
    st.subheader(f"🏢 Overall Fitness Score: {st.session_state.best_score:.2f}")

    shift_mapping = {
        "09:00-17:00": range(0, SHIFT_LENGTH),
        "14:00-22:00": range(14, 14+SHIFT_LENGTH)
    }

    summary_rows = []

    for dept in range(n_departments):
        n_employees = n_employees_per_dept[dept]
        employee_ids = [f"E{i+1}" for i in range(n_employees)]

        # Use off-schedule from ACO
        employee_off_schedule = best_off_schedules[dept]

        st.subheader(f"🏢 Department {dept+1}")
        rows = []
        total_shortage = 0

        # Heatmap data
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

                off_today = [employee_ids[e] for e in range(n_employees) if employee_off_schedule[e, d] == 1]

                total_shortage += shortage_total_shift
                heatmap_data[d, idx] = shortage_total_shift

                rows.append([
                    f"Day {d+1}",
                    shift_label,
                    ", ".join(sorted(assigned_emps)) if assigned_emps else "-",
                    ", ".join([f"{k}({v})" for k, v in shortage_periods.items()]) if shortage_periods else "-",
                    ", ".join(off_today) if off_today else "-"
                ])

        # Display schedule table
        df_dept = pd.DataFrame(
            rows,
            columns=["Day", "Shift", "Employees Assigned", "Shortage (People per Period)", "Employee Off"]
        )
        def highlight_shortage(val):
            return "background-color: red; color: white" if val != "-" else ""
        st.dataframe(df_dept.style.applymap(highlight_shortage, subset=["Shortage (People per Period)"]), use_container_width=True)
        st.markdown(f"**Total Shortage for Department {dept+1}: {total_shortage} people**")
        summary_rows.append([f"Department {dept+1}", total_shortage])

        # Heatmap
        st.subheader(f"🌡️ Shortage Heatmap - Department {dept+1}")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.imshow(heatmap_data, cmap="Reds", aspect="auto")
        ax.set_xticks(range(len(shift_mapping)))
        ax.set_xticklabels(list(shift_mapping.keys()))
        ax.set_yticks(range(n_days))
        ax.set_yticklabels([f"Day {i+1}" for i in range(n_days)])
        for i in range(n_days):
            for j in range(len(shift_mapping)):
                ax.text(j, i, int(heatmap_data[i, j]), ha="center", va="center", color="black")
        ax.set_xlabel("Shift")
        ax.set_ylabel("Day")
        st.pyplot(fig)

    # Summary table
    st.header("📊 Summary of Total Shortage")
    df_summary = pd.DataFrame(summary_rows, columns=["Department", "Total Shortage (People)"])
    st.dataframe(df_summary, use_container_width=True)

