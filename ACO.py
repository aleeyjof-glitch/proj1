# ================================
# ACO Streamlit Dashboard (Performance + Multi-objective)
# ================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# ================================
# CONFIG
# ================================
st.title("🐜 ACO Employee Shift Scheduling Dashboard")

n_departments = 6
n_days = 7
n_periods = 28
SHIFT_LENGTH = 14  # 09-17 = 0-13, 14-22 = 14-27

# ================================
# DEMAND (Simulated Example)
# ================================
DEMAND = np.random.randint(1, 4, size=(n_departments, n_days, n_periods))

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

# Multi-objective fitness
def fitness(schedule, demand, max_hours, w_shortage=1.0, w_workload=1.0, w_off=1.0):
    n_departments, days, periods, employees = schedule.shape
    penalty_shortage = 0
    penalty_workload = 0
    penalty_off = 0

    for dept in range(n_departments):
        for d in range(days):
            for t in range(periods):
                assigned = np.sum(schedule[dept, d, t, :])
                required = demand[dept, d, t]
                if assigned < required:
                    penalty_shortage += (required - assigned)

        # Workload per employee
        total_hours_per_employee = np.sum(schedule[dept], axis=(0,1))
        penalty_workload += np.var(total_hours_per_employee)

        # 1 day off per employee
        days_worked = np.sum(np.sum(schedule[dept], axis=1) > 0, axis=1)
        penalty_off += np.sum(np.abs(days_worked - (n_days-1)))

    total_penalty = w_shortage*penalty_shortage + w_workload*penalty_workload + w_off*penalty_off
    return total_penalty, penalty_shortage, penalty_workload, penalty_off

# ACO Scheduler
def ACO_scheduler(demand, n_employees, n_ants, n_iter, max_hours,
                  w_shortage=1.0, w_workload=1.0, w_off=1.0, early_stop=10):

    n_departments, days, periods = demand.shape
    best_schedule = None
    best_score = float("inf")
    no_improve_count = 0
    fitness_history = []

    for iter_num in range(n_iter):
        for ant in range(n_ants):
            schedule = np.zeros((n_departments, days, periods, n_employees))

            for dept in range(n_departments):
                # Generate 1 day off per employee balanced
                employee_off = np.zeros((n_employees, n_days), dtype=int)
                for e in range(n_employees):
                    off_day = e % n_days
                    employee_off[e, off_day] = 1
                for d in range(n_days):
                    np.random.shuffle(employee_off[:, d])

                # Assign shift 09-17 / 14-22
                for d in range(n_days):
                    available = [e for e in range(n_employees) if employee_off[e,d]==0]
                    random.shuffle(available)
                    half = len(available)//2
                    shift1 = available[:half]
                    shift2 = available[half:]
                    for e in shift1:
                        schedule[dept,d,0:SHIFT_LENGTH,e]=1
                    for e in shift2:
                        schedule[dept,d,14:14+SHIFT_LENGTH,e]=1

            score, _, _, _ = fitness(schedule, demand, max_hours, w_shortage, w_workload, w_off)
            fitness_history.append(score)

            if score < best_score:
                best_score = score
                best_schedule = schedule.copy()
                no_improve_count=0
            else:
                no_improve_count +=1

        if no_improve_count>=early_stop:
            break

    return best_schedule, best_score, fitness_history

# ================================
# STREAMLIT SIDEBAR
# ================================
st.sidebar.header("ACO Parameters")
n_employees = st.sidebar.slider("Employees per Department",5,50,20)
n_ants = st.sidebar.slider("Number of Ants",5,50,10)
n_iter = st.sidebar.slider("Iterations",10,200,50)
max_hours = st.sidebar.slider("Max Hours / Week",20,60,40)
early_stop = st.sidebar.slider("Early Stop Iterations",1,50,10)

st.sidebar.header("Multi-objective Weights")
w_shortage = st.sidebar.slider("Weight Shortage",0.0,5.0,1.0)
w_workload = st.sidebar.slider("Weight Workload Balance",0.0,5.0,1.0)
w_off = st.sidebar.slider("Weight 1 Day Off",0.0,5.0,1.0)

# ================================
# RUN BUTTON
# ================================
if st.sidebar.button("🚀 Run ACO"):
    with st.spinner("Optimizing schedule..."):
        start_time = time.time()
        best_schedule, best_score, fitness_history = ACO_scheduler(
            DEMAND, n_employees, n_ants, n_iter, max_hours,
            w_shortage, w_workload, w_off, early_stop
        )
        end_time = time.time()
        st.success(f"Best Fitness: {best_score:.2f} | Time: {end_time-start_time:.2f}s")
        st.session_state.best_schedule = best_schedule
        st.session_state.fitness_history = fitness_history

# ================================
# PERFORMANCE ANALYSIS: Convergence
# ================================
if "fitness_history" in st.session_state:
    st.subheader("📈 Convergence Curve")
    plt.figure(figsize=(10,4))
    plt.plot(st.session_state.fitness_history, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Score")
    plt.title("Fitness vs Iteration")
    st.pyplot(plt)

# ================================
# DISPLAY SCHEDULE & SHORTAGE DETAIL
# ================================
if "best_schedule" in st.session_state:
    best_schedule = st.session_state.best_schedule
    st.subheader("📋 Department Schedules & Shortage")
    shift_mapping = {"09:00-17:00": range(0,SHIFT_LENGTH),"14:00-22:00":range(14,14+SHIFT_LENGTH)}

    for dept in range(n_departments):
        st.markdown(f"### 🏢 Department {dept+1}")
        n_employees_curr = n_employees
        employee_ids = [f"E{i+1}" for i in range(n_employees_curr)]

        # Employee off schedule
        employee_off = np.zeros((n_employees_curr,n_days),dtype=int)
        for e in range(n_employees_curr):
            off_day = e % n_days
            employee_off[e,off_day]=1
        for d in range(n_days):
            np.random.shuffle(employee_off[:,d])

        rows=[]
        shortage_detail = {f"Day {d+1}":{} for d in range(n_days)}
        total_shortage=0

        for d in range(n_days):
            for shift_label, period_range in shift_mapping.items():
                assigned_emps=set()
                shortage_periods={}
                for t in period_range:
                    if t>=n_periods: continue
                    assigned=[employee_ids[e] for e in range(n_employees_curr) if best_schedule[dept,d,t,e]==1]
                    assigned_emps.update(assigned)
                    shortage=max(0,DEMAND[dept,d,t]-len(assigned))
                    if shortage>0: shortage_periods[f"P{t+1}"]=shortage
                    shortage_detail[f"Day {d+1}"][f"P{t+1}"]=shortage

                off_today=[employee_ids[e] for e in range(n_employees_curr) if employee_off[e,d]==1]
                rows.append([f"Day {d+1}",shift_label,", ".join(sorted(assigned_emps)) if assigned_emps else "-", 
                             ", ".join([f"{k}({v})" for k,v in shortage_periods.items()]) if shortage_periods else "-", 
                             ", ".join(off_today) if off_today else "-"])

        df=pd.DataFrame(rows,columns=["Day","Shift","Employees Assigned","Shortage(People per Period)","Employee Off"])
        st.dataframe(df,use_container_width=True)

        # Detailed shortage
        st.markdown(f"**Detailed Shortage for Department {dept+1}:**")
        for day, periods in shortage_detail.items():
            periods_str = ", ".join([f"{p}({v})" for p,v in periods.items() if v>0])
            st.markdown(f"{day}: {periods_str if periods_str else '-'}")
        st.markdown(f"**Total Shortage for Department {dept+1}: {sum([sum(v.values()) for v in shortage_detail.values()])} people**")
