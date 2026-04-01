from ortools.sat.python import cp_model


def solve_jssp_ortools(times, machines, time_limit_seconds=10.0, scale_factor=1000):
    """
    Args:
        times: [n_j, n_m] 浮点数或整数
        machines: [n_j, n_m] 机器索引 (必须是 0-based)
        scale_factor: 如果 times 是小数，需要乘以此系数转换为整数 (OR-Tools只吃整数)
    """
    n_j, n_m = times.shape
    model = cp_model.CpModel()

    # --- 1. 数据预处理 (Float -> Int) ---
    # 将时间放大并取整，以适应求解器
    times_int = (times * scale_factor).astype(int)
    horizon = int(times_int.sum())

    job_ops = {}
    machine_to_intervals = {m: [] for m in range(n_m)}

    for j in range(n_j):
        job_ops[j] = []
        for s in range(n_m):
            m_id = int(machines[j, s])
            duration = int(times_int[j, s])  # 使用整数时间

            suffix = f'_{j}_{s}'
            start_var = model.new_int_var(0, horizon, 'start' + suffix)
            end_var = model.new_int_var(0, horizon, 'end' + suffix)
            interval_var = model.new_interval_var(start_var, duration, end_var, 'interval' + suffix)

            job_ops[j].append({'end': end_var, 'start': start_var, 'interval': interval_var})
            machine_to_intervals[m_id].append(interval_var)

    # --- 2. 约束 ---
    # 工序顺序约束
    for j in range(n_j):
        for s in range(n_m - 1):
            model.add(job_ops[j][s + 1]['start'] >= job_ops[j][s]['end'])

    # 机器不重叠约束
    for m in range(n_m):
        model.add_no_overlap(machine_to_intervals[m])

    # --- 3. 目标 ---
    makespan_var = model.new_int_var(0, horizon, 'makespan')
    end_times = [job_ops[j][n_m - 1]['end'] for j in range(n_j)]
    model.add_max_equality(makespan_var, end_times)
    model.minimize(makespan_var)

    # --- 4. 求解 ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    status_code = solver.Solve(model)

    status_map = {cp_model.OPTIMAL: "OPTIMAL", cp_model.FEASIBLE: "FEASIBLE"}
    status = status_map.get(status_code, "NOT_FOUND")

    if status in ["OPTIMAL", "FEASIBLE"]:
        obj_int = solver.ObjectiveValue()
        # 将结果除以缩放因子，还原为浮点数
        return obj_int / scale_factor, status
    else:
        return float('inf'), status