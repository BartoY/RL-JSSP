import torch
import numpy as np
import pandas as pd
import time
import os
from torch_geometric.loader import DataLoader

from model_1 import JSSPActor
from utils import prio_lst_sched_bch
from data_utils import convert_to_pyg_data, get_initial_intput
from ortools_solver import solve_jssp_ortools

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_J = 10
N_M = 10
BATCH_SIZE = 512
seed = 200
OR_TOOLS_LIMIT = 5.0  # OR-Tools 每个算例限时5秒
SAMPLING_SIZE = 100
FILENAME = os.path.join("data_gen", f"generatedData{N_J}_{N_M}_Seed{seed}_bsz{BATCH_SIZE}.npy")


def run_benchmark():
    print(f"Loading data from {FILENAME}...")
    if not os.path.exists(FILENAME):
        print(f"Error: 文件 {FILENAME} 不存在！请检查路径。")
        return

    # 1. 加载 .npy 数据
    # 形状应该是 [Batch, 2, N_J, N_M]
    raw_data = np.load(FILENAME)
    batch_size = raw_data.shape[0]
    print(f"Data shape: {raw_data.shape} (Batch Size: {batch_size})")

    # 2. 准备 RL 模型
    model = JSSPActor(input_dim=2, hidden_dim=128, n_j=N_J, n_m=N_M).to(DEVICE)
    try:
        model_path = os.path.join("models_save", f"{N_J}_{N_M}_best_model_{BATCH_SIZE}.pth")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print("成功加载model")
    except:
        print("未找到model，将使用未训练模型测试！")

    model.eval()

    # 3. 数据预处理 (构建 PyG Batch)
    # 数据范围是 0.01 - 1.0
    pyg_data_list = []

    print("Preparing PyG data...")
    for i in range(batch_size):
        # 提取单个样本
        times = raw_data[i][0]
        machines = raw_data[i][1]

        adj, fea = get_initial_intput(n_j=N_J,n_m=N_M, data=(times, machines))

        # 转换 PyG 数据
        data = convert_to_pyg_data(adj, fea, N_J, N_M, machines - 1, times)
        pyg_data_list.append(data)

    test_loader = DataLoader(pyg_data_list, batch_size=batch_size, shuffle=False)

    # 4. 运行 RL
    print(f">>> Running RL on {batch_size} instances...")
    rl_start = time.time()

    rl_makespans = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(DEVICE)
            bsz = batch.num_graphs

            reshaped_durations = batch.raw_durations.view(bsz, N_J, N_M)
            reshaped_machines = batch.raw_machines.view(bsz, N_J, N_M)
            op_machine_idx = reshaped_machines.view(bsz, -1).long()
            op_proc_time = reshaped_durations.view(bsz, -1).float()

            # Greedy Rollout
            solutions, *_ = model(batch, op_machine_idx, op_proc_time, rollout=True)

            # 计算 Makespan
            _, costs = prio_lst_sched_bch(
                solutions, reshaped_durations, reshaped_machines, N_J, N_M
            )
            rl_makespans.extend(costs.cpu().tolist())

    rl_end = time.time()
    rl_total_time = rl_end - rl_start
    print(f"RL Finished. Total Time: {rl_total_time:.4f}s")

    # 5. 运行 OR-Tools (逐个求解)
    print(f">>> Running OR-Tools (Time Limit={OR_TOOLS_LIMIT}s)...")
    ortools_makespans = []
    ortools_times = []

    for i in range(batch_size):
        times = raw_data[i][0]
        machines = raw_data[i][1]

        t_start = time.time()
        # 必须把机器ID转为0-based
        val, status = solve_jssp_ortools(times, machines - 1, time_limit_seconds=OR_TOOLS_LIMIT)
        t_end = time.time()

        ortools_makespans.append(val)
        ortools_times.append(t_end - t_start)

        print(f"  Instance {i + 1}/{batch_size}: OR-Tools={val:.4f} (Status: {status})")

    # 6. 生成报告
    results = []
    for i in range(batch_size):
        rl_val = rl_makespans[i]
        or_val = ortools_makespans[i]

        # Gap 计算
        gap = ((rl_val - or_val) / or_val) * 100

        results.append({
            "Instance": i,
            "RL_Makespan": rl_val,
            "OR_Makespan": or_val,
            "Gap(%)": gap,
            "OR_Time": ortools_times[i]
        })

    df = pd.DataFrame(results)

    print("\n" + "=" * 50)
    print("             COMPARISON REPORT             ")
    print("=" * 50)
    print(f"Avg RL Makespan:       {np.mean(rl_makespans):.4f}")
    print(f"Avg OR-Tools Makespan: {np.mean(ortools_makespans):.4f}")
    print(f"Average Gap:           {df['Gap(%)'].mean():.2f}%")
    print("-" * 50)
    print(f"RL Avg Time/Instance:  {rl_total_time / batch_size:.6f} s")
    print(f"OR Avg Time/Instance:  {np.mean(ortools_times):.6f} s")
    print("=" * 50)

    df.to_csv("comparison_results.csv", index=False)


if __name__ == "__main__":
    run_benchmark()