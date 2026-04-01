import numpy as np
import torch
from torch_geometric.data import Batch
from JSSP_Env import SJSSP
from uniform_instance_gen import uni_instance_gen
from model_1 import JSSPActor
from utils import convert_state_to_pyg_data

def main():
    n_j = 10  # 作业数
    n_m = 10  # 机器数
    n_tasks = n_j * n_m  # 任务总数 (100)
    batch_size = 2  # 批次大小
    input_dim = 2  # 特征维度 (LBs/normalize_coef, finished_mark)
    hidden_dim = 128  # 隐藏层维度

    np.random.seed(200)
    torch.manual_seed(600)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"运行设备: {device}")
    print(f"测试配置: {n_j}x{n_m} JSSP, Batch Size: {batch_size}")

    env = SJSSP(n_j=n_j, n_m=n_m)

    print(f"\n生成 {batch_size} 个 JSSP 实例...")
    data_list = []
    batch_machines_info = []

    for i in range(batch_size):
        # 生成随机实例
        times, machines = uni_instance_gen(n_j=n_j, n_m=n_m, low=1, high=99)
        instance_data = (times, machines)

        # 保存机器分配信息（用于后续验证）
        batch_machines_info.append(machines.flatten())

        # 重置环境并获取初始状态
        adj, fea, omega, mask = env.reset(instance_data)

        # 转换为 PyG Data 对象
        pyg_data = convert_state_to_pyg_data(adj, fea, n_j, n_m, machines, times)
        data_list.append(pyg_data)

        print(f"  实例 {i+1}: 节点数={pyg_data.x.shape[0]}, 边数={pyg_data.edge_index.shape[1]}")

    batch = Batch.from_data_list(data_list).to(device)
    print(f"\nBatch 创建成功: 总节点数={batch.x.shape[0]} (期望 {batch_size * n_tasks})")

    model = JSSPActor(input_dim=input_dim, hidden_dim=hidden_dim,n_j=n_j, n_m=n_m).to(device)

    print("\n运行前向传播 (Rollout 模式)...")
    model.eval()
    with torch.no_grad():
        priority_list_jobs, log_probs = model(batch, rollout=True)

    priority_list_cpu = priority_list_jobs.cpu().numpy()

    for b in range(batch_size):
        job_sequence = priority_list_cpu[b]

        print(f"\n样本 {b + 1}:")
        print(f"生成的作业优先级序列 (Job IDs): \n{job_sequence}")

        # 验证完整性
        unique, counts = np.unique(job_sequence, return_counts=True)
        print(f"验证 counts (应全为 {n_m}): {counts}")


if __name__ == "__main__":
    main()
