import torch
from utils import prio_lst_sched_bch
import numpy as np


def validate_model(model, val_loader, device, n_j, n_m):
    """
    在验证集上评估模型，并返回所有样本的 Makespan 数组
    """
    model.eval()  # 切换到评估模式
    all_costs = []

    with torch.no_grad():  # 禁用梯度计算，节省显存和时间
        for batch in val_loader:
            batch = batch.to(device)

            # 维度处理
            bsz = batch.num_graphs
            reshaped_durations = batch.raw_durations.view(bsz, n_j, n_m)
            reshaped_machines = batch.raw_machines.view(bsz, n_j, n_m)

            op_machine_idx = reshaped_machines.view(bsz, -1).long()
            op_proc_time = reshaped_durations.view(bsz, -1).float()

            solutions, *_ = model(batch, op_machine_idx, op_proc_time, rollout=True)

            # 计算 Makespan
            _, costs = prio_lst_sched_bch(
                solutions, reshaped_durations, reshaped_machines, n_j, n_m
            )

            all_costs.extend(costs.cpu().numpy().tolist())

    return np.array(all_costs)