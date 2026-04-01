from uniform_instance_gen import uni_instance_gen
import torch
import numpy as np
from Params import configs


def get_initial_intput(n_j, n_m, data):
    m = data[-1]
    dur = data[0].astype(np.single)
    number_of_tasks = n_j * n_m
    # the task id for first column
    first_col = np.arange(start=0, stop=number_of_tasks, step=1).reshape(n_j, -1)[:, 0]

    conj_nei_up_stream = np.eye(number_of_tasks, k=-1, dtype=np.single)
    # first column does not have upper stream conj_nei
    conj_nei_up_stream[first_col] = 0

    self_as_nei = np.eye(number_of_tasks, dtype=np.single)
    adj = self_as_nei + conj_nei_up_stream

    #  展平机器矩阵，使其索引直接对应全局工序ID
    m_flat = m.flatten()
    unique_machines = np.unique(m_flat)

    # 遍历每台机器，批量更新adj
    for m_id in unique_machines:
        # 找到所有使用机器 m_id 的工序的全局 ID
        tasks_on_machine = np.where(m_flat == m_id)[0]
        if len(tasks_on_machine) < 2:
            continue

        # 构建网格索引
        grid_indices = np.ix_(tasks_on_machine, tasks_on_machine)

        # 提取子矩阵
        sub_adj = adj[grid_indices]
        mask = (sub_adj == 0)

        sub_adj[mask] = 2.0

        adj[grid_indices] = sub_adj

    # initialize features
    LBs = np.cumsum(dur, axis=1, dtype=np.single)
    finished_mark = np.zeros_like(m, dtype=np.single)

    fea = np.concatenate((#LBs.reshape(-1, 1),
                          LBs.reshape(-1, 1) / configs.et_normalize_coef,
                          # self.dur.reshape(-1, 1)/configs.high,
                          # wkr.reshape(-1, 1)/configs.wkr_normalize_coef,
                          finished_mark.reshape(-1, 1)), axis=1)
    return adj, fea


def convert_to_pyg_data(adj, fea, n_j, n_m, machines_np, duration_np):
    """
    将JSSP环境的状态转换为PyG的Data对象

    Args:
        adj: 邻接矩阵 [n_tasks, n_tasks] (包含了工序先后约束)
        fea: 节点特征
        machines_np: 机器分配矩阵 [n_j, n_m], machines_np[j,s]表示第j个作业第s道工序使用的机器 ID
        n_j: 作业数
        n_m: 机器数

    Returns:
        PyG Data 对象
    """
    from torch_geometric.data import Data

    # 处理基础特征和工序约束边，转换为 torch tensor
    adj_tensor = torch.from_numpy(adj).float()
    fea_tensor = torch.from_numpy(fea).float()

    # 从邻接矩阵提取边索引 (Value == 1)
    edge_index_conj = (adj_tensor == 1).nonzero(as_tuple=False).t().contiguous().long()
    num_conj_edges = edge_index_conj.shape[1]

    edge_attr_conj = torch.tensor([1, 0], dtype=torch.float).repeat(num_conj_edges, 1)

    # 提取机器连接边 (Value == 2)
    edge_index_mach = (adj_tensor == 2).nonzero(as_tuple=False).t().contiguous().long()
    num_mach_edges = edge_index_mach.shape[1]
    if num_mach_edges > 0:
        # 属性: [0, 1]
        edge_attr_mach = torch.tensor([0, 1], dtype=torch.float).repeat(num_mach_edges, 1)

        # 合并
        edge_index = torch.cat([edge_index_conj, edge_index_mach], dim=1)
        edge_attr = torch.cat([edge_attr_conj, edge_attr_mach], dim=0)
    else:
        edge_index = edge_index_conj
        edge_attr = edge_attr_conj

    data = Data(x=fea_tensor, edge_index=edge_index, edge_attr=edge_attr)

    # 将原始矩阵存为属性
    # PyG的Batch会把这些tensor沿着dim=0堆叠，
    # batch之后会变成 [Batch_Size, n_j, n_m]
    data.raw_machines = torch.from_numpy(machines_np).long()
    data.raw_durations = torch.from_numpy(duration_np).float()

    return data


def epoch_dataset_gen(n_samples, n_j, n_m):
    """
    为当前Epoch生成数据集
    """
    data_list = []

    for _ in range(n_samples):
        # 生成随机算例
        times, machines = uni_instance_gen(n_j=n_j, n_m=n_m, low=0.01, high=1.0)

        # 生成邻接矩阵和特征值
        adj, fea = get_initial_intput(n_j=n_j, n_m=n_m, data=(times, machines))

        # 转换为 PyG 数据
        data = convert_to_pyg_data(adj, fea, n_j, n_m, machines-1, times)

        data_list.append(data)

    return data_list


