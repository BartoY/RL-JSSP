import torch
from torch_geometric.data import Data, Dataset
from scipy.stats import ttest_rel


def chk_upd_bl(policy_costs, baseline_costs, alpha=0.05):
    """
    使用配对T检验判断Policy是否显著优于Baseline

    Args:
        policy_costs: 当前模型的验证集Makespan数组
        baseline_costs: Baseline模型的验证集Makespan数组
        alpha: 显著性水平

    Returns:
        bool: 是否应该更新Baseline
    """
    # 1. 简单的均值比较作为初步筛选
    # 如果当前模型均值比 Baseline 还差，直接不更新，省去 T-test
    if policy_costs.mean() >= baseline_costs.mean():
        return False

    # 2. 配对 T 检验 (Paired T-test)
    # H0: 两个模型均值相同
    # H1: 两个模型均值不同
    t_stat, p_value = ttest_rel(baseline_costs, policy_costs)

    # 逻辑：
    # t_stat > 0 表示 baseline_costs - policy_costs > 0，即 baseline 比 policy 大 (policy 更好)
    # p_value / 2 < alpha 表示单尾检验显著
    if t_stat > 0 and (p_value / 2) < alpha:
        return True

    return False


def prio_lst_sched_bch(job_sequence, duration, machine_alloc, n_j, n_m):
    """
    将 Batch 形式的 Tensor 输入转换为具体的调度方案和完工时间。

    Args:
        job_sequence: [B, Total_Ops] (LongTensor) 解码器生成的 Job ID 序列
        duration:     [B, n_j, n_m]  (Float/IntTensor) 每个工序的耗时
        machine_alloc:[B, n_j, n_m]  (LongTensor) 每个工序对应的机器ID
        n_j, n_m:     int, int

    Returns:
        batch_schedules: List[Dict] 长度为 B 的列表。
                         每个元素是一个字典 {machine_id: [task_info, ...]}
        batch_makespans: Tensor [B] 每个样本的最终完工时间
    """

    # 1. 预处理：将 Tensor 转移到 CPU 并转为 Numpy，方便构建字典结构
    # 避免在循环中频繁进行 device 转换
    seq_np = job_sequence.detach().cpu().numpy()
    dur_np = duration.detach().cpu().numpy()
    mach_np = machine_alloc.detach().cpu().numpy()

    batch_size = seq_np.shape[0]
    total_ops_in_seq = seq_np.shape[1]

    batch_schedules = []
    batch_makespans = []

    # 2. 遍历 Batch 中的每一个样本
    for b in range(batch_size):
        # 初始化当前样本的状态
        # 机器空闲时间 (Key: MachineID, Val: Time)
        machine_free_time = {m: 0.0 for m in range(n_m)}

        # Job 空闲时间 (Key: JobID, Val: Time)
        job_free_time = {j: 0.0 for j in range(n_j)}

        # Job 当前做到第几道工序 (Key: JobID, Val: Index 0~n_m)
        job_op_idx = {j: 0 for j in range(n_j)}

        # 结果容器：调度表
        current_schedule = {m: [] for m in range(n_m)}

        # --- 单个样本的调度模拟 ---
        for i in range(total_ops_in_seq):
            job_id = seq_np[b, i]  # 当前选中的Job

            # 获取该 Job 当前需要做的工序索引(op_k)
            op_k = job_op_idx[job_id]

            # 安全检查：如果该 Job 已经做完所有n_m道工序，跳过（防止越界）
            if op_k >= n_m:
                continue

            # 获取该工序的 机器ID 和 耗时
            # dur_np[batch, job, op]
            d = dur_np[b, job_id, op_k]
            m_id = int(mach_np[b, job_id, op_k])

            # 开始时间 = max(机器空闲, Job空闲)
            start_time = max(machine_free_time[m_id], job_free_time[job_id])
            end_time = start_time + d

            # --- 更新状态 ---
            machine_free_time[m_id] = end_time
            job_free_time[job_id] = end_time
            job_op_idx[job_id] += 1

            # --- 记录到调度表 ---
            current_schedule[m_id].append({
                'job': int(job_id),
                'op_idx': int(op_k),
                'start': float(start_time),
                'end': float(end_time),
                'duration': float(d)
            })

        # 计算当前样本的 Makespan
        makespan = max(machine_free_time.values()) if machine_free_time else 0

        batch_schedules.append(current_schedule)
        batch_makespans.append(makespan)

    # 将 Makespan 转回 Tensor (方便后续计算或与 Tensor 格式兼容)
    return batch_schedules, torch.tensor(batch_makespans, device=job_sequence.device)


'''
def validate_makespan_batch(job_sequence, duration, machine_alloc, n_j, n_m):
    """
    Args:
        job_sequence: [B, Total_Ops] (Job IDs)
        duration:     [B, n_j, n_m]
        machine_alloc:[B, n_j, n_m]
    """
    bsz, n_ops = job_sequence.size()
    device = job_sequence.device

    # 状态追踪
    job_op_idx = torch.zeros(bsz, n_j, dtype=torch.long, device=device)
    job_avail_time = torch.zeros(bsz, n_j, device=device)
    machine_avail_time = torch.zeros(bsz, n_m, device=device)

    # batch索引
    batch_idx = torch.arange(bsz, device=device)

    for step in range(n_ops):
        # 获取当前动作 (Job ID)
        selected_job = job_sequence[:, step]

        # 获取该Job当前的工序索引
        # curr_op_idx = job_op_idx[batch_idx, selected_job]
        curr_op_idx_raw = job_op_idx[batch_idx, selected_job]
        curr_op_idx = curr_op_idx_raw.clamp(max=n_m - 1)

        # 查表获取耗时(d)和机器(m)
        # duration 维度是 [B, n_j, n_m]
        d = duration[batch_idx, selected_job, curr_op_idx]
        m_id = machine_alloc[batch_idx, selected_job, curr_op_idx].long()

        # 计算开始时间
        t_job_ready = job_avail_time[batch_idx, selected_job]
        t_mach_ready = machine_avail_time[batch_idx, m_id]
        start_time = torch.max(t_job_ready, t_mach_ready)
        end_time = start_time + d

        # 更新状态
        # 更新 Job 可用时间
        job_avail_time[batch_idx, selected_job] = end_time
        # 更新 Machine 可用时间
        machine_avail_time[batch_idx, m_id] = end_time
        # 更新工序进度
        job_op_idx[batch_idx, selected_job] += 1

    makespan, _ = job_avail_time.max(dim=1)
    return makespan
'''
