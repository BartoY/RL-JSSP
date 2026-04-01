import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from copy import deepcopy

from uniform_instance_gen import uni_instance_gen
from model_1 import JSSPActor
from utils import prio_lst_sched_bch, chk_upd_bl
from plot import plot_learning_curves
from data_utils import get_initial_intput, epoch_dataset_gen, convert_to_pyg_data
from validate import validate_model

# --- 超参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
LR = 5e-5
BATCH_SIZE = 512
EPOCHS = 100
N_J = 10
N_M = 10
n_simple = 500000
ENTROPY_COEF = 0.01   # 熵正则化系数
TEMP_START = 1.2      # 初始温度
TEMP_END = 1.0


def main():
    # 初始化模型
    policy_model = JSSPActor(input_dim=2, hidden_dim=128, n_j=N_J, n_m=N_M).to(DEVICE)
    baseline_model = deepcopy(policy_model)
    baseline_model.eval()

    optimizer = optim.Adam(policy_model.parameters(), lr=LR)

    # 验证集
    val_data_list = epoch_dataset_gen(n_samples=200, n_j=N_J, n_m=N_M)
    val_loader = DataLoader(val_data_list, batch_size=BATCH_SIZE, shuffle=False)  # 验证集不需要 shuffle

    print("Evaluating initial baseline...")
    # 初始时 Baseline 和 Policy 一样，先算一次存起来
    baseline_val_costs = validate_model(baseline_model, val_loader, DEVICE, N_J, N_M)
    print(f"Initial Baseline Avg Mksp: {baseline_val_costs.mean():.2f}")

    # 初始化记录列表
    history_loss = []
    history_train_mksp = []
    history_val_mksp = []

    best_val_makespan = float('inf') # 用于记录历史最好成绩

    # 训练循环
    for epoch in range(EPOCHS):
        current_temp = TEMP_END + (TEMP_START - TEMP_END) * (1.0 - epoch / EPOCHS)

        train_data_list = epoch_dataset_gen(n_samples=n_simple, n_j=N_J, n_m=N_M)
        train_loader = DataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True)

        policy_model.train()
        total_loss = 0
        total_train_mksp = 0
        total_entropy = 0

        for batch in train_loader:
            batch = batch.to(DEVICE)

            # 获取当前的 Batch Size
            bsz = batch.num_graphs

            # PyG 默认把batch里的属性拼成了[Batch * n_j, n_m] (2维)
            # 需要将其reshape回[Batch, n_j, n_m](3维)以便使用
            reshaped_durations = batch.raw_durations.view(bsz, N_J, N_M)
            reshaped_machines = batch.raw_machines.view(bsz, N_J, N_M)
            op_machine_idx = reshaped_machines.view(bsz, -1).long()
            op_proc_time = reshaped_durations.view(bsz, -1).float()

            solutions, log_probs, entropies = policy_model(batch, op_machine_idx, op_proc_time, rollout=False, temperature=current_temp)

            # --- Reward Calculation ---
            m_schedule, costs = prio_lst_sched_bch(
                solutions,
                reshaped_durations,
                reshaped_machines,
                N_J, N_M
            )

            # --- Baseline Calculation ---
            with torch.no_grad():
                base_solutions, *_ = baseline_model(batch, op_machine_idx, op_proc_time, rollout=True)
                base_schedule, base_costs = prio_lst_sched_bch(
                    base_solutions,
                    reshaped_durations,
                    reshaped_machines,
                    N_J, N_M
                )

            # --- Loss ---
            advantage = (costs - base_costs).detach()

            # 对Advantage进行归一化
            # 减去均值，除以标准差，把数值缩放到0附近
            # if len(advantage) > 1:
            #     advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            # loss = (advantage * log_probs).mean()

            rl_loss = (advantage * log_probs).mean()
            entropy_bonus = entropies.mean()

            loss = rl_loss - ENTROPY_COEF * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_train_mksp += costs.mean().item()

        avg_loss = total_loss / len(train_loader)
        avg_train_mksp = total_train_mksp / len(train_loader)

        policy_val_costs = validate_model(policy_model, val_loader, DEVICE, N_J, N_M)
        avg_val_mksp = policy_val_costs.mean()

        print(f"Epoch {epoch + 1}: Loss {avg_loss:.4f} | Train Mksp {avg_train_mksp:.2f} | Val Mksp {avg_val_mksp:.2f}")

        # 记录当前 Epoch 的平均数据
        history_loss.append(avg_loss)
        history_train_mksp.append(avg_train_mksp)
        history_val_mksp.append(avg_val_mksp)

        # --- 保存最佳模型 ---
        if avg_val_mksp < best_val_makespan:
            best_val_makespan = avg_val_mksp
            torch.save(policy_model.state_dict(), f"/home/yifan/hang/fjsp/models_save/{N_J}_{N_M}_best_model_{BATCH_SIZE}.pth")
            print(f"  >>> New Best Model Saved! (Val Mksp: {best_val_makespan:.2f})")

        # --- Update Baseline ---
        should_update = chk_upd_bl(policy_val_costs, baseline_val_costs)

        if should_update:
        # if (epoch + 1) % 5 == 0:
            print("Updating Baseline...")
            baseline_model.load_state_dict(policy_model.state_dict())
            baseline_val_costs = policy_val_costs

        # 训练结束后调用画图函数
    print("Training finished. Plotting curves...")
    plot_learning_curves(history_loss, history_train_mksp, history_val_mksp)


if __name__ == "__main__":
    main()