import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn import LayerNorm
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import to_dense_batch
# from torch.nn import TransformerDecoderLayer, TransformerDecoder


class Attention(nn.Module):
    def __init__(self,
                 q_hidden_dim,
                 k_dim,
                 v_dim,
                 n_head,
                 k_hidden_dim=None,
                 v_hidden_dim=None):
        super().__init__()
        self.q_hidden_dim = q_hidden_dim
        self.k_hidden_dim = k_hidden_dim if k_hidden_dim else q_hidden_dim
        self.v_hidden_dim = v_hidden_dim if v_hidden_dim else q_hidden_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.n_head = n_head

        self.proj_q = nn.Linear(q_hidden_dim, k_dim * n_head, bias=False)
        self.proj_k = nn.Linear(self.k_hidden_dim, k_dim * n_head, bias=False)
        self.proj_v = nn.Linear(self.v_hidden_dim, v_dim * n_head, bias=False)
        self.proj_output = nn.Linear(v_dim * n_head,
                                     self.v_hidden_dim,
                                     bias=False)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None: k = q
        if v is None: v = k

        bsz, n_node, _ = k.size()

        # 计算 Q, K, V
        qs = torch.stack(torch.chunk(self.proj_q(q), self.n_head, dim=-1), dim=1)
        ks = torch.stack(torch.chunk(self.proj_k(k), self.n_head, dim=-1), dim=1)
        vs = torch.stack(torch.chunk(self.proj_v(v), self.n_head, dim=-1), dim=1)

        normalizer = self.k_dim ** 0.5
        u = torch.matmul(qs, ks.transpose(2, 3)) / normalizer

        if mask is not None:
            # mask shape 转换: [bsz, n_node] -> [bsz, 1, 1, n_node]
            mask = mask.unsqueeze(1).unsqueeze(1)
            u = u.masked_fill(mask, float('-inf'))

        att = torch.matmul(torch.softmax(u, dim=-1), vs)
        att = att.transpose(1, 2).reshape(bsz, -1, self.v_dim * self.n_head)
        att = self.proj_output(att)
        return att


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # 3层GATv2
        self.layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=True, edge_dim=2),
            GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=True, edge_dim=2),
            GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=2)
        ])

        # 前两层 concat=True 后的降维层
        self.projs = nn.ModuleList([
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Linear(hidden_dim * 4, hidden_dim)
        ])

        self.norms1 = nn.ModuleList([LayerNorm(hidden_dim) for _ in range(3)])
        self.norms2 = nn.ModuleList([LayerNorm(hidden_dim) for _ in range(3)])

        # FFN层
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            ) for _ in range(3)
        ])

    def forward(self, x, edge_index, edge_attr):
        h = self.embedding(x)

        for i, (conv, norm1, norm2, ffn) in enumerate(zip(self.layers, self.norms1, self.norms2, self.ffns)):
            h_in = h
            h = conv(h, edge_index, edge_attr=edge_attr)
            # hidden_dim*4 降维 hidden_dim
            if i < 2:
                h = self.projs[i](h)
            h = norm1(h+h_in)
            # h = F.relu(h)
            # h = h + h_in  # 残差连接
            # FFN
            h_in2 = h
            h = ffn(h)
            h = norm2(h + h_in2)
        return h


class Decoder(nn.Module):
    def __init__(self, hidden_dim, n_j, n_m):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_j = n_j
        self.n_m = n_m

        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        # decoder_layer = TransformerDecoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=512, batch_first=True)
        # self.transformer = TransformerDecoder(decoder_layer, num_layers=3)

        self.pointer_att = Attention(q_hidden_dim=hidden_dim,
                                     k_dim=hidden_dim // 8,
                                     v_dim=hidden_dim // 8,
                                     n_head=8)

        # self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.start_token = nn.Parameter(torch.randn(1, hidden_dim))
        # 初始隐藏状态 (h_0)
        self.h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.proj_k = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dyn_dim = 6
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim + self.dyn_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.tanh_clipping = 10.0

    def forward(self, encoder_out, mask_padding, op_machine_idx, op_proc_time, rollout=False, temperature=1.0):
        """
        encoder_out: [Batch, n_tasks, Hidden]
        Returns:
            job_indices: [Batch, n_tasks]
            sum_log_probs: [Batch]
        """
        bsz, n_node, _ = encoder_out.size()

        # 预计算用于 Logits 的 Key
        k_logits_all = self.proj_k(encoder_out)
        # 有效的节点掩码
        valid_mask = (~mask_padding).unsqueeze(-1).float()
        # 全局图特征 [B, H]
        graph_context = (encoder_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1)
        # 状态初始化
        curr_input = self.start_token.expand(bsz, -1)  # [B, H]
        # hx = self.h0.expand(bsz, -1)
        hx = graph_context

        job_next_op_local_idx = torch.zeros(bsz, self.n_j, dtype=torch.long, device=encoder_out.device)
        # [B, n_j] 标识Job是否已完成所有工序 (True表示已完成)
        mask_job_finished = torch.zeros(bsz, self.n_j, dtype=torch.bool, device=encoder_out.device)

        # Environment Time Trackers
        job_ready_time = torch.zeros(bsz, self.n_j, device=encoder_out.device)
        machine_avail_time = torch.zeros(bsz, self.n_m, device=encoder_out.device)

        job_indices_seq = []
        log_probs_seq = []
        entropies_seq = []

        # 每个Job当前待选工序的全局Node Index, base_indices: [0, n_m, 2*n_m, ...]
        base_indices = torch.arange(0, self.n_j * self.n_m, self.n_m, device=encoder_out.device)
        base_indices = base_indices.unsqueeze(0).expand(bsz, -1) # [B, N_J]

        for i in range(n_node):

            # Transformer提取当前步特征
            # h_context = self.transformer(tgt=curr_input,
            #                              memory=encoder_out,
            #                              memory_key_padding_mask=mask_padding)

            hx = self.gru(curr_input, hx)  # 输入: [B, H],[B, H] -> 输出: [B, H]
            # 计算 Glimpse：将 hx 升维到 [B, 1, H]
            q = hx.unsqueeze(1)
            glimpse = self.pointer_att(q=q, k=encoder_out, v=encoder_out, mask=mask_padding)

            current_global_indices = base_indices + job_next_op_local_idx
            safe_indices = current_global_indices.clamp(max=n_node - 1)

            # k_logits_all: [B, N_tasks, H] -> [B, n_j, H]
            safe_indices_expanded = safe_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            k_candidates = torch.gather(k_logits_all, 1, safe_indices_expanded)

            cand_mach = torch.gather(op_machine_idx, 1, safe_indices)  # [B, n_j]
            cand_pt = torch.gather(op_proc_time, 1, safe_indices)  # [B, n_j]

            cand_mach_avail = torch.gather(machine_avail_time, 1, cand_mach)  # [B, n_j]

            e_start = torch.max(job_ready_time, cand_mach_avail)  # 最早开始时间
            e_comp = e_start + cand_pt  # 预计完成时间
            wait_time = e_start - job_ready_time  # 工件等待时间
            idle_time = e_start - cand_mach_avail  # 机器空闲时间
            ops_left_ratio = (self.n_m - job_next_op_local_idx) / self.n_m  # 剩余工序比例
            pt_ratio = cand_pt / (cand_pt.mean(dim=-1, keepdim=True) + 1e-5) # 剩余加工时间

            norm_factor = torch.max(machine_avail_time, dim=1, keepdim=True)[0].clamp(min=1.0)
            dyn_feats = torch.stack([
                e_start / norm_factor,
                e_comp / norm_factor,
                wait_time / norm_factor,
                idle_time / norm_factor,
                ops_left_ratio.float(),
                pt_ratio
            ], dim=-1)  # [B, n_j, 6]

            cat_feats = torch.cat([k_candidates, dyn_feats], dim=-1)
            k_candidates_fused = self.fusion_net(cat_feats)  # [B, n_j, H]

            # ---计算 Logits---
            # u: [B, 1, n_j]
            u = torch.matmul(glimpse, k_candidates_fused.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
            u = torch.tanh(u) * self.tanh_clipping
            u = u / temperature
            u = u.squeeze(1)

            is_all_masked = mask_job_finished.all(dim=-1, keepdim=True)
            mask_job_finished = mask_job_finished.masked_fill(is_all_masked, False)

            # ---Mask掉已完成的Job---
            u = u.masked_fill(mask_job_finished, float('-inf'))

            # 采样/贪婪选择
            if rollout:
                selected_job = u.max(-1)[1]
            else:
                probs = F.softmax(u, dim=-1)
                m = Categorical(probs)
                selected_job = m.sample()
                log_probs_seq.append(m.log_prob(selected_job))
                entropies_seq.append(m.entropy())

            job_indices_seq.append(selected_job)

            selected_job_unsqueezed = selected_job.unsqueeze(1)  # [B, 1]

            chosen_pt = torch.gather(cand_pt, 1, selected_job_unsqueezed).squeeze(1)  # [B]
            chosen_mach = torch.gather(cand_mach, 1, selected_job_unsqueezed).squeeze(1)  # [B]

            chosen_job_ready = torch.gather(job_ready_time, 1, selected_job_unsqueezed).squeeze(1)  # [B]
            chosen_mach_avail = torch.gather(machine_avail_time, 1, chosen_mach.unsqueeze(1)).squeeze(1)  # [B]

            actual_start = torch.max(chosen_job_ready, chosen_mach_avail)
            actual_comp = actual_start + chosen_pt

            job_ready_time.scatter_(1, selected_job_unsqueezed, actual_comp.unsqueeze(1))
            machine_avail_time.scatter_(1, chosen_mach.unsqueeze(1), actual_comp.unsqueeze(1))

            # 更新索引和完成状态
            one_hot = F.one_hot(selected_job, num_classes=self.n_j)  # [B, n_j]
            job_next_op_local_idx = job_next_op_local_idx + one_hot
            mask_job_finished = (job_next_op_local_idx >= self.n_m)

            # 获取被选中的全局 Node ID
            selected_job_unsqueezed = selected_job.unsqueeze(1)  # [B, 1]
            selected_global_node_idx = torch.gather(current_global_indices, 1, selected_job_unsqueezed)  # [B, 1]

            # 获取该Node的Embedding
            idx_expanded = selected_global_node_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            curr_input = torch.gather(encoder_out, 1, idx_expanded)
            curr_input = curr_input.squeeze(1) # [B, H]

        # 拼接结果
        priority_job_list = torch.stack(job_indices_seq, dim=1)
        if not rollout:
            sum_log_probs = torch.stack(log_probs_seq, dim=1).sum(dim=1)
            sum_entropies = torch.stack(entropies_seq, dim=1).sum(dim=1)
            return priority_job_list, sum_log_probs, sum_entropies
        else:
            return priority_job_list, None, None


class JSSPActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_j, n_m):
        super().__init__()
        # Encoder: PyG GATv2
        self.encoder = GNNEncoder(input_dim, hidden_dim)
        # Decoder: Native Transformer + Attention Pointer
        self.decoder = Decoder(hidden_dim, n_j, n_m)

    def forward(self, pyg_batch, op_machine_idx, op_proc_time, rollout=False, temperature=1.0):
        """
        pyg_batch: torch_geometric.data.Batch
        """
        # 图编码
        node_emb_flat = self.encoder(pyg_batch.x, pyg_batch.edge_index,pyg_batch.edge_attr)

        # Sparse Graph -> Dense Batch
        x_dense, mask = to_dense_batch(node_emb_flat, pyg_batch.batch)
        mask_padding = ~mask

        # 解码
        return self.decoder(x_dense, mask_padding, op_machine_idx, op_proc_time, rollout=rollout, temperature=temperature)

