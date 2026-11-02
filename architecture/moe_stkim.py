import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from .network import DimReduction, Classifier_1fc
from .tam import configure_tam, modulate_attention

class AdvancedDimReduction(nn.Module):
    """
    改进的维度约简模块，参考ConvNeXt的思想
    先将特征维度提升到中间层（D_inner*4），再压缩到目标维度（D_inner）
    使用非线性激活函数保留更多特征信息
    """
    def __init__(self, n_channels, m_dim=512, expansion_factor=4):
        super(AdvancedDimReduction, self).__init__()
        self.expansion_factor = expansion_factor
        self.m_dim = m_dim
        self.expanded_dim = m_dim * expansion_factor
        
        # 第一层：将输入特征映射到扩展维度
        self.fc1 = nn.Linear(n_channels, self.expanded_dim, bias=False)
        self.norm1 = nn.LayerNorm(self.expanded_dim)
        self.act1 = nn.GELU()  # 使用GELU激活函数
        
        # 中间处理层（可选的残差块）
        self.middle_layer = nn.Linear(self.expanded_dim, self.expanded_dim, bias=False)
        self.norm2 = nn.LayerNorm(self.expanded_dim)
        self.act2 = nn.GELU()
        
        # 最终层：将扩展维度压缩到目标维度
        self.fc2 = nn.Linear(self.expanded_dim, m_dim, bias=False)
        self.norm3 = nn.LayerNorm(m_dim)
        self.act3 = nn.GELU()
        
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # 第一步：扩展到高维空间
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        # 中间处理
        residual = x
        x = self.middle_layer(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = x + residual  # 残差连接
        
        # 最后一步：压缩到目标维度
        x = self.fc2(x)
        x = self.norm3(x)
        x = self.act3(x)
        
        # 应用可学习的缩放
        x = x * self.scale
        
        return x

class MoEACMIL(nn.Module):
    """
    基于 Mixture of Experts 思想的 ACMIL 模型
    在原始 STKIM 的基础上，增加 Stratified STKIM（High / Mid / Low）
    """
    def __init__(self, conf, n_token=1, n_masked_patch=0, mask_drop=0, n_expand=2):
        super(MoEACMIL, self).__init__()

        if n_expand >= 1:
            self.dimreduction = AdvancedDimReduction(conf.D_feat, conf.D_inner, expansion_factor=n_expand)
        else:
            self.dimreduction = DimReduction(conf.D_feat, conf.D_inner)

        self.n_token = n_token
        self.n_class = conf.n_class

        # ===== 新增：分层 STKIM 配置（不影响原逻辑）=====
        self.use_stratified_stkim = getattr(conf, 'use_stratified_stkim', True)
        configure_tam(self, conf)

        # 多个注意力头
        self.attention_heads = nn.ModuleList()
        for _ in range(n_token):
            self.attention_heads.append(
                Attention_Gated(conf.D_inner, D=128, K=1)
            )

        # 多个分类器，每个注意力头对应一个分类器
        self.classifiers = nn.ModuleList()
        for _ in range(n_token):
            self.classifiers.append(
                Classifier_1fc(conf.D_inner, conf.n_class, 0.0)
            )

        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(conf.D_inner, n_token),
            nn.Softmax(dim=-1)
        )

        # slide-level 分类器
        self.Slide_classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0.0)

        # patch-level 门控网络（保留你原来的定义）
        self.patch_gate_network = nn.Sequential(
            nn.Linear(conf.D_inner, n_token),
            nn.Softmax(dim=-1)
        )

        # 原始 STKIM 参数
        self.n_masked_patch = n_masked_patch
        self.mask_drop = mask_drop

    def apply_stratified_stkim(self, A):
        """
        Stratified STKIM:
        按 attention score 排序后划分为 High / Mid / Low 三组，
        再分别按各自 mask rate 随机掩蔽。
        A: [1, N] (before softmax)
        """
        return modulate_attention(A, self.stratified_group_ratios, self.tam_mode,
                                  self.stratified_mask_rates, self.tam_rho)

    def apply_original_stkim(self, A):
        """
        你原来的 STKIM：只对 top-K attention instances 做随机 mask
        """
        k, n = A.shape
        n_masked_patch = min(self.n_masked_patch, n)
        if n_masked_patch <= 0:
            return A

        _, indices = torch.topk(A, n_masked_patch, dim=-1)
        rand_selected = torch.argsort(
            torch.rand(*indices.shape, device=A.device), dim=-1
        )[:, :int(n_masked_patch * self.mask_drop)]

        masked_indices = indices[
            torch.arange(indices.shape[0], device=A.device).unsqueeze(-1),
            rand_selected
        ]

        random_mask = torch.ones(k, n, device=A.device)
        random_mask.scatter_(-1, masked_indices, 0)
        A = A.masked_fill(random_mask == 0, -1e9)
        return A

    def forward(self, x):
        x = x[0]  # [N, D_feat]
        x = self.dimreduction(x)  # [N, D_inner]

        attention_weights = []
        attention_features = []

        for attention_head in self.attention_heads:
            A = attention_head(x)  # [1, N]

            # ===== 这里只改 STKIM，其他逻辑保持原样 =====
            if self.training or (self.use_stratified_stkim and self.tam_apply_in_eval):
                if self.use_stratified_stkim:
                    A = self.apply_stratified_stkim(A)
                elif self.n_masked_patch > 0:
                    A = self.apply_original_stkim(A)

            A = F.softmax(A, dim=1)  # [1, N]
            attention_weights.append(A)

            afeat = torch.mm(A, x)  # [1, D_inner]
            attention_features.append(afeat)

        # [1, n_token, D_inner]
        all_features = torch.stack(attention_features, dim=1)

        global_feat = torch.mean(x, dim=0, keepdim=True)  # [1, D_inner]
        gate_weights = self.gate_network(global_feat)  # [1, n_token]

        weighted_features = all_features * gate_weights.unsqueeze(-1)  # [1, n_token, D_inner]
        moe_features = torch.sum(weighted_features, dim=1, keepdim=True)  # [1, D_inner]

        sub_predictions = []
        for i, classifier in enumerate(self.classifiers):
            pred = classifier(attention_features[i])  # [1, n_class]
            sub_predictions.append(pred.squeeze(0))   # [n_class]

        sub_predictions = torch.stack(sub_predictions, dim=0)  # [n_token, n_class]

        gate_weights_expanded = gate_weights.squeeze(0).unsqueeze(-1)  # [n_token, 1]
        weighted_sub_pred = sub_predictions * gate_weights_expanded
        moe_sub_pred = torch.sum(weighted_sub_pred, dim=0, keepdim=True)  # [1, n_class]

        slide_pred = self.Slide_classifier(moe_features.squeeze(1))

        # 返回值保持你原样
        return moe_sub_pred, slide_pred, gate_weights.squeeze(0).unsqueeze(0)

class Attention_Gated(nn.Module):
    """
    门控注意力机制
    """
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()

        self.L = L
        self.D = D
        self.K = K

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x):
        ## x: N x L
        A_V = self.attention_V(x)  # NxD
        A_U = self.attention_U(x)  # NxD
        A = self.attention_weights(A_V * A_U) # NxK
        A = torch.transpose(A, 1, 0)  # KxN

        return A  ### K x N
