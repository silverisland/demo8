# Phys-PVGen：物理信息驱动的跨站点光伏生成模型复现指南

**密级：技术秘密** | **版本：V1.1** | **编制：Gemini CLI**

---

## 1. 算法架构与数学形式化

### 1.1 站点指纹提取 (Site Latent Codebook)
场站指纹通过一个可学习的记忆库 $\mathcal{C} \in \mathbb{R}^{K \times D}$ 提取，其中 $K=32$ 为原型数量，$D=64$ 为维度。
1. **序列编码**：利用双向 GRU 编码场站历史序列（气象+出力），取最后时刻隐藏状态拼接为查询向量 $q \in \mathbb{R}^D$。
2. **原型匹配**：采用缩放点积注意力机制计算权重：
   $$\alpha = \text{Softmax}\left(\frac{q \cdot \mathcal{C}^T}{\sqrt{D}}\right)$$
3. **特征合成**：生成最终指纹 $z_{site} = \alpha \cdot \mathcal{C}$。

### 1.2 物理增强扩散生成器 (DDPM)
模型通过 $T=100$ 步高斯去噪恢复功率序列 $x_0$。
- **输入层架构**：将带噪序列 $x_t \in \mathbb{R}^{L \times 1}$ 与物理锚点 $G_{cs} \in \mathbb{R}^{L \times 1}$ 在通道维度拼接，输入首层 1D 卷积。
- **条件注入逻辑**：
  - **FiLM 层**：调制 ResNet 内部特征图。
  - **Cross-Attention 层**：
    - **Query**: 扩散模型主干特征。
    - **Key/Value**: $\text{Concat}(\text{NWP Encoder}(Weather), z_{site})$。

---

## 2. 数据工程流水线 (Physics-Informed Pipeline)

### 2.1 物理锚点计算 (pvlib 整合)
必须集成 `pvlib` 库计算**晴空总辐射 (Clear-sky GHI)**。
- **输入**：场站经纬度、海拔、时区、本地时间戳。
- **算法**：采用 Ineichen 模型计算 GHI。
- **作用**：作为模型生成的“物理天花板”，确立能量守恒的硬约束边界。

### 2.2 特征工程规范
| 特征组 | 维度 | 处理逻辑 |
| :--- | :--- | :--- |
| NWP 基础 | 2 | GHI (Global Horizontal Irradiance), TEMP (Temperature) |
| 物理锚点 | 1 | Clear-sky GHI (通过 pvlib 实时计算) |
| 时间编码 | 4 | $\sin/\cos(Hour/24)$, $\sin/\cos(Month/12)$ |
| 观测序列 | 1 | 仅在 Stage 1 提取指纹时输入实测 Power |

---

## 3. 训练阶段复现步骤 (Training Protocol)

### Stage 1: 指纹空间对比预训练 (Contrastive Pre-training)
*   **目标**：强制 Codebook 学习场站的不变性特征。
*   **数据构造**：从同一场站随机抽取两个不重叠的时间窗口作为 Positive Pair ($v_1, v_2$)，不同场站间为 Negative Pairs。
*   **损失函数**：采用 InfoNCE Loss，温度系数 $\tau=0.07$。
    $$\mathcal{L}_{stage1} = -\log \frac{\exp(sim(z_1, z_2)/\tau)}{\sum \exp(sim(z_1, z_{neg})/\tau)}$$

### Stage 2: 物理约束生成训练 (Conditional Generation)
*   **损失函数**：$\mathcal{L}_{total} = \mathcal{L}_{mse} + 10.0 \cdot \mathcal{L}_{physics} + 0.5 \cdot \mathcal{L}_{fluctuation}$
*   **CPT Loss 实现关键点**：
    1.  **物理边界项**：$\text{Mean}(\text{ReLU}(P_{gen} - GHI_{cs})^2)$。
    2.  **反平滑波动项**：计算一阶差分 $\nabla P = P_t - P_{t-1}$，使 $\text{MAE}(|\nabla P_{gen}|, |\nabla P_{real}|)$ 最小化。

---

## 4. 关键代码实现片段 (Pseudo-Code)

### 核心扩散步 (EpsilonNet Forward)
```python
def forward(x_t, t, nwp, site_latent, ghi_clearsky):
    # 物理锚点拼接 (Channel Concat)
    x_input = torch.cat([x_t, ghi_clearsky], dim=-1)
    h = self.init_conv(x_input.transpose(1, 2))
    
    # 跨注意力融合 (Cross-Attention)
    # Context 融合了动态天气 NWP 和 静态站点指纹 site_latent
    context = torch.cat([self.nwp_encoder(nwp), site_latent.unsqueeze(1)], dim=1)
    h = self.cross_attn(query=h, key=context, value=context)
    
    return self.final_conv(h)
```

### 物理一致性截断 (Inference Wrapper)
```python
@torch.no_grad()
def physical_sample(model, nwp, site_latent, ghi_cs):
    # 1. 执行标准反向去噪步
    samples = model.sample(nwp, site_latent)
    # 2. 物理天花板强制约束 (Energy Conservation)
    samples = torch.clamp(samples, min=0.0)
    samples = torch.min(samples, ghi_cs) 
    return samples
```

---

## 5. 复现硬件与性能参考
*   **显存需求**：> 16GB (推荐 A100/RTX 3090)。
*   **训练时长**：Stage 1 约 50 Epochs，Stage 2 约 100 Epochs。
*   **收敛标志**：生成的功率曲线在高频段的 PSD (功率谱密度) 需与真实序列重合度 > 85%。

---

## 6. 运行与操作指南 (Execution Guide)

### 6.1 环境准备
```bash
# 安装依赖
pixi install
# 激活环境
pixi shell
```

### 6.2 阶段一：源站预训练 (Stage 1 Training)
用于训练站点记忆库 (Codebook) 和基础生成器权重。
```bash
python scripts/train_stage1.py \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0001 \
    --exp_name "base_pretrain"
```
*   **关键输出**：`outputs/stage1_base_pretrain/best_model.pth`

### 6.3 跨站点推理与数据合成 (Inference & Generation)
加载预训练模型，针对新站点提取指纹并合成长序列。
```bash
python scripts/run_inference.py \
    --checkpoint "outputs/stage1_base_pretrain/best_model.pth" \
    --num_sites 5 \
    --days 365 \
    --noise_scale 1.2
```
*   **参数说明**：
    *   `--noise_scale`：调整云层随机扰动强度（>1.0 增加波动性，<1.0 曲线更平滑）。
    *   `--days`：指定合成时长，系统将自动进行分块并行生成（Chunk-based Generation）。

### 6.4 下游任务衔接 (Downstream Training)
合成数据可通过 `scripts/pipeline.py` 导出为标准 CSV 格式，随后使用 `src/models/forecasting.py` 进行 Transformer 预测模型的增强训练。

---
**拟制人**：Gemini CLI | **技术审核**：模型算法组
