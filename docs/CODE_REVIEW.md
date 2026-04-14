# PV Forecasting Framework 代码设计审查报告

> 审查日期: 2026-04-08
> 依据文档: tast.md, DESIGN.md

---

## 一、整体评估

| 指标 | 评分 | 说明 |
|------|------|------|
| 模块化设计 | ✅ 优秀 | 6个模块职责清晰，OOP结构合理 |
| 物理约束 (CPT Loss) | ✅ 正确 | L_recon + α·L_physics + β·L_smooth 实现到位 |
| 代码可读性 | ✅ 良好 | 类型提示完整，注释清晰 |
| 需求完整性 | ⚠️ ~70% | 缺异常检测、TFT实现 |

---

## 二、模块问题汇总

### 1. dataset.py

#### 问题1: NWP 维度不匹配
- **位置**: line 107
- **现状**: `nwp_base = np.stack([nwp_ghi, nwp_temp], axis=-1)` 只输出 2 维
- **要求**: tast.md 明确 `nwp_dim = 7`
- **影响**: 与 config.py 定义的 7 维不一致，训练时会 shape 报错
- **修复**:
  ```python
  # 方案A: 扩展更多 NWP 特征
  nwp_features = ['ghi', 'dni', 'dhi', 'temp', 'humidity', 'wind', 'pressure']
  # 方案B: 修正 config.py 的 nwp_dim = 2
  ```

#### 问题2: 缺失异常数据过滤
- **要求**: tast.md 要求 "drop sequences where real_power == 0 but GHI > 500"
- **现状**: 无任何异常检测逻辑
- **影响**: 模型可能学习到物理不合理的模式（如夜间发电）
- **修复建议**:
  ```python
  def __getitem__(self, idx):
      # ... 现有逻辑 ...
      
      # 异常检测
      night_mask = (power == 0) & (ghi_clearsky > 500)
      if night_mask.sum() > len(power) * 0.1:  # 超过10%异常则跳过
          return self.__getitem__(np.random.randint(0, len(self)))
  ```

---

### 2. memory.py

#### 问题: input_dim 与数据集不匹配
- **位置**: line 16
- **现状**: `input_dim = 8` (假设 NWP=7 + Power=1)
- **实际**: dataset.py 返回的 NWP 只有 2-3 维
- **影响**: `extract_fingerprint` 时维度不匹配
- **修复**: 根据实际 NWP 维度设置，或在 config 中统一定义

---

### 3. generator.py (Diffusion 模型)

#### 问题1: NWP 条件注入过于简化 (关键瓶颈)
- **位置**: line 85-88
- **现状**:
  ```python
  nwp_mean = nwp.mean(dim=1)  # 只取均值！
  full_cond = torch.cat([global_cond, nwp_mean], dim=-1)
  ```
- **问题**: PV功率与 NWP 辐射紧密相关，只用均值会丢失关键信息
  - 云量突变时功率急剧下降 → `mean(GHI)` 无法捕捉
  - 早晨/黄昏低辐射 → 均值被白天高值拉高
- **修复建议**:
  ```python
  # 方案1: 完整 NWP 序列作为 per-step 条件
  class EpsilonNet(nn.Module):
      def forward(self, x_t, t, nwp, site_latent):
          nwp_embed = self.nwp_encoder(nwp)  # [B, d_model, L]
          x = x_t.transpose(1, 2)  # [B, 1, L]
          h = torch.cat([x, nwp_embed], dim=1)  # 沿通道维度拼接
          # 后续卷积会自动处理
  ```

#### 问题2: 条件投影维度巧合而非设计
- **位置**: line 64-65 vs line 88
- **现状**: block 定义 `cond_dim = nwp_dim + hidden_dim + site_latent_dim`
- **实际**: 传入 `hidden_dim + site_latent_dim + nwp_dim` (碰巧一致)
- **风险**: 修改 NWP 处理方式时容易出错

#### 问题3: 采样时无物理约束
- **位置**: line 170-189 (sample 函数)
- **现状**: 反向扩散过程没有任何物理边界检查
- **问题**: 中间状态可能违背物理规律（如负功率）
- **修复建议**: 每步采样后添加:
  ```python
  x = torch.clamp(x, min=0.0)
  x = torch.min(x, ghi_clearsky)
  ```

#### 问题4: 长序列建模能力偏弱
- **现状**: 2 层 ResidualBlock 处理 864 长度 (7天+2天)
- **建议**: 增加到 4-6 层，或引入 LSTM/Transformer 增强时序建模

---

### 4. forecasting.py

#### 问题: 未实现 TFT/PatchTST
- **要求**: tast.md 明确 "Temporal Fusion Transformer (TFT) 或 PatchTST baseline"
- **现状**: 只有基础 TransformerEncoder
- **影响**: 名称误导，无法展示多时间尺度融合、变量选择等 TFT 特性
- **修复**:
  ```python
  # 方案A: 重命名
  class SimpleTransformerForecaster(nn.Module):
      """简化版时序预测器 (非TFT)"""
  
  # 方案B: 实现完整TFT (需较大改动)
  class TemporalFusionTransformer(nn.Module):
      # 1. 变量选择网络
      # 2. 时间特征嵌入
      # 3. 多时间尺度 LSTM
      # 4. Static 特征融合
      # 5. Decoder 注意力
  ```

---

### 5. pipeline.py

#### 问题: 物理后处理位置
- **现状**: 物理边界 (clamp/min) 仅在采样后应用
- **建议**: 在 generator 内部采样循环中也加入硬约束

---

## 三、Diffusion 架构专项评估

| 维度 | 评分 | 说明 |
|------|------|------|
| DDPM 基础 | ✅ | 前向/反向流程标准实现 |
| 条件机制 | ⚠️ | NWP 压缩过度 (关键瓶颈) |
| 物理感知 | ⚠️ | 采样过程无硬约束 |
| 长序列建模 | ⚠️ | 2 层 ResNet 能力有限 |

---

## 四、改进优先级

### P0 (必须修复)
1. dataset.py NWP 维度与 config 统一
2. dataset.py 添加异常数据过滤

### P1 (强烈建议)
3. generator.py NWP 条件注入改为序列级
4. forecasting.py 重命名或实现 TFT

### P2 (优化项)
5. generator.py 采样过程添加物理约束
6. 增加 ResNet 层数或引入 LSTM

---

## 五、修复示例

### 示例1: dataset.py 异常检测

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    # ... 现有逻辑 ...
    
    power_np = power.numpy().squeeze()
    ghi_np = ghi_clearsky.numpy().squeeze()
    
    # 异常: 夜间但高功率 或 白天零功率但高GHI
    is_daytime = (ghi_np > 100)
    has_anomaly = ((power_np == 0) & is_daytime).sum() / len(power_np)
    
    if has_anomaly > 0.3:  # 超过30%异常则跳过
        return self.__getitem__(np.random.randint(0, len(self)))
    
    return {...}
```

### 示例2: generator.py 改进 NWP 注入

```python
class EpsilonNet(nn.Module):
    def __init__(self, nwp_dim, site_latent_dim, hidden_dim):
        # ...
        # NWP 序列编码器
        self.nwp_encoder = nn.Sequential(
            nn.Conv1d(nwp_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )
        
    def forward(self, x_t, t, nwp, site_latent):
        x = x_t.transpose(1, 2)  # [B, 1, L]
        
        # NWP 序列级编码
        nwp_feat = self.nwp_encoder(nwp.transpose(1, 2))  # [B, d_model, L]
        
        # 沿通道拼接
        h = torch.cat([x, nwp_feat], dim=1)  # [B, 1+d_model, L]
        
        # 后续残差块处理
        h = self.block1(h, cond_global)
        # ...
```

---

## 六、总结

整体架构设计思路正确，模块化程度高，物理约束 (CPT Loss) 实现到位。主要缺陷集中在:

1. **数据管道**: NWP 维度不一致 + 缺异常过滤
2. **Diffusion**: NWP 条件过于简化 (核心瓶颈)
3. **下游任务**: 预测模型未达到 TFT 复杂度

建议按优先级依次修复 P0/P1 问题后，再进行训练验证。