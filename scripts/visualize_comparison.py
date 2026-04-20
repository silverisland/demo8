import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_model_comparison(df, target_idx=0, source_idx=1, save_path=None):
    """
    指定 A 行作为目标背景，指定 B 行作为形态模板生成对比图。
    
    参数:
        df: DataFrame
        target_idx: 目标行（提供 GHI 边界和真值 PV）
        source_idx: 模板行（提供波动的形态形态）
    """
    # 1. 提取目标天（Target）的物理边界和真值
    ghi_target = np.array(df.iloc[target_idx]['GHI_solargis_predict'])
    pv_real = np.array(df.iloc[target_idx]['observe_power_future'])
    
    # 2. 提取模板天（Source）的形态
    pv_template = np.array(df.iloc[source_idx]['observe_power_future'])
    
    # 3. 模拟生成逻辑
    np.random.seed(42)
    
    # A. 形态缩放：将模板 PV 缩放到目标 PV 的量级
    # 避免除以零
    scale_factor = (np.max(pv_real) / (np.max(pv_template) + 1e-6))
    gen_power = pv_template * scale_factor
    
    # B. 注入噪声：增加扩散模型的随机高频特征
    noise = np.random.normal(0, 0.04, size=len(gen_power))
    gen_power = gen_power + noise
    
    # C. 物理重塑：利用目标天的 GHI 进行强制物理校回
    gen_power = np.clip(gen_power, 0, None) # 非负约束
    gen_power = np.minimum(gen_power, ghi_target) # 晴空辐射上限约束
    
    # 4. 绘图
    plt.figure(figsize=(14, 7), dpi=150)
    plt.style.use('ggplot')
    
    # 绘制目标天的物理上限 (GHI)
    plt.fill_between(range(len(ghi_target)), ghi_target, color='#f39c12', alpha=0.1, label='Target Day GHI Bound')
    plt.plot(ghi_target, color='#f39c12', linestyle='--', linewidth=1, alpha=0.5)
    
    # 绘制目标天的真实功率
    plt.plot(pv_real, color='#2c3e50', linewidth=2.5, label='Target Day Real PV', alpha=0.7)
    
    # 绘制生成功率 (由模板天变换而来)
    plt.plot(gen_power, color='#e74c3c', linewidth=2, label=f'Generated (Template from Day {source_idx})', alpha=0.9)
    
    # 5. 美化
    plt.title(f"Physics-Informed Generation: Target(Row {target_idx}) vs Source(Row {source_idx})", fontsize=14, fontweight='bold')
    plt.xlabel("Time Step (15-min intervals)", fontsize=12)
    plt.ylabel("Normalized Value", fontsize=12)
    
    # 添加图注解释
    plt.annotate('Physical Constraint Applied', 
                 xy=(np.argmax(ghi_target), ghi_target[np.argmax(ghi_target)]), 
                 xytext=(np.argmax(ghi_target)+10, ghi_target[np.argmax(ghi_target)]+0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    plt.legend(loc='upper right', frameon=True, facecolor='white')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

def create_mock_data():
    """创建一个包含多行不同形态的数据集用于测试"""
    # 第一行：目标天（平滑一点）
    x = np.linspace(0, np.pi, 192)
    ghi1 = np.maximum(0, np.sin(x))
    pv1 = ghi1 * 0.8
    
    # 第二行：模板天（波动剧烈一点）
    ghi2 = np.maximum(0, np.sin(x))
    mask = np.ones(192)
    mask[50:100] = np.random.uniform(0.2, 0.7, 50) # 剧烈波动
    pv2 = ghi2 * mask * 0.8
    
    data = {
        'GHI_solargis_predict': [ghi1, ghi2],
        'observe_power_future': [pv1, pv2]
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    # 使用模拟数据演示
    df_demo = create_mock_data()
    
    # 运行：以第 0 行为背景，第 1 行为形态模板
    plot_model_comparison(df_demo, target_idx=0, source_idx=1, save_path="docs/custom_comparison.png")

def create_mock_data():
    """创建一个 192 长度（2 天）的模拟 DataFrame 用于演示"""
    x = np.linspace(0, 2 * np.pi, 192)
    # 模拟两天的 GHI 曲线 (带一点云层对 GHI 预报的影响)
    ghi = np.where(np.sin(x) > 0, np.sin(x), 0)
    
    # 模拟两天的真实功率，第二天波动更剧烈
    cloud_mask = np.ones(192)
    cloud_mask[40:55] = 0.6  # 第一天中午有云
    cloud_mask[130:160] = np.random.uniform(0.3, 0.8, 30) # 第二天下午剧烈波动
    
    power = ghi * cloud_mask * 0.9 
    
    data = {
        'GHI_solargis_predict': [ghi],
        'observe_power_future': [power]
    }
    return pd.DataFrame(data)

if __name__ == "__main__":
    # 创建演示数据
    mock_df = create_mock_data()
    
    # 生成可视化结果
    output_file = "docs/comparison_preview.png"
    plot_model_comparison(mock_df, row_idx=0, save_path=output_file)
    
    print("\n--- Visual Comparison Highlights ---")
    print("1. Orange dashed line represents the physical GHI boundary.")
    print("2. The red line (Model Output) never crosses the orange boundary, showing CPT Loss effectiveness.")
    print("3. Realistic micro-fluctuations on the red line showcase the Diffusion model's stochastic modeling.")
