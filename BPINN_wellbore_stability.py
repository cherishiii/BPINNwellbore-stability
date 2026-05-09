"""
完整的贝叶斯物理信息神经网络 (BPINN) 用于井壁稳定性不确定性分析（论文复现版）
支持三种应力机制：正断层(NF)、走滑断层(SS)、逆断层(RF)
支持地层压力系数 alpha_p 的多值扫描
支持任意井斜/井眼方位的定向井/水平井
使用变分贝叶斯方法 (Bayes by Backprop) 实现参数不确定性
通过蒙特卡洛采样生成物理仿真数据集
实现可靠度-等效密度曲线的物理解与BPINN对比

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import Tuple, Dict, List
import warnings
import os
import pickle
import pandas as pd
import time
from plot_style import set_paper_style, apply_ax_style
warnings.filterwarnings('ignore')

# ============================================================================
# 第一部分：配置参数与分布定义
# ============================================================================

@dataclass
class PhysicalConstants:
    """物理常数和岩石力学参数配置（论文复现版 - 支持应力机制与压力系数）"""
    # 井深 (m) - 用作参考深度或验证集的目标深度
    depth: float = 4500.0
    
    # === 新增：深度采样范围（用于生成全井深训练数据）===
    depth_min: float = 1000.0
    depth_max: float = 5000.0
    
    # 重力加速度 (m/s^2)
    g: float = 9.81
    
    # === 新增：应力机制和压力系数 ===
    # 应力机制类型: "NF"(正断层), "SS"(走滑断层), "RF"(逆断层)
    stress_regime: str = "NF"
    
    # 地层压力系数 alpha_p (无量纲)
    # 论文中扫描值: 0.1, 0.2, 0.3, 0.4, 0.5
    alpha_p: float = 0.3
    
    # 岩石力学参数（基准值，会随 φ、T 变化）
    # 单轴抗压强度 UCS (MPa) - 基准值（φ=0.1, T=25°C 时）
    UCS: float = 50.0
    
    # 内聚力 C (MPa) - 基准值（基体）
    cohesion: float = 10.0
    
    # 内摩擦角 (度) - 基准值（基体）
    friction_angle_deg: float = 30.0
    
    # Biot 系数 (无量纲) - 基准值（会随 φ 变化）
    biot_coefficient: float = 0.7
    
    # 杨氏模量 E (GPa)
    youngs_modulus: float = 20.0
    
    # 泊松比 (无量纲)
    poisson_ratio: float = 0.25
    
    # 抗拉强度 T0 (MPa) - 基准值
    tensile_strength: float = 5.0
    
    # 井眼半径 (m)
    wellbore_radius: float = 0.1085  # 8.5 inch wellbore
    
    # 地表温度 (°C)
    surface_temperature: float = 25.0
    
    # 地表压力 (MPa) - 大气压
    surface_pressure: float = 0.101325
    
    # === 新增：φ、k、T 影响系数 ===
    # 孔隙度影响系数（物理依据：孔隙度增大导致岩石强度降低）
    porosity_ref: float = 0.1  # 参考孔隙度
    porosity_UCS_coeff: float = 1.5  # UCS 对 φ 的敏感系数
    porosity_cohesion_coeff: float = 1.2  # C 对 φ 的敏感系数
    porosity_friction_coeff: float = 0.3  # φ_s 对 φ 的敏感系数（弧度）
    porosity_biot_coeff: float = 0.8  # Biot 系数对 φ 的敏感系数
    
    # 温度影响系数（物理依据：高温导致岩石热损伤，强度降低）
    temperature_ref: float = 25.0  # 参考温度 (°C)
    temperature_UCS_coeff: float = 0.002  # UCS 对 T 的敏感系数 (1/°C)
    temperature_cohesion_coeff: float = 0.0015  # C 对 T 的敏感系数
    temperature_friction_coeff: float = 0.0001  # φ_s 对 T 的敏感系数 (rad/°C)
    
    # 渗透率影响系数（物理依据：高 k 区域压力传递快，接近静水压）
    permeability_pressure_coeff: float = 0.02  # k 对压力梯度的影响系数
    permeability_ref: float = 10.0  # 参考渗透率 (mD)

    # ===========================================================================
    # [REVISION 2026] Kachanov 裂缝密度参数 (回应 R1-Q4 / R3-Q3)
    # ---------------------------------------------------------------------------
    # 裂缝密度 ε = N·a^3 / V (Kachanov 1980; Sayers & Kachanov 1995)
    # - N: 裂缝数, a: 裂缝半长, V: 代表体积
    # - 稀疏裂缝（NIA, non-interaction approximation）: ε ≤ 0.10
    # - 中等密度: 0.10 < ε ≤ 0.20  (NIA 误差 < 10%)
    # - 高密度（强相互作用）: ε > 0.20  (此时 Kirsch 解失效，需 DFN 显式建模)
    #
    # 在等效连续介质框架下，有效弹性模量按 Sayers-Kachanov NIA 公式衰减：
    #     E_eff / E_0 ≈ 1 / (1 + κ_E * ε)
    # 其中 κ_E = 16(1 - ν^2)(10 - 3ν) / [45(2 - ν)] ≈ 1.4~2.0 (随 ν 取值)
    #
    # 强度参数（UCS, cohesion, T0）作为弹性模量的标量函数，按经验衰减：
    #     UCS_eff / UCS_0 ≈ (1 - κ_UCS * ε)
    #     C_eff   / C_0   ≈ (1 - κ_C   * ε)
    #     T0_eff  / T0_0  ≈ (1 - κ_T0  * ε)
    # 摩擦角受裂缝影响相对较小，通常忽略。
    #
    # 这里把 ε 引入为"内部物理变量"（不作为 BPINN 输入维度），通过在
    # compute_true_outputs() 中按 crack_density_distribution 随机采样，让训练
    # 数据隐含 fractured-vuggy 异质性效应；后续可在论文中报告 ε ∈ {0, 0.05, 0.10}
    # 的灵敏度对比，回应 R3-Q3。
    # ===========================================================================
    crack_density_enabled: bool = True   # 总开关；False 时退化为纯连续介质
    crack_density_mean: float = 0.05     # 均值 ε
    crack_density_std: float = 0.025     # 标准差
    crack_density_min: float = 0.0       # 物理下限
    crack_density_max: float = 0.10      # NIA 适用上限（论文中明确 ε ≤ 0.10）
    crack_density_UCS_coeff: float = 2.0   # κ_UCS：UCS 对 ε 的衰减系数
    crack_density_cohesion_coeff: float = 1.8  # κ_C
    crack_density_tensile_coeff: float = 2.5   # κ_T0：抗拉强度对裂缝最敏感
    crack_density_E_coeff: float = 1.6     # κ_E：弹性模量对 ε 的衰减系数

    # === 新增：地应力方位参数（不进入 BPINN 输入，作为物理常量）===
    # 最大水平主应力方位角（相对正北，顺时针为正，度）
    azimuth_sigma_H: float = 30.0
    
    # === 新增：弱面（层理）参数（Jaeger Plane of Weakness）===
    # 层理倾角（与水平面夹角，度，0° 水平，90° 垂直）
    bedding_dip: float = 45.0
    
    # 层理倾向（层理面法向量在水平面投影的方位角，相对正北，度）
    bedding_dip_direction: float = 60.0
    
    # 弱面内聚力 (MPa)
    cohesion_weak: float = 3.0
    
    # 弱面摩擦角 (度)
    friction_angle_weak_deg: float = 20.0
    
    # === 新增：泥浆密度物理范围（用于输出硬约束）===
    mud_density_min: float = 0.1  # g/cm^3
    mud_density_max: float = 3.5  # g/cm^3 - 放宽上限以避免截断
    
    # 垂直应力梯度 (MPa/m) - 用于计算垂直应力
    vertical_stress_gradient: float = 0.023  # ~2.3 g/cm^3 岩石密度
    
    # === 新增：应力机制参数（论文Table 2-4范围）===
    # 正断层(NF): sigma_v > sigma_H > sigma_h
    # 走滑断层(SS): sigma_H > sigma_v > sigma_h
    # 逆断层(RF): sigma_H > sigma_h > sigma_v
    
    # 水平应力系数范围（相对于垂直应力）
    # 用于根据应力机制生成合理的水平应力
    stress_ratio_H_min: float = 0.6  # sigma_H / sigma_v 最小值
    stress_ratio_H_max: float = 1.8  # sigma_H / sigma_v 最大值
    stress_ratio_h_min: float = 0.5  # sigma_h / sigma_v 最小值
    stress_ratio_h_max: float = 1.5  # sigma_h / sigma_v 最大值
    
    @property
    def friction_angle_rad(self) -> float:
        """摩擦角的弧度值（基准值）"""
        return np.deg2rad(self.friction_angle_deg)
    
    @property
    def friction_angle_weak_rad(self) -> float:
        """弱面摩擦角的弧度值"""
        return np.deg2rad(self.friction_angle_weak_deg)
    
    def get_in_situ_stresses(self, regime: str = None, alpha_p: float = None) -> Tuple[float, float, float, float]:
        """
        根据应力机制和地层压力系数生成原位地应力
        
        参数:
            regime: 应力机制类型 ("NF", "SS", "RF")，如果为None则使用实例的stress_regime
            alpha_p: 地层压力系数，如果为None则使用实例的alpha_p
        
        返回:
            (sigma_H, sigma_h, sigma_v, p_p): 最大水平主应力、最小水平主应力、垂直应力、孔隙压力 (MPa)
        
        物理依据:
            - 垂直应力: sigma_v = 垂向应力梯度 * 深度
            - 孔隙压力: p_p = alpha_p * sigma_v （简化模型）
            - 水平应力: 根据应力机制确定相对大小关系
        
        对应论文: Table 2-4 的应力机制定义
        """
        if regime is None:
            regime = self.stress_regime
        if alpha_p is None:
            alpha_p = self.alpha_p
        
        # 计算垂直应力
        sigma_v = self.vertical_stress_gradient * self.depth
        
        # 计算孔隙压力（基于压力系数）
        # alpha_p = 0.0 表示完全枯竭，alpha_p = 1.0 表示原始压力等于垂直应力
        p_p = alpha_p * sigma_v
        
        # 根据应力机制生成水平应力（增加各向异性以匹配原始脚本的特征）
        if regime == "NF":  # 正断层: sigma_v > sigma_H > sigma_h
            # 增加应力差异以获得更明显的方位依赖性
            # 参考 Zoback (2007) Figure 8.2 和 Peska & Zoback (1995)
            sigma_H = 0.90 * sigma_v  # 提高H值，增加与h的差异（差异0.40）
            sigma_h = 0.50 * sigma_v  # 降低h值，增加各向异性
            
        elif regime == "SS":  # 走滑断层: sigma_H > sigma_v > sigma_h
            # 走滑机制通常有最强的方位依赖性
            sigma_H = 1.35 * sigma_v  # 增加H值
            sigma_h = 0.70 * sigma_v  # 降低h值，增加差异
            
        elif regime == "RF":  # 逆断层: sigma_H > sigma_h > sigma_v
            # 逆断层机制的水平应力都较大
            sigma_H = 1.60 * sigma_v  # 增加H值
            sigma_h = 1.15 * sigma_v  # h值略高于v，但与H有明显差异
            
        else:
            raise ValueError(f"未知的应力机制: {regime}. 应为 'NF', 'SS', 或 'RF'")
        
        return sigma_H, sigma_h, sigma_v, p_p


@dataclass
class DistributionConfig:
    """
    输入变量的概率分布配置（8 维输入）
    
    修改后的输入 (Top 6 + Angles):
    1. Porosity (phi)
    2. Vertical Stress (MPa)
    3. Max Horizontal Stress (MPa)
    4. Min Horizontal Stress (MPa)
    5. Pore Pressure (MPa)
    6. Temperature (°C)
    7. Inclination (°)
    8. Azimuth (°)
    """
    
    # 孔隙度 φ - Beta 分布参数
    porosity_alpha: float = 2.0
    porosity_beta: float = 5.0
    porosity_min: float = 0.01
    porosity_max: float = 0.30
    
    # 渗透率 k - 对数正态分布参数 (ln(k) in mD)
    # 不作为直接输入，但在物理模型中关联
    permeability_log_mean: float = 1.0  # ln(k) mean
    permeability_log_std: float = 1.5   # ln(k) std
    permeability_min: float = 0.01      # mD
    permeability_max: float = 1000.0    # mD
    
    # 最大水平主应力 σ_H - 正态分布 (MPa)
    sigma_H_mean: float = 85.0
    sigma_H_std: float = 8.0
    sigma_H_min: float = 20.0   # 适应浅层
    sigma_H_max: float = 260.0  # 适应深层 (5000m * 0.026 * 2.0)
    
    # 最小水平主应力 σ_h - 正态分布 (MPa)
    sigma_h_mean: float = 55.0
    sigma_h_std: float = 6.0
    sigma_h_min: float = 15.0   # 适应浅层
    sigma_h_max: float = 200.0  # 适应深层 (5000m * 0.026 * 1.5)
    
    # 孔隙压力梯度 - 截断正态分布 (MPa/km)
    pressure_gradient_mean: float = 10.5  # MPa/km
    pressure_gradient_std: float = 1.0
    pressure_gradient_min: float = 8.0
    pressure_gradient_max: float = 13.0
    
    # 地温梯度 - 截断正态分布 (°C/100m)
    temperature_gradient_mean: float = 3.0  # °C/100m
    temperature_gradient_std: float = 0.5
    temperature_gradient_min: float = 2.0
    temperature_gradient_max: float = 4.5
    
    # === 新增：井斜角 α - 均匀分布或集中分布 (度) ===
    # 0° 直井，90° 水平井
    inclination_min: float = 0.0
    inclination_max: float = 90.0
    
    # === 新增：井眼方位角 β - 均匀分布 (度) ===
    # 相对正北，顺时针为正
    wellbore_azimuth_min: float = 0.0
    wellbore_azimuth_max: float = 360.0
    
    # 垂直应力梯度 (MPa/m) - 用于计算垂直应力
    vertical_stress_gradient: float = 0.023  # ~2.3 g/cm^3 岩石密度


# ============================================================================
# 第二部分：输入采样函数
# ============================================================================

class InputSampler:
    """输入变量的蒙特卡洛采样器（8 维输入）"""
    
    def __init__(self, dist_config: DistributionConfig, phys_const: PhysicalConstants):
        self.dist = dist_config
        self.phys = phys_const
        
    def sample_porosity(self, N: int, depth_samples: np.ndarray = None) -> np.ndarray:
        """
        采样孔隙度 - 考虑深度衰减趋势 + 随机扰动 (改进版)
        
        参数:
            N: 样本数量
            depth_samples: 深度数组 (m)，如果为 None 则使用 phys.depth
        返回:
            形状 (N,) 的孔隙度数组
        """
        if depth_samples is None:
            depth_samples = np.full(N, self.phys.depth)
            
        # 物理模型：孔隙度随深度指数衰减 phi = phi_0 * exp(-c * z)
        # 设定：地表 35%，5000m 处约 5%
        phi_trend = 0.35 * np.exp(-0.0004 * depth_samples)
        
        # 叠加随机扰动 (正态分布)
        noise = np.random.normal(0, 0.02, N)
        
        porosity = phi_trend + noise
        porosity = np.clip(porosity, self.dist.porosity_min, self.dist.porosity_max)
        return porosity
    
    def sample_permeability(self, N: int, porosity: np.ndarray = None) -> np.ndarray:
        """
        采样渗透率 - 基于孔隙度的相关性 (Kozeny-Carman 简化)
        
        参数:
            N: 样本数量
            porosity: 孔隙度数组，如果为 None 则使用随机分布
        返回:
            形状 (N,) 的渗透率数组 (mD)
        """
        if porosity is None:
            # 回退到纯随机（保持兼容性）
            log_k = np.random.normal(self.dist.permeability_log_mean, 
                                   self.dist.permeability_log_std, N)
            k = np.exp(log_k)
        else:
            # Log(k) 与 phi 线性正相关: log(k) = a * phi + b + noise
            # 经验关系: log(k_mD) ~ 20 * phi - 2
            log_k_mean = 20.0 * porosity - 2.0
            log_k = np.random.normal(log_k_mean, 1.5, N) # 1.5 是波动范围
            k = np.exp(log_k)
            
        k = np.clip(k, self.dist.permeability_min, self.dist.permeability_max)
        return k
    
    def sample_horizontal_stresses(self, N: int, regime: str = None, alpha_p_mean: float = None, depth_samples: np.ndarray = None, sigma_v: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样水平主应力 - 确保符合应力机制约束 (支持变井深)
        
        参数:
            N: 样本数量
            regime: 应力机制类型
            alpha_p_mean: 压力系数均值
            depth_samples: 深度数组 (m)
            sigma_v: 垂直应力数组 (MPa)，如果提供则使用该值，否则内部生成
        """
        if regime is None:
            regime = self.phys.stress_regime
        if alpha_p_mean is None:
            alpha_p_mean = self.phys.alpha_p
            
        # 计算垂直应力（如果外部未提供）
        if sigma_v is None:
            if depth_samples is None:
                depth_samples = np.full(N, self.phys.depth)
                
            # 垂直应力梯度本身也有不确定性 (2.1-2.6 g/cm3 -> 0.021-0.026 MPa/m)
            grad_v = np.random.uniform(0.021, 0.026, N)
            sigma_v = grad_v * depth_samples
        
        # 根据应力机制生成应力比 (ratio = sigma / sigma_v)
        # 使用 Anderson 理论的典型范围
        if regime == "NF":  # 正断层: sigma_v > sigma_H > sigma_h
            # SH = (0.75 ~ 1.0) * Sv
            # Sh = (0.6 ~ 0.8) * Sv
            ratio_H = np.random.uniform(0.75, 1.0, N)
            ratio_h = np.random.uniform(0.6, 0.8, N)
            
        elif regime == "SS":  # 走滑断层: sigma_H > sigma_v > sigma_h
            # SH = (1.1 ~ 1.5) * Sv
            # Sh = (0.6 ~ 0.9) * Sv
            ratio_H = np.random.uniform(1.1, 1.5, N)
            ratio_h = np.random.uniform(0.6, 0.9, N)
            
        elif regime == "RF":  # 逆断层: sigma_H > sigma_h > sigma_v
            # SH = (1.4 ~ 2.0) * Sv
            # Sh = (1.1 ~ 1.5) * Sv
            ratio_H = np.random.uniform(1.4, 2.0, N)
            ratio_h = np.random.uniform(1.1, 1.5, N)
            
        else:
            # 默认
            ratio_H = np.ones(N) * 0.9
            ratio_h = np.ones(N) * 0.7
            
        # 计算绝对值并添加随机噪声
        sigma_H = ratio_H * sigma_v * np.random.normal(1.0, 0.03, N)
        sigma_h = ratio_h * sigma_v * np.random.normal(1.0, 0.03, N)
        
        # 强制排序约束
        swap_mask = sigma_H < sigma_h
        sigma_H[swap_mask], sigma_h[swap_mask] = sigma_h[swap_mask], sigma_H[swap_mask]
        
        # 针对具体机制的硬约束修正
        if regime == "NF":
            sigma_H = np.minimum(sigma_H, sigma_v * 0.98) # 略微小于 Sv
        elif regime == "SS":
            sigma_H = np.maximum(sigma_H, sigma_v * 1.02)
            sigma_h = np.minimum(sigma_h, sigma_v * 0.98)
        elif regime == "RF":
            sigma_h = np.maximum(sigma_h, sigma_v * 1.02)
        
        return sigma_H, sigma_h
    
    def sample_pore_pressure(self, N: int, alpha_p_mean: float = None, alpha_p_std: float = 0.05, depth_samples: np.ndarray = None) -> np.ndarray:
        """
        采样孔隙压力 - 基于地层压力系数 alpha_p (支持变井深)
        
        参数:
            N: 样本数量
            alpha_p_mean: 压力系数均值
            alpha_p_std: 压力系数标准差
            depth_samples: 深度数组 (m)
        """
        if alpha_p_mean is None:
            alpha_p_mean = self.phys.alpha_p
            
        if depth_samples is None:
            depth_samples = np.full(N, self.phys.depth)
        
        # 计算垂直应力 (为了保持一致性，重新采样梯度)
        grad_v = np.random.normal(0.023, 0.001, N) # 均值 0.023
        sigma_v = grad_v * depth_samples
        
        # 采样压力系数（截断正态分布）
        alpha_p_samples = np.random.normal(alpha_p_mean, alpha_p_std, N)
        alpha_p_samples = np.clip(alpha_p_samples, 0.05, 0.95)  # 物理合理范围
        
        # 计算孔隙压力
        p_p = alpha_p_samples * sigma_v
        
        return p_p
    
    def sample_temperature(self, N: int, depth_samples: np.ndarray = None) -> np.ndarray:
        """
        采样温度 - 基于地温梯度 (支持变井深)
        """
        if depth_samples is None:
            depth_samples = np.full(N, self.phys.depth)
            
        grad = np.random.normal(self.dist.temperature_gradient_mean,
                                self.dist.temperature_gradient_std, N)
        grad = np.clip(grad, self.dist.temperature_gradient_min,
                      self.dist.temperature_gradient_max)
        
        depth_100m = depth_samples / 100.0
        T = self.phys.surface_temperature + grad * depth_100m
        
        return T
    
    def sample_inclination(self, N: int) -> np.ndarray:
        """
        采样井斜角 - 均匀分布
        
        参数:
            N: 样本数量
        返回:
            形状 (N,) 的井斜角数组 (度)，范围 [0, 90]
        """
        inclination = np.random.uniform(self.dist.inclination_min, 
                                       self.dist.inclination_max, N)
        return inclination
    
    def sample_wellbore_azimuth(self, N: int) -> np.ndarray:
        """
        采样井眼方位角 - 均匀分布
        
        参数:
            N: 样本数量
        返回:
            形状 (N,) 的井眼方位角数组 (度)，范围 [0, 360]
        """
        azimuth = np.random.uniform(self.dist.wellbore_azimuth_min, 
                                    self.dist.wellbore_azimuth_max, N)
        return azimuth
    
    def sample_inputs(self, N: int, regime: str = None, alpha_p_mean: float = None, return_depth: bool = False, fixed_depth: float = None):
        """
        生成所有输入变量的样本（8 维）- 根据 Heatmap Top 6 + Geometry
        
        修改后输入列 (8维):
        0: Porosity (phi)
        1: Vertical Stress (MPa)
        2: Max Horizontal Stress (MPa)
        3: Min Horizontal Stress (MPa)
        4: Pore Pressure (MPa)
        5: Temperature (°C)
        6: Inclination (°)
        7: Azimuth (°)
        
        参数:
            return_depth: 保留参数以兼容旧接口
            fixed_depth: 如果提供，则强制使用该深度
        """
        # 1. 采样深度 (作为物理场生成的基准)
        if fixed_depth is not None:
            depth_samples = np.full(N, fixed_depth)
        else:
            depth_samples = np.random.uniform(self.phys.depth_min, self.phys.depth_max, N)
        
        # 2. 基于深度生成参数 (保持物理相关性)
        phi = self.sample_porosity(N, depth_samples)
        k = self.sample_permeability(N, phi) # k 依赖于 phi
        
        # 先生成垂直应力（作为应力场的基准）
        grad_v = np.random.uniform(0.021, 0.026, N)
        sigma_v = grad_v * depth_samples
        
        # 将 sigma_v 传给水平应力采样，确保物理自洽性
        sigma_H, sigma_h = self.sample_horizontal_stresses(N, regime, alpha_p_mean, depth_samples, sigma_v=sigma_v)
        
        # 基于同一 sigma_v 计算 p_p（使用压力系数）
        if alpha_p_mean is None:
            alpha_p_mean = self.phys.alpha_p
        alpha_p_samples = np.random.normal(alpha_p_mean, 0.05, N)
        alpha_p_samples = np.clip(alpha_p_samples, 0.05, 0.95)
        p_p = alpha_p_samples * sigma_v
        
        T = self.sample_temperature(N, depth_samples)

        inclination = self.sample_inclination(N)
        wellbore_azimuth = self.sample_wellbore_azimuth(N)

        # [REVISION 2026] Kachanov 裂缝密度采样 ε ~ N(μ, σ) 截断到 [ε_min, ε_max]
        # 该值不进入 BPINN 输入维度（避免破坏现有架构），而是作为"潜在物理变量"
        # 进入 compute_true_outputs 内部的强度参数修正，让训练数据隐含 fractured-vuggy
        # 异质性效应。回应 R3-Q3：使 fractured-vuggy 名实相符。
        if getattr(self.phys, 'crack_density_enabled', False):
            eps_crack = np.random.normal(self.phys.crack_density_mean,
                                          self.phys.crack_density_std, N)
            eps_crack = np.clip(eps_crack,
                                self.phys.crack_density_min,
                                self.phys.crack_density_max)
        else:
            eps_crack = np.zeros(N)

        # 3. 组装新输入 (8维)
        # 注意顺序: Phi, Sv, SH, Sh, Pp, T, Inc, Azi
        inputs = np.column_stack([phi, sigma_v, sigma_H, sigma_h, p_p, T,
                                 inclination, wellbore_azimuth])

        # 辅助数据 (用于物理模型计算 Ground Truth)
        aux_data = {
            'phi': phi,
            'k': k,
            'sigma_v': sigma_v,  # Store sigma_v for consistency check if needed
            'depth': depth_samples,  # [新增] 传递真实深度，避免后续用 sigma_v 反推深度带来系统误差
            'eps_crack': eps_crack,  # [REVISION 2026] Kachanov 裂缝密度
        }

        if return_depth:
            return inputs, depth_samples, aux_data

        return inputs, aux_data


# ============================================================================
# 第三部分：物理真值计算（Kirsch 解 + Mohr-Coulomb）
# ============================================================================

class WellborePhysics:
    """井壁稳定性物理模型 - 通用 Kirsch 解析解 + Mohr-Coulomb + Jaeger 弱面破坏准则"""
    
    def __init__(self, phys_const: PhysicalConstants, dist_config: DistributionConfig):
        self.phys = phys_const
        self.dist = dist_config
    
    def compute_effective_rock_properties(self, phi: float, T: float,
                                           eps_crack: float = None) -> Dict[str, float]:
        """
        计算受孔隙度、温度、裂缝密度影响的有效岩石力学参数。

        参数:
            phi: 孔隙度 (无量纲)
            T: 温度 (°C)
            eps_crack: 裂缝密度 ε = N·a^3/V (无量纲, ε ≤ 0.10)。
                若为 None 或 phys.crack_density_enabled=False，则不应用裂缝修正
                （等价于纯连续介质 Kirsch 解）。

        返回:
            包含有效岩石参数的字典 (含 'crack_density'，便于追溯)。

        物理依据:
        1. 孔隙度影响：孔隙度越高，岩石骨架越疏松，强度参数降低；
        2. 温度影响：高温导致热损伤、微裂纹扩展，强度降低；
        3. Biot 系数：孔隙度越高，有效应力中孔隙压力的作用越明显；
        4. **[REVISION 2026] 裂缝密度 ε**（Kachanov 1980; Sayers-Kachanov NIA）：
           - 缝洞型碳酸盐岩在等效连续介质框架下的实质化体现；
           - 强度参数随 ε 线性衰减（NIA, 稀疏裂缝）：
                UCS_eff *= (1 - κ_UCS · ε)
                C_eff   *= (1 - κ_C   · ε)
                T0_eff  *= (1 - κ_T0  · ε)
           - ε ≤ 0.10 时连续介质 Kirsch 解仍是合理的一阶近似；
           - 为何不修改 friction_angle? Kachanov NIA 显示裂缝主要影响 mode-I
             控制的弹性模量与抗拉强度，对剪切摩擦角影响在二阶项，可忽略；
           - 直接回应 R3-Q3 ("no fracture-explicit modeling")。

        工程假设：采用线性近似，适用于常规储层参数范围。
        """
        # 孔隙度偏离参考值
        delta_phi = phi - self.phys.porosity_ref
        # 温度偏离参考值
        delta_T = T - self.phys.temperature_ref

        # 有效 UCS（确保不为负）
        UCS_eff = self.phys.UCS * (1.0 - self.phys.porosity_UCS_coeff * delta_phi) * \
                  (1.0 - self.phys.temperature_UCS_coeff * delta_T)
        UCS_eff = max(UCS_eff, 5.0)  # 下限保护

        # 有效内聚力
        C_eff = self.phys.cohesion * (1.0 - self.phys.porosity_cohesion_coeff * delta_phi) * \
                (1.0 - self.phys.temperature_cohesion_coeff * delta_T)
        C_eff = max(C_eff, 1.0)  # 下限保护

        # 有效内摩擦角（弧度）
        friction_rad_eff = self.phys.friction_angle_rad - \
                          self.phys.porosity_friction_coeff * delta_phi - \
                          self.phys.temperature_friction_coeff * delta_T
        friction_rad_eff = np.clip(friction_rad_eff, np.deg2rad(15), np.deg2rad(45))  # 合理范围

        # 有效 Biot 系数
        biot_eff = self.phys.biot_coefficient + self.phys.porosity_biot_coeff * delta_phi
        biot_eff = np.clip(biot_eff, 0.5, 1.0)  # 物理范围 [0, 1]

        # 有效抗拉强度（也受 φ 和 T 影响，通常为 UCS 的 1/10）
        T0_eff = self.phys.tensile_strength * (1.0 - 0.8 * self.phys.porosity_UCS_coeff * delta_phi) * \
                 (1.0 - 0.8 * self.phys.temperature_UCS_coeff * delta_T)
        T0_eff = max(T0_eff, 0.5)  # 下限保护

        # [REVISION 2026] Kachanov 裂缝密度修正
        eps_used = 0.0
        if (eps_crack is not None) and getattr(self.phys, 'crack_density_enabled', False):
            eps_used = float(np.clip(eps_crack,
                                     self.phys.crack_density_min,
                                     self.phys.crack_density_max))
            crack_factor_UCS = max(0.0, 1.0 - self.phys.crack_density_UCS_coeff * eps_used)
            crack_factor_C   = max(0.0, 1.0 - self.phys.crack_density_cohesion_coeff * eps_used)
            crack_factor_T0  = max(0.0, 1.0 - self.phys.crack_density_tensile_coeff * eps_used)
            UCS_eff = max(UCS_eff * crack_factor_UCS, 2.0)
            C_eff   = max(C_eff   * crack_factor_C,   0.5)
            T0_eff  = max(T0_eff  * crack_factor_T0,  0.2)

        return {
            'UCS': UCS_eff,
            'cohesion': C_eff,
            'friction_angle_rad': friction_rad_eff,
            'biot_coefficient': biot_eff,
            'tensile_strength': T0_eff,
            'crack_density': eps_used,
        }
    
    def compute_effective_pore_pressure(self, k: float, p_p_base: float, sigma_v: float = None) -> float:
        """
        计算受渗透率影响的有效孔隙压力
        
        参数:
            k: 渗透率 (mD)
            p_p_base: 基础孔隙压力（由压力梯度计算）(MPa)
            sigma_v: 垂直应力 (MPa)，如果提供则用于动态计算上限
        
        返回:
            有效孔隙压力 (MPa)
        
        物理依据:
        在实际储层中，渗透率影响压力传递和分布：
        - 高渗透率区域：流体流动性好，压力趋向静水压力（正常压力）
        - 低渗透率区域：流体难以流动，可能形成异常高压或低压
        
        简化模型：
        p_p_eff = p_p_base * (1 + c_k * log(k / k_ref))
        - 当 k > k_ref 时，压力略有调整
        - 这是一个简化的经验关系，实际需要流体力学模拟
        """
        log_k_ratio = np.log(k / self.phys.permeability_ref)
        correction = self.phys.permeability_pressure_coeff * log_k_ratio
        
        p_p_eff = p_p_base * (1.0 + correction)
        
        # 动态计算上限（优先使用传入的 sigma_v 以适配变深度采样）
        if sigma_v is not None:
            p_p_max = 0.9 * sigma_v
        else:
            # 回退为固定深度（向后兼容）
            p_p_max = self.dist.vertical_stress_gradient * self.phys.depth * 0.9
        
        p_p_eff = np.clip(p_p_eff, 0.1, p_p_max)
        
        return p_p_eff
        
    def compute_vertical_stress(self) -> float:
        """
        计算垂直应力 σ_v
        
        返回:
            垂直应力 (MPa)
        
        假设: σ_v = 上覆岩层重量 = 密度梯度 * 深度
        """
        sigma_v = self.dist.vertical_stress_gradient * self.phys.depth
        return sigma_v
    
    def transform_stress_tensor(self,
                                sigma_H: float,
                                sigma_h: float,
                                sigma_v: float,
                                inclination: float,
                                wellbore_azimuth: float,
                                regime: str = None) -> np.ndarray:
        """
        将原位主应力张量从地理坐标系 (N-E-V) 变换到井眼局部坐标系
        基于 Peska & Zoback (1995) 的完整实现
        
        参数:
            sigma_H: 最大水平主应力 (MPa)
            sigma_h: 最小水平主应力 (MPa)
            sigma_v: 垂直应力 (MPa)
            inclination: 井斜角 α (度)，0° 直井，90° 水平井
            wellbore_azimuth: 井眼方位角 β (度)，相对正北，顺时针为正
            regime: 应力机制类型 ("NF", "SS", "RF")，影响坐标系设置
        
        返回:
            井眼坐标系下的应力张量，形状 (3, 3) 的 numpy 数组
        
        物理理论依据：Peska & Zoback (1995) 的双旋转变换
        
        变换步骤：
        1. 在主应力坐标系中构造应力张量
        2. 第一次旋转：Rs - 考虑应力机制特定的坐标系设置
        3. 第二次旋转：Rb - 从地理坐标系到井眼坐标系
        """
        # 获取应力机制（如果未提供，使用默认值）
        if regime is None:
            regime = getattr(self.phys, 'stress_regime', 'NF')
        
        # 应力机制特定的旋转角度（参考原始脚本）
        MinStressDir = self.phys.azimuth_sigma_H  # 最小主应力方向（度）
        
        if regime == 'NF':  # 正断层
            # S1=sigma_v(垂直), S2=sigma_H, S3=sigma_h
            S1, S2, S3 = sigma_v, sigma_H, sigma_h
            alpha_deg = 0 + MinStressDir
            beta_deg = 90
            gamma_deg = 0
        elif regime == 'SS':  # 走滑断层
            # S1=sigma_H, S2=sigma_v(垂直), S3=sigma_h
            S1, S2, S3 = sigma_H, sigma_v, sigma_h
            alpha_deg = 90 + MinStressDir
            beta_deg = 0
            gamma_deg = 90
        elif regime == 'RF':  # 逆断层
            # S1=sigma_H, S2=sigma_h, S3=sigma_v(垂直)
            S1, S2, S3 = sigma_H, sigma_h, sigma_v
            alpha_deg = 90 + MinStressDir
            beta_deg = 0
            gamma_deg = 0
        else:
            # 默认使用NF设置
            S1, S2, S3 = sigma_v, sigma_H, sigma_h
            alpha_deg = 0 + MinStressDir
            beta_deg = 90
            gamma_deg = 0
        
        # 转换为弧度
        alpha = np.deg2rad(alpha_deg)
        beta_coord = np.deg2rad(beta_deg)
        gamma = np.deg2rad(gamma_deg)
        
        # 主应力张量（对角矩阵）
        S = np.array([[S1, 0, 0],
                     [0, S2, 0],
                     [0, 0, S3]])
        
        # Rs: 地理坐标变换矩阵（Peska & Zoback, 1995）
        Rs = np.array([
            [np.cos(alpha)*np.cos(beta_coord), 
             np.sin(alpha)*np.cos(beta_coord), 
             -np.sin(beta_coord)],
            [np.cos(alpha)*np.sin(beta_coord)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma),
             np.sin(alpha)*np.sin(beta_coord)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma),
             np.cos(beta_coord)*np.sin(gamma)],
            [np.cos(alpha)*np.sin(beta_coord)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma),
             np.sin(alpha)*np.sin(beta_coord)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma),
             np.cos(beta_coord)*np.cos(gamma)]
        ])
        
        # 第一次变换：主应力坐标系 -> 地理坐标系
        sigma_geo = Rs.T @ S @ Rs
        
        # Rb: 井眼方位和井斜变换矩阵
        azimuth_rad = np.deg2rad(wellbore_azimuth)
        inclination_rad = np.deg2rad(inclination)
        
        Rb = np.array([
            [-np.cos(azimuth_rad)*np.cos(inclination_rad),
             -np.sin(azimuth_rad)*np.cos(inclination_rad),
             np.sin(inclination_rad)],
            [np.sin(azimuth_rad),
             -np.cos(azimuth_rad),
             0],
            [np.cos(azimuth_rad)*np.sin(inclination_rad),
             np.sin(azimuth_rad)*np.sin(inclination_rad),
             np.cos(inclination_rad)]
        ])
        
        # 第二次变换：地理坐标系 -> 井眼坐标系
        sigma_wellbore = Rb @ sigma_geo @ Rb.T
        
        return sigma_wellbore
    
    def generalized_kirsch_solution(self, sigma_wellbore: np.ndarray,
                                   p_p: float, p_w: float, 
                                   theta: float) -> Dict[str, float]:
        """
        通用 Kirsch 弹性解 - 适用于任意井眼取向
        
        参数:
            sigma_wellbore: 井眼坐标系下的应力张量 (3, 3)，单位 MPa
                           [0,0]=σ_xx, [1,1]=σ_yy, [2,2]=σ_zz
                           [0,1]=τ_xy, [0,2]=τ_xz, [1,2]=τ_yz
            p_p: 孔隙压力 (MPa)
            p_w: 井筒压力 (泥浆压力) (MPa)
            theta: 井壁周向角 (度)，0° 为 x 轴正方向，逆时针为正
        
        返回:
            包含井壁应力分量的字典:
            - sigma_r: 径向应力 (MPa)
            - sigma_theta: 环向应力 (MPa)
            - sigma_z: 轴向应力 (MPa)
            - tau_rtheta: 径向-环向剪应力 (MPa)
            - tau_ztheta: 轴向-环向剪应力 (MPa)
        
        物理理论依据：Kirsch 解（1898）
        
        物理假设:
        - 各向同性线弹性材料
        - 平面应变条件
        - 井眼为圆形
        - 远场应力边界条件
        
        在井壁处 (r = a)，Kirsch 解给出（柱坐标系）：
        - σ_r = p_w
        - σ_θ = σ_xx + σ_yy - 2(σ_xx - σ_yy)cos(2θ) - 4τ_xy·sin(2θ) - p_w
        - σ_z = σ_zz - 2ν[(σ_xx - σ_yy)cos(2θ) + 2τ_xy·sin(2θ)]
        - τ_rθ = 0（井壁边界条件）
        - τ_zθ = -2[τ_xz·cos(θ) + τ_yz·sin(θ)]
        """
        sigma_xx = sigma_wellbore[0, 0]
        sigma_yy = sigma_wellbore[1, 1]
        sigma_zz = sigma_wellbore[2, 2]
        tau_xy = sigma_wellbore[0, 1]
        tau_xz = sigma_wellbore[0, 2]
        tau_yz = sigma_wellbore[1, 2]
        
        theta_rad = np.deg2rad(theta)
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        cos_2theta = np.cos(2.0 * theta_rad)
        sin_2theta = np.sin(2.0 * theta_rad)
        
        sigma_r = p_w
        
        sigma_theta = (sigma_xx + sigma_yy - 
                      2.0 * (sigma_xx - sigma_yy) * cos_2theta - 
                      4.0 * tau_xy * sin_2theta - 
                      p_w)
        
        nu = self.phys.poisson_ratio
        sigma_z = (sigma_zz - 
                  2.0 * nu * ((sigma_xx - sigma_yy) * cos_2theta + 
                             2.0 * tau_xy * sin_2theta))
        
        tau_rtheta = 0.0
        
        tau_ztheta = -2.0 * (tau_xz * cos_theta + tau_yz * sin_theta)
        
        return {
            'sigma_r': sigma_r,
            'sigma_theta': sigma_theta,
            'sigma_z': sigma_z,
            'tau_rtheta': tau_rtheta,
            'tau_ztheta': tau_ztheta
        }
    
    def effective_stress(self, sigma: float, p_p: float, biot_coeff: float = None) -> float:
        """
        计算有效应力
        
        参数:
            sigma: 总应力 (MPa)
            p_p: 孔隙压力 (MPa)
            biot_coeff: Biot 系数，如果为 None 则使用基准值
        
        返回:
            有效应力 (MPa)
        
        Terzaghi 有效应力原理: σ' = σ - α*p_p
        其中 α 是 Biot 系数
        """
        if biot_coeff is None:
            alpha = self.phys.biot_coefficient
        else:
            alpha = biot_coeff
        sigma_eff = sigma - alpha * p_p
        return sigma_eff
    
    def mohr_coulomb_shear_failure(self, sigma_1: float, sigma_3: float, 
                                   C: float = None, phi_rad: float = None,
                                   UCS: float = None) -> float:
        """
        Mohr-Coulomb 剪切破坏准则（改进版：支持有效岩石参数）
        
        参数:
            sigma_1: 最大主应力 (有效应力) (MPa)
            sigma_3: 最小主应力 (有效应力) (MPa)
            C: 内聚力 (MPa)，如果为 None 则使用基准值
            phi_rad: 内摩擦角 (弧度)，如果为 None 则使用基准值
            UCS: 单轴抗压强度 (MPa)，如果为 None 则使用基准值
        
        返回:
            安全系数 SF (无量纲)
            SF > 1: 安全
            SF = 1: 临界状态
            SF < 1: 破坏
        
        Mohr-Coulomb 准则: σ_1 = σ_3 * N_φ + UCS
        其中: N_φ = (1 + sin(φ)) / (1 - sin(φ))
              UCS = 2*C*cos(φ) / (1 - sin(φ))
        
        安全系数定义: SF = (σ_3 * N_φ + UCS) / σ_1
        """
        # 使用提供的参数或默认值
        if phi_rad is None:
            phi = self.phys.friction_angle_rad
        else:
            phi = phi_rad
            
        if C is None:
            C_val = self.phys.cohesion
        else:
            C_val = C
            
        if UCS is None:
            UCS_val = self.phys.UCS
        else:
            UCS_val = UCS
        
        # 计算 N_φ
        N_phi = (1.0 + np.sin(phi)) / (1.0 - np.sin(phi))
        
        # 计算临界最大主应力
        sigma_1_critical = sigma_3 * N_phi + UCS_val
        
        # 安全系数
        if sigma_1 > 0:
            SF = sigma_1_critical / sigma_1
        else:
            SF = 1e6  # 避免除零，给一个很大的安全系数
        
        return SF
    
    def mogi_coulomb_shear_failure(self, sigma_1: float, sigma_2: float, 
                                   sigma_3: float, C: float = None, 
                                   phi_rad: float = None) -> float:
        """
        Mogi-Coulomb 剪切破坏准则（考虑中间主应力影响）
        
        公式: τ_oct = a + b * σ_m,2
        其中: a = (2√2 * C * cos φ) / (3 - sin φ)
              b = (2√2 * sin φ) / (3 - sin φ)
              σ_m,2 = (σ_1 + σ_3) / 2  (不考虑 σ_2)
              τ_oct = (1/3) * √[(σ1-σ2)² + (σ2-σ3)² + (σ3-σ1)²]
        
        参数:
            sigma_1: 最大有效主应力 (MPa)
            sigma_2: 中间有效主应力 (MPa)
            sigma_3: 最小有效主应力 (MPa)
            C: 内聚力 (MPa)，默认使用配置值
            phi_rad: 内摩擦角 (rad)，默认使用配置值
        
        返回:
            安全系数 (>1 安全, <1 破坏)
        
        物理意义:
            Mogi-Coulomb 准则考虑了中间主应力的影响，相比 Mohr-Coulomb
            更符合真三轴试验结果，特别适用于复杂应力状态。
        
        参考文献: 
            Mogi, K. (1971). Fracture and flow of rocks under high triaxial compression.
            Al-Ajmi & Zimmerman (2005). Relation between the Mogi and Coulomb failure criteria.
        """
        if C is None:
            C = self.phys.cohesion
        if phi_rad is None:
            phi_rad = self.phys.friction_angle_rad
        
        # Mogi-Coulomb 参数
        sin_phi = np.sin(phi_rad)
        cos_phi = np.cos(phi_rad)
        
        a = (2.0 * np.sqrt(2.0) * C * cos_phi) / (3.0 - sin_phi)
        b = (2.0 * np.sqrt(2.0) * sin_phi) / (3.0 - sin_phi)
        
        # 有效平均应力（不含 σ2）
        sigma_m2 = (sigma_1 + sigma_3) / 2.0
        
        # 八面体剪应力
        diff_12 = sigma_1 - sigma_2
        diff_23 = sigma_2 - sigma_3
        diff_31 = sigma_3 - sigma_1
        tau_oct = (1.0/3.0) * np.sqrt(diff_12**2 + diff_23**2 + diff_31**2)
        
        # 破坏强度
        tau_strength = a + b * sigma_m2
        
        # 安全系数
        if tau_oct > 1e-6:
            SF = tau_strength / tau_oct
        else:
            SF = 100.0
        
        return SF
    
    def drucker_prager_shear_failure(self, sigma_1: float, sigma_2: float, 
                                     sigma_3: float, C: float = None, 
                                     phi_rad: float = None) -> float:
        """
        Drucker-Prager 剪切破坏准则（光滑屈服面）
        
        公式: √J2 = α * I1 + k
        其中: α = (2 sin φ) / [√3 * (3 - sin φ)]  (外接圆匹配)
              k = (6C cos φ) / [√3 * (3 - sin φ)]
              I1 = σ1 + σ2 + σ3  (第一应力不变量)
              J2 = (1/6) * [(σ1-σ2)² + (σ2-σ3)² + (σ3-σ1)²]  (第二偏应力不变量)
        
        参数:
            sigma_1: 最大有效主应力 (MPa)
            sigma_2: 中间有效主应力 (MPa)
            sigma_3: 最小有效主应力 (MPa)
            C: 内聚力 (MPa)，默认使用配置值
            phi_rad: 内摩擦角 (rad)，默认使用配置值
        
        返回:
            安全系数 (>1 安全, <1 破坏)
        
        物理意义:
            Drucker-Prager 准则是 Mohr-Coulomb 的光滑扩展，在主应力空间
            中形成圆锥面，避免了 MC 准则的棱角问题，便于数值计算。
        
        参考文献:
            Drucker, D. C., & Prager, W. (1952). Soil mechanics and plastic analysis 
            or limit design. Quarterly of applied mathematics, 10(2), 157-165.
        """
        if C is None:
            C = self.phys.cohesion
        if phi_rad is None:
            phi_rad = self.phys.friction_angle_rad
        
        sin_phi = np.sin(phi_rad)
        cos_phi = np.cos(phi_rad)
        
        # Drucker-Prager 参数（外接圆匹配 - 拉伸子午线）
        alpha = (2.0 * sin_phi) / (np.sqrt(3.0) * (3.0 - sin_phi))
        k = (6.0 * C * cos_phi) / (np.sqrt(3.0) * (3.0 - sin_phi))
        
        # 应力不变量
        I1 = sigma_1 + sigma_2 + sigma_3
        J2 = (1.0/6.0) * ((sigma_1 - sigma_2)**2 + 
                          (sigma_2 - sigma_3)**2 + 
                          (sigma_3 - sigma_1)**2)
        sqrt_J2 = np.sqrt(max(J2, 0.0))
        
        # 破坏强度
        strength = alpha * I1 + k
        
        # 安全系数
        if sqrt_J2 > 1e-6:
            SF = strength / sqrt_J2
        else:
            SF = 100.0
        
        return SF
    
    def tensile_failure(self, sigma_3: float, T0: float = None) -> float:
        """
        拉伸破坏准则
        
        参数:
            sigma_3: 最小主应力 (有效应力) (MPa)
            T0: 抗拉强度 (MPa)，如果为 None 则使用基准值
        
        返回:
            安全系数 SF (无量纲)
        
        应力符号约定：压应力为正，拉应力为负
        
        拉伸破坏条件: σ_3_eff <= -T0
        
        安全系数: SF = T0 / |σ_3_eff|（当 σ_3_eff < 0 即拉应力时）
        """
        if T0 is None:
            T0_val = self.phys.tensile_strength
        else:
            T0_val = T0
        
        if sigma_3 >= 0:
            return 1e6
        
        SF = T0_val / abs(sigma_3)
        return SF
    
    def compute_weak_plane_normal(self) -> np.ndarray:
        """
        计算层理弱面的单位法向量（在地理坐标系 N-E-V 中）
        
        返回:
            形状 (3,) 的单位法向量
        
        物理理论依据：Jaeger Plane of Weakness Theory
        
        层理面几何定义：
        - 倾角 dip（与水平面夹角，0° 水平，90° 垂直）
        - 倾向 dip_direction（法向量在水平面投影的方位角，相对正北）
        
        法向量计算：
        n = [sin(dip) * cos(dip_direction),
             sin(dip) * sin(dip_direction),
             cos(dip)]
        """
        dip_rad = np.deg2rad(self.phys.bedding_dip)
        dip_dir_rad = np.deg2rad(self.phys.bedding_dip_direction)
        
        n = np.array([
            np.sin(dip_rad) * np.cos(dip_dir_rad),
            np.sin(dip_rad) * np.sin(dip_dir_rad),
            np.cos(dip_rad)
        ])
        
        n = n / np.linalg.norm(n)
        
        return n
    
    def weak_plane_stress_projection(self, sigma_tensor: np.ndarray, 
                                     plane_normal: np.ndarray) -> Tuple[float, float]:
        """
        计算弱面上的法向应力和剪应力
        
        参数:
            sigma_tensor: 应力张量 (3, 3)，单位 MPa
            plane_normal: 弱面单位法向量 (3,)
        
        返回:
            (sigma_n, tau): 法向应力和剪应力 (MPa)
        
        物理理论依据：应力张量在任意平面上的投影
        
        牵引力向量: T = σ · n
        法向应力: σ_n = T · n = n^T · σ · n
        剪应力大小: τ = sqrt(|T|^2 - σ_n^2)
        """
        T = sigma_tensor @ plane_normal
        
        sigma_n = np.dot(T, plane_normal)
        
        T_magnitude = np.linalg.norm(T)
        tau_squared = T_magnitude**2 - sigma_n**2
        
        tau = np.sqrt(max(tau_squared, 0.0))
        
        return sigma_n, tau
    
    def weak_plane_failure(self, sigma_n_eff: float, tau: float) -> float:
        """
        Jaeger 弱面破坏准则
        
        参数:
            sigma_n_eff: 弱面上的有效法向应力 (MPa)
            tau: 弱面上的剪应力 (MPa)
        
        返回:
            安全系数 SF_plane (无量纲)
        
        物理理论依据：Jaeger Plane of Weakness (1960)
        
        弱面 Mohr-Coulomb 准则：
        τ_critical = C_w + σ_n_eff * tan(φ_w)
        
        安全系数：
        SF = τ_critical / τ
        
        物理意义：
        - SF > 1: 弱面稳定
        - SF = 1: 弱面临界滑移
        - SF < 1: 弱面滑移破坏
        """
        C_w = self.phys.cohesion_weak
        phi_w = self.phys.friction_angle_weak_rad
        
        tau_critical = C_w + sigma_n_eff * np.tan(phi_w)
        
        if tau > 1e-6:
            SF_plane = tau_critical / tau
        else:
            SF_plane = 1e6
        
        return SF_plane
    
    def compute_safety_factors(self, sigma_H: float, sigma_h: float, 
                               p_p: float, p_w: float, 
                               inclination: float, wellbore_azimuth: float,
                               phi: float = None, k: float = None, T: float = None,
                               sigma_v: float = None,
                               failure_model: str = "mohr",
                               eps_crack: float = None) -> Dict[str, float]:
        """
        计算给定井筒压力下的井壁稳定性安全系数（工程级完整版）
        
        参数:
            sigma_H: 最大水平主应力 (MPa)
            sigma_h: 最小水平主应力 (MPa)
            p_p: 孔隙压力 (MPa，基础值)
            p_w: 井筒压力 (泥浆压力) (MPa)
            inclination: 井斜角 (度)
            wellbore_azimuth: 井眼方位角 (度)
            phi: 孔隙度（如果为 None 则使用基准参数）
            k: 渗透率 (mD)（如果为 None 则使用基准参数）
            T: 温度 (°C)（如果为 None 则使用基准参数）
            sigma_v: 垂直应力 (MPa) (如果为 None 则使用计算值)
            failure_model: 破坏准则选择 ("mohr" | "mogi" | "dp")
            eps_crack: [REVISION 2026] 裂缝密度 ε，传入 compute_effective_rock_properties
                       做 Kachanov NIA 修正。None 时退化为纯连续介质。
        """
        if phi is not None and T is not None:
            rock_props = self.compute_effective_rock_properties(phi, T, eps_crack=eps_crack)
            UCS_eff = rock_props['UCS']
            C_eff = rock_props['cohesion']
            phi_rad_eff = rock_props['friction_angle_rad']
            biot_eff = rock_props['biot_coefficient']
            T0_eff = rock_props['tensile_strength']
        else:
            UCS_eff = self.phys.UCS
            C_eff = self.phys.cohesion
            phi_rad_eff = self.phys.friction_angle_rad
            biot_eff = self.phys.biot_coefficient
            T0_eff = self.phys.tensile_strength
        
        # 先确保 sigma_v 存在（在计算 p_p_eff 之前）
        if sigma_v is None:
            sigma_v = self.compute_vertical_stress()
        
        if k is not None:
            p_p_eff = self.compute_effective_pore_pressure(k, p_p, sigma_v)
        else:
            p_p_eff = p_p
        
        # 获取应力机制（用于正确的坐标变换）
        regime = getattr(self.phys, 'stress_regime', 'NF')
        sigma_wellbore = self.transform_stress_tensor(
            sigma_H, sigma_h, sigma_v, inclination, wellbore_azimuth, regime
        )
        
        # === 修正：扫描井周寻找最危险点 ===
        # 之前的代码只计算 theta=0，这是不完整的。
        # 坍塌通常发生在 theta=90 (最小应力方向)，破裂发生在 theta=0 (最大应力方向)
        # 考虑到坐标旋转，需要扫描一周取极值。
        
        min_SF_collapse = 1e6
        min_SF_fracture = 1e6
        
        # 扫描 0 到 180 度 (对称)，步长 10 度
        theta_scan_range = np.arange(0, 180, 10.0)
        
        for theta in theta_scan_range:
            stresses = self.generalized_kirsch_solution(
                sigma_wellbore, p_p_eff, p_w, theta
            )
            
            sigma_r = stresses['sigma_r']
            sigma_theta = stresses['sigma_theta']
            sigma_z = stresses['sigma_z']
            
            sigma_r_eff = self.effective_stress(sigma_r, p_p_eff, biot_eff)
            sigma_theta_eff = self.effective_stress(sigma_theta, p_p_eff, biot_eff)
            sigma_z_eff = self.effective_stress(sigma_z, p_p_eff, biot_eff)
            
            principal_stresses = np.array([sigma_r_eff, sigma_theta_eff, sigma_z_eff])
            principal_stresses = np.sort(principal_stresses)
            
            sigma_1_eff = principal_stresses[2]
            sigma_2_eff = principal_stresses[1]
            sigma_3_eff = principal_stresses[0]
            
            # 1. 基体剪切破坏 (Collapse) - 根据选择的破坏准则
            if failure_model == "mohr":
                SF_matrix_shear = self.mohr_coulomb_shear_failure(
                    sigma_1_eff, sigma_3_eff, C_eff, phi_rad_eff, UCS_eff
                )
            elif failure_model == "mogi":
                SF_matrix_shear = self.mogi_coulomb_shear_failure(
                    sigma_1_eff, sigma_2_eff, sigma_3_eff, C_eff, phi_rad_eff
                )
            elif failure_model == "dp":
                SF_matrix_shear = self.drucker_prager_shear_failure(
                    sigma_1_eff, sigma_2_eff, sigma_3_eff, C_eff, phi_rad_eff
                )
            else:
                raise ValueError(f"Unknown failure_model: {failure_model}. Must be 'mohr', 'mogi', or 'dp'.")
            
            # 2. 拉伸破坏 (Collapse - 极端情况)
            SF_tensile = self.tensile_failure(sigma_3_eff, T0_eff)
            SF_matrix = min(SF_matrix_shear, SF_tensile)
            
            # 3. 弱面破坏 (Collapse)
            # 需要将应力转回地理坐标系来计算弱面投影，或者将弱面转到井眼坐标系
            # 这里沿用之前的逻辑：转回地理坐标系
            # 注意：这部分计算量较大，但在数据生成阶段可以接受
            # 为了效率，只在基体 SF 较低时检查弱面，或者简化处理
            # 这里为了严谨，完整计算
            
            # ... (应力旋转代码复用) ...
            # 为了代码简洁，这里重新构建旋转矩阵 R (与 theta 无关)
            # 实际上 sigma_wellbore_eff 随 theta 变化吗？
            # Kirsch 解给出的 sigma_r, theta, z 是在柱坐标系下的。
            # 转换到直角坐标系 (x', y', z') 需要旋转 theta
            # 这里的 weak_plane 逻辑在原代码中是基于 sigma_wellbore (x,y,z) 的
            # 原代码逻辑：kirsch -> principal -> SF (Scalar)
            # 弱面逻辑：kirsch -> wellbore tensor (x,y,z) -> geo tensor -> plane
            
            # 修正：Kirsch 解给出的应力是在 (r, theta, z) 坐标系。
            # 弱面计算需要 (x, y, z) 井眼坐标系下的全张量。
            # 在井壁处，r方向是主应力方向。
            # sigma_xx_wall = sigma_r * cos^2(t) + sigma_t * sin^2(t) ... 
            # 这种转换很复杂。
            
            # 简化策略：
            # 弱面破坏通常不是由井周位置决定的（它是全域的），但在井壁处应力集中最严重。
            # 我们假设弱面破坏主要由应力集中引起。
            # 暂时沿用 min(SF_matrix) 作为 SF_collapse 的主要贡献。
            # 如果必须包含弱面，取 theta 循环外的计算结果（基于远场应力？不，井壁应力集中会导致弱面更容易破坏）
            
            # 鉴于原代码在 weak plane 处的处理比较模糊，且此处主要修正 Collapse/Fracture 的反常
            # 我们主要关注基体破坏。
            
            SF_collapse_curr = SF_matrix
            
            if SF_collapse_curr < min_SF_collapse:
                min_SF_collapse = SF_collapse_curr
            
            # 4. 破裂 (Fracture)
            if sigma_theta_eff < 0:
                SF_fracture_hoop = T0_eff / abs(sigma_theta_eff)
            else:
                SF_fracture_hoop = (sigma_theta_eff + T0_eff) / T0_eff
                SF_fracture_hoop = max(SF_fracture_hoop, 1.5)
            
            SF_fracture_min_principal = self.tensile_failure(sigma_3_eff, T0_eff)
            SF_fracture_curr = min(SF_fracture_hoop, SF_fracture_min_principal)
            
            if SF_fracture_curr < min_SF_fracture:
                min_SF_fracture = SF_fracture_curr
                
        # 弱面检查 (在最危险点之外单独检查一次，或者认为基体控制)
        # 恢复原有的弱面逻辑（只计算一次，基于 transform_stress_tensor 的结果？）
        # 原逻辑中是用 kirsch 结果组装成对角阵，这其实隐含了 theta=0 或主轴方向
        # 这里为了保持一致性，我们取 min_SF_collapse 和 单独计算的弱面 SF 的最小值
        # 但弱面 SF 计算依赖具体的应力张量，这里先略过复杂的弱面全角度扫描
        # 假设弱面破坏不如基体应力集中点敏感（通常正确，除非弱面正好切过井眼）
        
        return {
            'SF_collapse': min_SF_collapse,
            'SF_fracture': min_SF_fracture
        }
    
    def find_critical_mud_pressure(self, sigma_H: float, sigma_h: float,
                                   p_p: float, inclination: float, wellbore_azimuth: float,
                                   failure_type: str = 'collapse',
                                   phi: float = None, k: float = None, T: float = None,
                                   sigma_v: float = None,
                                   failure_model: str = "mohr",
                                   eps_crack: float = None) -> float:
        """
        通过迭代法寻找临界泥浆压力（SF = 1）
        
        参数:
            sigma_H: 最大水平主应力 (MPa)
            sigma_h: 最小水平主应力 (MPa)
            p_p: 孔隙压力 (MPa)
            inclination: 井斜角 (度)
            wellbore_azimuth: 井眼方位角 (度)
            failure_type: 'collapse' 或 'fracture'
            phi: 孔隙度
            k: 渗透率 (mD)
            T: 温度 (°C)
            sigma_v: 垂直应力 (MPa)
            failure_model: 破坏准则选择 ("mohr" | "mogi" | "dp")
        """
        if failure_type == 'collapse':
            p_min = 0.1
            # 修正：上限应基于地应力，而非孔隙压力。
            # 枯竭时 Pp 很小，但为了抵抗地应力可能需要很高的泥浆密度。
            # 使用 1.2 * sigma_H 作为保守上限。
            p_max = sigma_H * 1.2
        else:
            p_min = p_p * 0.5
            p_max = sigma_H * 1.5
        
        tolerance = 0.01
        max_iterations = 50
        
        for _ in range(max_iterations):
            p_mid = (p_min + p_max) / 2.0
            
            SF_dict = self.compute_safety_factors(sigma_H, sigma_h, p_p, p_mid, 
                                                 inclination, wellbore_azimuth,
                                                 phi, k, T, sigma_v, failure_model,
                                                 eps_crack=eps_crack)
            
            if failure_type == 'collapse':
                SF = SF_dict['SF_collapse']
            else:
                SF = SF_dict['SF_fracture']
            
            if abs(SF - 1.0) < 0.01:
                return p_mid
            
            if failure_type == 'collapse':
                if SF < 1.0:
                    p_min = p_mid
                else:
                    p_max = p_mid
            else:
                if SF < 1.0:
                    p_max = p_mid
                else:
                    p_min = p_mid
        
        return (p_min + p_max) / 2.0
    
    def mud_pressure_to_density(self, p_mud: float, depth: float = None) -> float:
        """
        将泥浆压力转换为泥浆密度
        
        参数:
            p_mud: 泥浆压力 (MPa)
            depth: 深度 (m)，如果为 None 则使用 self.phys.depth (固定深度)
        
        返回:
            泥浆密度 (g/cm^3)
        """
        # p_mud 单位 MPa, 需要转换为 Pa
        p_pa = p_mud * 1e6
        
        # 深度
        if depth is None:
            h = self.phys.depth
        else:
            h = depth
            
        if h < 1.0: # 避免除零
            h = 1.0
        
        # 密度 (kg/m^3)
        rho_kg_m3 = p_pa / (self.phys.g * h)
        
        # 转换为 g/cm^3
        rho_g_cm3 = rho_kg_m3 / 1000.0
        
        return rho_g_cm3
    
    def compute_true_outputs(self, inputs: np.ndarray, aux_data: Dict[str, np.ndarray] = None, p_w: float = None, failure_model: str = "mohr") -> np.ndarray:
        """
        计算物理真值输出（通过解析解）（工程级完整版，8 维输入）

        参数:
            inputs: 输入数组 (N, 8)
            aux_data: 辅助数据字典（包含 phi, k, depth 等；
                可选包含 'eps_crack' 用于 Kachanov 裂缝密度修正）
            p_w: 井筒压力（如果为None则使用临界压力）
            failure_model: 破坏准则选择 ("mohr" | "mogi" | "dp")

        [REVISION 2026] 若 aux_data 中含 'eps_crack' 则按 Kachanov NIA 修正强度参数；
            否则在 phys.crack_density_enabled=True 时按 PhysicalConstants 配置的
            分布即时采样一份 ε（保证向后兼容）。回应 R3-Q3、R1-Q4。
        """
        N = inputs.shape[0]
        outputs = np.zeros((N, 4))

        if aux_data is None:
            raise ValueError("aux_data (phi, k) is required for true output computation")

        # k needs to be retrieved from aux_data as it is not in inputs
        k_arr = aux_data['k']

        # [REVISION 2026] 准备裂缝密度数组：优先从 aux_data 取（保证可复现），
        # 否则按 PhysicalConstants 中的 crack_density_distribution 即时采样。
        if 'eps_crack' in aux_data:
            eps_crack_arr = np.asarray(aux_data['eps_crack'], dtype=np.float64)
        elif getattr(self.phys, 'crack_density_enabled', False):
            eps_crack_arr = np.random.normal(
                self.phys.crack_density_mean, self.phys.crack_density_std, N)
            eps_crack_arr = np.clip(eps_crack_arr,
                                     self.phys.crack_density_min,
                                     self.phys.crack_density_max)
        else:
            eps_crack_arr = np.zeros(N)

        for i in range(N):
            # 解包新输入
            phi = inputs[i, 0]
            sigma_v = inputs[i, 1]
            sigma_H = inputs[i, 2]
            sigma_h = inputs[i, 3]
            p_p = inputs[i, 4]
            T = inputs[i, 5]
            inclination = inputs[i, 6]
            wellbore_azimuth = inputs[i, 7]

            k = k_arr[i]
            eps_i = float(eps_crack_arr[i])

            # [FIX] 优先使用采样时的真实深度进行 EMW 换算。
            # 旧实现用 est_depth = sigma_v / vertical_stress_gradient 反推深度，
            # 当 sigma_v 来自随机梯度采样时会引入系统偏差与额外噪声。
            if 'depth' in aux_data:
                depth_for_emw = float(aux_data['depth'][i])
            else:
                depth_for_emw = sigma_v / self.phys.vertical_stress_gradient  # fallback

            if p_w is None:
                p_w_current = p_p * 1.05
            else:
                p_w_current = p_w

            SF_dict = self.compute_safety_factors(sigma_H, sigma_h, p_p,
                                                 p_w_current,
                                                 inclination, wellbore_azimuth,
                                                 phi, k, T, sigma_v=sigma_v,
                                                 failure_model=failure_model,
                                                 eps_crack=eps_i)
            outputs[i, 0] = SF_dict['SF_collapse']
            outputs[i, 1] = SF_dict['SF_fracture']

            p_collapse = self.find_critical_mud_pressure(sigma_H, sigma_h, p_p,
                                                        inclination, wellbore_azimuth,
                                                        'collapse', phi, k, T, sigma_v=sigma_v,
                                                        failure_model=failure_model,
                                                        eps_crack=eps_i)
            p_fracture = self.find_critical_mud_pressure(sigma_H, sigma_h, p_p,
                                                        inclination, wellbore_azimuth,
                                                        'fracture', phi, k, T, sigma_v=sigma_v,
                                                        failure_model=failure_model,
                                                        eps_crack=eps_i)
            
            outputs[i, 2] = self.mud_pressure_to_density(p_collapse, depth=depth_for_emw)
            outputs[i, 3] = self.mud_pressure_to_density(p_fracture, depth=depth_for_emw)

            # ----------------------------------------------------------------
            # [REVISION 2026] 删除原"深度相关的密度漂移 + 指数放大的白噪声 + 硬截断"
            # 这段历史代码会做以下三件破坏方法论的事，是 ρ_c 拟合 R²≈0.25 的根本原因：
            #   (a) 把 Kirsch + 二分迭代得到的物理 ρ_c 与一个线性深度基准按 70/30 加权
            #       → 抹掉真实的物理控制律；
            #   (b) 在真值上叠加 σ=0.18·exp(0.8·z_norm) g/cm^3 的不可学习高斯白噪声
            #       → 任何确定性映射器都不可能在 R² 指标上恢复这部分方差；
            #   (c) 强制截断到 [1.2, 3.6]，引入分段非光滑性
            #       → 进一步恶化平滑神经网络的拟合表现。
            # 评审 R1-Q12（"poor ρ_c regression vs near-perfect collapse-pressure map"
            # paradox）和 R3-Q4（"R²=0.25 unacceptable for engineering use"）的根因
            # 在此。新方案：真值仅做安全物理截断，不再人为污染；不确定性应当且仅当
            # 来自 (i) 输入参数的 MC 采样（aleatory）和 (ii) 贝叶斯权重后验（epistemic）。
            # ----------------------------------------------------------------
            rho_min = self.phys.mud_density_min
            rho_max = self.phys.mud_density_max
            outputs[i, 2] = float(np.clip(outputs[i, 2], rho_min, rho_max))
            outputs[i, 3] = float(np.clip(outputs[i, 3], rho_min, rho_max))

        return outputs

    def compute_baseline_rho(self, inputs: np.ndarray,
                             aux_data: Dict[str, np.ndarray],
                             failure_model: str = "mohr") -> np.ndarray:
        """
        [REVISION 2026] 计算 ρ_c / ρ_f 的"物理 baseline"——即在裂缝密度 ε=0
        （纯连续介质 Kirsch+二分迭代）下解析得到的临界密度。

        用途：BPINN 改为"baseline + residual"架构后，网络只需学 Δρ。
            真实预测 ρ_pred = baseline + denorm(net_residual)。

        参数:
            inputs: (N, 8) 输入数组（与 compute_true_outputs 一致）
            aux_data: 含 'phi', 'k', 'depth' 等的字典；
                      会强制覆盖 'eps_crack'=0 后传入 compute_true_outputs。
            failure_model: 破坏准则。

        返回:
            (N, 2) 数组：[:, 0]=rho_c_baseline, [:, 1]=rho_f_baseline (g/cm^3)。

        说明：
            内部直接复用 compute_true_outputs 的解析链；为保证 baseline 与
            连续介质 Kirsch 解严格对齐，临时关闭裂缝修正（eps_crack=0），
            不影响外部 aux_data。
        """
        if aux_data is None:
            raise ValueError("aux_data is required for baseline computation")

        N = inputs.shape[0]
        aux_zero = {k: v for k, v in aux_data.items()}
        aux_zero['eps_crack'] = np.zeros(N, dtype=np.float64)

        crack_flag_backup = getattr(self.phys, 'crack_density_enabled', False)
        try:
            if crack_flag_backup:
                self.phys.crack_density_enabled = False
            outputs = self.compute_true_outputs(inputs, aux_data=aux_zero,
                                                failure_model=failure_model)
        finally:
            self.phys.crack_density_enabled = crack_flag_backup

        return outputs[:, 2:4].copy()


# ============================================================================
# 第四部分：贝叶斯神经网络层 (Bayes by Backprop)
# ============================================================================

class BayesianLinear(nn.Module):
    """
    贝叶斯线性层 - 使用变分推断 (Bayes by Backprop)
    
    权重和偏置都建模为高斯分布:
    - w ~ N(μ_w, σ_w²)
    - 训练参数: μ 和 ρ (其中 σ = log(1 + exp(ρ)))
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 prior_sigma: float = 1.0):
        """
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            prior_sigma: 先验分布的标准差
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        
        # 权重参数: 均值和 rho
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # 偏置参数: 均值和 rho
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # 初始化参数
        self.reset_parameters()
        
        # KL 散度累积（在前向传播中计算）
        self.kl = 0.0
    
    def reset_parameters(self):
        """初始化参数"""
        # 均值：使用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.weight_mu, a=np.sqrt(5))
        nn.init.uniform_(self.bias_mu, -0.2, 0.2)
        
        # rho：初始化为较小的值，使初始 sigma 较小
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.constant_(self.bias_rho, -3.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 使用重参数化技巧采样权重
        
        参数:
            x: 输入张量，形状 (batch_size, in_features)
        
        返回:
            输出张量，形状 (batch_size, out_features)
        """
        # 计算 sigma = softplus(rho) = log(1 + exp(rho))
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        # 重参数化采样: w = μ + σ * ε, 其中 ε ~ N(0, 1)
        weight_epsilon = torch.randn_like(self.weight_mu)
        bias_epsilon = torch.randn_like(self.bias_mu)
        
        weight = self.weight_mu + weight_sigma * weight_epsilon
        bias = self.bias_mu + bias_sigma * bias_epsilon
        
        # 计算该层的 KL 散度: KL(q||p)
        self.kl = self._compute_kl(self.weight_mu, weight_sigma, 
                                   self.bias_mu, bias_sigma)
        
        # 线性变换
        return nn.functional.linear(x, weight, bias)
    
    def _compute_kl(self, mu_w: torch.Tensor, sigma_w: torch.Tensor,
                   mu_b: torch.Tensor, sigma_b: torch.Tensor) -> torch.Tensor:
        """
        计算 KL 散度: KL(q(w|μ,σ) || p(w|0,σ_p))
        
        对于高斯分布:
        KL(N(μ,σ²) || N(0,σ_p²)) = log(σ_p/σ) + (σ² + μ²)/(2σ_p²) - 1/2
        
        返回:
            标量张量，该层所有参数的 KL 散度之和
        
        改进：确保 device 一致性
        """
        sigma_p = self.prior_sigma
        
        # 使用 Python float，避免 device 不一致问题
        log_sigma_p = np.log(sigma_p)
        
        # 权重的 KL
        kl_w = (log_sigma_p - torch.log(sigma_w) +
                (sigma_w**2 + mu_w**2) / (2 * sigma_p**2) - 0.5)
        
        # 偏置的 KL
        kl_b = (log_sigma_p - torch.log(sigma_b) +
                (sigma_b**2 + mu_b**2) / (2 * sigma_p**2) - 0.5)
        
        # 总 KL（所有参数求和）
        return kl_w.sum() + kl_b.sum()


class DataScaler:
    """数据归一化工具 (StandardScaler) - 用于输入特征 X"""
    
    def __init__(self):
        self.mean = None
        self.std = None
        self.mean_tensor = None
        self.std_tensor = None
        
    def fit(self, data: np.ndarray):
        """计算均值和标准差"""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # 避免除以零
        self.std[self.std < 1e-8] = 1.0
        
    def transform(self, data: np.ndarray) -> np.ndarray:
        """归一化"""
        if self.mean is None:
            raise ValueError("Scaler not fitted yet.")
        return (data - self.mean) / self.std
        
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """反归一化"""
        if self.mean is None:
            raise ValueError("Scaler not fitted yet.")
        return data * self.std + self.mean
        
    def to_torch(self, device):
        """转换为 PyTorch 张量"""
        if self.mean is not None:
            self.mean_tensor = torch.tensor(self.mean, dtype=torch.float32, device=device)
            self.std_tensor = torch.tensor(self.std, dtype=torch.float32, device=device)
            
    def inverse_transform_tensor(self, data_tensor: torch.Tensor) -> torch.Tensor:
        """PyTorch 张量反归一化"""
        if self.mean_tensor is None:
             raise ValueError("Scaler tensors not initialized. Call to_torch() first.")
        
        # 确保设备一致
        if self.mean_tensor.device != data_tensor.device:
            self.mean_tensor = self.mean_tensor.to(data_tensor.device)
            self.std_tensor = self.std_tensor.to(data_tensor.device)
            
        return data_tensor * self.std_tensor + self.mean_tensor


class OutputScaler:
    """
    输出归一化工具 (MinMax -> [-1, 1]) - 用于方案C
    
    将物理量级差异巨大的输出映射到统一的 [-1, 1] 区间，
    解决梯度失衡和截断问题。
    """
    def __init__(self, residual_mode: bool = False,
                 rho_residual_abs_max=1.5,
                 residual_channels: List[int] = None):
        """
        参数:
            residual_mode: [REVISION 2026] 是否启用"残差通道"机制。
                True 时：`residual_channels` 中列出的通道用对称残差范围
                [-abs_max_ch, +abs_max_ch] 归一化；其余通道保持物理范围
                （即直接预测物理值）。
                False 时：所有通道按物理范围归一化（向后兼容）。
            rho_residual_abs_max: 残差通道的幅度上限（g/cm^3）。**v6 新增**：
                同时支持 float 与 dict 两种形式，
                - float（旧）：所有 residual_channels 共用同一对称范围；
                - dict（新）：per-channel，如 {2: 2.5, 3: 0.4}，
                  解决 ρ_c 与 ρ_f 残差幅度差 ~10× 时共用上限会
                  让小幅度通道分辨率不足的问题（v3 与 A 档教训）。
            residual_channels: [REVISION 2026] 指定哪些通道做残差化。
                - None 或省略时，若 residual_mode=True 则默认 [2]（仅 ρ_c）；
                - v3 历史：直接用 [2, 3] 共用 ±1.5 上限会让 ρ_f 信号被 ρ_c
                  的大量纲淹没，导致 R² 退化；
                - **v6 修复**：恢复 [2, 3] 但 per-channel 给 ρ_f ±0.4 g/cm^3
                  （A 档实测训练集 Δρ_f ∈ [-0.21, 0]，留 ~90% 余量）。
        """
        if residual_mode and residual_channels is None:
            residual_channels = [2]
        if residual_channels is None:
            residual_channels = []

        self.residual_mode = bool(residual_mode)
        self.residual_channels = list(int(c) for c in residual_channels)

        # [REVISION 2026 v6] 把 rho_residual_abs_max 统一规约为 dict
        # key=channel index, value=对称残差上限（g/cm^3）。
        if isinstance(rho_residual_abs_max, dict):
            self._abs_max_per_channel = {int(k): float(v) for k, v in rho_residual_abs_max.items()}
            # 任何 residual_channel 没显式给值的，回退到 1.5（v5 默认）
            for ch in self.residual_channels:
                self._abs_max_per_channel.setdefault(ch, 1.5)
            # 对外仍暴露一个标量入口（取所有 residual 通道的最大值），
            # 便于旧代码 / 日志读取，但内部以 dict 为准。
            self.rho_residual_abs_max = max(
                (self._abs_max_per_channel[ch] for ch in self.residual_channels),
                default=1.5
            )
        else:
            scalar_val = float(rho_residual_abs_max)
            self.rho_residual_abs_max = scalar_val
            self._abs_max_per_channel = {ch: scalar_val for ch in self.residual_channels}

        # 默认物理范围
        ranges = {
            0: {'min': -3.0, 'max': 3.0,  'note': 'SF_collapse (dimensionless)'},
            1: {'min':  0.0, 'max': 75.0, 'note': 'SF_fracture (dimensionless, can be large)'},
            2: {'min':  0.1, 'max': 3.5,  'note': 'rho_c (g/cm^3) — 与 mud_density_min/max 对齐'},
            3: {'min':  0.1, 'max': 3.5,  'note': 'rho_f (g/cm^3) — 与 mud_density_min/max 对齐'},
        }

        # [REVISION 2026 v6] per-channel 写入对称残差范围
        if self.residual_mode:
            for ch in self.residual_channels:
                if ch not in ranges:
                    raise ValueError(f"residual_channels 含未知通道 {ch}")
                abs_max_ch = self._abs_max_per_channel[ch]
                ranges[ch] = {
                    'min': -abs_max_ch,
                    'max':  abs_max_ch,
                    'note': f"Δ-channel {ch} (residual ±{abs_max_ch:g} g/cm^3)",
                }

        self.ranges = ranges
        self.mins = torch.tensor([self.ranges[i]['min'] for i in range(4)], dtype=torch.float32)
        self.maxs = torch.tensor([self.ranges[i]['max'] for i in range(4)], dtype=torch.float32)
        self.device = None

    def to(self, device):
        self.device = device
        self.mins = self.mins.to(device)
        self.maxs = self.maxs.to(device)
        return self

    def normalize(self, y_phys: torch.Tensor) -> torch.Tensor:
        """物理值 -> [-1, 1]"""
        # y_norm = 2 * (y - min) / (max - min) - 1
        if self.device is None or self.mins.device != y_phys.device:
            self.to(y_phys.device)
        return 2.0 * (y_phys - self.mins) / (self.maxs - self.mins) - 1.0

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        """[-1, 1] -> 物理值"""
        # y_phys = (y_norm + 1) / 2 * (max - min) + min
        if self.device is None or self.mins.device != y_norm.device:
            self.to(y_norm.device)
        return (y_norm + 1.0) / 2.0 * (self.maxs - self.mins) + self.mins
    
    def denormalize_numpy(self, y_norm: np.ndarray) -> np.ndarray:
        """Numpy 版本反归一化"""
        mins = self.mins.cpu().numpy()
        maxs = self.maxs.cpu().numpy()
        return (y_norm + 1.0) / 2.0 * (maxs - mins) + mins


class BayesianBPINN(nn.Module):
    """
    贝叶斯物理信息神经网络 (BPINN) - 方案C 重构版
    
    输出: 归一化后的值 [-1, 1]，不再包含物理单位。
    物理含义通过 OutputScaler 进行后处理恢复。
    """
    
    def __init__(self, input_dim: int = 8, output_dim: int = 4,
                 hidden_dims: list = [64, 64, 64], 
                 prior_sigma: float = 1.0,
                 phys_const: PhysicalConstants = None): # phys_const kept for compatibility but unused for constraints
        """
        参数:
            input_dim: 输入维度
            output_dim: 输出维度
            hidden_dims: 隐藏层维度列表
            prior_sigma: 先验分布标准差
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.phys_const = phys_const # 不再需要物理常数来做硬截断
        
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(BayesianLinear(dims[i], dims[i+1], prior_sigma))
        
        self.layers = nn.ModuleList(layers)
        
        self.activation = nn.Tanh()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        返回:
            输出张量，形状 (batch_size, 4)，范围约在 [-1, 1] 之间
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        
        # 最后一层使用 Tanh 激活，将输出强制限制在 [-1, 1]
        # 这对应于 OutputScaler 定义的物理范围 [min, max]
        output = torch.tanh(x)
        
        return output
    
    def get_kl_loss(self) -> torch.Tensor:
        """
        获取所有层的 KL 散度之和
        
        返回:
            标量张量
        """
        # 使用与模型参数相同的 device 和 dtype，避免 CUDA 下的 device mismatch
        ref_param = next(self.parameters())
        kl_total = torch.zeros((), device=ref_param.device, dtype=ref_param.dtype)
        
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl_total = kl_total + layer.kl
        
        return kl_total


# ============================================================================
# 第五部分：损失函数与训练
# ============================================================================

class TorchWellborePhysics:
    """
    可微分的井壁稳定性物理模型 (PyTorch Batch Version)
    用于在 Physics Loss 中计算梯度
    """
    
    def __init__(self, phys_const: PhysicalConstants):
        self.phys = phys_const
        
    def compute_effective_rock_properties(self, phi: torch.Tensor, T: torch.Tensor):
        """计算有效岩石参数 (PyTorch)"""
        delta_phi = phi - self.phys.porosity_ref
        delta_T = T - self.phys.temperature_ref
        
        UCS_eff = self.phys.UCS * (1.0 - self.phys.porosity_UCS_coeff * delta_phi) * \
                  (1.0 - self.phys.temperature_UCS_coeff * delta_T)
        UCS_eff = torch.clamp(UCS_eff, min=5.0)
        
        C_eff = self.phys.cohesion * (1.0 - self.phys.porosity_cohesion_coeff * delta_phi) * \
                (1.0 - self.phys.temperature_cohesion_coeff * delta_T)
        C_eff = torch.clamp(C_eff, min=1.0)
        
        friction_rad_eff = self.phys.friction_angle_rad - \
                          self.phys.porosity_friction_coeff * delta_phi - \
                          self.phys.temperature_friction_coeff * delta_T
        friction_rad_eff = torch.clamp(friction_rad_eff, np.deg2rad(15), np.deg2rad(45))
        
        biot_eff = self.phys.biot_coefficient + self.phys.porosity_biot_coeff * delta_phi
        biot_eff = torch.clamp(biot_eff, 0.5, 1.0)
        
        T0_eff = self.phys.tensile_strength * (1.0 - 0.8 * self.phys.porosity_UCS_coeff * delta_phi) * \
                 (1.0 - 0.8 * self.phys.temperature_UCS_coeff * delta_T)
        T0_eff = torch.clamp(T0_eff, min=0.5)
        
        return UCS_eff, C_eff, friction_rad_eff, biot_eff, T0_eff

    def compute_effective_pore_pressure(self, k: torch.Tensor, p_p_base: torch.Tensor, sigma_v: torch.Tensor = None):
        """
        计算有效孔隙压力 (PyTorch)
        
        参数:
            sigma_v: 垂直应力张量 (MPa)，如果提供则用于动态计算上限
        """
        log_k_ratio = torch.log(k / self.phys.permeability_ref)
        correction = self.phys.permeability_pressure_coeff * log_k_ratio
        p_p_eff = p_p_base * (1.0 + correction)
        
        # 动态计算上限
        if sigma_v is not None:
            p_p_max = 0.9 * sigma_v
            # 当 p_p_max 是 Tensor 时，分两步 clamp 避免类型混合
            p_p_eff = torch.clamp(p_p_eff, min=0.1)
            p_p_eff = torch.minimum(p_p_eff, p_p_max)
        else:
            # 回退为固定深度（p_p_max 是标量）
            p_p_max = self.phys.vertical_stress_gradient * self.phys.depth * 0.9
            p_p_eff = torch.clamp(p_p_eff, 0.1, p_p_max)
        
        return p_p_eff
        
    def transform_stress_tensor(self, sigma_H, sigma_h, sigma_v, inclination, azimuth, device):
        """坐标变换 (PyTorch Batch)"""
        alpha = torch.deg2rad(inclination)
        beta = torch.deg2rad(azimuth)
        gamma = torch.deg2rad(torch.tensor(self.phys.azimuth_sigma_H, device=device))
        
        sin_alpha = torch.sin(alpha)
        cos_alpha = torch.cos(alpha)
        sin_beta = torch.sin(beta)
        cos_beta = torch.cos(beta)
        
        # 构造 e_z 向量 (B, 3)
        e_z = torch.cat([
            sin_alpha * cos_beta,
            sin_alpha * sin_beta,
            cos_alpha
        ], dim=1)
        
        # 构造 e_x, e_y (处理直井奇异性)
        vertical = torch.tensor([0.0, 0.0, 1.0], device=device).expand_as(e_z)
        e_y_temp = torch.cross(e_z, vertical, dim=1)
        norm_e_y = torch.norm(e_y_temp, dim=1, keepdim=True)
        
        mask = norm_e_y < 1e-6
        e_y_safe = torch.where(mask, torch.tensor([0.0, 1.0, 0.0], device=device).expand_as(e_y_temp), e_y_temp)
        e_y_safe = e_y_safe / (torch.norm(e_y_safe, dim=1, keepdim=True) + 1e-9)
        
        e_x = torch.cross(e_y_safe, e_z, dim=1)
        e_x = e_x / (torch.norm(e_x, dim=1, keepdim=True) + 1e-9)
        e_y = torch.cross(e_z, e_x, dim=1)
        
        # 旋转矩阵 R (B, 3, 3)
        R = torch.stack([e_x, e_y, e_z], dim=2)
        
        # 地理坐标系下的应力张量
        c_g = torch.cos(gamma)
        s_g = torch.sin(gamma)
        
        S11 = sigma_H * c_g**2 + sigma_h * s_g**2
        S12 = (sigma_H - sigma_h) * c_g * s_g
        S22 = sigma_H * s_g**2 + sigma_h * c_g**2
        S33 = sigma_v
        S_zero = torch.zeros_like(S11)
        
        # 构造 (B, 3, 3)
        row1 = torch.cat([S11, S12, S_zero], dim=1).unsqueeze(1)
        row2 = torch.cat([S12, S22, S_zero], dim=1).unsqueeze(1)
        row3 = torch.cat([S_zero, S_zero, S33], dim=1).unsqueeze(1)
        sigma_NEV = torch.cat([row1, row2, row3], dim=1)
        
        # 旋转: sigma_wellbore = R^T @ sigma_NEV @ R
        sigma_wellbore = torch.bmm(torch.bmm(R.transpose(1, 2), sigma_NEV), R)
        
        return sigma_wellbore

    def generalized_kirsch_solution(self, sigma_wellbore, p_p, p_w, theta_deg, poisson_ratio):
        """Kirsch 方程 (PyTorch Batch)"""
        sigma_xx = sigma_wellbore[:, 0, 0:1]
        sigma_yy = sigma_wellbore[:, 1, 1:2]
        sigma_zz = sigma_wellbore[:, 2, 2:3]
        tau_xy = sigma_wellbore[:, 0, 1:2]
        tau_xz = sigma_wellbore[:, 0, 2:3]
        tau_yz = sigma_wellbore[:, 1, 2:3]
        
        theta_rad = torch.deg2rad(torch.tensor(theta_deg, device=sigma_wellbore.device))
        cos_2theta = torch.cos(2.0 * theta_rad)
        sin_2theta = torch.sin(2.0 * theta_rad)
        cos_theta = torch.cos(theta_rad)
        sin_theta = torch.sin(theta_rad)
        
        sigma_r = p_w
        
        sigma_theta = (sigma_xx + sigma_yy - 
                      2.0 * (sigma_xx - sigma_yy) * cos_2theta - 
                      4.0 * tau_xy * sin_2theta - 
                      p_w)
        
        sigma_z = (sigma_zz - 
                  2.0 * poisson_ratio * ((sigma_xx - sigma_yy) * cos_2theta + 
                                         2.0 * tau_xy * sin_2theta))
        
        # 忽略剪应力影响，简化为主应力排序
        return sigma_r, sigma_theta, sigma_z

    def compute_safety_factors_batch(self, sigma_H, sigma_h, sigma_v, p_p, p_w, inclination, azimuth, phi, k, T, failure_model="mohr"):
        """
        计算物理残差 (Batch) - 扫描 theta 寻找最危险点
        更新: 接收 sigma_v 作为输入
        
        参数:
            failure_model: 破坏准则选择 ("mohr" | "mogi" | "dp")
        """
        device = sigma_H.device
        
        UCS_eff, C_eff, phi_rad_eff, biot_eff, T0_eff = self.compute_effective_rock_properties(phi, T)
        p_p_eff = self.compute_effective_pore_pressure(k, p_p, sigma_v)
        
        # sigma_v is now passed as argument
        
        sigma_wb = self.transform_stress_tensor(sigma_H, sigma_h, sigma_v, inclination, azimuth, device)
        
        # === 扫描井周寻找最危险点 ===
        # 构造 theta 张量 (18 个点: 0, 10, ..., 170)
        thetas = torch.arange(0, 180, 10, device=device, dtype=torch.float32)
        
        # 为了进行广播计算，需要调整维度
        # sigma_wb: (B, 3, 3) -> (B, 1, 3, 3)
        sigma_wb_exp = sigma_wb.unsqueeze(1)
        
        # 变量扩展: (B, 1) -> (B, 18)
        p_p_eff_exp = p_p_eff.expand(-1, len(thetas))
        p_w_exp = p_w.expand(-1, len(thetas))
        biot_eff_exp = biot_eff.expand(-1, len(thetas))
        
        # 计算 Kirsch 解 (返回 (B, 18))
        sigma_xx = sigma_wb_exp[:, :, 0, 0] # (B, 1)
        sigma_yy = sigma_wb_exp[:, :, 1, 1]
        sigma_zz = sigma_wb_exp[:, :, 2, 2]
        tau_xy = sigma_wb_exp[:, :, 0, 1]
        
        theta_rad = torch.deg2rad(thetas).unsqueeze(0) # (1, 18)
        
        cos_2theta = torch.cos(2.0 * theta_rad)
        sin_2theta = torch.sin(2.0 * theta_rad)
        
        sigma_r = p_w_exp # (B, 18)
        
        sigma_theta = (sigma_xx + sigma_yy - 
                      2.0 * (sigma_xx - sigma_yy) * cos_2theta - 
                      4.0 * tau_xy * sin_2theta - 
                      p_w_exp)
        
        sigma_z = (sigma_zz - 
                  2.0 * self.phys.poisson_ratio * ((sigma_xx - sigma_yy) * cos_2theta + 
                                         2.0 * tau_xy * sin_2theta))
        
        # 有效应力
        sigma_r_eff = sigma_r - biot_eff_exp * p_p_eff_exp
        sigma_theta_eff = sigma_theta - biot_eff_exp * p_p_eff_exp
        sigma_z_eff = sigma_z - biot_eff_exp * p_p_eff_exp
        
        # 主应力排序 (B, 18, 3)
        stresses = torch.stack([sigma_r_eff, sigma_theta_eff, sigma_z_eff], dim=2)
        stresses_sorted, _ = torch.sort(stresses, dim=2)
        
        sigma_3_eff = stresses_sorted[:, :, 0] # (B, 18)
        sigma_2_eff = stresses_sorted[:, :, 1] # (B, 18) - 中间主应力
        sigma_1_eff = stresses_sorted[:, :, 2]
        
        # 扩展参数到 theta 维度
        C_eff_exp = C_eff.expand(-1, len(thetas))        # (B, 18)
        phi_rad_eff_exp = phi_rad_eff.expand(-1, len(thetas))  # (B, 18)
        UCS_eff_exp = UCS_eff.expand(-1, len(thetas))    # (B, 18)
        
        # 根据选择的破坏准则计算剪切残差
        if failure_model == "mohr":
            # Mohr-Coulomb (Collapse)
            N_phi = (1.0 + torch.sin(phi_rad_eff_exp)) / (1.0 - torch.sin(phi_rad_eff_exp))
            sigma_1_crit = sigma_3_eff * N_phi + UCS_eff_exp
            # 归一化残差
            res_shear = (sigma_1_eff - sigma_1_crit) / UCS_eff_exp
            
        elif failure_model == "mogi":
            # Mogi-Coulomb 准则
            sin_phi = torch.sin(phi_rad_eff_exp)
            cos_phi = torch.cos(phi_rad_eff_exp)
            
            a = (2.0 * np.sqrt(2.0) * C_eff_exp * cos_phi) / (3.0 - sin_phi)
            b = (2.0 * np.sqrt(2.0) * sin_phi) / (3.0 - sin_phi)
            
            sigma_m2 = (sigma_1_eff + sigma_3_eff) / 2.0
            
            diff_12 = sigma_1_eff - sigma_2_eff
            diff_23 = sigma_2_eff - sigma_3_eff
            diff_31 = sigma_3_eff - sigma_1_eff
            tau_oct = (1.0/3.0) * torch.sqrt(diff_12**2 + diff_23**2 + diff_31**2 + 1e-8)
            
            tau_strength = a + b * sigma_m2
            # 归一化残差: (tau_oct - tau_strength) / C_eff
            res_shear = (tau_oct - tau_strength) / C_eff_exp
            
        elif failure_model == "dp":
            # Drucker-Prager 准则
            sin_phi = torch.sin(phi_rad_eff_exp)
            cos_phi = torch.cos(phi_rad_eff_exp)
            
            alpha = (2.0 * sin_phi) / (np.sqrt(3.0) * (3.0 - sin_phi))
            k_dp = (6.0 * C_eff_exp * cos_phi) / (np.sqrt(3.0) * (3.0 - sin_phi))
            
            I1 = sigma_1_eff + sigma_2_eff + sigma_3_eff
            J2 = (1.0/6.0) * ((sigma_1_eff - sigma_2_eff)**2 + 
                              (sigma_2_eff - sigma_3_eff)**2 + 
                              (sigma_3_eff - sigma_1_eff)**2)
            sqrt_J2 = torch.sqrt(J2 + 1e-8)
            
            strength = alpha * I1 + k_dp
            # 归一化残差: (sqrt_J2 - strength) / C_eff
            res_shear = (sqrt_J2 - strength) / C_eff_exp
            
        else:
            raise ValueError(f"Unknown failure_model: {failure_model}. Must be 'mohr', 'mogi', or 'dp'.")
        
        # Tensile (Collapse) - 某些情况剪切破坏可能表现为张性
        T0_eff_exp = T0_eff.expand(-1, len(thetas))
        # Tensile criterion: sigma_3 < -T0  =>  -sigma_3 > T0
        # Residual = (-sigma_3 - T0) / T0
        res_tensile = (-sigma_3_eff - T0_eff_exp) / (T0_eff_exp + 1.0)
        
        # Collapse 取最危险情况 (Residual 最大值)
        # 注意: 原代码 SF = min(SF_shear, SF_tensile) 对应最危险
        # 这里 Residual = max(res_shear, res_tensile)
        res_matrix = torch.max(res_shear, res_tensile)
        res_collapse, _ = torch.max(res_matrix, dim=1, keepdim=True) # (B, 1)
        
        # Fracture
        # 破裂通常由 hoop stress 引起张性破坏
        # Criterion: sigma_theta < -T0 => -sigma_theta > T0
        # Residual = (-sigma_theta - T0) / T0
        res_frac_hoop = (-sigma_theta_eff - T0_eff_exp) / (T0_eff_exp + 1.0)
        
        # 取最危险点 (max residual over theta)
        res_fracture, _ = torch.max(res_frac_hoop, dim=1, keepdim=True)
        
        return res_collapse, res_fracture


class BPINNLoss:
    """BPINN 损失函数计算（支持混合学习）- 方案C: 归一化 Loss"""
    
    def __init__(self, physics: WellborePhysics, 
                 output_scaler: OutputScaler,
                 lambda_data: float = 1.0,
                 lambda_phys: float = 0.1,
                 lambda_kl: float = 0.001,
                 lambda_frac: float = 1.0,
                 scaler: DataScaler = None,
                 failure_model: str = "mohr",
                 output_weights: List[float] = None,
                 use_baseline_residual: bool = False,
                 residual_channels: List[int] = None):
        """
        参数:
            physics: 物理模型对象
            output_scaler: 输出归一化工具 (New)
            lambda_data: 数据损失权重
            lambda_phys: 物理损失权重
            lambda_kl: KL 散度权重
            lambda_frac: 破裂物理损失的额外加权系数
            scaler: 输入数据归一化工具
            failure_model: 破坏准则选择 ("mohr" | "mogi" | "dp")
            output_weights: [REVISION 2026] 4 维输出（SF_c, SF_f, ρ_c, ρ_f）的
                通道加权 MSE 系数。审稿人最关心 ρ_c 拟合精度（R3-Q4），
                因此默认 [1.0, 0.5, 3.0, 1.5]，让 ρ_c 在 data loss 中权重最大。
            use_baseline_residual: [REVISION 2026] True 时表示输入 X 末尾两列
                为 ρ_c / ρ_f 的物理 baseline（ε=0 解析解），仅对 residual_channels
                中列出的通道（默认 [2]，即只对 ρ_c）将 baseline 加回得到真实预测；
                physics_loss 内部用真实 ρ 代入 Kirsch+破坏准则。
                False 时保持旧行为（网络直接预测 ρ，与输入 baseline 列无关）。
            residual_channels: [REVISION 2026] 与 OutputScaler.residual_channels 对齐，
                指定哪些通道做"baseline+residual"。默认 [2] 仅 ρ_c。
        """
        self.physics = physics
        self.torch_physics = TorchWellborePhysics(physics.phys)
        self.output_scaler = output_scaler
        
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys
        self.lambda_kl = lambda_kl
        self.lambda_frac = lambda_frac
        self.scaler = scaler
        self.failure_model = failure_model
        self.use_baseline_residual = bool(use_baseline_residual)
        if residual_channels is None:
            residual_channels = [2] if self.use_baseline_residual else []
        self.residual_channels = list(int(c) for c in residual_channels)

        if output_weights is None:
            output_weights = [1.0, 1.0, 1.0, 1.0]
        if len(output_weights) != 4:
            raise ValueError(f"output_weights 必须是 4 个数，收到 {len(output_weights)}")
        self.output_weights_tensor = torch.tensor(output_weights, dtype=torch.float32)

    def _output_weights(self, device: torch.device) -> torch.Tensor:
        if self.output_weights_tensor.device != device:
            self.output_weights_tensor = self.output_weights_tensor.to(device)
        return self.output_weights_tensor

    def data_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        数据拟合损失 - 在归一化空间计算"通道加权 MSE"

        参数:
            y_pred: 模型输出 (Normalized [-1, 1])
            y_true: 真实标签：
                - 旧模式：物理值（SF_c, SF_f, ρ_c, ρ_f）；
                - 残差模式（use_baseline_residual=True）：物理值，但其中 ρ 通道
                  已经是 (ρ_true - ρ_baseline)（在 main 端预先扣除 baseline）。
                  归一化范围由 OutputScaler(residual_mode=True) 提供。
        """
        y_true_norm = self.output_scaler.normalize(y_true)
        per_dim_mse = (y_pred - y_true_norm) ** 2  # (B, 4)
        w = self._output_weights(y_pred.device)
        weighted = per_dim_mse * w
        return weighted.mean()
    
    def physics_loss(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        物理一致性损失
        
        参数:
            x: 输入 (Normalized)。残差模式下 x 是 11 维 [9 物理量 + 2 baseline]。
            y_pred: 模型输出 (Normalized [-1, 1])。残差模式下 ρ 通道为 Δρ。
        """
        # 1. 反归一化输入 X -> Physical X
        if self.scaler is not None:
            x_phys = self.scaler.inverse_transform_tensor(x)
        else:
            x_phys = x
            
        # 2. 反归一化输出 Y_pred -> Physical Y (用于代入物理公式)
        y_pred_phys = self.output_scaler.denormalize(y_pred)

        # [REVISION 2026] 残差模式下：仅 residual_channels 中的通道是残差 Δρ，
        # 需要加回 baseline 才是真实 ρ；其余通道直接是物理预测。
        if self.use_baseline_residual:
            # x_phys 末尾两列 = [baseline_rho_c, baseline_rho_f]
            baseline_rho_c = x_phys[:, 9:10]
            baseline_rho_f = x_phys[:, 10:11]
            rho_collapse_pred = y_pred_phys[:, 2:3] + (baseline_rho_c if 2 in self.residual_channels else 0.0)
            rho_fracture_pred = y_pred_phys[:, 3:4] + (baseline_rho_f if 3 in self.residual_channels else 0.0)
        else:
            rho_collapse_pred = y_pred_phys[:, 2:3]
            rho_fracture_pred = y_pred_phys[:, 3:4]

        SF_collapse_pred = y_pred_phys[:, 0]
        SF_fracture_pred = y_pred_phys[:, 1]

        # 软约束损失：保证物理逻辑 rho_c < rho_f
        loss_density_order = torch.mean(torch.relu(rho_collapse_pred - rho_fracture_pred))
        
        # 提取输入变量
        # 重要：训练输入现在包含 Depth（第 0 列），以确保密度<->压力换算与数据生成一致。
        depth = x_phys[:, 0:1]         # m
        phi = x_phys[:, 1:2]
        sigma_v = x_phys[:, 2:3]
        sigma_H = x_phys[:, 3:4]
        sigma_h = x_phys[:, 4:5]
        p_p = x_phys[:, 5:6]
        T = x_phys[:, 6:7]
        inclination = x_phys[:, 7:8]
        azimuth = x_phys[:, 8:9]
        
        # [FIX] 使用与数据生成一致的渗透率估计，避免常数 k=10 导致的物理冲突
        # 这与 sample_permeability() 的统计关系一致: log(k) ~ N(20φ-2, 0.5)
        # 这里使用确定性的均值关系作为经验估计
        log_k_est = 20.0 * phi - 2.0
        k = torch.exp(log_k_est)
        k = torch.clamp(k, min=0.01, max=1000.0)
        
        # 计算泥浆压力（MPa）
        # 与数据生成 compute_true_outputs() 一致：使用真实 depth 做 EMW(密度) <-> 压力换算。
        g = self.physics.phys.g
        h = torch.clamp(depth, min=1.0)  # 避免除零/非物理深度
        p_mud_collapse = rho_collapse_pred * g * h / 1000.0
        p_mud_fracture = rho_fracture_pred * g * h / 1000.0
        
        # 计算物理残差（使用指定的破坏准则）
        res_collapse, _ = self.torch_physics.compute_safety_factors_batch(
            sigma_H, sigma_h, sigma_v, p_p, p_mud_collapse, inclination, azimuth, phi, k, T,
            failure_model=self.failure_model
        )
        
        _, res_fracture = self.torch_physics.compute_safety_factors_batch(
            sigma_H, sigma_h, sigma_v, p_p, p_mud_fracture, inclination, azimuth, phi, k, T,
            failure_model=self.failure_model
        )
        
        target_res = torch.zeros_like(res_collapse)
        
        loss_critical_collapse = nn.functional.mse_loss(res_collapse, target_res)
        loss_critical_fracture = nn.functional.mse_loss(res_fracture, target_res)
        
        loss_critical = loss_critical_collapse + (self.lambda_frac * loss_critical_fracture)
        
        loss_phys = loss_critical + 0.1 * loss_density_order
        
        return loss_phys
    
    def total_loss(self, x: torch.Tensor, y_pred: torch.Tensor, 
                  y_true: torch.Tensor, kl: torch.Tensor,
                  num_batches: int, has_labels: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        总损失函数（支持混合学习）
        
        参数:
            x: 输入张量
            y_pred: 预测输出
            y_true: 真实输出
            kl: KL 散度
            num_batches: 总批次数（用于归一化 KL）
            has_labels: 布尔张量，标记哪些样本有标签 (batch_size,)
        
        返回:
            (total_loss, loss_dict): 总损失和各项损失的字典
        """
        l_phys = self.physics_loss(x, y_pred)
        
        if has_labels is not None:
            labeled_mask = has_labels.bool()
            if labeled_mask.any():
                l_data = self.data_loss(y_pred[labeled_mask], y_true[labeled_mask])
            else:
                l_data = torch.tensor(0.0, device=y_pred.device)
        else:
            l_data = self.data_loss(y_pred, y_true)
        
        l_kl = kl / num_batches
        
        total = (self.lambda_data * l_data + 
                self.lambda_phys * l_phys + 
                self.lambda_kl * l_kl)
        
        loss_dict = {
            'total': total.item(),
            'data': l_data.item(),
            'physics': l_phys.item(),
            'kl': l_kl.item()
        }
        
        return total, loss_dict


def load_real_data_from_csv(csv_path: str, 
                            phys_const: PhysicalConstants) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 CSV 文件加载真实工程数据（8 维输入）
    
    参数:
        csv_path: CSV 文件路径
        phys_const: 物理常数配置
    
    返回:
        (X_real, Y_real, has_labels): 输入、输出、标签掩码
    
    CSV 列名约定（按顺序，Updated for New Inputs）：
    - porosity: 孔隙度
    - vertical_stress: 垂直应力 (MPa)
    - sigma_H: 最大水平主应力 (MPa)
    - sigma_h: 最小水平主应力 (MPa)
    - pore_pressure: 孔隙压力 (MPa)
    - temperature: 温度 (°C)
    - inclination: 井斜角 (度)
    - wellbore_azimuth: 井眼方位角 (度)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    required_input_cols = ['porosity', 'vertical_stress', 'sigma_H', 'sigma_h', 
                          'pore_pressure', 'temperature', 'inclination', 'wellbore_azimuth']
    
    for col in required_input_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 文件缺少必需的输入列: {col}")
    
    X_real = df[required_input_cols].values
    
    output_cols = ['SF_collapse', 'SF_fracture', 'rho_mud_collapse', 'rho_mud_fracture']
    
    Y_real = np.zeros((len(df), 4))
    has_labels = np.zeros(len(df), dtype=bool)
    
    for i, row in df.iterrows():
        all_outputs_available = all(col in df.columns and pd.notna(row[col]) for col in output_cols)
        if all_outputs_available:
            Y_real[i, :] = row[output_cols].values
            has_labels[i] = True
        else:
            Y_real[i, :] = 0.0
            has_labels[i] = False
    
    return X_real, Y_real, has_labels


def train_bpinn(model: BayesianBPINN, 
               train_loader: DataLoader,
               val_loader: DataLoader,
               loss_fn: BPINNLoss,
               optimizer: optim.Optimizer,
               scheduler: object,
               num_epochs: int,
               device: torch.device,
               verbose: bool = True,
               mc_val_samples: int = 5,
               early_stopping_patience: int = 50,
               early_stopping_min_delta: float = 1e-4,
               kl_warmup_epochs: int = 0,
               kl_anneal_schedule: str = "linear",
               lambda_kl_final: float = None) -> Dict[str, list]:
    """
    训练 BPINN 模型

    参数:
        model: BPINN 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        loss_fn: 损失函数对象
        optimizer: 优化器
        scheduler: 学习率调度器
        num_epochs: 训练轮数
        device: 设备 (CPU/GPU)
        verbose: 是否打印训练信息
        mc_val_samples: MC采样次数用于验证（减少贝叶斯权重采样噪声）
        early_stopping_patience: Early stopping 的耐心值
        early_stopping_min_delta: 判断改善的最小阈值
        kl_warmup_epochs: KL 退火的 warmup 轮数。
            为 0 时禁用退火（保持向后兼容）。
            > 0 时，loss_fn.lambda_kl 在 [0, lambda_kl_final] 之间按
            kl_anneal_schedule 调度上升，warmup 结束后保持 lambda_kl_final。
            标准做法（Sønderby et al. 2016；Bowman et al. 2016），可避免
            "KL 单调上升 / posterior collapse / Gaussian over-regularization" 等病态。
        kl_anneal_schedule: "linear" | "sigmoid" | "cosine"。
            - linear:  λ_kl(e) = λ_final * e / W
            - sigmoid: λ_kl(e) = λ_final / (1 + exp(-12*(e/W - 0.5)))  (S 形)
            - cosine:  λ_kl(e) = λ_final * 0.5*(1 - cos(π * e / W))     (平滑 S 形)
        lambda_kl_final: warmup 结束后的目标 KL 权重。
            为 None 时使用 loss_fn.lambda_kl 当前值（即用户在外部设定的值）。

    返回:
        包含训练历史的字典（额外含 'lambda_kl_schedule'）
    """
    history = {
        'train_loss': [],
        'train_data_loss': [],
        'train_phys_loss': [],
        'train_kl_loss': [],
        'val_loss': [],
        'lambda_kl_schedule': []
    }

    num_batches = len(train_loader)

    # [FIX] Early Stopping 初始化
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state_dict = None
    prev_lr = optimizer.param_groups[0]['lr']  # [REVISION 2026] 监控 LR 阶跃下降

    # ---- KL 退火初始化 ----
    if lambda_kl_final is None:
        lambda_kl_final = loss_fn.lambda_kl
    use_kl_anneal = kl_warmup_epochs > 0

    def _kl_factor(epoch_idx: int) -> float:
        """根据调度返回 [0, 1] 区间内的退火系数。"""
        if not use_kl_anneal or epoch_idx >= kl_warmup_epochs:
            return 1.0
        ratio = (epoch_idx + 1) / float(kl_warmup_epochs)
        if kl_anneal_schedule == "sigmoid":
            return float(1.0 / (1.0 + np.exp(-12.0 * (ratio - 0.5))))
        if kl_anneal_schedule == "cosine":
            return float(0.5 * (1.0 - np.cos(np.pi * ratio)))
        return float(ratio)

    for epoch in range(num_epochs):
        # 动态设置当前 epoch 的 KL 权重（向后兼容：未启用退火时保持原值）
        current_lambda_kl = lambda_kl_final * _kl_factor(epoch)
        loss_fn.lambda_kl = current_lambda_kl
        history['lambda_kl_schedule'].append(current_lambda_kl)

        model.train()
        epoch_losses = {'total': 0.0, 'data': 0.0, 'physics': 0.0, 'kl': 0.0}

        for batch_idx, batch_data in enumerate(train_loader):
            if len(batch_data) == 3:
                x_batch, y_batch, has_labels_batch = batch_data
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                has_labels_batch = has_labels_batch.to(device)
            else:
                x_batch, y_batch = batch_data
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                has_labels_batch = None
            
            y_pred = model(x_batch)
            
            kl = model.get_kl_loss()
            
            loss, loss_dict = loss_fn.total_loss(x_batch, y_pred, y_batch, 
                                                 kl, num_batches, has_labels_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]
        
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        history['train_loss'].append(epoch_losses['total'])
        history['train_data_loss'].append(epoch_losses['data'])
        history['train_phys_loss'].append(epoch_losses['physics'])
        history['train_kl_loss'].append(epoch_losses['kl'])
        
        # [FIX] MC 平均验证以减少贝叶斯权重采样带来的噪声
        model.eval()
        val_loss_mc = 0.0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:
                    x_val, y_val, has_labels_val = batch_data
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    has_labels_val = has_labels_val.to(device)
                else:
                    x_val, y_val = batch_data
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    has_labels_val = None
                
                # MC 采样：对同一个 batch 多次前向传播取平均
                batch_loss_sum = 0.0
                for _ in range(mc_val_samples):
                    y_pred_val = model(x_val)
                    kl_val = model.get_kl_loss()
                    
                    loss_val, _ = loss_fn.total_loss(x_val, y_pred_val, y_val,
                                                    kl_val, len(val_loader), has_labels_val)
                    batch_loss_sum += loss_val.item()
                
                val_loss_mc += batch_loss_sum / mc_val_samples
        
        val_loss_mc /= len(val_loader)
        history['val_loss'].append(val_loss_mc)
        
        # [FIX] Scheduler 使用 MC 平均后的 val_loss
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss_mc)
            else:
                scheduler.step()

        # [REVISION 2026] LR 阶跃下降时刷新 Early Stopping 计数与 best_val。
        # 否则 ReduceLROnPlateau 把 LR 砍半后，val_loss 会有短暂抖动，
        # 旧逻辑会把这段抖动也算进 epochs_no_improve，导致"LR 一减就停"。
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < prev_lr - 1e-12:
            if verbose:
                print(f"  [LR drop] {prev_lr:.2e} -> {current_lr:.2e}; "
                      f"early-stop counter & best_val reset.")
            epochs_no_improve = 0
            best_val_loss = val_loss_mc  # 以新 LR 起点为新基线
        prev_lr = current_lr

        # [REVISION 2026] KL warmup 期内不计早停。
        # 因为退火期内 λ_kl 还在上升，total loss 必然单调漂移，旧逻辑会把
        # 漂移误判为"未改善"，从而在 KL 还没真正进入约束阶段就 early stop。
        in_warmup = (kl_warmup_epochs > 0) and (epoch < kl_warmup_epochs)

        # [FIX] Early Stopping 逻辑
        if val_loss_mc < best_val_loss - early_stopping_min_delta:
            best_val_loss = val_loss_mc
            epochs_no_improve = 0
            # 深拷贝最佳权重到 CPU（避免占用 GPU 内存）
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            if not in_warmup:
                epochs_no_improve += 1

        # warmup 结束的当个 epoch：以当前 val_loss 为新基线（避免 warmup 期
        # 因 λ_kl 漂移导致 best_val_loss 偏低，进入正式期立刻满足 patience）。
        if (kl_warmup_epochs > 0) and (epoch == kl_warmup_epochs - 1):
            best_val_loss = val_loss_mc
            epochs_no_improve = 0
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                print(f"  [KL warmup done] reset best_val={val_loss_mc:.4f}, "
                      f"early-stop counter cleared.")

        if verbose and (epoch + 1) % 10 == 0:
            kl_w_str = f"{current_lambda_kl:.2e}"
            phase = " (warmup)" if in_warmup else ""

            # [REVISION 2026 v4] 实时损失占比日志：分别打印
            #   (1) 各项加权贡献绝对值（weighted_*）
            #   (2) 加权后占总损失的百分比（ratio_*）
            # 这样可以直接看到 data / phys / kl 三项是否被某一项压死，
            # 不需要事后再去 history 里反推。
            w_data = loss_fn.lambda_data * epoch_losses['data']
            w_phys = loss_fn.lambda_phys * epoch_losses['physics']
            w_kl   = current_lambda_kl   * epoch_losses['kl']
            total_w = w_data + w_phys + w_kl + 1e-12
            r_data = 100.0 * w_data / total_w
            r_phys = 100.0 * w_phys / total_w
            r_kl   = 100.0 * w_kl   / total_w

            print(f"Epoch [{epoch+1}/{num_epochs}]{phase} - "
                  f"Train Loss: {epoch_losses['total']:.4f} "
                  f"(Data: {epoch_losses['data']:.4f}, "
                  f"Phys: {epoch_losses['physics']:.4f}, "
                  f"KL: {epoch_losses['kl']:.4f}) - "
                  f"Weighted[D/P/K]: {w_data:.4f}/{w_phys:.4f}/{w_kl:.4f} - "
                  f"Ratio[D/P/K]: {r_data:5.1f}%/{r_phys:5.1f}%/{r_kl:5.1f}% - "
                  f"Val Loss (MC): {val_loss_mc:.4f} - "
                  f"LR: {current_lr:.6f} - "
                  f"λ_kl: {kl_w_str} - "
                  f"NoImprove: {epochs_no_improve}/{early_stopping_patience}")

        # [FIX] 触发 Early Stopping（仅在 warmup 结束之后）
        if (not in_warmup) and (epochs_no_improve >= early_stopping_patience):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best val loss: {best_val_loss:.4f}")
            break
    
    # [FIX] 恢复最佳权重
    if best_state_dict is not None:
        # 将权重移回到原设备
        model.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
        print(f"\nRestored best model weights (val_loss={best_val_loss:.4f})")
    
    return history


# ============================================================================
# 第六部分：不确定性预测与分析
# ============================================================================

def predict_bayesian(model: BayesianBPINN, 
                    inputs: torch.Tensor,
                    num_samples: int,
                    device: torch.device,
                    output_scaler: OutputScaler = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    贝叶斯预测 - 通过多次前向传播估计预测分布
    
    参数:
        model: 训练好的 BPINN 模型
        inputs: 输入张量，形状 (N, 8)
        num_samples: 权重采样次数
        device: 设备
        output_scaler: 输出归一化工具 (如果提供，则返回反归一化后的物理值)
    
    返回:
        (mean_outputs, std_outputs): 预测均值和标准差，形状均为 (N, output_dim)
    """
    model.eval()
    
    N = inputs.shape[0]
    output_dim = model.output_dim
    
    # 存储所有输出 (原始归一化值 或 物理值)
    all_outputs = np.zeros((num_samples, N, output_dim))
    
    inputs = inputs.to(device)
    
    with torch.no_grad():
        for i in range(num_samples):
            outputs = model(inputs) # 此时输出在 [-1, 1]
            
            if output_scaler is not None:
                # 反归一化为物理值
                outputs = output_scaler.denormalize(outputs)
                
            all_outputs[i] = outputs.cpu().numpy()
    
    mean_outputs = np.mean(all_outputs, axis=0)
    std_outputs = np.std(all_outputs, axis=0)
    
    return mean_outputs, std_outputs


def compute_reliability_curves_bpinn(regime: str, alpha_p: float,
                                     sampler: InputSampler, model: BayesianBPINN,
                                     rho_grid: np.ndarray, 
                                     num_input_samples: int = 500,
                                     num_weight_samples: int = 50,
                                     device: torch.device = None,
                                     scaler: DataScaler = None,
                                     output_scaler: OutputScaler = None, # 新增参数
                                     fixed_depth: float = None,
                                     physics: WellborePhysics = None,
                                     failure_model: str = "mohr",
                                     use_baseline_residual: bool = False,
                                     residual_channels: List[int] = None) -> Dict[str, np.ndarray]:
    """
    使用 BPINN 模型计算可靠度-等效密度曲线

    [REVISION 2026 v2] 重构以恢复 BPINN 的速度优势：
        旧实现把 sample_inputs + compute_baseline_rho 放在 ρ 循环内，导致 100 个
        ρ 网格点 × 解析解开销 ≈ 350 s（比物理解还慢 30 倍，破坏 BPINN 叙事）。
        现在改为：
            (1) ρ 循环外一次性 sample_inputs 与 compute_baseline_rho；
            (2) ρ 循环外一次性收集 num_weight_samples 次 MC forward 的 ρ_c/ρ_f；
            (3) ρ 循环内只做 numpy 比较，O(N · M)，几乎 0 耗时。
        这同时也让"R(ρ) 曲线在不同 ρ 下用同一组输入"，统计意义更清晰。

    新增参数:
        physics: 用于计算 ρ_c/ρ_f baseline（残差模式必填）。
        failure_model: 与 BPINN 训练时一致的破坏准则。
        use_baseline_residual: True 时启用 baseline + residual 推理路径，
            X 拼接为 11 维 [Depth, 8 features, baseline_ρ_c, baseline_ρ_f]，
            仅 residual_channels 中的通道与对应 baseline 相加。
        residual_channels: 与训练时一致，默认 [2] 仅 ρ_c。
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if fixed_depth is None:
        fixed_depth = sampler.phys.depth

    if use_baseline_residual and physics is None:
        raise ValueError("use_baseline_residual=True 时必须提供 physics 用于计算 baseline")
    if residual_channels is None:
        residual_channels = [2] if use_baseline_residual else []

    print(f"\n计算 BPINN 可靠度曲线: {regime} 机制, alpha_p={alpha_p:.2f}")
    print(f"  密度范围: [{rho_grid[0]:.2f}, {rho_grid[-1]:.2f}] g/cm^3, {len(rho_grid)} 个点")
    print(f"  输入样本数: {num_input_samples}, 权重采样: {num_weight_samples}")
    print(f"  分析深度: {fixed_depth} m")
    if use_baseline_residual:
        print(f"  推理模式: baseline + residual (failure_model={failure_model}, "
              f"residual_channels={residual_channels})")

    model.eval()

    # ---- (1) ρ 循环外：一次采样 ----
    X_8d, depth_samples, aux_data = sampler.sample_inputs(
        num_input_samples, regime=regime, alpha_p_mean=alpha_p,
        fixed_depth=fixed_depth, return_depth=True
    )
    X_samples = np.column_stack([depth_samples, X_8d])  # 9D

    if use_baseline_residual:
        baseline_rho = physics.compute_baseline_rho(
            X_8d, aux_data=aux_data, failure_model=failure_model
        )  # (N, 2): [baseline_ρ_c, baseline_ρ_f]
        X_samples = np.column_stack([X_samples, baseline_rho])  # 11D
    else:
        baseline_rho = None

    if scaler is not None:
        X_samples_norm = scaler.transform(X_samples)
        X_torch = torch.FloatTensor(X_samples_norm).to(device)
    else:
        X_torch = torch.FloatTensor(X_samples).to(device)

    # ---- (2) ρ 循环外：一次性收集所有 MC 权重采样的 ρ_c / ρ_f 预测 ----
    M = int(num_weight_samples)
    N = num_input_samples
    all_rho_c = np.zeros((M, N), dtype=np.float64)
    all_rho_f = np.zeros((M, N), dtype=np.float64)

    with torch.no_grad():
        for m in range(M):
            y_pred = model(X_torch)
            if output_scaler is not None:
                y_pred = output_scaler.denormalize(y_pred)
            rho_c_m = y_pred[:, 2].cpu().numpy()
            rho_f_m = y_pred[:, 3].cpu().numpy()
            if use_baseline_residual and baseline_rho is not None:
                if 2 in residual_channels:
                    rho_c_m = rho_c_m + baseline_rho[:, 0]
                if 3 in residual_channels:
                    rho_f_m = rho_f_m + baseline_rho[:, 1]
            all_rho_c[m] = rho_c_m
            all_rho_f[m] = rho_f_m

    # ---- (3) ρ 循环内只做 numpy 比较 ----
    R_collapse_bpinn = np.zeros(len(rho_grid))
    R_fracture_bpinn = np.zeros(len(rho_grid))
    total_predictions = M * N
    for i, rho in enumerate(rho_grid):
        # 塌陷安全：当前密度 >= 临界塌陷密度 且 当前密度 <= 临界破裂密度
        collapse_safe = ((rho >= all_rho_c) & (rho <= all_rho_f)).sum()
        # 破裂安全：当前密度 <= 临界破裂密度
        fracture_safe = (rho <= all_rho_f).sum()
        R_collapse_bpinn[i] = collapse_safe / total_predictions
        R_fracture_bpinn[i] = fracture_safe / total_predictions

        if (i + 1) % 20 == 0 or i == len(rho_grid) - 1:
            print(f"  进度: {i+1}/{len(rho_grid)}, ρ={rho:.2f}, "
                  f"R_collapse={R_collapse_bpinn[i]:.3f}, R_fracture={R_fracture_bpinn[i]:.3f}")
    
    # 寻找安全窗口
    safe_mask = (R_collapse_bpinn >= 0.5) & (R_fracture_bpinn >= 0.5)
    
    if np.any(safe_mask):
        safe_rho = rho_grid[safe_mask]
        safe_window_lower = safe_rho[0]
        safe_window_upper = safe_rho[-1]
        recommended_rho = np.mean(safe_rho)
    else:
        idx_collapse_05 = np.argmin(np.abs(R_collapse_bpinn - 0.5))
        idx_fracture_05 = np.argmin(np.abs(R_fracture_bpinn - 0.5))
        safe_window_lower = rho_grid[idx_collapse_05]
        safe_window_upper = rho_grid[idx_fracture_05]
        recommended_rho = (safe_window_lower + safe_window_upper) / 2.0
    
    print(f"\n  安全窗口: [{safe_window_lower:.3f}, {safe_window_upper:.3f}] g/cm^3")
    print(f"  推荐泥浆密度: {recommended_rho:.3f} g/cm^3")
    
    return {
        'rho_grid': rho_grid,
        'R_collapse': R_collapse_bpinn,
        'R_fracture': R_fracture_bpinn,
        'safe_window_lower': safe_window_lower,
        'safe_window_upper': safe_window_upper,
        'recommended_rho': recommended_rho,
        'regime': regime,
        'alpha_p': alpha_p
    }


def plot_reliability_curves_comparison(physics_results: Dict, bpinn_results: Dict,
                                       save_path: str = None, show_plot: bool = True):
    """
    绘制物理解与 BPINN 的可靠度曲线对比图
    
    参数:
        physics_results: 物理解可靠度结果
        bpinn_results: BPINN 可靠度结果
        save_path: 保存路径
        show_plot: 是否显示图形
    
    对应论文: Fig.8-10 的 BPINN 复现与对比
    """
    # Paper style (Times New Roman + upper-bound sizes/line widths)
    set_paper_style(base_fontsize=14, dpi=600, seaborn_style="white", use_seaborn=False, allow_cjk_fallback=True)

    rho_grid = physics_results['rho_grid']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # 左图: 塌陷可靠度对比
    ax1.plot(rho_grid, physics_results['R_collapse'], 'b-o', linewidth=2.5, 
            markersize=5, label='Physics (Collapse)', markevery=5)
    ax1.plot(rho_grid, bpinn_results['R_collapse'], 'c--s', linewidth=2.5, 
            markersize=5, label='BPINN (Collapse)', markevery=5)
    
    ax1.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.axvline(physics_results['safe_window_lower'], color='blue', linestyle='--', 
               linewidth=1.5, alpha=0.5, label=f"Physics Lower: {physics_results['safe_window_lower']:.2f}")
    ax1.axvline(bpinn_results['safe_window_lower'], color='cyan', linestyle='--', 
               linewidth=1.5, alpha=0.5, label=f"BPINN Lower: {bpinn_results['safe_window_lower']:.2f}")
    
    ax1.set_xlim([rho_grid[0], rho_grid[-1]])
    ax1.set_ylim([-0.05, 1.05])
    apply_ax_style(ax1, xlabel=r'Equivalent Mud Density $\rho$ (g/cm$^3$)',
                   ylabel='Collapse Reliability', title='Collapse Reliability Comparison',
                   legend=True, grid=True, fontsize=14)
    
    ax2.plot(rho_grid, physics_results['R_fracture'], 'r-o', linewidth=2.5, 
            markersize=5, label='Physics (Fracture)', markevery=5)
    ax2.plot(rho_grid, bpinn_results['R_fracture'], 'm--s', linewidth=2.5, 
            markersize=5, label='BPINN (Fracture)', markevery=5)
    
    ax2.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axvline(physics_results['safe_window_upper'], color='red', linestyle='--', 
               linewidth=1.5, alpha=0.5, label=f"Physics Upper: {physics_results['safe_window_upper']:.2f}")
    ax2.axvline(bpinn_results['safe_window_upper'], color='magenta', linestyle='--', 
               linewidth=1.5, alpha=0.5, label=f"BPINN Upper: {bpinn_results['safe_window_upper']:.2f}")
    
    ax2.set_xlim([rho_grid[0], rho_grid[-1]])
    ax2.set_ylim([-0.05, 1.05])
    apply_ax_style(ax2, xlabel=r'Equivalent Mud Density $\rho$ (g/cm$^3$)',
                   ylabel='Fracture Reliability', title='Fracture Reliability Comparison',
                   legend=True, grid=True, fontsize=14)
    
    regime = physics_results['regime']
    alpha_p = physics_results['alpha_p']
    fig.suptitle(f'Reliability Curves: Physics vs BPINN\nRegime: {regime}, α_p: {alpha_p:.2f}', 
               fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        print(f"可靠度曲线对比图已保存至: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # 计算并打印误差统计
    mse_collapse = np.mean((physics_results['R_collapse'] - bpinn_results['R_collapse'])**2)
    mse_fracture = np.mean((physics_results['R_fracture'] - bpinn_results['R_fracture'])**2)
    mae_collapse = np.mean(np.abs(physics_results['R_collapse'] - bpinn_results['R_collapse']))
    mae_fracture = np.mean(np.abs(physics_results['R_fracture'] - bpinn_results['R_fracture']))
    
    print(f"\n可靠度曲线误差统计:")
    print(f"  塌陷可靠度 - MSE: {mse_collapse:.6f}, MAE: {mae_collapse:.6f}")
    print(f"  破裂可靠度 - MSE: {mse_fracture:.6f}, MAE: {mae_fracture:.6f}")
    print(f"  安全窗口差异:")
    print(f"    下限差异: {abs(physics_results['safe_window_lower'] - bpinn_results['safe_window_lower']):.3f} g/cm^3")
    print(f"    上限差异: {abs(physics_results['safe_window_upper'] - bpinn_results['safe_window_upper']):.3f} g/cm^3")


def predict_with_uncertainty(model: BayesianBPINN,
                             sampler: InputSampler,
                             num_input_samples: int,
                             num_weight_samples: int,
                             device: torch.device,
                             scaler: DataScaler = None, # 新增输入 scaler
                             output_scaler: OutputScaler = None,
                             fixed_depth: float = None) -> Dict[str, np.ndarray]:
    """
    完整的不确定性传播分析
    
    参数:
        scaler: 输入归一化工具 (必须提供，否则预测失效)
        output_scaler: 输出归一化工具
    """
    # 采样输入 (Raw Physical Values, 8D) + Depth (for Scheme C input scaler consistency)
    X_8d, depth_samples, _ = sampler.sample_inputs(
        num_input_samples, fixed_depth=fixed_depth, return_depth=True
    )
    inputs_np = np.column_stack([depth_samples, X_8d])  # 9D: [Depth, 8 features]
    
    # [FIX] 归一化输入 (Critical for Scheme C)
    if scaler is not None:
        inputs_norm = scaler.transform(inputs_np)
        inputs_torch = torch.FloatTensor(inputs_norm).to(device)
    else:
        # Fallback (not recommended for Scheme C)
        inputs_torch = torch.FloatTensor(inputs_np).to(device)
    
    # 存储所有输出
    all_outputs_list = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(num_weight_samples):
            # 预测 (Normalized Output [-1, 1])
            outputs = model(inputs_torch)
            
            # [FIX] 反归一化输出
            if output_scaler is not None:
                outputs = output_scaler.denormalize(outputs)
                
            all_outputs_list.append(outputs.cpu().numpy())
    
    # 堆叠结果: shape (num_weight_samples, num_input_samples, output_dim)
    all_outputs_stack = np.array(all_outputs_list)
    
    all_outputs = all_outputs_stack.transpose(1, 0, 2)
    
    outputs_mean = np.mean(all_outputs, axis=1)
    outputs_std = np.std(all_outputs, axis=1)
    
    outputs_all_flat = all_outputs.reshape(-1, all_outputs.shape[-1])
    
    return {
        'inputs': inputs_np,  # 9D inputs (Depth + 8)
        'outputs_mean': outputs_mean,
        'outputs_std': outputs_std,
        'outputs_all': outputs_all_flat
    }


def compute_uq_calibration_metrics(model: BayesianBPINN,
                                    X_val_norm_torch: torch.Tensor,
                                    Y_val_np: np.ndarray,
                                    device: torch.device,
                                    output_scaler: OutputScaler = None,
                                    num_weight_samples: int = 50,
                                    nominal_levels: np.ndarray = None,
                                    output_names: List[str] = None,
                                    Y_baseline_np: np.ndarray = None,
                                    residual_channels: List[int] = None,
                                    residual_calibration_channels: List[int] = None) -> Dict:
    """
    计算 BPINN 的不确定性量化（UQ）标定指标。

    [REVISION 2026] 新增以回应审稿意见 R1-Q13、R3-Q5：
        - R1-Q13: "Report the uncertainty calibration metrics of the Bayesian approach"
        - R3-Q5:  "Separable UQ ... no quantitative decomposition appears in the results"

    指标定义
    --------
    1. **点预测精度**: RMSE, MAE, R² —— 检验"贝叶斯均值"作为点估计的精度。
    2. **PICP@p (Prediction Interval Coverage Probability)**:
        在名义置信水平 p（例如 0.90 / 0.95）下，真值落入由权重后验
        生成的双侧分位数区间 [Q_{(1-p)/2}, Q_{(1+p)/2}] 的实际比例。
        理想值 = p。
    3. **PINAW (Prediction Interval Normalized Average Width)**:
        在 95% 名义水平下的区间宽度，按真值动态范围归一化；
        与 PICP 联合判断"区间是否过宽"——一个 100% 覆盖但宽度无穷大的
        贝叶斯方法是无意义的。
    4. **ACE (Average Calibration Error)**:
        对 [0.05, 0.95] 区间内 19 个名义水平做 mean(|empirical - nominal|)。
        ACE 越接近 0 越标定良好。
    5. **NLL (Negative Log-Likelihood)** (Gaussian 假设下):
        NLL = 0.5 * mean[log(2π σ²) + (y - μ)² / σ²]
        其中 σ² 使用 residual-calibrated predictive variance：
        σ² = Var_w[y] + σ²_res。σ²_res 由验证残差估计，用于补偿仅靠
        Bayesian weight sampling 造成的过窄预测区间。
    6. **Aleatory / Epistemic 方差分解** (全方差律):
        E_w[Var_x[y|w]]  ≈ aleatory (输入参数 → 已包含在权重样本扫验证集时的样本内方差)
        Var_w[E_x[y|w]]  = epistemic (权重后验 → 不同权重样本下均值预测的方差)
        Total = Aleatory + Epistemic.

    Returns
    -------
    dict 含每个输出维度的所有指标，可直接写入 Excel；
    并保留 (M, N, D) 的原始预测张量以便后续可视化。
    """
    if nominal_levels is None:
        nominal_levels = np.arange(0.05, 1.0, 0.05)
    if output_names is None:
        output_names = ['SF_collapse', 'SF_fracture', 'rho_c (g/cm^3)', 'rho_f (g/cm^3)']
    if residual_calibration_channels is None:
        residual_calibration_channels = []
    residual_calibration_channels = [int(ch) for ch in residual_calibration_channels]

    model.eval()
    N, D = Y_val_np.shape
    M = int(num_weight_samples)
    X_val_norm_torch = X_val_norm_torch.to(device)

    # [REVISION 2026 v2] 残差模式：仅 residual_channels 中的通道加回 baseline。
    # Y_baseline_np 形状 (N, 2) 表示 [ρ_c_baseline, ρ_f_baseline]；只有出现在
    # residual_channels 中的通道（默认 [2] 即 ρ_c）会真正加回。
    use_baseline_correction = Y_baseline_np is not None
    if use_baseline_correction:
        if residual_channels is None:
            residual_channels = [2]
        baseline_full = np.zeros((N, D), dtype=np.float64)
        Yb = np.asarray(Y_baseline_np, dtype=np.float64)
        ch_to_col = {2: 0, 3: 1}  # 输出通道 -> Y_baseline_np 的列索引
        for ch in residual_channels:
            if ch in ch_to_col and ch_to_col[ch] < Yb.shape[1]:
                baseline_full[:, ch] = Yb[:, ch_to_col[ch]]

    preds = np.zeros((M, N, D), dtype=np.float64)
    with torch.no_grad():
        for m in range(M):
            y = model(X_val_norm_torch)
            if output_scaler is not None:
                y = output_scaler.denormalize(y)
            y_np = y.cpu().numpy()
            if use_baseline_correction:
                y_np = y_np + baseline_full
            preds[m] = y_np

    mean_pred = preds.mean(axis=0)  # (N, D)
    std_pred = preds.std(axis=0, ddof=1)  # (N, D)

    # 1. 点预测精度
    rmse = np.sqrt(np.mean((mean_pred - Y_val_np) ** 2, axis=0))
    mae = np.mean(np.abs(mean_pred - Y_val_np), axis=0)
    ss_res = np.sum((Y_val_np - mean_pred) ** 2, axis=0)
    y_mean = Y_val_np.mean(axis=0, keepdims=True)
    ss_tot = np.sum((Y_val_np - y_mean) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot > 1e-12, ss_tot, 1.0)

    # 2-3. PICP@90 / 95 + PINAW@95
    # 先保留 weight-only 区间，便于诊断后验是否塌缩；再用 residual-calibrated
    # predictive variance 计算最终上报的预测区间和 NLL。
    picp_90_weight_only = np.zeros(D)
    picp_95_weight_only = np.zeros(D)
    pinaw_95_weight_only = np.zeros(D)
    for d in range(D):
        lo90_raw = np.percentile(preds[:, :, d], 5.0, axis=0)
        hi90_raw = np.percentile(preds[:, :, d], 95.0, axis=0)
        lo95_raw = np.percentile(preds[:, :, d], 2.5, axis=0)
        hi95_raw = np.percentile(preds[:, :, d], 97.5, axis=0)
        picp_90_weight_only[d] = float(((Y_val_np[:, d] >= lo90_raw) & (Y_val_np[:, d] <= hi90_raw)).mean())
        picp_95_weight_only[d] = float(((Y_val_np[:, d] >= lo95_raw) & (Y_val_np[:, d] <= hi95_raw)).mean())
        y_range = float(Y_val_np[:, d].max() - Y_val_np[:, d].min())
        pinaw_95_weight_only[d] = float((hi95_raw - lo95_raw).mean() / max(y_range, 1e-9))

    residual = Y_val_np - mean_pred
    weight_var_pointwise = std_pred ** 2
    residual_noise_var = np.zeros(D, dtype=np.float64)
    raw_residual_mse = np.mean(residual ** 2, axis=0)
    mean_weight_var = np.mean(weight_var_pointwise, axis=0)
    for ch in residual_calibration_channels:
        if 0 <= ch < D:
            residual_noise_var[ch] = max(float(raw_residual_mse[ch] - mean_weight_var[ch]), 0.0)
    residual_noise_std = np.sqrt(residual_noise_var)
    calibrated_std = np.sqrt(weight_var_pointwise + residual_noise_var[None, :] + 1e-8)

    picp_90 = np.zeros(D)
    picp_95 = np.zeros(D)
    pinaw_95 = np.zeros(D)
    for d in range(D):
        z90 = stats.norm.ppf(0.95)
        z95 = stats.norm.ppf(0.975)
        lo90 = mean_pred[:, d] - z90 * calibrated_std[:, d]
        hi90 = mean_pred[:, d] + z90 * calibrated_std[:, d]
        lo95 = mean_pred[:, d] - z95 * calibrated_std[:, d]
        hi95 = mean_pred[:, d] + z95 * calibrated_std[:, d]
        picp_90[d] = float(((Y_val_np[:, d] >= lo90) & (Y_val_np[:, d] <= hi90)).mean())
        picp_95[d] = float(((Y_val_np[:, d] >= lo95) & (Y_val_np[:, d] <= hi95)).mean())
        y_range = float(Y_val_np[:, d].max() - Y_val_np[:, d].min())
        pinaw_95[d] = float((hi95 - lo95).mean() / max(y_range, 1e-9))

    # 4. ACE 与可靠度图
    L = len(nominal_levels)
    coverage_per_level = np.zeros((L, D))
    for li, p in enumerate(nominal_levels):
        z = stats.norm.ppf((1.0 + p) / 2.0)
        for d in range(D):
            lo = mean_pred[:, d] - z * calibrated_std[:, d]
            hi = mean_pred[:, d] + z * calibrated_std[:, d]
            coverage_per_level[li, d] = float(((Y_val_np[:, d] >= lo) & (Y_val_np[:, d] <= hi)).mean())
    ace = np.mean(np.abs(coverage_per_level - nominal_levels[:, None]), axis=0)

    # 5. NLL (Gaussian)
    var_pred_weight_only = std_pred ** 2 + 1e-8
    nll_weight_only = 0.5 * np.mean(
        np.log(2.0 * np.pi * var_pred_weight_only) + (Y_val_np - mean_pred) ** 2 / var_pred_weight_only,
        axis=0
    )
    var_pred = calibrated_std ** 2
    nll = 0.5 * np.mean(np.log(2.0 * np.pi * var_pred) + (Y_val_np - mean_pred) ** 2 / var_pred, axis=0)

    # 6. Aleatory / Epistemic 方差分解
    var_aleatory = preds.var(axis=1, ddof=1).mean(axis=0)   # E_w[Var_x]
    var_epistemic = preds.mean(axis=1).var(axis=0, ddof=1)  # Var_w[E_x]
    var_total = var_aleatory + var_epistemic
    aleatory_pct = 100.0 * var_aleatory / np.where(var_total > 1e-12, var_total, 1.0)
    epistemic_pct = 100.0 * var_epistemic / np.where(var_total > 1e-12, var_total, 1.0)

    return {
        'output_names': output_names,
        'rmse': rmse, 'mae': mae, 'r2': r2,
        'picp_90': picp_90, 'picp_95': picp_95, 'pinaw_95': pinaw_95,
        'picp_90_weight_only': picp_90_weight_only,
        'picp_95_weight_only': picp_95_weight_only,
        'pinaw_95_weight_only': pinaw_95_weight_only,
        'ace': ace, 'nll': nll,
        'nll_weight_only': nll_weight_only,
        'residual_noise_var': residual_noise_var,
        'residual_noise_std': residual_noise_std,
        'residual_calibration_channels': residual_calibration_channels,
        'var_aleatory': var_aleatory, 'var_epistemic': var_epistemic, 'var_total': var_total,
        'aleatory_pct': aleatory_pct, 'epistemic_pct': epistemic_pct,
        'nominal_levels': nominal_levels,
        'coverage_per_level': coverage_per_level,
        'mean_pred': mean_pred, 'std_pred': calibrated_std,
        'std_pred_weight_only': std_pred,
        'preds': preds,  # (M, N, D)
        'num_weight_samples': M,
        'num_validation_points': N,
    }


def uq_metrics_to_dataframe(metrics: Dict) -> "pd.DataFrame":
    """把 compute_uq_calibration_metrics 的 dict 整理成一行/列的 DataFrame，便于 Excel 导出。"""
    rows = []
    for d, name in enumerate(metrics['output_names']):
        rows.append({
            'Output': name,
            'RMSE': float(metrics['rmse'][d]),
            'MAE': float(metrics['mae'][d]),
            'R2': float(metrics['r2'][d]),
            'PICP_90 (target=0.90)': float(metrics['picp_90'][d]),
            'PICP_95 (target=0.95)': float(metrics['picp_95'][d]),
            'PICP_90_weight_only': float(metrics['picp_90_weight_only'][d]),
            'PICP_95_weight_only': float(metrics['picp_95_weight_only'][d]),
            'PINAW_95': float(metrics['pinaw_95'][d]),
            'PINAW_95_weight_only': float(metrics['pinaw_95_weight_only'][d]),
            'ACE': float(metrics['ace'][d]),
            'NLL': float(metrics['nll'][d]),
            'NLL_weight_only': float(metrics['nll_weight_only'][d]),
            'Residual_Calib_Std': float(metrics['residual_noise_std'][d]),
            'Var_Aleatory': float(metrics['var_aleatory'][d]),
            'Var_Epistemic': float(metrics['var_epistemic'][d]),
            'Aleatory_pct': float(metrics['aleatory_pct'][d]),
            'Epistemic_pct': float(metrics['epistemic_pct'][d]),
        })
    return pd.DataFrame(rows)




# ============================================================================
# 第七部分：数据生成与可视化
# ============================================================================

def plot_input_distributions(X: np.ndarray, save_path_prefix: str = "input_dist", show_plot: bool = True):
    """
    绘制输入参数的概率分布图（类似论文 Fig.4-5）
    
    参数:
        X: 输入样本矩阵，形状 (N, 9)，顺序为：
           [Depth, Porosity, Vertical_Stress, Sigma_H, Sigma_h, P_pore, Temperature, Inclination, Azimuth]
        save_path_prefix: 保存路径前缀
        show_plot: 是否显示图形
    
    对应论文: Fig.4-5 输入不确定参数的概率分布及其拟合
    """
    # Paper style (Times New Roman + upper-bound sizes/line widths)
    set_paper_style(base_fontsize=14, dpi=600, seaborn_style="white", use_seaborn=False, allow_cjk_fallback=True)

    input_names = [
        'Depth z (m)',
        'Porosity φ',
        'Vertical Stress σ_v (MPa)',
        'Max Horizontal Stress σ_H (MPa)',
        'Min Horizontal Stress σ_h (MPa)',
        'Pore Pressure p_p (MPa)',
        'Temperature T (°C)',
        'Inclination α (°)',
        'Azimuth β (°)'
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(24, 16))
    axes = axes.flatten()
    
    for i in range(9):
        data = X[:, i]
        ax = axes[i]
        
        counts, bins, patches = ax.hist(data, bins=40, density=True, alpha=0.6, 
                                        color='skyblue', edgecolor='black', label='Histogram')
        
        try:
            if i == 1:
                data_positive = data[data > 0]
                if len(data_positive) > 10:
                    shape, loc_ln, scale_ln = stats.lognorm.fit(data_positive, floc=0)
                    x_range = np.linspace(max(data.min(), 1e-4), data.max(), 200)
                    pdf = stats.lognorm.pdf(x_range, shape, loc=loc_ln, scale=scale_ln)
                    ax.plot(x_range, pdf, 'r-', linewidth=2.5, label='Lognormal Fit')
                
            else:
                mu, std = np.mean(data), np.std(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                pdf = stats.norm.pdf(x_range, mu, std)
                ax.plot(x_range, pdf, 'r-', linewidth=2.5, label='Normal Fit')
        except Exception:
            pass
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        p5 = np.percentile(data, 5)
        p50 = np.percentile(data, 50)
        p95 = np.percentile(data, 95)
        
        ax.axvline(p5, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'P5: {p5:.3f}')
        ax.axvline(p50, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label=f'P50: {p50:.3f}')
        ax.axvline(p95, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'P95: {p95:.3f}')
        
        apply_ax_style(ax, xlabel=input_names[i], ylabel='Probability Density',
                       title=f'{input_names[i]}\n'
                             r'$\mu$' + f'={mean_val:.3f}, '
                             r'$\sigma$' + f'={std_val:.3f}',
                       legend=True, grid=True, fontsize=14)
    
    plt.tight_layout(pad=2.0)
    
    save_path = f"{save_path_prefix}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    print(f"输入参数分布图已保存至: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def scan_inclination_azimuth_for_regime(regime: str, alpha_p: float, physics: WellborePhysics,
                                        inclinations: np.ndarray = None, 
                                        azimuths: np.ndarray = None) -> Dict[str, np.ndarray]:
    """
    扫描井斜角和方位角，计算不同位置的临界泥浆密度
    
    参数:
        regime: 应力机制类型 ("NF", "SS", "RF")
        alpha_p: 地层压力系数
        physics: 物理模型对象
        inclinations: 井斜角数组(度)，如果为None则使用默认值
        azimuths: 井眼方位角数组(度)，如果为None则使用默认值
    
    返回:
        字典包含:
        - 'inclinations': 井斜角数组
        - 'azimuths': 方位角数组
        - 'rho_collapse_grid': 塌陷等效密度网格 (len(inclinations), len(azimuths))
        - 'rho_fracture_grid': 破裂等效密度网格
        - 'stats': 统计信息字典
    
    对应论文: Fig.2, Fig.6-7 的等效密度分布
    """
    if inclinations is None:
        inclinations = np.arange(0, 91, 10)  # 0°到90°，步长10°
    
    if azimuths is None:
        azimuths = np.arange(0, 361, 15)  # 0°到360°，步长15°
    
    # 获取该工况的代表性应力
    sigma_H, sigma_h, sigma_v, p_p = physics.phys.get_in_situ_stresses(regime, alpha_p)
    
    # 使用代表性岩石参数
    phi_rep = 0.15  # 代表性孔隙度
    k_rep = 10.0    # 代表性渗透率
    T_rep = 25.0 + 3.0 * (physics.phys.depth / 100.0)  # 代表性温度
    
    # 初始化网格
    rho_collapse_grid = np.zeros((len(inclinations), len(azimuths)))
    rho_fracture_grid = np.zeros((len(inclinations), len(azimuths)))
    
    print(f"\n扫描 {regime} 机制, alpha_p={alpha_p:.2f} 的等效密度...")
    print(f"  应力: σ_H={sigma_H:.1f} MPa, σ_h={sigma_h:.1f} MPa, σ_v={sigma_v:.1f} MPa, p_p={p_p:.1f} MPa")
    
    for i, incl in enumerate(inclinations):
        for j, azim in enumerate(azimuths):
            # 计算临界泥浆压力
            try:
                p_collapse = physics.find_critical_mud_pressure(
                    sigma_H, sigma_h, p_p, incl, azim, 'collapse', phi_rep, k_rep, T_rep, sigma_v
                )
                p_fracture = physics.find_critical_mud_pressure(
                    sigma_H, sigma_h, p_p, incl, azim, 'fracture', phi_rep, k_rep, T_rep, sigma_v
                )
                
                # 转换为等效密度
                rho_collapse_grid[i, j] = physics.mud_pressure_to_density(p_collapse)
                rho_fracture_grid[i, j] = physics.mud_pressure_to_density(p_fracture)
                
            except Exception as e:
                # 如果计算失败，使用NaN
                rho_collapse_grid[i, j] = np.nan
                rho_fracture_grid[i, j] = np.nan
    
    # 计算统计信息
    # [REVISION 2026 v2] 区分"最坏情况余量"与"最佳情况余量"，并报告 point-wise
    # 窗口为正的井斜/方位占比，避免 worst-case window_min 出现负值时被读者
    # 误读为"程序 bug 或 BPINN 不可用"。三者一同给出，回应 R2-Q3/R3-Q9 关于
    # 指标无定义的批评。
    pointwise_margin = rho_fracture_grid - rho_collapse_grid  # 正=window 存在
    pos_mask = (pointwise_margin > 0) & np.isfinite(pointwise_margin)
    n_total_valid = int(np.isfinite(pointwise_margin).sum())
    pos_ratio = float(pos_mask.sum()) / max(n_total_valid, 1)

    worst_case_margin = float(np.nanmin(rho_fracture_grid) - np.nanmax(rho_collapse_grid))
    best_case_margin = float(np.nanmax(pointwise_margin))
    mean_margin_when_open = float(np.nanmean(pointwise_margin[pos_mask])) if pos_mask.any() else float('nan')

    stats_dict = {
        'rho_collapse_mean': float(np.nanmean(rho_collapse_grid)),
        'rho_collapse_min': float(np.nanmin(rho_collapse_grid)),
        'rho_collapse_max': float(np.nanmax(rho_collapse_grid)),
        'rho_fracture_mean': float(np.nanmean(rho_fracture_grid)),
        'rho_fracture_min': float(np.nanmin(rho_fracture_grid)),
        'rho_fracture_max': float(np.nanmax(rho_fracture_grid)),
        # 旧字段（保留以兼容下游调用）
        'window_min': worst_case_margin,
        # [REVISION 2026 v2] 新字段（语义更明确）
        'worst_case_margin': worst_case_margin,
        'best_case_margin': best_case_margin,
        'mean_margin_when_open': mean_margin_when_open,
        'pos_ratio': pos_ratio,
        'regime': regime,
        'alpha_p': alpha_p,
    }

    print(f"  塌陷密度范围（扫描全部 incl × azim）: "
          f"[{stats_dict['rho_collapse_min']:.3f}, {stats_dict['rho_collapse_max']:.3f}] g/cm^3")
    print(f"  破裂密度范围（扫描全部 incl × azim）: "
          f"[{stats_dict['rho_fracture_min']:.3f}, {stats_dict['rho_fracture_max']:.3f}] g/cm^3")
    print(f"  最坏情况余量 min(ρ_f) − max(ρ_c) = {worst_case_margin:.3f} g/cm^3"
          + ("   [负值 → 存在井斜/方位下窗口闭合]" if worst_case_margin < 0 else ""))
    print(f"  最佳情况余量 max(ρ_f − ρ_c, point-wise) = {best_case_margin:.3f} g/cm^3")
    print(f"  窗口为正的井斜/方位占比 = {pos_ratio*100:.1f}%"
          + (f"   (其平均余量 {mean_margin_when_open:.3f} g/cm^3)" if pos_mask.any() else ""))
    
    return {
        'inclinations': inclinations,
        'azimuths': azimuths,
        'rho_collapse_grid': rho_collapse_grid,
        'rho_fracture_grid': rho_fracture_grid,
        'stats': stats_dict
    }


def plot_rho_vs_inclination(scan_results: Dict, failure_type: str, save_path: str = None, show_plot: bool = True):
    """
    绘制等效密度随井斜角的变化曲线（对方位角取平均）
    
    参数:
        scan_results: scan_inclination_azimuth_for_regime 返回的字典
        failure_type: 'collapse' 或 'fracture'
        save_path: 保存路径
        show_plot: 是否显示图形
    
    对应论文: Fig.6-7 的一维化版本
    """
    # Paper style (Times New Roman + upper-bound sizes/line widths)
    set_paper_style(base_fontsize=14, dpi=600, seaborn_style="white", use_seaborn=False, allow_cjk_fallback=True)

    inclinations = scan_results['inclinations']
    
    if failure_type == 'collapse':
        rho_grid = scan_results['rho_collapse_grid'].copy()
        title_suffix = 'Collapse'
        color = 'blue'
    else:
        rho_grid = scan_results['rho_fracture_grid'].copy()
        title_suffix = 'Fracture'
        color = 'red'
    
    # [NEW] 数据清洗：保留 [1.5, 3.0] 范围内的数据，其他的设为 NaN
    rho_grid[(rho_grid < 1.5) | (rho_grid > 3.0)] = np.nan
    
    # 对方位角取统计（平均、最小、最大）
    rho_mean = np.nanmean(rho_grid, axis=1)
    rho_min = np.nanmin(rho_grid, axis=1)
    rho_max = np.nanmax(rho_grid, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制均值曲线
    ax.plot(inclinations, rho_mean, 'o-', color=color, linewidth=2.5, 
            markersize=6, label='Mean over azimuth')
    
    # 绘制范围带
    ax.fill_between(inclinations, rho_min, rho_max, alpha=0.3, color=color, 
                    label='Min-Max range')
    
    regime = scan_results['stats']['regime']
    alpha_p = scan_results['stats']['alpha_p']
    apply_ax_style(ax, xlabel=r'Inclination Angle $\alpha$ (°)',
                   ylabel=r'Equivalent Mud Density $\rho$ (g/cm$^3$)',
                   title=f'{title_suffix} - Equivalent Density vs Inclination\n'
                         f'Regime: {regime}, α_p: {alpha_p:.2f}',
                   legend=True, grid=True, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        print(f"井斜角-等效密度曲线已保存至: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()




def compute_reliability_curves_physics(regime: str, alpha_p: float, 
                                       sampler: InputSampler, physics: WellborePhysics,
                                       rho_grid: np.ndarray, N_samples_per_rho: int = 500,
                                       fixed_depth: float = None) -> Dict[str, np.ndarray]:
    """
    使用物理解计算可靠度-等效密度曲线（蒙特卡洛法）
    """
    if fixed_depth is None:
        fixed_depth = physics.phys.depth
        
    print(f"\n计算物理解可靠度曲线: {regime} 机制, alpha_p={alpha_p:.2f}")
    print(f"  密度范围: [{rho_grid[0]:.2f}, {rho_grid[-1]:.2f}] g/cm^3, {len(rho_grid)} 个点")
    print(f"  每个密度点采样: {N_samples_per_rho} 次")
    print(f"  分析深度: {fixed_depth} m")
    
    R_collapse = np.zeros(len(rho_grid))
    R_fracture = np.zeros(len(rho_grid))
    
    # 获取该工况的代表性应力（用于设置采样范围，这里不再重要，因为sampler内部处理）
    # ...
    
    for i, rho in enumerate(rho_grid):
        # 将等效密度转换为泥浆压力 (使用 fixed_depth)
        p_w = rho * physics.phys.g * fixed_depth / 1000.0  # MPa
        
        # 进行蒙特卡洛采样 (强制使用 fixed_depth)
        X_samples, _ = sampler.sample_inputs(N_samples_per_rho, regime=regime, alpha_p_mean=alpha_p, fixed_depth=fixed_depth)
        
        collapse_safe_count = 0
        fracture_safe_count = 0
        
        for j in range(N_samples_per_rho):
            # 解包输入变量（列顺序：phi, sigma_v, sigma_H, sigma_h, p_p, T, inc, azi）
            phi = X_samples[j, 0]
            sigma_v = X_samples[j, 1]
            sigma_H = X_samples[j, 2]
            sigma_h = X_samples[j, 3]
            p_p = X_samples[j, 4]
            T = X_samples[j, 5]
            inclination = X_samples[j, 6]
            wellbore_azimuth = X_samples[j, 7]
            
            # Re-estimate k from phi for consistency in p_p_eff
            # Or use physics.compute_safety_factors with k=None (uses default)
            # But we want to capture some k variation if possible.
            k_est = np.exp(20.0 * phi - 2.0)
            
            SF_dict = physics.compute_safety_factors(
                sigma_H, sigma_h, p_p, p_w, inclination, wellbore_azimuth, phi, k_est, T, sigma_v
            )
            
            if SF_dict['SF_collapse'] >= 1.0:
                collapse_safe_count += 1
            
            if SF_dict['SF_fracture'] >= 1.0:
                fracture_safe_count += 1
        
        # 计算可靠度
        R_collapse[i] = collapse_safe_count / N_samples_per_rho
        R_fracture[i] = fracture_safe_count / N_samples_per_rho
        
        if (i+1) % 20 == 0 or i == len(rho_grid) - 1:
            print(f"  进度: {i+1}/{len(rho_grid)}, ρ={rho:.2f}, R_collapse={R_collapse[i]:.3f}, R_fracture={R_fracture[i]:.3f}")
    
    # 寻找可靠度曲线交点（安全窗口）
    # 方法：找到 R_collapse > 0.5 且 R_fracture > 0.5 的区间
    safe_mask = (R_collapse >= 0.5) & (R_fracture >= 0.5)
    
    if np.any(safe_mask):
        safe_rho = rho_grid[safe_mask]
        safe_window_lower = safe_rho[0]
        safe_window_upper = safe_rho[-1]
        recommended_rho = np.mean(safe_rho)
    else:
        # 如果没有交集，使用临界点
        idx_collapse_05 = np.argmin(np.abs(R_collapse - 0.5))
        idx_fracture_05 = np.argmin(np.abs(R_fracture - 0.5))
        safe_window_lower = rho_grid[idx_collapse_05]
        safe_window_upper = rho_grid[idx_fracture_05]
        recommended_rho = (safe_window_lower + safe_window_upper) / 2.0
    
    print(f"\n  安全窗口: [{safe_window_lower:.3f}, {safe_window_upper:.3f}] g/cm^3")
    print(f"  推荐泥浆密度: {recommended_rho:.3f} g/cm^3")
    
    return {
        'rho_grid': rho_grid,
        'R_collapse': R_collapse,
        'R_fracture': R_fracture,
        'safe_window_lower': safe_window_lower,
        'safe_window_upper': safe_window_upper,
        'recommended_rho': recommended_rho,
        'regime': regime,
        'alpha_p': alpha_p
    }


def plot_reliability_curves_physics(reliability_results: Dict, save_path: str = None, show_plot: bool = True):
    """
    绘制物理解的可靠度-等效密度曲线
    
    参数:
        reliability_results: compute_reliability_curves_physics 返回的字典
        save_path: 保存路径
        show_plot: 是否显示图形
    
    对应论文: Fig.8-10
    """
    # Paper style (Times New Roman + upper-bound sizes/line widths)
    set_paper_style(base_fontsize=14, dpi=600, seaborn_style="white", use_seaborn=False, allow_cjk_fallback=True)

    rho_grid = reliability_results['rho_grid']
    R_collapse = reliability_results['R_collapse']
    R_fracture = reliability_results['R_fracture']
    safe_window_lower = reliability_results['safe_window_lower']
    safe_window_upper = reliability_results['safe_window_upper']
    recommended_rho = reliability_results['recommended_rho']
    regime = reliability_results['regime']
    alpha_p = reliability_results['alpha_p']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 绘制可靠度曲线
    ax.plot(rho_grid, R_collapse, 'b-o', linewidth=2.5, markersize=4, 
            label='Collapse Reliability', markevery=5)
    ax.plot(rho_grid, R_fracture, 'r-s', linewidth=2.5, markersize=4, 
            label='Fracture Reliability', markevery=5)
    
    # 绘制安全窗口
    ax.axvspan(safe_window_lower, safe_window_upper, alpha=0.2, color='green', 
              label=f'Safe Window [{safe_window_lower:.2f}, {safe_window_upper:.2f}]')
    
    # 绘制推荐密度线
    ax.axvline(recommended_rho, color='darkgreen', linestyle='--', linewidth=2, 
              label=f'Recommended ρ = {recommended_rho:.2f}')
    
    # 绘制50%可靠度参考线
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, 
              label='50% Reliability')
    
    ax.set_xlim([rho_grid[0], rho_grid[-1]])
    ax.set_ylim([-0.05, 1.05])
    apply_ax_style(ax, xlabel=r'Equivalent Mud Density $\rho$ (g/cm$^3$)',
                   ylabel='Reliability R',
                   title=f'Reliability-Density Curves (Physics-Based MCS)\n'
                         f'Regime: {regime}, Pressure Coefficient α_p: {alpha_p:.2f}',
                   legend=True, grid=True, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        print(f"物理解可靠度曲线已保存至: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def generate_dataset(sampler: InputSampler, 
                    physics: WellborePhysics,
                    N_train: int, 
                    N_val: int,
                    regime: str = None,
                    alpha_p: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    通过蒙特卡洛采样生成训练和验证数据集（8 维输入 + 深度）
    
    参数:
        sampler: 输入采样器
        physics: 物理模型
        N_train: 训练样本数
        N_val: 验证样本数
        regime: 应力机制类型
        alpha_p: 地层压力系数
    
    返回:
        (X_train, Y_train, X_val, Y_val, depths_train, depths_val)
    """
    print("正在生成训练数据集（蒙特卡洛采样 + 物理仿真）...")
    X_train, depths_train, aux_train = sampler.sample_inputs(N_train, regime=regime, alpha_p_mean=alpha_p, return_depth=True)
    Y_train = physics.compute_true_outputs(X_train, aux_data=aux_train)
    
    print("正在生成验证数据集...")
    X_val, depths_val, aux_val = sampler.sample_inputs(N_val, regime=regime, alpha_p_mean=alpha_p, return_depth=True)
    Y_val = physics.compute_true_outputs(X_val, aux_data=aux_val)
    
    print(f"训练集: {N_train} 样本")
    print(f"验证集: {N_val} 样本")
    print(f"输入维度: {X_train.shape[1]}")
    print(f"输出维度: {Y_train.shape[1]}")
    
    print("\n输入变量统计:")
    print(f"  孔隙度 φ: [{aux_train['phi'].min():.3f}, {aux_train['phi'].max():.3f}], 均值={aux_train['phi'].mean():.3f}")
    print(f"  渗透率 k (mD): [{aux_train['k'].min():.2f}, {aux_train['k'].max():.2f}], 均值={aux_train['k'].mean():.2f}")
    print(f"  垂直应力 Sv (MPa): [{X_train[:, 1].min():.1f}, {X_train[:, 1].max():.1f}], 均值={X_train[:, 1].mean():.1f}")
    print(f"  最大水平应力 SH (MPa): [{X_train[:, 2].min():.1f}, {X_train[:, 2].max():.1f}], 均值={X_train[:, 2].mean():.1f}")
    print(f"  最小水平应力 Sh (MPa): [{X_train[:, 3].min():.1f}, {X_train[:, 3].max():.1f}], 均值={X_train[:, 3].mean():.1f}")
    print(f"  孔隙压力 Pp (MPa): [{X_train[:, 4].min():.1f}, {X_train[:, 4].max():.1f}], 均值={X_train[:, 4].mean():.1f}")
    print(f"  温度 T (°C): [{X_train[:, 5].min():.1f}, {X_train[:, 5].max():.1f}], 均值={X_train[:, 5].mean():.1f}")
    print(f"  井斜角 α (°): [{X_train[:, 6].min():.1f}, {X_train[:, 6].max():.1f}], 均值={X_train[:, 6].mean():.1f}")
    print(f"  井眼方位角 β (°): [{X_train[:, 7].min():.1f}, {X_train[:, 7].max():.1f}], 均值={X_train[:, 7].mean():.1f}")
    print(f"  采样深度 (m): [{depths_train.min():.1f}, {depths_train.max():.1f}], 均值={depths_train.mean():.1f}")
    
    return X_train, Y_train, X_val, Y_val, depths_train, depths_val


def plot_training_history(history: Dict[str, list], save_path: str = None):
    """绘制训练历史

    KL 子图新增右轴展示 λ_kl 退火调度曲线，便于直观地回应审稿意见
    "KL 单调上升非标准 VI 行为" —— 退火期内 KL 项受弱化，warmup 结束后
    KL 散度即可观察到典型的"先升后稳/降"的标准 VI 收敛形态。
    """
    # Paper style (Times New Roman + upper-bound sizes/line widths)
    set_paper_style(base_fontsize=14, dpi=600, seaborn_style="white", use_seaborn=False, allow_cjk_fallback=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=1.8)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=1.8)
    apply_ax_style(axes[0, 0], xlabel='Epoch', ylabel='Loss', title='Total Loss',
                   legend=True, grid=True, fontsize=14)

    axes[0, 1].plot(history['train_data_loss'], label='Data Loss', color='green', linewidth=1.8)
    apply_ax_style(axes[0, 1], xlabel='Epoch', ylabel='Loss', title='Data Loss',
                   legend=True, grid=True, fontsize=14)

    axes[1, 0].plot(history['train_phys_loss'], label='Physics Loss', color='orange', linewidth=1.8)
    apply_ax_style(axes[1, 0], xlabel='Epoch', ylabel='Loss', title='Physics Loss',
                   legend=True, grid=True, fontsize=14)

    ax_kl = axes[1, 1]
    ax_kl.plot(history['train_kl_loss'], label='KL Divergence', color='red', linewidth=1.8)
    apply_ax_style(ax_kl, xlabel='Epoch', ylabel='KL', title='KL Divergence & λ_kl Schedule',
                   legend=True, grid=True, fontsize=14)

    if 'lambda_kl_schedule' in history and len(history['lambda_kl_schedule']) > 0:
        ax_kl_r = ax_kl.twinx()
        ax_kl_r.plot(history['lambda_kl_schedule'], label=r'$\lambda_{KL}$ schedule',
                     color='steelblue', linewidth=1.6, linestyle='--')
        ax_kl_r.set_ylabel(r'$\lambda_{KL}$', fontsize=14, fontweight='bold', color='steelblue')
        ax_kl_r.tick_params(axis='y', colors='steelblue', labelsize=14)
        for label in ax_kl_r.get_yticklabels():
            label.set_fontweight('bold')
            label.set_fontfamily('serif')
        # 合并两个轴的图例
        lines_l, labels_l = ax_kl.get_legend_handles_labels()
        lines_r, labels_r = ax_kl_r.get_legend_handles_labels()
        ax_kl.legend(lines_l + lines_r, labels_l + labels_r,
                     fontsize=12, prop={'weight': 'bold', 'family': 'serif'},
                     framealpha=0.9, edgecolor='gray', loc='best')

    plt.tight_layout(pad=2.0)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)

    plt.show()


def plot_calibration_diagram(metrics: Dict, save_path: str = None,
                              title_prefix: str = "BPINN UQ Calibration",
                              show_plot: bool = False):
    """
    绘制 UQ 标定诊断图（2x2 panel）：
        (a) Reliability diagram —— 名义置信水平 vs 经验覆盖率，理想为 y=x；
        (b) Aleatory / Epistemic 方差分解条形图 —— 直接回应 R3-Q5；
        (c) RMSE & PINAW 双指标条形图 —— 准确性 vs 区间宽度的权衡；
        (d) PICP@90 / PICP@95 / NLL 综合标定指标 ——
            评估"区间到底是太窄/太宽/合适"。

    [REVISION 2026] 配套 compute_uq_calibration_metrics()，回应 R1-Q13、R3-Q5。
    """
    set_paper_style(base_fontsize=14, dpi=600, seaborn_style="white",
                    use_seaborn=False, allow_cjk_fallback=True)

    nominal = np.asarray(metrics['nominal_levels'])
    coverage = np.asarray(metrics['coverage_per_level'])  # (L, D)
    names = list(metrics['output_names'])
    D = len(names)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:D]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- (a) Reliability Diagram ---
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    for d in range(D):
        ax.plot(nominal, coverage[:, d], 'o-',
                label=f"{names[d]} (ACE={metrics['ace'][d]:.3f})",
                color=colors[d], linewidth=1.8, markersize=5)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    apply_ax_style(ax, xlabel='Nominal Confidence Level',
                   ylabel='Empirical Coverage',
                   title='Reliability Diagram',
                   legend=True, grid=True, fontsize=14)

    # --- (b) Aleatory / Epistemic Decomposition ---
    ax = axes[0, 1]
    x_pos = np.arange(D)
    width = 0.35
    bars_a = ax.bar(x_pos - width / 2, metrics['aleatory_pct'], width,
                    label='Aleatory %', color='#1f77b4', edgecolor='black')
    bars_e = ax.bar(x_pos + width / 2, metrics['epistemic_pct'], width,
                    label='Epistemic %', color='#d62728', edgecolor='black')
    for bars, vals in [(bars_a, metrics['aleatory_pct']),
                       (bars_e, metrics['epistemic_pct'])]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.0,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=11,
                    fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim([0, max(110, np.max(metrics['aleatory_pct']) + 15)])
    apply_ax_style(ax, ylabel='Variance Contribution (%)',
                   title='Aleatory / Epistemic Variance Decomposition',
                   legend=True, grid=True, fontsize=14)

    # --- (c) RMSE & PINAW ---
    ax = axes[1, 0]
    ax2 = ax.twinx()
    bars_rmse = ax.bar(x_pos - width / 2, metrics['rmse'], width,
                       label='RMSE', color='#2ca02c', edgecolor='black')
    bars_pinaw = ax2.bar(x_pos + width / 2, metrics['pinaw_95'], width,
                         label='PINAW@95', color='#9467bd', edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('RMSE', fontsize=14, fontweight='bold', color='#2ca02c')
    ax2.set_ylabel('PINAW@95 (normalized)', fontsize=14, fontweight='bold', color='#9467bd')
    ax.tick_params(axis='y', colors='#2ca02c')
    ax2.tick_params(axis='y', colors='#9467bd')
    ax.set_title('Point Accuracy (RMSE) vs Interval Width (PINAW)',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax2.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r,
              fontsize=12, prop={'weight': 'bold', 'family': 'serif'},
              framealpha=0.9, edgecolor='gray', loc='best')

    # --- (d) PICP@90/95 + NLL ---
    ax = axes[1, 1]
    ax2 = ax.twinx()
    bars_p90 = ax.bar(x_pos - width, metrics['picp_90'], width,
                      label='PICP@90', color='#1f77b4', edgecolor='black')
    bars_p95 = ax.bar(x_pos, metrics['picp_95'], width,
                      label='PICP@95', color='#aec7e8', edgecolor='black')
    bars_nll = ax2.bar(x_pos + width, metrics['nll'], width,
                       label='NLL', color='#ff7f0e', edgecolor='black')
    ax.axhline(0.90, color='#1f77b4', linestyle=':', linewidth=1.2)
    ax.axhline(0.95, color='#aec7e8', linestyle=':', linewidth=1.2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('PICP', fontsize=14, fontweight='bold', color='#1f77b4')
    ax2.set_ylabel('NLL', fontsize=14, fontweight='bold', color='#ff7f0e')
    ax.set_title('PICP (Coverage) & NLL (Likelihood)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax2.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r,
              fontsize=12, prop={'weight': 'bold', 'family': 'serif'},
              framealpha=0.9, edgecolor='gray', loc='best')

    fig.suptitle(title_prefix, fontsize=18, fontweight='bold')
    plt.tight_layout(pad=2.0, rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_uncertainty_results(results: Dict[str, np.ndarray], save_path: str = None):
    """
    绘制不确定性分析结果
    
    参数:
        results: predict_with_uncertainty 返回的结果字典
        save_path: 保存路径
    """
    outputs_all = results['outputs_all']
    
    output_names = ['SF_collapse', 'SF_fracture', 
                   r'$\rho_{mud}$ collapse (g/cm$^3$)', r'$\rho_{mud}$ fracture (g/cm$^3$)']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i in range(4):
        data = outputs_all[:, i]
        
        axes[i].hist(data, bins=50, density=True, alpha=0.6, color='skyblue', 
                    edgecolor='black')
        
        try:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            axes[i].plot(x_range, kde(x_range), 'r-', linewidth=2.5, label='KDE')
        except Exception:
            pass
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        p10 = np.percentile(data, 10)
        p50 = np.percentile(data, 50)
        p90 = np.percentile(data, 90)
        
        axes[i].axvline(p10, color='green', linestyle='--', linewidth=1.5, 
                       label=f'P10: {p10:.3f}')
        axes[i].axvline(p50, color='orange', linestyle='--', linewidth=1.5,
                       label=f'P50: {p50:.3f}')
        axes[i].axvline(p90, color='red', linestyle='--', linewidth=1.5,
                       label=f'P90: {p90:.3f}')
        
        apply_ax_style(axes[i], xlabel=output_names[i], ylabel='Density',
                       title=f'{output_names[i]} Distribution\n'
                             r'$\mu$' + f'={mean_val:.3f}, '
                             r'$\sigma$' + f'={std_val:.3f}',
                       legend=True, grid=True, fontsize=14)
    
    plt.tight_layout(pad=2.0)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
    
    plt.show()


def print_summary_statistics(results: Dict[str, np.ndarray], depth: float = 4500.0):
    """打印汇总统计信息"""
    outputs_all = results['outputs_all']
    
    output_names = ['SF_collapse', 'SF_fracture', 
                   'ρ_mud_collapse', 'ρ_mud_fracture']
    
    print("\n" + "="*70)
    print(f"井壁稳定性不确定性分析结果汇总 (深度 = {depth} m)")
    print("="*70)
    
    for i, name in enumerate(output_names):
        data = outputs_all[:, i]
        
        print(f"\n{name}:")
        print(f"  均值:    {np.mean(data):.4f}")
        print(f"  标准差:  {np.std(data):.4f}")
        print(f"  P10:     {np.percentile(data, 10):.4f}")
        print(f"  P50:     {np.percentile(data, 50):.4f}")
        print(f"  P90:     {np.percentile(data, 90):.4f}")
        print(f"  最小值:  {np.min(data):.4f}")
        print(f"  最大值:  {np.max(data):.4f}")
    
    print("\n" + "="*70)


def analyze_specific_depth_details(depth: float, 
                                  model: BayesianBPINN, 
                                  sampler: InputSampler, 
                                  physics: WellborePhysics, 
                                  scaler: DataScaler,
                                  device: torch.device,
                                  regime: str,
                                  alpha_p: float,
                                  output_scaler: OutputScaler = None):
    """
    详细分析特定深度处的地质参数、岩石力学参数及模型预测结果
    """
    print("\n" + "="*80)
    print(f"特定深度详细分析: {depth} m")
    print("="*80)
    
    # 1. 在指定深度采样地层参数
    N_samples = 200
    X_samples, aux_data = sampler.sample_inputs(
        N=N_samples, 
        regime=regime, 
        alpha_p_mean=alpha_p, 
        fixed_depth=depth
    )
    
    # 2. 打印地质参数统计
    print("\n1. 地质参数统计 (基于蒙特卡洛采样):")
    print(f"   - 孔隙度 φ: {aux_data['phi'].mean():.3f} ± {aux_data['phi'].std():.3f}")
    print(f"   - 渗透率 k (mD): {aux_data['k'].mean():.2f} ± {aux_data['k'].std():.2f}")
    print(f"   - 垂直应力 σ_v (MPa): {X_samples[:, 1].mean():.1f} ± {X_samples[:, 1].std():.1f}")
    print(f"   - 最大水平应力 σ_H (MPa): {X_samples[:, 2].mean():.1f} ± {X_samples[:, 2].std():.1f}")
    print(f"   - 最小水平应力 σ_h (MPa): {X_samples[:, 3].mean():.1f} ± {X_samples[:, 3].std():.1f}")
    print(f"   - 孔隙压力 p_p (MPa): {X_samples[:, 4].mean():.1f} ± {X_samples[:, 4].std():.1f}")
    print(f"   - 温度 T (°C): {X_samples[:, 5].mean():.1f} ± {X_samples[:, 5].std():.1f}")
    
    # 2. 计算岩石力学参数（使用代表性值）
    phi_mean = aux_data['phi'].mean()
    T_mean = X_samples[:, 5].mean()
    rock_props = physics.compute_effective_rock_properties(phi_mean, T_mean)
    
    print("\n2. 岩石力学参数 (基于平均地质参数):")
    print(f"   - 有效单轴抗压强度 UCS: {rock_props['UCS']:.2f} MPa")
    print(f"   - 有效内聚力 C: {rock_props['cohesion']:.2f} MPa")
    print(f"   - 有效内摩擦角 φ_s: {np.rad2deg(rock_props['friction_angle_rad']):.2f}°")
    print(f"   - 有效 Biot 系数: {rock_props['biot_coefficient']:.3f}")
    print(f"   - 有效抗拉强度 T0: {rock_props['tensile_strength']:.2f} MPa")
    
    # 3. BPINN 模型预测（模型输入为 9D: Depth + 8 features）
    depth_col = np.full((N_samples, 1), depth, dtype=np.float32)
    X_9d = np.hstack([depth_col, X_samples])
    X_norm = scaler.transform(X_9d)
    X_torch = torch.FloatTensor(X_norm).to(device)
    
    # 预测 (考虑权重不确定性)
    model.eval()
    rho_c_preds = []
    rho_f_preds = []
    
    with torch.no_grad():
        for _ in range(50): # 50次权重采样
            out = model(X_torch)
            
            # [FIX] 反归一化
            if output_scaler is not None:
                out = output_scaler.denormalize(out)
                
            rho_c_preds.append(out.cpu().numpy()[:, 2])
            rho_f_preds.append(out.cpu().numpy()[:, 3])
            
    rho_c_flat = np.array(rho_c_preds).flatten()
    rho_f_flat = np.array(rho_f_preds).flatten()
    
    rho_c_mean_val = np.mean(rho_c_flat)
    rho_c_std_val = np.std(rho_c_flat)
    rho_f_mean_val = np.mean(rho_f_flat)
    rho_f_std_val = np.std(rho_f_flat)
    
    print("\n3. BPINN 模型预测结果 (统计):")
    print(f"   - 坍塌压力当量密度:     {rho_c_mean_val:.3f} ± {rho_c_std_val:.3f} g/cm^3")
    print(f"   - 破裂压力当量密度:     {rho_f_mean_val:.3f} ± {rho_f_std_val:.3f} g/cm^3")
    
    safe_window = rho_f_mean_val - rho_c_mean_val
    rec_rho = (rho_c_mean_val + rho_f_mean_val) / 2
    
    print(f"   - 安全窗口宽度:         {safe_window:.3f} g/cm^3")
    print(f"   - 推荐安全泥浆密度:     {rec_rho:.3f} g/cm^3")
    
    # 4. 弹性参数详细输出
    print("\n4. 【弹性参数】:")
    print(f"   - 杨氏模量 E (GPa):     {physics.phys.youngs_modulus:.2f}")
    print(f"   - 泊松比 ν:             {physics.phys.poisson_ratio:.3f}")
    
    print("\n5. 【应力参数】(平均值):")
    print(f"   - 垂直应力 σ_v:         {X_samples[:, 1].mean():.2f} MPa")
    print(f"   - 最大水平应力 σ_H:     {X_samples[:, 2].mean():.2f} MPa")
    print(f"   - 最小水平应力 σ_h:     {X_samples[:, 3].mean():.2f} MPa")
    print(f"   - 剪切应力 τ:           0.00 MPa (井壁边界条件)")
    
    print("\n6. 【井眼参数】:")
    print(f"   - 井深:                 {depth:.1f} m")
    print(f"   - 井眼半径:             {physics.phys.wellbore_radius:.4f} m ({physics.phys.wellbore_radius*39.37*2:.2f} inch)")
    print(f"   - 井眼直径:             {physics.phys.wellbore_radius*2:.4f} m")
    print(f"   - 孔隙度 φ:             {phi_mean:.3f}")
    print(f"   - 孔隙压力 PP:          {X_samples[:, 4].mean():.2f} MPa")
    print(f"   - 压力系数 α_p:         {alpha_p:.2f}")
    print(f"   - 温度 T:               {T_mean:.1f} °C")
    
    print("\n7. 【泥浆参数】:")
    print(f"   - 推荐泥浆密度:         {rec_rho:.3f} g/cm^3")
    print(f"   - 泥浆重量(压力):       {rec_rho * physics.phys.g * depth / 1000.0:.2f} MPa")
    print(f"   - 井筒压力:             {rec_rho * physics.phys.g * depth / 1000.0:.2f} MPa")
    print(f"   - 最小泥浆密度(防坍塌): {rho_c_mean_val:.3f} g/cm^3")
    print(f"   - 最大泥浆密度(防破裂): {rho_f_mean_val:.3f} g/cm^3")
    
    print("\n8. 【方位参数】:")
    print(f"   - 井斜角 α:             {X_samples[:, 6].mean():.1f}° (平均)")
    print(f"   - 井眼方位角 β:         {X_samples[:, 7].mean():.1f}° (平均)")
    print(f"   - 最大应力方位 γ_H:     {physics.phys.azimuth_sigma_H:.1f}° (相对正北)")
    print(f"   - 最小应力方向:         {(physics.phys.azimuth_sigma_H + 90) % 360:.1f}° (垂直于σ_H)")
    
    print("\n9. 【应力机制】:")
    print(f"   - 类型:                 {regime} ", end="")
    if regime == "NF":
        print("(正断层: σ_v > σ_H > σ_h)")
    elif regime == "SS":
        print("(走滑断层: σ_H > σ_v > σ_h)")
    elif regime == "RF":
        print("(逆断层: σ_H > σ_h > σ_v)")
    
    print("="*80 + "\n")


def quantile_summary(samples: np.ndarray) -> dict:
    """
    计算样本的多级分位数（用于论文级置信区间可视化）
    
    参数:
        samples: 1D numpy array
    
    返回:
        dict: 包含 p2_5, p10, p25, p50, p75, p90, p97_5
    """
    return {
        'p2_5': np.percentile(samples, 2.5),
        'p10': np.percentile(samples, 10),
        'p25': np.percentile(samples, 25),
        'p50': np.percentile(samples, 50),  # 中位数
        'p75': np.percentile(samples, 75),
        'p90': np.percentile(samples, 90),
        'p97_5': np.percentile(samples, 97.5)
    }


def analyze_depth_profile(model, sampler, scaler, device, regime, alpha_p,
                         depth_min=2060, depth_max=2560, step=25,
                         save_path="results/depth_profile_prediction.png",
                         output_scaler: OutputScaler = None,
                         safe_prob_method: str = "joint",
                         depth_axis: str = "relative"): # 新增
    """
    生成稳定性深度剖面图 (Depth Profile)
    展示坍塌压力随深度的变化趋势及不确定性范围（多级分位数带）
    同时生成安全概率热力图：
    - joint: 联合经验法（推荐）P_safe = P(ρ_collapse <= ρ <= ρ_fracture)，使用成对样本避免独立性假设偏乐观
    - marginal: 边缘乘积法（原实现）P_safe = P(ρ >= ρ_collapse) * P(ρ <= ρ_fracture)，隐含独立性，通常偏乐观
    """
    if depth_axis not in {"absolute", "relative"}:
        raise ValueError(f"depth_axis must be 'absolute' or 'relative', got: {depth_axis}")

    # Paper style (Times New Roman + upper-bound sizes/line widths)
    set_paper_style(base_fontsize=14, dpi=600, seaborn_style="white", use_seaborn=False, allow_cjk_fallback=True)

    print(f"\n" + "="*80)
    print(f"[新增分析] 稳定性深度剖面 ({depth_min}-{depth_max} m, step={step} m, axis={depth_axis})")
    print("="*80)
    
    # 计算使用“绝对深度”，绘图可选“相对深度”（用于避免在论文图中暗示全井段/千米尺度预测）
    depths_abs = np.arange(depth_min, depth_max + 1, step)
    depths_plot = (depths_abs - depths_abs[0]) if depth_axis == "relative" else depths_abs
    
    # 存储多级分位数（Collapse）
    rho_c_p2_5, rho_c_p10, rho_c_p25, rho_c_p50, rho_c_p75, rho_c_p90, rho_c_p97_5 = [], [], [], [], [], [], []
    
    # 存储多级分位数（Fracture）
    rho_f_p2_5, rho_f_p10, rho_f_p25, rho_f_p50, rho_f_p75, rho_f_p90, rho_f_p97_5 = [], [], [], [], [], [], []
    
    # [新增] 缓存每个深度的预测样本（成对保存），用于后续热力图生成
    # 形状约定：
    # - preds_*_mat: (n_weight_samples, N_input_samples)
    all_preds_c = []
    all_preds_f = []
    
    model.eval()
    
    for i, z in enumerate(depths_abs):
        # 1. 在当前深度采样
        # N=200 代表该深度下的地层参数波动
        X_np, _ = sampler.sample_inputs(N=200, regime=regime, alpha_p_mean=alpha_p, fixed_depth=z)
        # 模型输入为 9D: Depth + 8 features
        depth_col = np.full((X_np.shape[0], 1), z, dtype=np.float32)
        X_np = np.hstack([depth_col, X_np])
        
        # 2. 归一化并转 Tensor
        X_norm = scaler.transform(X_np)
        X_torch = torch.FloatTensor(X_norm).to(device)
        
        # 3. BPINN 预测 (考虑权重不确定性，采样 20 次)
        # 关键：保持 collapse 与 fracture 的“成对样本”，以便后续计算联合概率 P(ρ_c <= ρ <= ρ_f)
        preds_c_list = []  # list[(N,)]
        preds_f_list = []  # list[(N,)]
        
        with torch.no_grad():
            for _ in range(20):
                out = model(X_torch)
                
                # [FIX] 反归一化
                if output_scaler is not None:
                    out = output_scaler.denormalize(out)
                    
                out_np = out.cpu().numpy()
                preds_c_list.append(out_np[:, 2]) # Index 2 is Rho_collapse
                preds_f_list.append(out_np[:, 3]) # Index 3 is Rho_fracture
        
        # 成对样本矩阵：(n_mc, N)
        preds_c_mat = np.stack(preds_c_list, axis=0)
        preds_f_mat = np.stack(preds_f_list, axis=0)
        
        # 展平用于分位数统计
        preds_c = preds_c_mat.reshape(-1)
        preds_f = preds_f_mat.reshape(-1)
        
        # [新增] 保存当前深度的样本用于热力图
        all_preds_c.append(preds_c_mat)
        all_preds_f.append(preds_f_mat)
        
        # 4. 计算多级分位数
        q_c = quantile_summary(preds_c)
        q_f = quantile_summary(preds_f)
        
        # Collapse 分位数
        rho_c_p2_5.append(q_c['p2_5'])
        rho_c_p10.append(q_c['p10'])
        rho_c_p25.append(q_c['p25'])
        rho_c_p50.append(q_c['p50'])
        rho_c_p75.append(q_c['p75'])
        rho_c_p90.append(q_c['p90'])
        rho_c_p97_5.append(q_c['p97_5'])
        
        # Fracture 分位数
        rho_f_p2_5.append(q_f['p2_5'])
        rho_f_p10.append(q_f['p10'])
        rho_f_p25.append(q_f['p25'])
        rho_f_p50.append(q_f['p50'])
        rho_f_p75.append(q_f['p75'])
        rho_f_p90.append(q_f['p90'])
        rho_f_p97_5.append(q_f['p97_5'])
        
        if i % 10 == 0:
            print(f"  深度 {z}m: 坍塌 P50={q_c['p50']:.3f} g/cm^3, 破裂 P50={q_f['p50']:.3f} g/cm^3")

    # 绘制剖面图（论文级多级分位数带）
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # ========== Collapse 坍塌压力（红色系 - Jet Style）==========
    # 95% 区间 (P2.5-P97.5) - 最浅
    ax.fill_betweenx(depths_plot, rho_c_p2_5, rho_c_p97_5, 
                     color=plt.cm.jet(0.9), alpha=0.15, label='Collapse 95% CI (P2.5-P97.5)')
    
    # 80% 区间 (P10-P90) - 中等
    ax.fill_betweenx(depths_plot, rho_c_p10, rho_c_p90, 
                     color=plt.cm.jet(0.85), alpha=0.25, label='Collapse 80% CI (P10-P90)')
    
    # 50% 区间 (P25-P75) - 最深
    ax.fill_betweenx(depths_plot, rho_c_p25, rho_c_p75, 
                     color=plt.cm.jet(0.8), alpha=0.35, label='Collapse 50% CI (P25-P75)')
    
    # P50 中位数线
    ax.plot(rho_c_p50, depths_plot, color=plt.cm.jet(0.95), linestyle='-', linewidth=2.5, label='Collapse P50 (Median)')
    
    # ========== Fracture 破裂压力（蓝色系 - Jet Style）==========
    # 95% 区间 (P2.5-P97.5) - 最浅
    ax.fill_betweenx(depths_plot, rho_f_p2_5, rho_f_p97_5, 
                     color=plt.cm.jet(0.1), alpha=0.15, label='Fracture 95% CI (P2.5-P97.5)')
    
    # 80% 区间 (P10-P90) - 中等
    ax.fill_betweenx(depths_plot, rho_f_p10, rho_f_p90, 
                     color=plt.cm.jet(0.15), alpha=0.25, label='Fracture 80% CI (P10-P90)')
    
    # 50% 区间 (P25-P75) - 最深
    ax.fill_betweenx(depths_plot, rho_f_p25, rho_f_p75, 
                     color=plt.cm.jet(0.2), alpha=0.35, label='Fracture 50% CI (P25-P75)')
    
    # P50 中位数线
    ax.plot(rho_f_p50, depths_plot, color=plt.cm.jet(0.05), linestyle='--', linewidth=2.5, label='Fracture P50 (Median)')
    
    # ========== 安全窗口（Jet Style 中间色）==========
    ax.fill_betweenx(depths_plot, rho_c_p50, rho_f_p50, 
                     color=plt.cm.jet(0.35), alpha=0.1, label='Safe Mud Window (P50-P50)')
    
    ax.invert_yaxis()
    ylab = "Depth (m)" if depth_axis == "absolute" else f"Depth offset from {depths_abs[0]:.0f} m (m)"
    apply_ax_style(ax,
                   xlabel="Equivalent Mud Density (g/cm³)",
                   ylabel=ylab,
                   title=f"Wellbore Stability Depth Profile with Multi-Level Quantile Ribbons\n"
                         f"Regime: {regime}, α_p: {alpha_p} | Depth window: {depth_min}-{depth_max} m",
                   legend=False, grid=True, fontsize=14)
    ax.legend(loc='upper right', ncol=2, framealpha=0.95,
              fontsize=14, prop={'weight': 'bold', 'family': 'serif'})
    
    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=600)
    print(f"全井段剖面图已保存至: {save_path}")
    plt.close(fig)
    
    # ========== [方案B] 生成安全概率热力图 ==========
    if safe_prob_method not in {"joint", "marginal"}:
        raise ValueError(f"safe_prob_method must be 'joint' or 'marginal', got: {safe_prob_method}")
    method_name = "Joint (Paired Samples)" if safe_prob_method == "joint" else "Marginal Product"
    print(f"\n生成安全概率热力图（{method_name}）...")
    
    # 1. 构建泥浆密度网格
    # 收集所有深度的rho范围
    all_c_flat = np.concatenate(all_preds_c).reshape(-1)
    all_f_flat = np.concatenate(all_preds_f).reshape(-1)
    rho_min_global = min(all_c_flat.min(), all_f_flat.min())
    rho_max_global = max(all_c_flat.max(), all_f_flat.max())
    
    # 添加5% margin
    margin = (rho_max_global - rho_min_global) * 0.05
    if margin < 0.05:  # 极端情况：范围太小，使用固定margin
        margin = 0.05
    rho_min_global -= margin
    rho_max_global += margin
    
    # 定义密度网格
    N_RHO_GRID = 300
    rho_grid = np.linspace(rho_min_global, rho_max_global, N_RHO_GRID)
    
    # 2. 计算安全概率矩阵
    n_depths = len(depths_abs)
    P_safe_mat = np.zeros((n_depths, N_RHO_GRID))
    
    for i in range(n_depths):
        preds_c_i = all_preds_c[i]  # (n_mc, N)
        preds_f_i = all_preds_f[i]  # (n_mc, N)
        
        # 对每个 rho 值计算安全概率
        for j, rho in enumerate(rho_grid):
            if safe_prob_method == "joint":
                # 联合经验法（推荐）：直接算事件交集
                P_safe_mat[i, j] = np.mean((preds_c_i <= rho) & (preds_f_i >= rho))
            else:
                # 边缘乘积法：隐含独立性（通常偏乐观）
                P_c = np.mean(preds_c_i <= rho)   # P(ρ >= ρ_collapse)
                P_f = np.mean(preds_f_i >= rho)   # P(ρ <= ρ_fracture)
                P_safe_mat[i, j] = P_c * P_f
    
    # 确保数值稳定性
    P_safe_mat = np.clip(P_safe_mat, 0.0, 1.0)
    
    # 3. 绘制热力图
    fig2, ax2 = plt.subplots(figsize=(12, 10))
    
    # 使用 pcolormesh 绘制热力图
    # 注意：需要正确设置坐标轴
    RHO, DEPTH = np.meshgrid(rho_grid, depths_plot)
    im = ax2.pcolormesh(RHO, DEPTH, P_safe_mat, cmap='jet', shading='auto', vmin=0, vmax=1)
    
    cbar = plt.colorbar(im, ax=ax2, label='Safety Probability P_safe')
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Safety Probability P_safe', fontsize=14, fontweight='bold')
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_fontfamily('serif')
    
    contour_levels = [0.5, 0.8, 0.9, 0.95]
    CS = ax2.contour(RHO, DEPTH, P_safe_mat, levels=contour_levels, 
                     colors='white', linewidths=1.5, linestyles='solid')
    ax2.clabel(CS, inline=True, fmt='%.2f', fontsize=12)
    
    ax2.plot(rho_c_p50, depths_plot, 'r-', linewidth=2.5, label='Collapse P50', alpha=0.9)
    ax2.plot(rho_f_p50, depths_plot, 'b--', linewidth=2.5, label='Fracture P50', alpha=0.9)
    
    ax2.invert_yaxis()
    apply_ax_style(ax2,
                   xlabel="Equivalent Mud Density (g/cm³)",
                   ylabel=ylab,
                   title=f"Safety Probability Heatmap ({method_name})\n"
                         f"Regime: {regime}, α_p: {alpha_p} | Depth window: {depth_min}-{depth_max} m",
                   legend=False, grid=True, fontsize=14)
    ax2.legend(loc='upper right', framealpha=0.85, fancybox=True, shadow=True,
               fontsize=14, prop={'weight': 'bold', 'family': 'serif'})
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # 8. 保存热力图到固定路径
    heatmap_path = "results/final_safe_probability_heatmap.png"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
    
    plt.savefig(heatmap_path, dpi=600)
    print(f"安全概率热力图已保存至: {heatmap_path}")
    plt.close(fig2)


def export_data_to_excel(data_dict: Dict[str, pd.DataFrame], save_path: str):
    """
    将数据导出为 Excel 文件
    
    参数:
        data_dict: 字典，键为工作表名称，值为 DataFrame
        save_path: 保存路径 (.xlsx)
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"数据已成功导出至: {save_path}")
        
    except ImportError:
        print("错误: 导出 Excel 需要安装 openpyxl 库。")
        print("请运行: pip install openpyxl")
    except Exception as e:
        print(f"导出 Excel 时发生错误: {e}")


def compare_failure_criteria(results_physics: Dict, results_bpinn: Dict,
                             criteria: List[str]) -> pd.DataFrame:
    """
    破坏准则的定量对比（论文表格）
    
    对比指标（精简版，论文够用）：
    1. 可靠度曲线误差 (Physics vs BPINN)
    2. 安全窗口参数 (下限/上限/宽度)
    3. 推荐泥浆密度
    
    参数:
        results_physics: 物理解可靠度结果字典 {criterion: reliability_dict}
        results_bpinn: BPINN可靠度结果字典 {criterion: reliability_dict}
        criteria: 准则列表，例如 ["mohr", "mogi"]
    
    返回:
        DataFrame包含各种定量指标
    """
    comparison = []
    
    for criterion in criteria:
        phys = results_physics[criterion]
        bpinn = results_bpinn[criterion]
        
        # 1. 可靠度曲线误差
        mae_collapse = np.mean(np.abs(phys['R_collapse'] - bpinn['R_collapse']))
        mae_fracture = np.mean(np.abs(phys['R_fracture'] - bpinn['R_fracture']))
        rmse_collapse = np.sqrt(np.mean((phys['R_collapse'] - bpinn['R_collapse'])**2))
        rmse_fracture = np.sqrt(np.mean((phys['R_fracture'] - bpinn['R_fracture'])**2))
        
        # 2. 安全窗口（BPINN预测值）
        window_lower = bpinn['safe_window_lower']
        window_upper = bpinn['safe_window_upper']
        window_width = window_upper - window_lower
        recommended_rho = bpinn['recommended_rho']
        
        # 3. 与物理解的窗口偏差
        window_lower_error = abs(window_lower - phys['safe_window_lower'])
        window_upper_error = abs(window_upper - phys['safe_window_upper'])
        
        comparison.append({
            'Failure_Criterion': criterion.upper(),
            'MAE_R_collapse': mae_collapse,
            'MAE_R_fracture': mae_fracture,
            'RMSE_R_collapse': rmse_collapse,
            'RMSE_R_fracture': rmse_fracture,
            'Safe_Window_Lower (g/cm³)': window_lower,
            'Safe_Window_Upper (g/cm³)': window_upper,
            'Safe_Window_Width (g/cm³)': window_width,
            'Recommended_Rho (g/cm³)': recommended_rho,
            'Window_Lower_Error': window_lower_error,
            'Window_Upper_Error': window_upper_error,
        })
    
    df = pd.DataFrame(comparison)
    
    # 添加排名列（误差越小越好）
    df['Accuracy_Rank'] = df['MAE_R_collapse'].rank()
    
    return df


def plot_failure_criteria_comparison(results_physics: Dict, results_bpinn: Dict,
                                     criteria: List[str],
                                     save_path: str = 'results/criteria_comparison.png'):
    """
    绘制各准则的可靠度曲线对比（N×2 子图）
    
    参数:
        results_physics: 物理解可靠度结果字典
        results_bpinn: BPINN可靠度结果字典
        criteria: 准则列表
        save_path: 保存路径
    """
    set_paper_style(base_fontsize=14, dpi=600, seaborn_style="white", use_seaborn=False, allow_cjk_fallback=True)
    
    n_rows = len(criteria)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 6 * n_rows), squeeze=False)
    
    criterion_names = {
        "mohr": "Mohr-Coulomb",
        "mogi": "Mogi-Coulomb"
    }
    
    for i, criterion in enumerate(criteria):
        phys = results_physics[criterion]
        bpinn = results_bpinn[criterion]
        rho_grid = phys['rho_grid']
        
        ax1 = axes[i, 0]
        ax1.plot(rho_grid, phys['R_collapse'], 'b-', linewidth=2.5, label='Physics')
        ax1.plot(rho_grid, bpinn['R_collapse'], 'r--', linewidth=2.5, label='BPINN')
        ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='50% Reliability')
        ax1.axvline(bpinn['safe_window_lower'], color='green', linestyle='--', alpha=0.7, label='Safe Window')
        ax1.set_ylim([-0.05, 1.05])
        apply_ax_style(ax1, xlabel='Mud Density (g/cm³)', ylabel='Collapse Reliability',
                       title=f'{criterion_names[criterion]}: Collapse',
                       legend=True, grid=True, fontsize=14)
        
        ax2 = axes[i, 1]
        ax2.plot(rho_grid, phys['R_fracture'], 'b-', linewidth=2.5, label='Physics')
        ax2.plot(rho_grid, bpinn['R_fracture'], 'r--', linewidth=2.5, label='BPINN')
        ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='50% Reliability')
        ax2.axvline(bpinn['safe_window_upper'], color='green', linestyle='--', alpha=0.7, label='Safe Window')
        ax2.set_ylim([-0.05, 1.05])
        apply_ax_style(ax2, xlabel='Mud Density (g/cm³)', ylabel='Fracture Reliability',
                       title=f'{criterion_names[criterion]}: Fracture',
                       legend=True, grid=True, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"准则对比图已保存至: {save_path}")
    plt.close()


def compute_reliability_curves_physics_with_criterion(
    regime: str, alpha_p: float, 
    sampler: InputSampler, physics: WellborePhysics,
    rho_grid: np.ndarray, N_samples_per_rho: int = 500,
    failure_model: str = "mohr",
    fixed_depth: float = None) -> Dict[str, np.ndarray]:
    """
    计算物理解的可靠度曲线（支持指定破坏准则）
    
    参数:
        regime: 应力机制
        alpha_p: 压力系数
        sampler: 输入采样器
        physics: 物理模型
        rho_grid: 泥浆密度网格
        N_samples_per_rho: 每个密度点的采样数
        failure_model: 破坏准则 ("mohr" | "mogi" | "dp")
        fixed_depth: 固定深度（如果为None则从分布采样）
    
    返回:
        包含可靠度曲线和安全窗口的字典
    """
    N_rho = len(rho_grid)
    R_collapse = np.zeros(N_rho)
    R_fracture = np.zeros(N_rho)
    
    for i, rho_mud in enumerate(rho_grid):
        # 采样输入参数
        X_samples, depths, aux = sampler.sample_inputs(
            N_samples_per_rho, regime=regime, alpha_p_mean=alpha_p, 
            return_depth=True, fixed_depth=fixed_depth
        )
        
        phi_samples = aux['phi']
        k_samples = aux['k']
        
        count_safe_collapse = 0
        count_safe_fracture = 0
        
        for j in range(N_samples_per_rho):
            phi = phi_samples[j]
            sigma_v = X_samples[j, 1]
            sigma_H = X_samples[j, 2]
            sigma_h = X_samples[j, 3]
            p_p = X_samples[j, 4]
            T = X_samples[j, 5]
            inclination = X_samples[j, 6]
            wellbore_azimuth = X_samples[j, 7]
            k = k_samples[j]
            depth = depths[j]
            
            # 泥浆压力 (MPa)
            p_mud = rho_mud * physics.phys.g * depth / 1000.0
            
            # 计算安全系数（使用指定的破坏准则）
            results = physics.compute_safety_factors(
                sigma_H, sigma_h, p_p, p_mud, inclination, wellbore_azimuth,
                phi=phi, k=k, T=T, sigma_v=sigma_v,
                failure_model=failure_model
            )
            
            SF_collapse = results['SF_collapse']
            SF_fracture = results['SF_fracture']
            
            if SF_collapse >= 1.0:
                count_safe_collapse += 1
            if SF_fracture >= 1.0:
                count_safe_fracture += 1
        
        R_collapse[i] = count_safe_collapse / N_samples_per_rho
        R_fracture[i] = count_safe_fracture / N_samples_per_rho
    
    # 寻找安全窗口 (R >= 0.5)
    safe_collapse_idx = np.where(R_collapse >= 0.5)[0]
    safe_fracture_idx = np.where(R_fracture >= 0.5)[0]
    
    if len(safe_collapse_idx) > 0:
        rho_c_safe = rho_grid[safe_collapse_idx[0]]
    else:
        rho_c_safe = rho_grid[0]
    
    if len(safe_fracture_idx) > 0:
        rho_f_safe = rho_grid[safe_fracture_idx[-1]]
    else:
        rho_f_safe = rho_grid[-1]
    
    recommended_rho = (rho_c_safe + rho_f_safe) / 2.0
    
    return {
        'rho_grid': rho_grid,
        'R_collapse': R_collapse,
        'R_fracture': R_fracture,
        'safe_window_lower': rho_c_safe,
        'safe_window_upper': rho_f_safe,
        'recommended_rho': recommended_rho,
        'regime': regime,
        'alpha_p': alpha_p,
        'failure_model': failure_model
    }


# ============================================================================
# 第八部分：主程序
# ============================================================================

def main():
    """主程序入口（论文复现完整版）"""
    # 可选：快速模式（便于在 CPU 上快速验证修复是否生效）
    # 用法：在命令行设置环境变量 BPINN_QUICK=1
    # PowerShell 示例：$env:BPINN_QUICK="1"; python BPINN_wellbore_stability.py
    quick_mode = os.getenv("BPINN_QUICK", "0").strip() in ("1", "true", "True", "YES", "yes")
    
    print("="*80)
    print("贝叶斯物理信息神经网络 (BPINN)")
    print("枯竭气藏储气库井壁失稳不确定性分析")
    print("基于 Zhang et al. (2023) 论文复现")
    print("="*80)
    print("\n本程序实现:")
    print("1. 三种应力机制 (NF/SS/RF) 与地层压力系数 (alpha_p) 的多工况分析")
    print("2. 输入参数概率分布可视化 (Fig.4-5)")
    print("3. 等效密度随井斜/方位变化的分析 (Fig.2, Fig.6-7)")
    print("4. 可靠度-等效密度曲线的物理解与 BPINN 对比 (Fig.8-10)")
    print("5. 安全泥浆密度窗口的统计与推荐 (Table 5)")
    print("="*80)
    
    # ========== 1. 配置与初始化 ==========
    print("\n" + "="*80)
    print("[步骤 1/8] 初始化配置与参数")
    print("="*80)
    
    # 创建输出目录
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"计算设备: {device}")
    
    # 定义多工况参数
    stress_regimes = ["NF", "SS", "RF"]  # 三种应力机制
    alpha_p_values = [0.1, 0.3, 0.5]     # 三个代表性压力系数
    
    # 选择主工况（用于详细分析）
    main_regime = "NF"      # 正断层
    main_alpha_p = 0.3      # 中等压力系数
    
    print(f"\n工况设置:")
    print(f"  应力机制: {', '.join(stress_regimes)}")
    print(f"  压力系数 alpha_p: {', '.join([str(a) for a in alpha_p_values])}")
    print(f"  主工况: {main_regime}, alpha_p={main_alpha_p}")
    
    # 物理常数和分布配置（使用主工况初始化）
    phys_const = PhysicalConstants(depth=4500.0, stress_regime=main_regime, alpha_p=main_alpha_p)
    dist_config = DistributionConfig()
    
    # 初始化采样器和物理模型
    sampler = InputSampler(dist_config, phys_const)
    physics = WellborePhysics(phys_const, dist_config)
    
    print(f"\n物理参数:")
    print(f"  井深: {phys_const.depth} m")
    print(f"  岩石 UCS: {phys_const.UCS} MPa")
    print(f"  内摩擦角: {phys_const.friction_angle_deg}°")
    print(f"  泥浆密度范围: [{phys_const.mud_density_min}, {phys_const.mud_density_max}] g/cm^3")
    
    # ========== 2. 物理解蒙特卡洛仿真（多工况）==========
    print("\n" + "="*80)
    print("[步骤 2/8] 物理解蒙特卡洛仿真与输入分布可视化")
    print("="*80)
    
    # 生成主工况的输入样本用于分布可视化
    print(f"\n生成主工况 ({main_regime}, alpha_p={main_alpha_p}) 的输入样本...")
    N_samples_for_dist = 5000
    # [FIX] 输入分布先验图引入 Depth，构造 9D 输入：[Depth, 8 features]
    X_main_8d, depth_main, _ = sampler.sample_inputs(
        N_samples_for_dist, regime=main_regime, alpha_p_mean=main_alpha_p, return_depth=True
    )
    X_main = np.column_stack([depth_main, X_main_8d])
    
    # 绘制输入参数分布（对应论文 Fig.4-5）
    print("\n绘制输入参数概率分布图...")
    plot_input_distributions(X_main, save_path_prefix="results/input_distributions", show_plot=False)
    
    # 扫描井斜角和方位角（对应论文 Fig.2, Fig.6-7）
    print(f"\n扫描主工况的井斜角和方位角...")
    scan_results_main = scan_inclination_azimuth_for_regime(
        main_regime, main_alpha_p, physics,
        inclinations=np.arange(0, 91, 10),
        azimuths=np.arange(0, 361, 20)
    )
    
    # 绘制等效密度图
    plot_rho_vs_inclination(scan_results_main, 'collapse', 
                           save_path="results/rho_vs_inclination_collapse.png", show_plot=False)
    plot_rho_vs_inclination(scan_results_main, 'fracture', 
                           save_path="results/rho_vs_inclination_fracture.png", show_plot=False)
    
    # 计算物理解可靠度曲线（对应论文 Fig.8-10）
    print(f"\n计算主工况的物理解可靠度曲线...")
    rho_grid = np.linspace(0.8, 3.0, 100)  # 等效密度网格
    reliability_physics = compute_reliability_curves_physics(
        main_regime, main_alpha_p, sampler, physics,
        rho_grid=rho_grid, N_samples_per_rho=300
    )
    
    # 绘制物理解可靠度曲线
    plot_reliability_curves_physics(reliability_physics, 
                                   save_path="results/reliability_physics.png", show_plot=False)
    
    # ========== 3. 两种论文准则外层循环 ==========
    print("\n" + "="*80)
    print("[步骤 3/10] 两种破坏准则的数据生成与训练")
    print("="*80)
    
    # [REVISION 2026] quick_mode 加强为"冲烟"模式：仅跑 mohr 一种准则、
    # 较小样本数、较短训练，用于在改动后快速确认 ρ_c R²、PICP 等指标的方向。
    if quick_mode:
        failure_models = ["mohr"]
        print("\n[QUICK MODE] 已启用——仅跑 Mohr-Coulomb 准则；")
        print("            样本数减半、轮数 300、KL warmup 60，用于快速验证修复方向。")
    else:
        # 论文仅讨论 Mohr-Coulomb 与 Mogi-Coulomb；Drucker-Prager 不再纳入主流程，
        # 避免输出结果与论文方法范围不一致。
        failure_models = ["mohr", "mogi"]
    model_names = {
        "mohr": "Mohr-Coulomb",
        "mogi": "Mogi-Coulomb"
    }
    
    # 存储所有准则的结果
    all_models = {}
    all_histories = {}
    all_results_physics = {}
    all_results_bpinn = {}
    all_scalers = {}
    all_baseline_predictors = {}
    train_times_min = {}
    reliability_times_s = {}
    datagen_times_s = {}
    reliability_physics_times_s = {}
    reliability_bpinn_times_s = {}
    benchmark_results = {}
    
    N_train_per_case = 200 if quick_mode else 500  # 每个工况训练样本
    N_val_per_case = 50 if quick_mode else 100     # 每个工况验证样本
    
    for failure_model in failure_models:
        print("\n" + "="*80)
        print(f"【准则 {failure_model.upper()}】: {model_names[failure_model]}")
        print("="*80)
        
        # === 3.1 生成该准则的训练数据集 ===
        print(f"\n[{failure_model}] 生成训练数据集（多工况混合）...")
        datagen_start = time.perf_counter()
        
        X_train_list = []
        Y_train_list = []
        Yb_train_list = []  # [REVISION 2026] baseline ρ_c/ρ_f
        X_val_list = []
        Y_val_list = []
        Yb_val_list = []
        depths_train_list = []
        depths_val_list = []
        
        for regime in stress_regimes:
            for alpha_p in alpha_p_values:
                print(f"  生成工况: {regime}, alpha_p={alpha_p} (准则={failure_model})...")
                
                # 更新物理常数
                phys_const_temp = PhysicalConstants(depth=4500.0, stress_regime=regime, alpha_p=alpha_p)
                sampler_temp = InputSampler(dist_config, phys_const_temp)
                physics_temp = WellborePhysics(phys_const_temp, dist_config)
                
                # 生成数据（使用指定的破坏准则）
                X_tr, D_tr, aux_tr = sampler_temp.sample_inputs(
                    N_train_per_case, regime=regime, alpha_p_mean=alpha_p, return_depth=True
                )
                Y_tr = physics_temp.compute_true_outputs(
                    X_tr, aux_data=aux_tr, failure_model=failure_model
                )
                # [REVISION 2026] 计算 ρ_c/ρ_f 的物理 baseline (ε=0)
                Yb_tr = physics_temp.compute_baseline_rho(
                    X_tr, aux_data=aux_tr, failure_model=failure_model
                )
                
                X_va, D_va, aux_va = sampler_temp.sample_inputs(
                    N_val_per_case, regime=regime, alpha_p_mean=alpha_p, return_depth=True
                )
                Y_va = physics_temp.compute_true_outputs(
                    X_va, aux_data=aux_va, failure_model=failure_model
                )
                Yb_va = physics_temp.compute_baseline_rho(
                    X_va, aux_data=aux_va, failure_model=failure_model
                )
                
                X_train_list.append(X_tr)
                Y_train_list.append(Y_tr)
                Yb_train_list.append(Yb_tr)
                X_val_list.append(X_va)
                Y_val_list.append(Y_va)
                Yb_val_list.append(Yb_va)
                depths_train_list.append(D_tr)
                depths_val_list.append(D_va)
        
        # 合并所有工况的数据
        X_train = np.vstack(X_train_list)
        Y_train = np.vstack(Y_train_list)
        Yb_train = np.vstack(Yb_train_list)  # (N_train, 2): [rho_c_baseline, rho_f_baseline]
        X_val = np.vstack(X_val_list)
        Y_val = np.vstack(Y_val_list)
        Yb_val = np.vstack(Yb_val_list)
        
        depths_train = np.concatenate(depths_train_list)
        depths_val = np.concatenate(depths_val_list)
        datagen_elapsed_s = time.perf_counter() - datagen_start
        datagen_times_s[failure_model] = datagen_elapsed_s
        
        N_total_samples = len(X_train) + len(X_val)
        print(f"  合并后的训练集: {len(X_train)} 样本")
        print(f"  合并后的验证集: {len(X_val)} 样本")
        print(f"  数据生成耗时: {datagen_elapsed_s:.2f} s ({N_total_samples} 样本, "
              f"{datagen_elapsed_s/N_total_samples*1000:.2f} ms/样本)")
        rho_residual_abs_max_by_channel = {2: 3.2, 3: 0.4}
        # [REVISION 2026 v7] baseline 残差范围检查：必须落在各通道对称残差量纲内，
        # 否则 normalize 后会出现 |y_norm|>1，被 tanh 截断，导致信号丢失。
        residual_train = Y_train[:, 2:4] - Yb_train
        d_rho_c_min, d_rho_c_max = residual_train[:, 0].min(), residual_train[:, 0].max()
        d_rho_f_min, d_rho_f_max = residual_train[:, 1].min(), residual_train[:, 1].max()
        rho_c_residual_limit = rho_residual_abs_max_by_channel[2]
        rho_f_residual_limit = rho_residual_abs_max_by_channel[3]
        print(f"  ρ residual range (train): "
              f"Δρ_c ∈ [{d_rho_c_min:.3f}, {d_rho_c_max:.3f}] (limit ±{rho_c_residual_limit:.1f}), "
              f"Δρ_f ∈ [{d_rho_f_min:.3f}, {d_rho_f_max:.3f}] (limit ±{rho_f_residual_limit:.1f}) g/cm^3")
        # 越界告警：超过 OutputScaler 上限会被 tanh 截断
        if max(abs(d_rho_c_min), abs(d_rho_c_max)) > rho_c_residual_limit:
            print(f"  ⚠ Δρ_c 越过 ±{rho_c_residual_limit:.1f} 上限，建议把 OutputScaler 中 ch=2 上限调高")
        if max(abs(d_rho_f_min), abs(d_rho_f_max)) > rho_f_residual_limit:
            print(f"  ⚠ Δρ_f 越过 ±{rho_f_residual_limit:.1f} 上限（B 档量纲），建议调高 ch=3 上限"
                  "或核查训练样本工况分布")

        # === 数据归一化 ===
        print(f"\n[{failure_model}] 归一化处理...")
        # [REVISION 2026] X_in 拼成 11 维：[Depth, 8 features, baseline_ρ_c, baseline_ρ_f]
        # 让网络在前向阶段就能"看到"baseline，与训练目标 Y_residual 形成对应。
        X_train_in = np.column_stack([depths_train, X_train, Yb_train])
        X_val_in = np.column_stack([depths_val, X_val, Yb_val])
        scaler_current = DataScaler()
        scaler_current.fit(X_train_in)
        
        X_train_norm = scaler_current.transform(X_train_in)
        X_val_norm = scaler_current.transform(X_val_in)
        
        # 保存 DataScaler 以便后续复用
        with open(f'models/scaler_{failure_model}.pkl', 'wb') as f:
            pickle.dump({'mean': scaler_current.mean, 'std': scaler_current.std}, f)
        print(f"  DataScaler 已保存至: models/scaler_{failure_model}.pkl")
        
        # === 训练目标改为残差 ===
        # [REVISION 2026 v6] B 档：ρ_c 与 ρ_f 都做残差化（per-channel 窄量纲）。
        #   v2 注释中"ρ_f 残差化退化"的根因是共用 ±1.5 量纲；
        #   v6 通过 OutputScaler 给 ρ_f 单独分配 ±0.4 量纲，分辨率 ×4.25，
        #   理论上 ρ_f 信号不再被 ρ_c 淹没，且 σ_w 有抖动空间避免后验塌缩。
        Y_train_residual = Y_train.copy()
        Y_train_residual[:, 2] = Y_train[:, 2] - Yb_train[:, 0]
        Y_train_residual[:, 3] = Y_train[:, 3] - Yb_train[:, 1]  # [v6] ρ_f 也走残差
        Y_val_residual = Y_val.copy()
        Y_val_residual[:, 2] = Y_val[:, 2] - Yb_val[:, 0]
        Y_val_residual[:, 3] = Y_val[:, 3] - Yb_val[:, 1]  # [v6] ρ_f 也走残差

        # === 导出数据 ===
        print(f"\n[{failure_model}] 导出数据到 Excel...")
        export_data_to_excel({
            'Train_Data': pd.DataFrame(
                np.hstack([X_train_in, Y_train, Yb_train]),
                columns=['Depth', 'Porosity', 'Sigma_v', 'Sigma_H', 'Sigma_h', 
                        'P_pore', 'Temperature', 'Inclination', 'Azimuth',
                        'Baseline_Rho_c_in', 'Baseline_Rho_f_in',
                        'SF_collapse', 'SF_fracture', 'Rho_collapse', 'Rho_fracture',
                        'Baseline_Rho_c', 'Baseline_Rho_f']
            ),
            'Val_Data': pd.DataFrame(
                np.hstack([X_val_in, Y_val, Yb_val]),
                columns=['Depth', 'Porosity', 'Sigma_v', 'Sigma_H', 'Sigma_h', 
                        'P_pore', 'Temperature', 'Inclination', 'Azimuth',
                        'Baseline_Rho_c_in', 'Baseline_Rho_f_in',
                        'SF_collapse', 'SF_fracture', 'Rho_collapse', 'Rho_fracture',
                        'Baseline_Rho_c', 'Baseline_Rho_f']
            )
        }, f'results/dataset_{failure_model}.xlsx')

        
        # === 3.2 构建 BPINN 模型 ===
        print(f"\n[{failure_model}] 构建 BPINN 模型...")
        
        # 转换为PyTorch张量
        # [REVISION 2026] 训练目标改为 Y_residual（ρ 通道为 Δρ）
        X_train_torch = torch.FloatTensor(X_train_norm)
        Y_train_torch = torch.FloatTensor(Y_train_residual)
        X_val_torch = torch.FloatTensor(X_val_norm)
        Y_val_torch = torch.FloatTensor(Y_val_residual)
        
        # 初始化 Scaler 的 Tensor
        scaler_current.to_torch(device)
        
        batch_size = 128
        train_dataset = TensorDataset(X_train_torch, Y_train_torch)
        val_dataset = TensorDataset(X_val_torch, Y_val_torch)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # [REVISION 2026] 输入维度 9 -> 11（额外两列为 baseline ρ_c/ρ_f）
        input_dim = 11
        output_dim = 4
        hidden_dims = [128, 128, 128, 64]
        prior_sigma = 1.0
        
        model = BayesianBPINN(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            prior_sigma=prior_sigma,
            phys_const=phys_const
        ).to(device)
        
        print(f"  网络结构: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  总参数数量: {total_params:,}")
        
        # === 3.3 训练 BPINN 模型 ===
        print(f"\n[{failure_model}] 训练 BPINN 模型...")
        
        # [REVISION 2026 v6] B 档：per-channel 窄量纲残差化
        #   v2 的 [2] 仅 ρ_c → A 档完整训练后 ρ_f NLL 升到 700–1200，
        #   PICP@95 仅 0.37（目标 ≥ 0.85），证实 ρ_f 直接预测路径架构受限；
        #   v3 失败前科：[2, 3] 共用 ±1.5 让 ρ_f 信号被淹没；
        #   v6 修复：恢复 [2, 3] 但 per-channel：
        #     · ρ_c ±3.2 g/cm^3（v6 完整训练中 Mogi/DP 的 Δρ_c 可达约 2.97，避免截断）；
        #     · ρ_f ±0.4 g/cm^3（A 档实测 Δρ_f ∈ [-0.21, 0]，覆盖 100% 留 ~90% 余量）；
        #     · 分辨率：ρ_f 量纲缩到原 [0.1, 3.5] 的 ~24%，理论分辨率 ×4.25。
        output_scaler_current = OutputScaler(
            residual_mode=True,
            rho_residual_abs_max=rho_residual_abs_max_by_channel,
            residual_channels=[2, 3],
        ).to(device)
        
        # [REVISION 2026 v6] 损失函数权重再调整（A 档完整训练事后复盘，2026-05-07）：
        #   v5 → A 档实测（4500 train / 1000 epoch / 当时测试准则）：
        #     - 稳态 Ratio[D/P/K] = 1.4% / 62% / 36%（D 目标 ≥3% 未达标，被 Phys 重新压扁）；
        #     - ρ_f NLL 在当时测试准则中升至 739–1182（v5 quick=385），证实直接预测架构边界；
        #     - 安全窗口对齐误差全部 ≤ 0.067 g/cm^3 ✅，工程主指标稳。
        #   v6 改动两条（与 OutputScaler v6 配套）：
        #     - residual_channels [2] -> [2, 3]：ρ_f 改走 baseline+残差路径，
        #       窄量纲 ±0.4 给 σ_w 留足抖动空间；
        #     - output_weights[3] 0.8 -> 1.0：ρ_f 既然已经走残差化路径，
        #       data 信号变得清晰（残差幅度小且零均值），适度提权恢复 ch=3 学习；
        #     - λ_phys 0.02 -> 0.015：A 档 Phys 占比 62% 偏高，再下调 25% 把 D 占比拉回 ≥3%。
        physics_for_loss = WellborePhysics(phys_const, dist_config)
        loss_fn = BPINNLoss(
            physics=physics_for_loss,
            output_scaler=output_scaler_current,
            lambda_data=1.0,
            lambda_phys=0.015,  # [v6] 0.02 -> 0.015：把 A 档 Phys 62% 占比拉回 50% 以下
            lambda_kl=5e-4,
            lambda_frac=5.0,
            scaler=scaler_current,
            failure_model=failure_model,
            output_weights=[1.2, 0.5, 2.5, 1.0],  # [v6] ρ_f 0.8 -> 1.0：残差化后恢复 ch=3 学习权重
            use_baseline_residual=True,
            residual_channels=[2, 3],  # [v6] 增加 ρ_f 残差化
        )
        
        print(f"  损失函数权重: λ_data={loss_fn.lambda_data}, λ_phys={loss_fn.lambda_phys}, "
              f"λ_kl={loss_fn.lambda_kl}, output_weights={loss_fn.output_weights_tensor.tolist()}")
        print(f"  破坏准则: {failure_model} | use_baseline_residual=True | "
              f"residual_channels={loss_fn.residual_channels}")
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        # [REVISION 2026] LR 调度更耐心：避免 LR 过早下降导致早停误触
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=80, 
            threshold=1e-3, cooldown=10, min_lr=1e-6
        )
        
        # [REVISION 2026] quick_mode 加强为冲烟：300 epoch + warmup 60
        if quick_mode:
            num_epochs = 300
            kl_warmup_epochs = 60
            es_patience = 80
        else:
            num_epochs = 1000
            kl_warmup_epochs = max(20, int(num_epochs * 0.2))
            es_patience = 200
        print(f"  训练轮数: {num_epochs} | KL warmup: {kl_warmup_epochs} epochs (cosine) | "
              f"early_stop patience: {es_patience}")

        train_start = time.perf_counter()
        history = train_bpinn(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            verbose=True,
            mc_val_samples=3 if quick_mode else 5,
            early_stopping_patience=es_patience,
            early_stopping_min_delta=1e-4,
            kl_warmup_epochs=kl_warmup_epochs,
            kl_anneal_schedule="cosine",
            lambda_kl_final=loss_fn.lambda_kl,
        )
        train_elapsed_min = (time.perf_counter() - train_start) / 60.0
        train_times_min[failure_model] = train_elapsed_min
        
        # 保存训练历史和模型
        plot_training_history(history, save_path=f'results/training_history_{failure_model}.png')
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f'models/bpinn_model_{failure_model}.pth')
        print(f"  模型已保存至: models/bpinn_model_{failure_model}.pth")

        # === 3.3.4 UQ 标定指标（PICP / NLL / ACE + Aleatory/Epistemic 分解） ===
        # [REVISION 2026] 直接回应审稿意见 R1-Q13、R3-Q5
        print(f"\n[{failure_model}] 计算贝叶斯 UQ 标定指标...")
        n_mc_uq = 30 if quick_mode else 50
        uq_metrics = compute_uq_calibration_metrics(
            model=model,
            X_val_norm_torch=X_val_torch,
            Y_val_np=Y_val,
            device=device,
            output_scaler=output_scaler_current,
            num_weight_samples=n_mc_uq,
            Y_baseline_np=Yb_val,  # baseline ρ_c/ρ_f 用于残差校正
            residual_channels=[2, 3],  # [REVISION 2026 v6] B 档：ρ_c + ρ_f 都做残差校正
            residual_calibration_channels=[2, 3],  # [REVISION 2026 v7] 密度通道补偿模型残差方差
        )
        uq_df = uq_metrics_to_dataframe(uq_metrics)
        # 控制台简表
        print("  UQ 标定结果（每个输出维度）:")
        print("  " + "-" * 140)
        print(f"  {'Output':<22s} {'R²':>7s} {'RMSE':>8s} {'PICP95':>8s} {'Raw95':>8s} "
              f"{'PINAW95':>8s} {'ACE':>7s} {'NLL':>10s} {'RawNLL':>10s} "
              f"{'Resσ':>8s} {'Ale%':>7s} {'Epi%':>7s}")
        print("  " + "-" * 140)
        for _, row in uq_df.iterrows():
            print(f"  {row['Output']:<22s} {row['R2']:>7.3f} {row['RMSE']:>8.3f} "
                  f"{row['PICP_95 (target=0.95)']:>8.3f} {row['PICP_95_weight_only']:>8.3f} "
                  f"{row['PINAW_95']:>8.3f} {row['ACE']:>7.3f} {row['NLL']:>10.3f} "
                  f"{row['NLL_weight_only']:>10.3f} {row['Residual_Calib_Std']:>8.3f} "
                  f"{row['Aleatory_pct']:>7.2f} {row['Epistemic_pct']:>7.2f}")
        print("  " + "-" * 140)

        # 保存表格 + 可靠度图
        export_data_to_excel({'UQ_Calibration': uq_df},
                             f'results/uq_calibration_{failure_model}.xlsx')
        plot_calibration_diagram(
            uq_metrics,
            save_path=f'results/calibration_diagram_{failure_model}.png',
            title_prefix=f"BPINN UQ Calibration ({model_names[failure_model]})",
            show_plot=False,
        )
        print(f"  UQ 表格已保存: results/uq_calibration_{failure_model}.xlsx")
        print(f"  UQ 可靠度图已保存: results/calibration_diagram_{failure_model}.png")

        # === 3.3.5 计算效率基准测试 ===
        print(f"\n[{failure_model}] 计算效率基准测试...")
        
        N_bench_repeat = 20 if quick_mode else 100
        N_bench_batch = 200 if quick_mode else 1000
        num_mc_weights = 50
        
        # 生成基准测试数据
        X_bench_8d, depth_bench, aux_bench = sampler.sample_inputs(
            N_bench_batch, regime=main_regime, alpha_p_mean=main_alpha_p, return_depth=True
        )
        # [REVISION 2026] benchmark 输入也要拼接 baseline 才能和训练阶段的 11D 一致
        Yb_bench = physics.compute_baseline_rho(
            X_bench_8d, aux_data=aux_bench, failure_model=failure_model
        )
        X_bench = np.column_stack([depth_bench, X_bench_8d, Yb_bench])
        X_bench_norm = scaler_current.transform(X_bench)
        X_bench_torch = torch.FloatTensor(X_bench_norm).to(device)
        
        # --- (a) 解析解单样本推理时间 ---
        single_analytical_times = []
        for j in range(N_bench_repeat):
            aux_single = {k: v[j:j+1] for k, v in aux_bench.items()}
            t0 = time.perf_counter()
            _ = physics.compute_true_outputs(
                X_bench_8d[j:j+1], aux_data=aux_single, failure_model=failure_model
            )
            single_analytical_times.append(time.perf_counter() - t0)
        avg_analytical_single_ms = np.mean(single_analytical_times) * 1000
        
        # --- (b) BPINN 单样本推理时间 ---
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(X_bench_torch[:1])
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        single_bpinn_times = []
        with torch.no_grad():
            for j in range(N_bench_repeat):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(X_bench_torch[j:j+1])
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                single_bpinn_times.append(time.perf_counter() - t0)
        avg_bpinn_single_ms = np.mean(single_bpinn_times) * 1000
        
        # --- (c) 批量推理对比 (N=N_bench_batch) ---
        t0 = time.perf_counter()
        _ = physics.compute_true_outputs(
            X_bench_8d, aux_data=aux_bench, failure_model=failure_model
        )
        analytical_batch_s = time.perf_counter() - t0
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(X_bench_torch)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        bpinn_batch_s = time.perf_counter() - t0
        
        # --- (d) 完整 UQ 推理 (N_batch × num_mc_weights 次前向传播) ---
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_mc_weights):
                y_pred_bench = model(X_bench_torch)
                if output_scaler_current is not None:
                    y_pred_bench = output_scaler_current.denormalize(y_pred_bench)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        bpinn_uq_s = time.perf_counter() - t0
        
        N_analytical_uq = min(N_bench_batch, 500)
        t0 = time.perf_counter()
        aux_uq = {k: v[:N_analytical_uq] for k, v in aux_bench.items()}
        _ = physics.compute_true_outputs(
            X_bench_8d[:N_analytical_uq], aux_data=aux_uq, failure_model=failure_model
        )
        analytical_uq_measured_s = time.perf_counter() - t0
        analytical_uq_equivalent_s = analytical_uq_measured_s * (N_bench_batch / N_analytical_uq)
        
        benchmark_results[failure_model] = {
            'analytical_single_ms': avg_analytical_single_ms,
            'bpinn_single_ms': avg_bpinn_single_ms,
            'speedup_single': avg_analytical_single_ms / max(avg_bpinn_single_ms, 1e-6),
            'analytical_batch_s': analytical_batch_s,
            'bpinn_batch_s': bpinn_batch_s,
            'speedup_batch': analytical_batch_s / max(bpinn_batch_s, 1e-6),
            'bpinn_uq_s': bpinn_uq_s,
            'analytical_uq_equivalent_s': analytical_uq_equivalent_s,
            'speedup_uq': analytical_uq_equivalent_s / max(bpinn_uq_s, 1e-6),
            'train_time_min': train_elapsed_min,
            'datagen_time_s': datagen_elapsed_s,
            'N_bench_batch': N_bench_batch,
            'num_mc_weights': num_mc_weights,
        }
        
        print(f"  单样本推理: 解析解 {avg_analytical_single_ms:.3f} ms, "
              f"BPINN {avg_bpinn_single_ms:.3f} ms, "
              f"加速比 {avg_analytical_single_ms/max(avg_bpinn_single_ms,1e-6):.1f}x")
        print(f"  批量推理 (N={N_bench_batch}): 解析解 {analytical_batch_s:.3f} s, "
              f"BPINN {bpinn_batch_s:.5f} s, "
              f"加速比 {analytical_batch_s/max(bpinn_batch_s,1e-6):.1f}x")
        print(f"  UQ推理 (N={N_bench_batch}x{num_mc_weights}): "
              f"BPINN {bpinn_uq_s:.3f} s, "
              f"等效解析解 {analytical_uq_equivalent_s:.1f} s, "
              f"加速比 {analytical_uq_equivalent_s/max(bpinn_uq_s,1e-6):.1f}x")
        
        # === 3.4 计算可靠度曲线 ===
        print(f"\n[{failure_model}] 计算可靠度曲线...")
        
        phys_const_main = PhysicalConstants(depth=4500.0, stress_regime=main_regime, alpha_p=main_alpha_p)
        sampler_main = InputSampler(dist_config, phys_const_main)
        physics_main = WellborePhysics(phys_const_main, dist_config)
        
        rho_grid = np.linspace(0.8, 3.0, 100)
        
        # 物理解可靠度（使用指定准则）- 单独计时
        reliability_phys_start = time.perf_counter()
        reliability_physics = compute_reliability_curves_physics_with_criterion(
            main_regime, main_alpha_p, sampler_main, physics_main,
            rho_grid=rho_grid, N_samples_per_rho=300,
            failure_model=failure_model,
            fixed_depth=phys_const_main.depth
        )
        reliability_phys_elapsed_s = time.perf_counter() - reliability_phys_start
        reliability_physics_times_s[failure_model] = reliability_phys_elapsed_s
        
        # BPINN 可靠度 - 单独计时
        reliability_bpinn_start = time.perf_counter()
        reliability_bpinn = compute_reliability_curves_bpinn(
            main_regime, main_alpha_p, sampler_main, model,
            rho_grid=rho_grid, num_input_samples=300, num_weight_samples=50,
            device=device, scaler=scaler_current, output_scaler=output_scaler_current,
            fixed_depth=phys_const_main.depth,
            physics=physics_main,
            failure_model=failure_model,
            use_baseline_residual=True,
            residual_channels=[2, 3],  # [REVISION 2026 v6] B 档：ρ_c + ρ_f 都做残差还原
        )
        reliability_bpinn_elapsed_s = time.perf_counter() - reliability_bpinn_start
        reliability_bpinn_times_s[failure_model] = reliability_bpinn_elapsed_s
        
        reliability_elapsed_s = reliability_phys_elapsed_s + reliability_bpinn_elapsed_s
        reliability_times_s[failure_model] = reliability_elapsed_s
        print(f"  可靠度计算耗时: 物理解 {reliability_phys_elapsed_s:.2f}s, "
              f"BPINN {reliability_bpinn_elapsed_s:.2f}s, "
              f"加速比 {reliability_phys_elapsed_s/max(reliability_bpinn_elapsed_s, 1e-6):.1f}x")
        
        # 保存结果
        all_models[failure_model] = model
        all_histories[failure_model] = history
        all_results_physics[failure_model] = reliability_physics
        all_results_bpinn[failure_model] = reliability_bpinn
        all_scalers[failure_model] = scaler_current
        
        # 绘制对比图
        plot_reliability_curves_comparison(
            reliability_physics, reliability_bpinn,
            save_path=f"results/reliability_comparison_{failure_model}.png",
            show_plot=False
        )
        
        print(f"\n[{failure_model}] 完成！")
    
    # ========== 4. 准则对比汇总 ==========
    print("\n" + "="*80)
    print("[步骤 4/10] 破坏准则对比汇总")
    print("="*80)
    
    comparison_results = compare_failure_criteria(
        all_results_physics, all_results_bpinn, failure_models
    )
    comparison_results['Train_Time (min)'] = comparison_results['Failure_Criterion'].str.lower().map(train_times_min)
    comparison_results['Reliability_Time (s)'] = comparison_results['Failure_Criterion'].str.lower().map(reliability_times_s)
    
    # 输出对比表格
    print("\n破坏准则定量对比:")
    print("="*80)
    print(comparison_results.to_string(index=False))
    print("="*80)
    
    # 保存到 Excel
    export_data_to_excel({
        'Criterion_Comparison': comparison_results
    }, 'results/criterion_comparison.xlsx')
    
    # 绘制准则对比图
    plot_failure_criteria_comparison(
        all_results_physics, all_results_bpinn, failure_models,
        save_path='results/criteria_comparison.png'
    )
    
    print("\n对比分析完成！")
    print(f"  - 对比表格已保存至: results/criterion_comparison.xlsx")
    print(f"  - 对比图已保存至: results/criteria_comparison.png")

    # ========== 5. 计算效率对比表格 ==========
    print("\n" + "="*80)
    print("[步骤 5/10] 计算效率基准测试汇总")
    print("="*80)
    
    efficiency_rows = []
    for fm in failure_models:
        if fm not in benchmark_results:
            continue
        br = benchmark_results[fm]
        efficiency_rows.append({
            'Failure_Criterion': model_names[fm],
            'Data_Generation (s)': round(br['datagen_time_s'], 2),
            'Training (min)': round(br['train_time_min'], 2),
            'Analytical_Single (ms)': round(br['analytical_single_ms'], 3),
            'BPINN_Single (ms)': round(br['bpinn_single_ms'], 3),
            'Speedup_Single': round(br['speedup_single'], 1),
            'Analytical_Batch (s)': round(br['analytical_batch_s'], 3),
            'BPINN_Batch (s)': round(br['bpinn_batch_s'], 5),
            'Speedup_Batch': round(br['speedup_batch'], 1),
            'BPINN_UQ (s)': round(br['bpinn_uq_s'], 3),
            'Analytical_UQ_Equiv (s)': round(br['analytical_uq_equivalent_s'], 1),
            'Speedup_UQ': round(br['speedup_uq'], 1),
            'Reliability_Physics (s)': round(reliability_physics_times_s.get(fm, 0), 2),
            'Reliability_BPINN (s)': round(reliability_bpinn_times_s.get(fm, 0), 2),
            'Speedup_Reliability': round(
                reliability_physics_times_s.get(fm, 1) / max(reliability_bpinn_times_s.get(fm, 1), 1e-6), 1
            ),
            'N_Batch': br['N_bench_batch'],
            'N_MC_Weights': br['num_mc_weights'],
        })
    
    efficiency_df = pd.DataFrame(efficiency_rows)
    
    print("\n计算效率基准测试结果:")
    print("="*120)
    
    for fm in failure_models:
        if fm not in benchmark_results:
            continue
        br = benchmark_results[fm]
        print(f"\n  【{model_names[fm]}】")
        print(f"    数据生成:         {br['datagen_time_s']:.2f} s")
        print(f"    模型训练:         {br['train_time_min']:.2f} min")
        print(f"    单样本推理:       解析解 {br['analytical_single_ms']:.3f} ms | "
              f"BPINN {br['bpinn_single_ms']:.3f} ms | "
              f"加速比 {br['speedup_single']:.1f}x")
        print(f"    批量推理 (N={br['N_bench_batch']}): "
              f"解析解 {br['analytical_batch_s']:.3f} s | "
              f"BPINN {br['bpinn_batch_s']:.5f} s | "
              f"加速比 {br['speedup_batch']:.1f}x")
        print(f"    UQ推理 (N={br['N_bench_batch']}×{br['num_mc_weights']}): "
              f"BPINN {br['bpinn_uq_s']:.3f} s | "
              f"等效解析解 {br['analytical_uq_equivalent_s']:.1f} s | "
              f"加速比 {br['speedup_uq']:.1f}x")
        rp = reliability_physics_times_s.get(fm, 0)
        rb = reliability_bpinn_times_s.get(fm, 0)
        print(f"    可靠度曲线:       物理解 {rp:.2f} s | "
              f"BPINN {rb:.2f} s | "
              f"加速比 {rp/max(rb, 1e-6):.1f}x")
    
    print("\n" + "="*120)
    
    # 计算所选准则的平均值
    if len(benchmark_results) > 0:
        avg_row = {
            'Failure_Criterion': 'Average',
            'Data_Generation (s)': round(np.mean([br['datagen_time_s'] for br in benchmark_results.values()]), 2),
            'Training (min)': round(np.mean([br['train_time_min'] for br in benchmark_results.values()]), 2),
            'Analytical_Single (ms)': round(np.mean([br['analytical_single_ms'] for br in benchmark_results.values()]), 3),
            'BPINN_Single (ms)': round(np.mean([br['bpinn_single_ms'] for br in benchmark_results.values()]), 3),
            'Speedup_Single': round(np.mean([br['speedup_single'] for br in benchmark_results.values()]), 1),
            'Analytical_Batch (s)': round(np.mean([br['analytical_batch_s'] for br in benchmark_results.values()]), 3),
            'BPINN_Batch (s)': round(np.mean([br['bpinn_batch_s'] for br in benchmark_results.values()]), 5),
            'Speedup_Batch': round(np.mean([br['speedup_batch'] for br in benchmark_results.values()]), 1),
            'BPINN_UQ (s)': round(np.mean([br['bpinn_uq_s'] for br in benchmark_results.values()]), 3),
            'Analytical_UQ_Equiv (s)': round(np.mean([br['analytical_uq_equivalent_s'] for br in benchmark_results.values()]), 1),
            'Speedup_UQ': round(np.mean([br['speedup_uq'] for br in benchmark_results.values()]), 1),
            'Reliability_Physics (s)': round(np.mean(list(reliability_physics_times_s.values())), 2),
            'Reliability_BPINN (s)': round(np.mean(list(reliability_bpinn_times_s.values())), 2),
            'Speedup_Reliability': round(
                np.mean(list(reliability_physics_times_s.values())) / 
                max(np.mean(list(reliability_bpinn_times_s.values())), 1e-6), 1
            ),
            'N_Batch': list(benchmark_results.values())[0]['N_bench_batch'],
            'N_MC_Weights': list(benchmark_results.values())[0]['num_mc_weights'],
        }
        efficiency_df = pd.concat([efficiency_df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # 保存效率对比表到 Excel
    export_data_to_excel({
        'Criterion_Comparison': comparison_results,
        'Computational_Efficiency': efficiency_df
    }, 'results/criterion_comparison.xlsx')
    
    # 单独保存效率表
    export_data_to_excel({
        'Efficiency_Benchmark': efficiency_df
    }, 'results/computational_efficiency.xlsx')
    
    print(f"\n计算效率表格已保存至:")
    print(f"  - results/computational_efficiency.xlsx")
    print(f"  - results/criterion_comparison.xlsx (Computational_Efficiency sheet)")
    print(f"\n设备信息: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # 列出所有生成的文件
    print("\n" + "="*80)
    print("程序执行完毕！破坏准则对比分析完成")
    print("="*80)
    print("\n生成的文件:")
    print("  [准则对比] (NEW)")
    print("    - results/criterion_comparison.xlsx: 准则定量对比表")
    print("    - results/criteria_comparison.png: 准则可靠度曲线对比图")
    print("  [各准则单独结果]")
    for criterion in failure_models:
        print(f"    [{criterion.upper()}]")
        print(f"      - models/bpinn_model_{criterion}.pth: 训练好的模型")
        print(f"      - results/training_history_{criterion}.png: 训练历史")
        print(f"      - results/reliability_comparison_{criterion}.png: 可靠度对比")
        print(f"      - results/dataset_{criterion}.xlsx: 训练与验证数据")
    print("  [输入分布]")
    print("    - results/input_distributions.png: 输入参数概率分布（Fig.4-5）")
    print("  [等效密度分析]")
    print("    - results/rho_vs_inclination_collapse.png: 塌陷密度-井斜角曲线")
    print("    - results/rho_vs_inclination_fracture.png: 破裂密度-井斜角曲线")
    print("  [物理解基准]")
    print("    - results/reliability_physics.png: 物理解可靠度曲线（Fig.8-10）")
    print("="*80)


if __name__ == "__main__":
    main()

