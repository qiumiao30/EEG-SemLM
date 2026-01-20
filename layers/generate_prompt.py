# import torch
# import numpy as np
# from typing import Dict, Optional

# class PromptGenerator:
#     """Prompt generator for wind farm time series anomaly detection"""

#     def __init__(self, seq_len: int, domain_info: Optional[Dict] = None, variable_descriptions: Optional[Dict] = None):
#         """
#         初始化 PromptGenerator
        
#         Args:
#             seq_len: 输入序列长度
#             domain_info: 领域相关信息字典
#             variable_descriptions: 变量描述字典
#         """
#         self.seq_len = seq_len
#         self.domain_info = domain_info or self._get_default_domain_info()
#         self.variable_descriptions = variable_descriptions or self._get_default_variable_descriptions()

#     def _get_default_domain_info(self) -> Dict:
#         """获取默认的领域信息"""
#         return {
#             "dataset_name": "WT03",
#             "full_name": "Main Bearing Fault Dataset",
#             "location": "Coast, near Dongtai Offshore Wind Farm",
#             "type": "DFIG (Doubly-fed Induction Generator)",
#             "installed_capacity": "2.5 MW",
#             "commissioning_date": "2015",
#             "maintenance_records": "Routine inspection every 6 months",
#             "sampling_intervals": "10 minutes",
#             "fault_type": "Main Bearing Fault",
#             "total_variables": "10",
#             "domain_characteristics": [
#                 "Bearing temperature fluctuation",
#                 "Rotor speed variations",
#                 "Active power differences between generator and grid",
#                 "Wind speed impact on drivetrain",
#                 "Ambient temperature fluctuations",
#                 "Frequent pitch adjustments due to variable wind direction",
#                 "High humidity and saline conditions due to coastal location"
#             ]
#         }

#     def _get_default_variable_descriptions(self) -> Dict:
#         """获取默认的变量描述"""
#         return {
#             "ambient_temp": "Ambient temperature measured at the weather station (°C)",
#             "wind_speed": "Wind speed measured at the weather station (m/s)",
#             "rotor_speed": "Rotor speed measured at the hub (rpm)",
#             "gen_power": "Active power at the generator side (kW)",
#             "grid_power": "Active power at the grid side (kW)",
#             "bearing_temp_mid": "Bearing temperature at the middle of drivetrain (°C)",
#             "bearing_temp_front": "Bearing temperature at the front-end of drivetrain (°C)",
#             "bearing_temp_back": "Bearing temperature at the back-end of drivetrain (°C)",
#             "bearing_temp_env": "Bearing temperature of the environment (°C)",
#             "bearing_temp_diff": "Temperature difference between front and rear bearings (°C)"
#         }

#     def calculate_statistics(self, x_enc: torch.Tensor) -> Dict:
#         """
#         计算统计信息
        
#         Args:
#             x_enc: shape [B, T, N]
        
#         Returns:
#             Dict with statistical information
#         """
#         B, T, N = x_enc.size()
        
#         basic_stats = {
#             "min": torch.min(x_enc, dim=1)[0],
#             "max": torch.max(x_enc, dim=1)[0],
#             "mean": torch.mean(x_enc, dim=1),
#             "median": torch.median(x_enc, dim=1).values,
#             "std": torch.std(x_enc, dim=1),
#         }
        
#         diff = x_enc.diff(dim=1)
#         trend_stats = {
#             "trend": diff.mean(dim=1),
#             "trend_std": diff.std(dim=1),
#             "trend_direction": torch.sign(diff.mean(dim=1)),
#         }
        
#         z_scores = (x_enc - basic_stats["mean"].unsqueeze(1)) / basic_stats["std"].unsqueeze(1)
#         anomaly_stats = {
#             "num_anomalies": torch.sum(torch.abs(z_scores) > 3, dim=1),
#             "max_deviation": torch.max(torch.abs(z_scores), dim=1)[0]
#         }
        
#         return {
#             "basic_stats": basic_stats,
#             "trend_stats": trend_stats,
#             "anomaly_stats": anomaly_stats
#         }

#     def generate_prompt(self, x_enc: torch.Tensor) -> str:
#         """
#         为一个batch生成异常检测的prompt
        
#         Args:
#             x_enc: shape [B, T, N]
        
#         Returns:
#             Generated prompt string
#         """
#         stats = self.calculate_statistics(x_enc)
#         B, N = x_enc.size(0), x_enc.size(2)
        
#         prompts = []
        
#         for i in range(B):  # 为每个batch生成prompt
#             basic_info = (
#                 f"Dataset: {self.domain_info['dataset_name']} ({self.domain_info['full_name']})\n"
#                 f"Wind Farm Location: {self.domain_info['location']}, Type: {self.domain_info['type']}\n"
#                 f"Installed Capacity: {self.domain_info['installed_capacity']}, Commissioned: {self.domain_info['commissioning_date']}\n"
#                 f"Maintenance Records: {self.domain_info['maintenance_records']}\n"
#                 f"Frequency: {self.domain_info['sampling_intervals']}\n"
#                 f"Fault: {self.domain_info['fault_type']}\n"
#                 f"Context window: Previous {self.seq_len} steps ({self.seq_len * 10} minutes)\n"
#                 f"Task: Anomaly detection for next time step.\n"
#             )

#             # 统计信息部分
#             stats_info = (
#                 "Statistical Information:\n"
#                 f"- Range: [{stats['basic_stats']['min'][i].min().item():.4f}, {stats['basic_stats']['max'][i].max().item():.4f}]\n"
#                 f"- Central Tendency: mean={stats['basic_stats']['mean'][i].mean().item():.4f}, median={stats['basic_stats']['median'][i].median().item():.4f}\n"
#                 f"- Variability: std={stats['basic_stats']['std'][i].mean().item():.4f}\n"
#                 f"- Trend: {'increasing' if stats['trend_stats']['trend'][i].mean().item() > 0 else 'decreasing'} "
#                 f"with magnitude {abs(stats['trend_stats']['trend'][i].mean().item()):.4f}\n"
#             )
            
#             # 异常信息
#             anomaly_info = (
#                 "Anomaly Information:\n"
#                 f"- Number of anomalies: {stats['anomaly_stats']['num_anomalies'][i].sum().item()}\n"
#                 f"- Maximum deviation: {stats['anomaly_stats']['max_deviation'][i].max().item():.4f} standard deviations\n"
#             )

#             # 领域特征
#             domain_characteristics = (
#                 "Domain Characteristics:\n" +
#                 "\n".join(f"- {char}" for char in self.domain_info['domain_characteristics'])
#             )

#             # 环境信息
#             environmental_factors = (
#                 "Environmental Factors:\n"
#                 f"- Wind speed: Critical factor affecting drivetrain and power output\n"
#                 f"- Ambient temperature: Impacts bearing temperature and rotor performance\n"
#                 # f"- Humidity: Coastal location prone to corrosion, affects bearing lifespan\n"
#             )

#             # 组合prompt
#             prompt = (
#                 f"{basic_info}\n"
#                 f"{stats_info}\n"
#                 f"{anomaly_info}\n"
#                 f"{domain_characteristics}\n"
#                 f"{environmental_factors}\n"
#                 # "Task: Detect if there is any anomaly in the current time series data.\n"
#                 "<|<end_prompt>|>"
#             )

#             prompts.append(prompt)
        
#         return prompts






###################################
import torch
import numpy as np
from typing import Dict, Optional

class PromptGenerator:
    """Prompt generator for EEG dataset anomaly detection"""

    def __init__(self, seq_len: int, domain_info: Optional[Dict] = None, variable_descriptions: Optional[Dict] = None):
        """
        Initialize PromptGenerator for EEG data.
        
        Args:
            seq_len: Input sequence length
            domain_info: Domain-specific information dictionary
            variable_descriptions: Variable descriptions dictionary
        """
        self.seq_len = seq_len
        self.domain_info = domain_info or self._get_default_domain_info()
        self.variable_descriptions = variable_descriptions or self._get_default_variable_descriptions()

    def _get_default_domain_info(self) -> Dict:
        """Get default domain information for EEG data."""
        return {
            "dataset_name": "EEG",
            "full_name": "Electroencephalography Dataset",
            "location": "Clinical testbed for neural monitoring",
            "type": "Brainwave activity recording",
            "sampling_intervals": "500 ms",
            "data_types": ["EEG channels"],
            "conditions_monitored": [
                "Normal brain activity",
                "Abnormal events (e.g., seizures, anomalies)"
            ]
        }

    def _get_default_variable_descriptions(self) -> Dict:
        """Get default variable descriptions for EEG channels."""
        return {f"Ch-{i+1}": f"EEG Channel {i+1}" for i in range(21)}  # Assuming 21 EEG channels

    def calculate_statistics(self, x_enc: torch.Tensor) -> Dict:
        """
        Calculate statistical information for EEG data.
        
        Args:
            x_enc: shape [B, T, N]
        
        Returns:
            Dict with statistical information
        """
        B, T, N = x_enc.size()
        
        basic_stats = {
            "min": torch.min(x_enc, dim=1)[0],
            "max": torch.max(x_enc, dim=1)[0],
            "mean": torch.mean(x_enc, dim=1),
            "median": torch.median(x_enc, dim=1).values,
            "std": torch.std(x_enc, dim=1),
        }
        
        diff = x_enc.diff(dim=1)
        trend_stats = {
            "trend": diff.mean(dim=1),
            "trend_std": diff.std(dim=1),
            "trend_direction": torch.sign(diff.mean(dim=1)),
        }
        
        z_scores = (x_enc - basic_stats["mean"].unsqueeze(1)) / basic_stats["std"].unsqueeze(1)
        anomaly_stats = {
            "num_anomalies": torch.sum(torch.abs(z_scores) > 3, dim=1),
            "max_deviation": torch.max(torch.abs(z_scores), dim=1)[0]
        }
        
        return {
            "basic_stats": basic_stats,
            "trend_stats": trend_stats,
            "anomaly_stats": anomaly_stats
        }

    def generate_prompt(self, x_enc: torch.Tensor) -> str:
        """
        Generate anomaly detection prompt for a batch of EEG data.
        
        Args:
            x_enc: shape [B, T, N]
        
        Returns:
            Generated prompt string
        """
        stats = self.calculate_statistics(x_enc)
        B, N = x_enc.size(0), x_enc.size(2)
        
        prompts = []
        
        for i in range(B):  # Generate prompt for each batch
            basic_info = (
                f"Dataset: {self.domain_info['dataset_name']} ({self.domain_info['full_name']})\n"
                f"Recording Site: {self.domain_info['location']}\n"
                f"System Type: {self.domain_info['type']}\n"
                f"Conditions Monitored: {', '.join(self.domain_info['conditions_monitored'])}\n"
                f"Sampling Interval: {self.domain_info['sampling_intervals']}\n"
                f"Context window: Previous {self.seq_len} samples ({self.seq_len} x 500 ms)\n"
                f"Task: Detect abnormal brainwave patterns.\n"
            )

            # Statistical information
            stats_info = (
                "Statistical Information:\n"
                f"- Range: [{stats['basic_stats']['min'][i].min().item():.4f}, {stats['basic_stats']['max'][i].max().item():.4f}]\n"
                f"- Central Tendency: mean={stats['basic_stats']['mean'][i].mean().item():.4f}, median={stats['basic_stats']['median'][i].median().item():.4f}\n"
                f"- Variability: std={stats['basic_stats']['std'][i].mean().item():.4f}\n"
                f"- Trend: {'increasing' if stats['trend_stats']['trend'][i].mean().item() > 0 else 'decreasing'} "
                f"with magnitude {abs(stats['trend_stats']['trend'][i].mean().item()):.4f}\n"
            )
            
            # Anomaly information
            anomaly_info = (
                "Anomaly Information:\n"
                f"- Number of anomalies: {stats['anomaly_stats']['num_anomalies'][i].sum().item()}\n"
                f"- Maximum deviation: {stats['anomaly_stats']['max_deviation'][i].max().item():.4f} standard deviations\n"
            )

            # Combine prompt
            prompt = (
                f"{basic_info}\n"
                f"{stats_info}\n"
                f"{anomaly_info}\n"
                "<|<end_prompt>|>"
            )

            prompts.append(prompt)
        
        return prompts
