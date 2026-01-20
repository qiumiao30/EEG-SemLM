import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

import args

parser = args.get_parser()
args = parser.parse_args()

if args.dataset == "WADI":
    class PromptGenerator:
        def __init__(self):
            """
            Initialize the WADI Dataset Prompt Generator for anomaly detection tasks.
            Includes 118 features and relevant contextual details.
            """
            # Complete feature list for WADI dataset
            self.features = ['1_AIT_001_PV', '1_AIT_003_PV', '1_AIT_005_PV', '1_FIT_001_PV', '1_LS_001_AL', '1_LS_002_AL', '1_LT_001_PV', 
                            '1_MV_001_STATUS', '1_MV_002_STATUS', '1_MV_003_STATUS', '1_MV_004_STATUS', '1_P_001_STATUS', '1_P_002_STATUS', 
                            '1_P_003_STATUS', '1_P_004_STATUS', '1_P_005_STATUS', '1_P_006_STATUS', '2_DPIT_001_PV', '2_FIC_101_CO', 
                            '2_FIC_101_PV', '2_FIC_101_SP', '2_FIC_201_CO', '2_FIC_201_PV', '2_FIC_201_SP', '2_FIC_301_CO', '2_FIC_301_PV', 
                            '2_FIC_301_SP', '2_FIC_401_CO', '2_FIC_401_PV', '2_FIC_401_SP', '2_FIC_501_CO', '2_FIC_501_PV', '2_FIC_501_SP', 
                            '2_FIC_601_CO', '2_FIC_601_PV', '2_FIC_601_SP', '2_FIT_001_PV', '2_FIT_002_PV', '2_FIT_003_PV', '2_FQ_101_PV', 
                            '2_FQ_201_PV', '2_FQ_301_PV', '2_FQ_401_PV', '2_FQ_501_PV', '2_FQ_601_PV', '2_LS_101_AH', '2_LS_101_AL', 
                            '2_LS_201_AH', '2_LS_201_AL', '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', 
                            '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_LT_001_PV', '2_LT_002_PV', '2_MCV_007_CO', '2_MCV_101_CO', 
                            '2_MCV_201_CO', '2_MCV_301_CO', '2_MCV_401_CO', '2_MCV_501_CO', '2_MCV_601_CO', '2_MV_001_STATUS', '2_MV_002_STATUS', 
                            '2_MV_003_STATUS', '2_MV_004_STATUS', '2_MV_005_STATUS', '2_MV_006_STATUS', '2_MV_009_STATUS', '2_MV_101_STATUS', 
                            '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_SPEED', 
                            '2_P_003_STATUS', '2_P_004_SPEED', '2_P_004_STATUS', '2_PIC_003_CO', '2_PIC_003_PV', '2_PIC_003_SP', '2_PIT_001_PV', 
                            '2_PIT_002_PV', '2_PIT_003_PV', '2_SV_101_STATUS', '2_SV_201_STATUS', '2_SV_301_STATUS', '2_SV_401_STATUS', 
                            '2_SV_501_STATUS', '2_SV_601_STATUS', '2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV', '2A_AIT_004_PV', 
                            '2B_AIT_001_PV', '2B_AIT_003_PV', '3_AIT_001_PV', '3_AIT_002_PV', '3_AIT_003_PV', '3_AIT_005_PV', '3_FIT_001_PV', 
                            '3_LS_001_AL', '3_LT_001_PV', '3_MV_001_STATUS', '3_MV_002_STATUS', '3_MV_003_STATUS', '3_P_001_STATUS', 
                            '3_P_002_STATUS', '3_P_003_STATUS', '3_P_004_STATUS', 'LEAK_DIFF_PRESSURE', 'PLANT_START_STOP_LOG', 
                            'TOTAL_CONS_REQUIRED_FLOW']
            
            # Feature groups (arbitrarily grouped; update as needed)
            self.feature_groups = {
                'AIT Sensors': [feat for feat in self.features if "AIT" in feat],
                'Flow Measurements': [feat for feat in self.features if "FIT" in feat or "FQ" in feat],
                'Tank Levels': [feat for feat in self.features if "LT" in feat or "LS" in feat],
                'Valves': [feat for feat in self.features if "MV" in feat or "SV" in feat or "MCV" in feat],
                'Pumps': [feat for feat in self.features if "P_" in feat],
                'Pressure Monitoring': [feat for feat in self.features if "PIT" in feat or "DPIT" in feat],
                'System Logs': [feat for feat in self.features if "LOG" in feat],
            }

        def format_testbed_data(self, timestamp_data):
            """
            Format data for a single timestamp, displaying only the values.
            """
            return ", ".join([f"{value:.2f}" for value in timestamp_data])

        def get_group_summary(self, timestamp_data):
            """
            Generate summaries for each feature group.
            """
            summaries = []
            for group, features in self.feature_groups.items():
                values = [timestamp_data[self.features.index(feat)] for feat in features if feat in self.features]
                avg_value = sum(values) / len(values) if values else 0
                summaries.append(f"{group} Group Avg: {avg_value:.2f}")
            return "\n".join(summaries)

        def generate_prompt(self, input_data, window_size=10):
            """
            Generate professional prompts for anomaly detection tasks in the WADI dataset.
            Args:
                input_data: Shape (batch_size, sequence_length, len(features)) data.
                window_size: Time window size for trend analysis.
            """
            prompts = []
            
            feature_names = ", ".join(self.features)

            for batch_idx in range(input_data.shape[0]):
                # Task context and background
                context = (
                    "Task: Anomaly Detection in WADI Testbed\n"
                    "Analyze the following time-series data from critical sensors and actuators to predict the next state and identify potential anomalies.\n\n"
                    "Background Information:\n"
                    "- The WADI dataset represents data collected from a water distribution and processing system.\n"
                    "- Sampling rate: 1-second intervals.\n"
                    "- Data includes normal operations and simulated attacks.\n\n"
                    "Normal Behavior Expectations:\n"
                    "- Tank levels should correlate with pump activity.\n"
                    "- Flow rates should align with valve operations.\n"
                    "- Pressure sensors should reflect normal operational ranges.\n"
                    "- AIT sensors should report stable readings under normal water quality conditions.\n\n"
                    "Anomaly Indicators:\n"
                    "- Tank overflows or underflows outside expected ranges.\n"
                    "- Flow inconsistencies not justified by valve or pump actions.\n"
                    "- Spikes in pressure or water quality metrics.\n\n"
                    f"Monitored Variables: {feature_names}\n\n"
                    "Historical Data:\n"
                )
                
                # Add time-series data
                time_series_data = []
                for time_step in range(input_data.shape[1]):
                    timestamp_data = input_data[batch_idx, time_step]
                    formatted_data = self.format_testbed_data(timestamp_data)
                    time_series_data.append(f"\nTimestamp {time_step + 1}: {formatted_data}")
                    
                    # Add group summaries at the end of each window
                    if (time_step + 1) % window_size == 0 or time_step == input_data.shape[1] - 1:
                        time_series_data.append("\nFeature Group Analysis:")
                        time_series_data.append(self.get_group_summary(timestamp_data))
                
                context += "\n".join(time_series_data)
                
                # Add specific prediction and anomaly detection request
                request = (
                    "\n\nAnomaly Detection Request:\n"
                    "Using the historical data, predict the next timestamp's values for all variables and identify any potential anomalies.\n"
                    "For anomalies, provide the following details:\n"
                    "- Which variables are anomalous.\n"
                    "- The expected normal behavior and how it differs from the observed values.\n"
                    "<|<end_prompt>|>"
                )
                
                prompts.append(context + request)
            
            return prompts


elif args.dataset == "SWAT":
    class PromptGenerator:
        def __init__(self):
            """
            Initialize the SWaT Testbed Prompt Generator for anomaly detection tasks.
            Includes all 51 features and relevant contextual details.
            """
            # Complete feature list from the dataset (51 features from Table 2)
            self.features = [
                'FIT-101', 'LIT-101', 'MV-101', 'P-101', 'P-102', 'AIT-201',
                'AIT-202', 'AIT-203', 'FIT-201', 'MV-201', 'P-201', 'P-202',
                'DPIT-301', 'FIT-301', 'LIT-301', 'MV-301', 'MV-302', 'MV-303',
                'MV-304', 'P-301', 'P-302', 'AIT-401', 'FIT-401', 'LIT-401',
                'P-401', 'P-402', 'AIT-402', 'UV-401', 'PIT-501', 'AIT-501',
                'FIT-501', 'PIT-502', 'AIT-502', 'FIT-502', 'FIT-503', 'FIT-504',
                'P-501', 'P-502', 'PIT-503', 'AIT-503', 'AIT-504', 'FIT-601',
                'P-601', 'P-602', 'P-603'
            ]
            
            # Feature groups for context
            self.feature_groups = {
                'Flow Measurements': ['FIT-101', 'FIT-201', 'FIT-301', 'FIT-401', 'FIT-501', 'FIT-502', 'FIT-503', 'FIT-504', 'FIT-601'],
                'Tank Levels': ['LIT-101', 'LIT-301', 'LIT-401'],
                'Actuators': ['MV-101', 'MV-201', 'MV-301', 'MV-302', 'MV-303', 'MV-304', 'P-101', 'P-102', 'P-201', 'P-301', 'P-302', 'P-401', 'P-402', 'P-501', 'P-502', 'P-601', 'P-602', 'P-603'],
                'Water Quality': ['AIT-201', 'AIT-202', 'AIT-203', 'AIT-401', 'AIT-402', 'AIT-501', 'AIT-502', 'AIT-503', 'AIT-504'],
                'Pressure Monitoring': ['DPIT-301', 'PIT-501', 'PIT-502', 'PIT-503'],
                'Dechlorination': ['UV-401']
            }

        # def format_testbed_data(self, timestamp_data):
        #     """
        #     Format data for a single timestamp, displaying only the values.
        #     """
        #     return ", ".join([f"{value:.2f}" for value in timestamp_data])

        def get_group_summary(self, timestamp_data):
            """
            Generate summaries for each feature group.
            """
            summaries = []
            for group, features in self.feature_groups.items():
                values = [timestamp_data[self.features.index(feat)] for feat in features if feat in self.features]
                avg_value = sum(values) / len(values) if values else 0
                summaries.append(f"{group} Group Avg: {avg_value:.2f}")
            return "\n".join(summaries)

        def generate_prompt(self, input_data, window_size=10):
            """
            Generate professional prompts for SWaT Testbed anomaly detection tasks.
            Args:
                input_data: Shape (batch_size, sequence_length, len(features)) testbed data.
                window_size: Time window size for trend analysis.
            """
            prompts = []
            
            feature_names = ", ".join(self.features)

            for batch_idx in range(input_data.shape[0]):
                # Task context and background
                context = (
                    "Task: Anomaly Detection in SWaT Testbed\n"
                    "Analyze the following time-series data from critical sensors and actuators to predict the next state and identify potential anomalies.\n\n"
                    "Background Information:\n"
                    "- The data includes normal operations and attack scenarios.\n"
                    "- Sampling rate: 1-second intervals.\n"
                    "- Anomalies may arise from inconsistencies in sensor-actuator relationships or unexpected changes in sensor readings.\n\n"
                    "Normal Behavior Expectations:\n"
                    "- Tank levels (LIT sensors) should correspond with pump activity.\n"
                    "- Flow rates (FIT sensors) should match valve positions.\n"
                    "- Water quality sensors (AIT sensors) should remain stable unless dosing is applied.\n\n"
                    "Anomaly Indicators:\n"
                    "- Tank overflow (e.g., LIT-101 > threshold while pump P-101 remains active).\n"
                    "- Tank underflow (e.g., LIT-301 < threshold while pump P-301 is off).\n"
                    "- Unusual flow rates (e.g., FIT-401 inconsistent with valve MV-401 positions).\n\n"
                    f"Monitored Variables: {feature_names}\n\n"
                    "Historical Data:\n"
                )
                
                # Add time-series data
                time_series_data = []
                for time_step in range(input_data.shape[1]):
                    timestamp_data = input_data[batch_idx, time_step]
                    # formatted_data = self.format_testbed_data(timestamp_data)
                    # time_series_data.append(f"\nTimestamp {time_step + 1}: {formatted_data}")
                    
                    # Add group summaries at the end of each window
                    if (time_step + 1) % window_size == 0 or time_step == input_data.shape[1] - 1:
                        time_series_data.append("\nFeature Group Analysis:")
                        time_series_data.append(self.get_group_summary(timestamp_data))
                
                context += "\n".join(time_series_data)
                
                # Add specific prediction and anomaly detection request
                request = (
                    "\n\nAnomaly Detection Request:\n"
                    "Using the historical data, predict the next timestamp's values for all variables and identify any potential anomalies.\n"
                    "For anomalies, provide the following details:\n"
                    "- Which variables are anomalous.\n"
                    "- The expected normal behavior and how it differs from the observed values.\n"
                    "<|<end_prompt>|>"
                )
                
                prompts.append(context + request)
            
            return prompts

elif args.dataset == "PSM":
    class PromptGenerator:
        def __init__(self):
            """
            Initialize the New Dataset Testbed Prompt Generator for anomaly detection tasks.
            Includes 25 variables and relevant contextual details.
            """
            # Complete feature list from the new dataset (25 unknown variables)
            self.features = [f"Var-{i+1}" for i in range(25)]  # Var-1, Var-2, ..., Var-25
            
            # Placeholder for feature groups if applicable
            self.feature_groups = {
                'Group 1': self.features[:5],  # Example grouping
                'Group 2': self.features[5:10],
                'Group 3': self.features[10:15],
                'Group 4': self.features[15:20],
                'Group 5': self.features[20:],
            }

        def format_testbed_data(self, timestamp_data):
            """
            Format data for a single timestamp, displaying only the values.
            """
            return ", ".join([f"{value:.2f}" for value in timestamp_data])

        def get_group_summary(self, timestamp_data):
            """
            Generate summaries for each feature group.
            """
            summaries = []
            for group, features in self.feature_groups.items():
                values = [timestamp_data[self.features.index(feat)] for feat in features if feat in self.features]
                avg_value = sum(values) / len(values) if values else 0
                summaries.append(f"{group} Group Avg: {avg_value:.2f}")
            return "\n".join(summaries)

        def generate_prompt(self, input_data, window_size=10):
            """
            Generate professional prompts for anomaly detection tasks in the new dataset.
            Args:
                input_data: Shape (batch_size, sequence_length, len(features)) data.
                window_size: Time window size for trend analysis.
            """
            prompts = []
            
            feature_names = ", ".join(self.features)

            for batch_idx in range(input_data.shape[0]):
                # Task context and background
                context = (
                    "Task: Anomaly Detection in New Dataset\n"
                    "Analyze the following time-series data from critical variables to predict the next state and identify potential anomalies.\n\n"
                    "Background Information:\n"
                    "- The data includes normal operations and abnormal events.\n"
                    "- Sampling rate: 1-second intervals.\n"
                    "- Anomalies may arise from inconsistencies in variable relationships or unexpected changes in readings.\n\n"
                    "Normal Behavior Expectations:\n"
                    "- Variables within the same group should correlate with each other.\n"
                    "- Certain variables should show predictable behavior based on time of day, operational states, etc.\n\n"
                    "Anomaly Indicators:\n"
                    "- Sudden spikes or drops in any variable.\n"
                    "- Variables not correlating with the expected trends.\n\n"
                    f"Monitored Variables: {feature_names}\n\n"
                    "Historical Data:\n"
                )
                
                # Add time-series data
                time_series_data = []
                for time_step in range(input_data.shape[1]):
                    timestamp_data = input_data[batch_idx, time_step]
                    formatted_data = self.format_testbed_data(timestamp_data)
                    time_series_data.append(f"\nTimestamp {time_step + 1}: {formatted_data}")
                    
                    # Add group summaries at the end of each window
                    if (time_step + 1) % window_size == 0 or time_step == input_data.shape[1] - 1:
                        time_series_data.append("\nFeature Group Analysis:")
                        time_series_data.append(self.get_group_summary(timestamp_data))
                
                context += "\n".join(time_series_data)
                
                # Add specific prediction and anomaly detection request
                request = (
                    "\n\nAnomaly Detection Request:\n"
                    "Using the historical data, predict the next timestamp's values for all variables and identify any potential anomalies.\n"
                    "For anomalies, provide the following details:\n"
                    "- Which variables are anomalous.\n"
                    "- The expected normal behavior and how it differs from the observed values.\n"
                    "<|<end_prompt>|>"
                )
                
                prompts.append(context + request)
            
            return prompts


elif args.dataset == "SERVERMACHINEDATASET":
    class PromptGenerator:
        def __init__(self):
            """
            Initialize the SMD (Server Machine Dataset) Prompt Generator for anomaly detection tasks.
            Includes 38 features and relevant contextual details for a server monitoring environment.
            """
            # Complete feature list for SMD dataset
            self.features = [f"Feature_{i+1}" for i in range(38)]  # Placeholder feature names; replace if actual names are available
            
            # Feature groups (assume arbitrary grouping for server metrics; modify based on actual features)
            self.feature_groups = {
                'CPU Metrics': self.features[:10],  # Assume first 10 features relate to CPU metrics
                'Memory Metrics': self.features[10:20],  # Assume features 11-20 relate to memory usage
                'Disk Metrics': self.features[20:30],  # Assume features 21-30 relate to disk usage
                'Network Metrics': self.features[30:38],  # Assume features 31-38 relate to network activity
            }

        def format_data(self, timestamp_data):
            """
            Format data for a single timestamp, displaying only the values.
            """
            return ", ".join([f"{value:.2f}" for value in timestamp_data])

        def get_group_summary(self, timestamp_data):
            """
            Generate summaries for each feature group.
            """
            summaries = []
            for group, features in self.feature_groups.items():
                values = [timestamp_data[self.features.index(feat)] for feat in features if feat in self.features]
                avg_value = sum(values) / len(values) if values else 0
                summaries.append(f"{group} Group Avg: {avg_value:.2f}")
            return "\n".join(summaries)

        def generate_prompt(self, input_data, window_size=10):
            """
            Generate professional prompts for anomaly detection tasks in the SMD dataset.
            Args:
                input_data: Shape (batch_size, sequence_length, len(features)) data.
                window_size: Time window size for trend analysis.
            """
            prompts = []
            
            feature_names = ", ".join(self.features)

            for batch_idx in range(input_data.shape[0]):
                # Task context and background
                context = (
                    "Task: Anomaly Detection in SMD (Server Machine Dataset)\n"
                    "Analyze the following time-series data collected from 28 server machines to predict the next state and identify potential anomalies.\n\n"
                    "Background Information:\n"
                    "- The dataset represents server performance metrics collected over a 5-week period.\n"
                    "- Sampling is done at intervals of 10 points.\n"
                    "- Each machine has its own subset of data; here, all subsets have been combined into one dataset.\n"
                    "- Anomalies may result from unusual server behavior, such as sudden CPU spikes, memory leaks, or abnormal network activity.\n\n"
                    "Normal Behavior Expectations:\n"
                    "- CPU usage should remain within expected ranges under typical workloads.\n"
                    "- Memory usage should grow predictably without sudden large increases.\n"
                    "- Disk metrics should reflect normal read/write activity patterns.\n"
                    "- Network metrics should show consistent throughput and latency without unexpected spikes.\n\n"
                    "Anomaly Indicators:\n"
                    "- Sudden CPU or memory usage spikes.\n"
                    "- Unexpected disk I/O patterns.\n"
                    "- Unusual network activity such as packet loss or high latency.\n\n"
                    f"Monitored Variables: {feature_names}\n\n"
                    "Historical Data:\n"
                )
                
                # Add time-series data
                time_series_data = []
                for time_step in range(input_data.shape[1]):
                    timestamp_data = input_data[batch_idx, time_step]
                    formatted_data = self.format_data(timestamp_data)
                    time_series_data.append(f"\nTimestamp {time_step + 1}: {formatted_data}")
                    
                    # Add group summaries at the end of each window
                    if (time_step + 1) % window_size == 0 or time_step == input_data.shape[1] - 1:
                        time_series_data.append("\nFeature Group Analysis:")
                        time_series_data.append(self.get_group_summary(timestamp_data))
                
                context += "\n".join(time_series_data)
                
                # Add specific prediction and anomaly detection request
                request = (
                    "\n\nAnomaly Detection Request:\n"
                    "Using the historical data, predict the next timestamp's values for all variables and identify any potential anomalies.\n"
                    "For anomalies, provide the following details:\n"
                    "- Which variables are anomalous.\n"
                    "- The expected normal behavior and how it differs from the observed values.\n"
                    "<|<end_prompt>|>"
                )
                
                prompts.append(context + request)
            
            return prompts


elif args.dataset == "WT03" or args.dataset == "WT23":
    class PromptGenerator:
        def __init__(self):
            """
            初始化风机prompt生成器，包含标准特征名称和相关信息
            """
            # 10个关键特征名称
            self.features = [
                'wind_speed', 'wind_direction', 'power_output', 
                'rotor_speed', 'blade_pitch', 'nacelle_position',
                'generator_speed', 'generator_temp', 'ambient_temp',
                'vibration_level'
            ]
            
            # 定义特征分组，用于提供更有意义的上下文
            self.feature_groups = {
                'Environmental': ['wind_speed', 'wind_direction', 'ambient_temp'],
                'Performance': ['power_output', 'rotor_speed', 'generator_speed'],
                'Control': ['blade_pitch', 'nacelle_position'],
                'Health': ['generator_temp', 'vibration_level']
            }

        def format_turbine_data(self, timestamp_data):
            """
            格式化单个时间点的风机数据，不显示特征名称
            """
            return ", ".join([f"{value:.2f}" for value in timestamp_data])

        def get_group_summary(self, timestamp_data):
            """
            生成按特征组分类的数据摘要
            """
            summaries = []
            for group, features in self.feature_groups.items():
                values = [timestamp_data[self.features.index(feat)] for feat in features]
                avg_value = sum(values) / len(values)
                summaries.append(f"{group} Group Avg: {avg_value:.2f}")
            return "\n".join(summaries)

        def generate_prompt(self, input_data, window_size=5):
            """
            生成风机预测的专业prompt
            Args:
                input_data: shape为(batch_size, sequence_length, 10)的风机数据
                window_size: 用于计算趋势的时间窗口大小
            """
            prompts = []
            
            # 列出所有的特征名称，后续数据中将不再重复显示
            feature_names = ", ".join(self.features)

            for batch_idx in range(input_data.shape[0]):
                # 构建任务描述和上下文
                context = (
                    "Task: Wind Turbine Parameter Prediction\n"
                    "Based on the following wind turbine measurements from 10 key sensors, "
                    "predict the next time point's values for all parameters.\n\n"
                    "Important Context:\n"
                    "- Values are normalized to their respective operating ranges\n"
                    "- Sampling rate: 10-minute intervals\n"
                    "- Consider both individual parameter trends and cross-parameter relationships\n\n"
                    f"Monitored Parameters: {feature_names}\n\n"
                    "Historical Readings:\n"
                )
                
                # 添加时间序列数据
                time_series_data = []
                for time_step in range(input_data.shape[1]):
                    timestamp_data = input_data[batch_idx, time_step]
                    
                    # 添加详细的时间点数据（只显示数据值，不显示特征名）
                    formatted_data = self.format_turbine_data(timestamp_data)
                    time_series_data.append(f"\nTimestamp {time_step + 1}:")
                    time_series_data.append(formatted_data)
                    
                    # 每个时间窗口结束时添加特征组摘要
                    if (time_step + 1) % window_size == 0 or time_step == input_data.shape[1] - 1:
                        time_series_data.append("\nParameter Group Analysis:")
                        time_series_data.append(self.get_group_summary(timestamp_data))
                
                # 将时间序列数据添加到上下文中
                context += "\n".join(time_series_data)
                
                # 添加具体预测要求
                request = (
                    "\n\nPrediction Request:\n"
                    "Based on the above wind turbine patterns, predict the next timestamp's values "
                    "for all 10 parameters, considering:\n"
                    "1. Individual parameter trends and patterns\n"
                    "2. Inter-parameter relationships within each group\n"
                    "3. Overall turbine operating state\n"
                    "4. Environmental conditions impact\n\n"
                    "Provide predictions as numerical values for each parameter, "
                    "maintaining the same parameter order as input data. "
                    "<|<end_prompt>|>"
                )
                
                prompts.append(context + request)
            
            return prompts
    

elif args.dataset == "WIND":
    None

elif args.dataset == "WT":
    None

# elif args.dataset == "EEG":
#     class PromptGenerator:
#         def __init__(self):
#             """
#             Initialize the EEG Prompt Generator for anomaly detection tasks.
#             Includes 21 EEG channels and relevant contextual details.
#             """
#             # Complete feature list for EEG channels (21 channels)
#             self.features = [
#                 'EEG-01', 'EEG-02', 'EEG-03', 'EEG-04', 'EEG-05', 'EEG-06',
#                 'EEG-07', 'EEG-08', 'EEG-09', 'EEG-10', 'EEG-11', 'EEG-12',
#                 'EEG-13', 'EEG-14', 'EEG-15', 'EEG-16', 'EEG-17', 'EEG-18',
#                 'EEG-19', 'EEG-20', 'EEG-21'
#             ]
            
#             # EEG data groupings for context (e.g., brain regions or signal types)
#             self.feature_groups = {
#                 'Frontal': ['EEG-01', 'EEG-02', 'EEG-03', 'EEG-04', 'EEG-05'],
#                 'Parietal': ['EEG-06', 'EEG-07', 'EEG-08', 'EEG-09', 'EEG-10'],
#                 'Occipital': ['EEG-11', 'EEG-12', 'EEG-13', 'EEG-14', 'EEG-15'],
#                 'Temporal': ['EEG-16', 'EEG-17', 'EEG-18', 'EEG-19', 'EEG-20'],
#                 'Central': ['EEG-21']
#             }

#         # def format_eeg_data(self, timestamp_data):
#         #     """
#         #     Format EEG data for a single timestamp, displaying only the values.
#         #     """
#         #     return ", ".join([f"{value:.2f}" for value in timestamp_data])

#         def get_group_summary(self, timestamp_data):
#             """
#             Generate summaries for each feature group in EEG data.
#             """
#             summaries = []
#             for group, features in self.feature_groups.items():
#                 values = [timestamp_data[self.features.index(feat)] for feat in features if feat in self.features]
#                 avg_value = sum(values) / len(values) if values else 0
#                 summaries.append(f"{group} Group Avg: {avg_value:.2f}")
#             return "\n".join(summaries)

#         def generate_prompt(self, input_data, window_size=10):
#             """
#             Generate professional prompts for EEG anomaly detection tasks.
#             Args:
#                 input_data: Shape (batch_size, sequence_length, len(features)) EEG data.
#                 window_size: Time window size for trend analysis.
#             """
#             prompts = []
            
#             feature_names = ", ".join(self.features)

#             for batch_idx in range(input_data.shape[0]):
#                 # Task context and background
#                 context = (
#                     "Task: Anomaly Detection in EEG Data\n"
#                     "Analyze the following time-series data from EEG channels to predict the next state and identify potential anomalies.\n\n"
#                     "Background Information:\n"
#                     "- The data includes both normal brain activity and potential abnormal events (e.g., seizures, sleep disorders).\n"
#                     "- Sampling rate: 500Hz, with data downsampled every 10 points to 50Hz.\n"
#                     "- Anomalies may arise from sudden changes in signal patterns or outlier values across channels.\n\n"
#                     "Normal Behavior Expectations:\n"
#                     "- Frontal channels (EEG-01 to EEG-05) typically show consistent alpha and beta waves during awake states.\n"
#                     "- Parietal channels (EEG-06 to EEG-10) should reflect sensory processing activity.\n"
#                     "- Temporal and occipital channels (EEG-11 to EEG-20) are involved in higher cognitive functions and visual processing.\n"
#                     "- Central channel (EEG-21) typically represents motor control activity.\n\n"
#                     "Anomaly Indicators:\n"
#                     "- Sudden spike or drop in signal amplitude across one or more channels.\n"
#                     "- Disruption in expected frequency patterns (e.g., alpha waves turning into high-frequency beta or gamma waves).\n"
#                     "- Persistent abnormal patterns across multiple channels, suggesting seizure-like activity or sleep disruptions.\n\n"
#                     f"Monitored Variables: {feature_names}\n\n"
#                     "Historical Data:\n"
#                 )
                
#                 # Add time-series data
#                 time_series_data = []
#                 for time_step in range(input_data.shape[1]):
#                     timestamp_data = input_data[batch_idx, time_step]
#                     # formatted_data = self.format_eeg_data(timestamp_data)
#                     # time_series_data.append(f"\nTimestamp {time_step + 1}: {formatted_data}")
                    
#                     # Add group summaries at the end of each window
#                     if (time_step + 1) % window_size == 0 or time_step == input_data.shape[1] - 1:
#                         time_series_data.append("\nFeature Group Analysis:")
#                         time_series_data.append(self.get_group_summary(timestamp_data))
                
#                 context += "\n".join(time_series_data)
                
#                 # Add specific prediction and anomaly detection request
#                 request = (
#                     "\n\nAnomaly Detection Request:\n"
#                     "Using the historical data, predict the next timestamp's values for all variables and identify any potential anomalies.\n"
#                     "For anomalies, provide the following details:\n"
#                     "- Which variables are anomalous.\n"
#                     "- The expected normal behavior and how it differs from the observed values.\n"
#                     "<|<end_prompt>|>"
#                 )
                
#                 prompts.append(context + request)
            
#             return prompts

elif args.dataset == "EEG":
    class PromptGenerator:
        def __init__(self):
            # 完整的EEG通道列表
            self.features = [
                'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4',
                'P3', 'P4', 'O1', 'O2', 'T5', 'T6', 'Pz', 'Cz', 'Fz', 'Oz', 'A1'
            ]
            
            # EEG数据组（按脑区分类）
            self.feature_groups = {
                'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8'],
                'Temporal': ['T3', 'T4', 'T5', 'T6'],
                'Central': ['C3', 'C4', 'Cz'],
                'Parietal': ['P3', 'P4', 'Pz'],
                'Occipital': ['O1', 'O2', 'Oz'],
                'Midline': ['Fz']
            }

        def get_group_summary(self, timestamp_data):
            summaries = []
            for group, features in self.feature_groups.items():
                values = [timestamp_data[self.features.index(feat)] for feat in features if feat in self.features]
                avg_value = sum(values) / len(values) if values else 0
                summaries.append(f"{group} Group Avg: {avg_value:.2f}")
            return "\n".join(summaries)

        def generate_prompt(self, input_data, window_size=25):
            prompts = []
            
            feature_names = ", ".join(self.features)

            for batch_idx in range(input_data.shape[0]):
                context = (
                    "Task: Anomaly Detection in EEG Data\n"
                    "Analyze the following time-series data from EEG channels and predict anomalies.\n"
                    "Normal Behavior: Frontal (alpha/beta), Parietal (sensory), Temporal/Occipital (cognitive/visual), Central (motor control).\n"
                    "Anomaly Indicators: Sudden signal spikes, disrupted frequency patterns, persistent abnormal patterns.\n\n"
                    f"Monitored Variables: {feature_names}\n\n"
                    "Historical Data:\n"
                )
                
                # Add time-series data with group summaries
                time_series_data = []
                for time_step in range(input_data.shape[1]):
                    timestamp_data = input_data[batch_idx, time_step]
                    
                    if (time_step + 1) % window_size == 0 or time_step == input_data.shape[1] - 1:
                        time_series_data.append("\nFeature Group Analysis:")
                        time_series_data.append(self.get_group_summary(timestamp_data))
                
                context += "\n".join(time_series_data)
                
                # Add anomaly detection request
                request = (
                    "\n\nAnomaly Detection Request:\n"
                    "Predict the next timestamp's values and identify anomalies. Provide details on:\n"
                    "- Anomalous variables\n"
                    "- Expected vs observed behavior\n"
                    "<|<end_prompt>|>"
                )
                
                prompts.append(context + request)
            
            return prompts

class PromptGenerator:
    def __init__(self):
        # 完整的EEG通道列表
        self.features = [
            'Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4',
            'P3', 'P4', 'O1', 'O2', 'T5', 'T6', 'Pz', 'Cz', 'Fz', 'Oz', 'A1'
        ]
        
        # EEG数据组（按脑区分类）
        self.feature_groups = {
            'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8'],
            'Temporal': ['T3', 'T4', 'T5', 'T6'],
            'Central': ['C3', 'C4', 'Cz'],
            'Parietal': ['P3', 'P4', 'Pz'],
            'Occipital': ['O1', 'O2', 'Oz'],
            'Midline': ['Fz']
        }

    def get_group_summary(self, timestamp_data):
        summaries = []
        for group, features in self.feature_groups.items():
            values = [timestamp_data[self.features.index(feat)] for feat in features if feat in self.features]
            avg_value = sum(values) / len(values) if values else 0
            summaries.append(f"{group} Group Avg: {avg_value:.2f}")
        return "\n".join(summaries)

    def generate_prompt(self, input_data, window_size=25):
        prompts = []
        
        feature_names = ", ".join(self.features)

        for batch_idx in range(input_data.shape[0]):
            context = (
                "Task: Anomaly Detection in EEG Data\n"
                "Analyze the following time-series data from EEG channels and predict anomalies.\n"
                "Normal Behavior: Frontal (alpha/beta), Parietal (sensory), Temporal/Occipital (cognitive/visual), Central (motor control).\n"
                "Anomaly Indicators: Sudden signal spikes, disrupted frequency patterns, persistent abnormal patterns.\n\n"
                f"Monitored Variables: {feature_names}\n\n"
                "Historical Data:\n"
            )
            
            # Add time-series data with group summaries
            time_series_data = []
            for time_step in range(input_data.shape[1]):
                timestamp_data = input_data[batch_idx, time_step]
                
                if (time_step + 1) % window_size == 0 or time_step == input_data.shape[1] - 1:
                    time_series_data.append("\nFeature Group Analysis:")
                    time_series_data.append(self.get_group_summary(timestamp_data))
            
            context += "\n".join(time_series_data)
            
            # Add anomaly detection request
            request = (
                "\n\nAnomaly Detection Request:\n"
                "Predict the next timestamp's values and identify anomalies. Provide details on:\n"
                "- Anomalous variables\n"
                "- Expected vs observed behavior\n"
                "<|<end_prompt>|>"
            )
            
            prompts.append(context + request)
        
        return prompts

import numpy as np
input_data = np.random.rand(1, 50, 21)
prompt_generator = PromptGenerator()

prompts = prompt_generator.generate_prompt(input_data)

print(prompts[0])
print("\nNumber of Prompts:", len(prompts[0]))


# WT03


# SWaT数据集的prompt生成器


# PSM
# class PromptGenerator:
#     def __init__(self):
#         """
#         Initialize the New Dataset Testbed Prompt Generator for anomaly detection tasks.
#         Includes 25 variables and relevant contextual details.
#         """
#         # Complete feature list from the new dataset (25 unknown variables)
#         self.features = [f"Var-{i+1}" for i in range(25)]  # Var-1, Var-2, ..., Var-25
        
#         # Placeholder for feature groups if applicable
#         self.feature_groups = {
#             'Group 1': self.features[:5],  # Example grouping
#             'Group 2': self.features[5:10],
#             'Group 3': self.features[10:15],
#             'Group 4': self.features[15:20],
#             'Group 5': self.features[20:],
#         }

#     def format_testbed_data(self, timestamp_data):
#         """
#         Format data for a single timestamp, displaying only the values.
#         """
#         return ", ".join([f"{value:.2f}" for value in timestamp_data])

#     def get_group_summary(self, timestamp_data):
#         """
#         Generate summaries for each feature group.
#         """
#         summaries = []
#         for group, features in self.feature_groups.items():
#             values = [timestamp_data[self.features.index(feat)] for feat in features if feat in self.features]
#             avg_value = sum(values) / len(values) if values else 0
#             summaries.append(f"{group} Group Avg: {avg_value:.2f}")
#         return "\n".join(summaries)

#     def generate_prompt(self, input_data, window_size=10):
#         """
#         Generate professional prompts for anomaly detection tasks in the new dataset.
#         Args:
#             input_data: Shape (batch_size, sequence_length, len(features)) data.
#             window_size: Time window size for trend analysis.
#         """
#         prompts = []
        
#         feature_names = ", ".join(self.features)

#         for batch_idx in range(input_data.shape[0]):
#             # Task context and background
#             context = (
#                 "Task: Anomaly Detection in New Dataset\n"
#                 "Analyze the following time-series data from critical variables to predict the next state and identify potential anomalies.\n\n"
#                 "Background Information:\n"
#                 "- The data includes normal operations and abnormal events.\n"
#                 "- Sampling rate: 1-second intervals.\n"
#                 "- Anomalies may arise from inconsistencies in variable relationships or unexpected changes in readings.\n\n"
#                 "Normal Behavior Expectations:\n"
#                 "- Variables within the same group should correlate with each other.\n"
#                 "- Certain variables should show predictable behavior based on time of day, operational states, etc.\n\n"
#                 "Anomaly Indicators:\n"
#                 "- Sudden spikes or drops in any variable.\n"
#                 "- Variables not correlating with the expected trends.\n\n"
#                 f"Monitored Variables: {feature_names}\n\n"
#                 "Historical Data:\n"
#             )
            
#             # Add time-series data
#             time_series_data = []
#             for time_step in range(input_data.shape[1]):
#                 timestamp_data = input_data[batch_idx, time_step]
#                 formatted_data = self.format_testbed_data(timestamp_data)
#                 time_series_data.append(f"\nTimestamp {time_step + 1}: {formatted_data}")
                
#                 # Add group summaries at the end of each window
#                 if (time_step + 1) % window_size == 0 or time_step == input_data.shape[1] - 1:
#                     time_series_data.append("\nFeature Group Analysis:")
#                     time_series_data.append(self.get_group_summary(timestamp_data))
            
#             context += "\n".join(time_series_data)
            
#             # Add specific prediction and anomaly detection request
#             request = (
#                 "\n\nAnomaly Detection Request:\n"
#                 "Using the historical data, predict the next timestamp's values for all variables and identify any potential anomalies.\n"
#                 "For anomalies, provide the following details:\n"
#                 "- Which variables are anomalous.\n"
#                 "- The expected normal behavior and how it differs from the observed values.\n"
#                 "<|<end_prompt>|>"
#             )
            
#             prompts.append(context + request)
        
#         return prompts

# WADI
# class PromptGenerator:
#     def __init__(self):
#         """
#         Initialize the WADI Dataset Prompt Generator for anomaly detection tasks.
#         Includes 118 features and relevant contextual details.
#         """
#         # Complete feature list for WADI dataset
#         self.features = ['1_AIT_001_PV', '1_AIT_003_PV', '1_AIT_005_PV', '1_FIT_001_PV', '1_LS_001_AL', '1_LS_002_AL', '1_LT_001_PV', 
#                          '1_MV_001_STATUS', '1_MV_002_STATUS', '1_MV_003_STATUS', '1_MV_004_STATUS', '1_P_001_STATUS', '1_P_002_STATUS', 
#                          '1_P_003_STATUS', '1_P_004_STATUS', '1_P_005_STATUS', '1_P_006_STATUS', '2_DPIT_001_PV', '2_FIC_101_CO', 
#                          '2_FIC_101_PV', '2_FIC_101_SP', '2_FIC_201_CO', '2_FIC_201_PV', '2_FIC_201_SP', '2_FIC_301_CO', '2_FIC_301_PV', 
#                          '2_FIC_301_SP', '2_FIC_401_CO', '2_FIC_401_PV', '2_FIC_401_SP', '2_FIC_501_CO', '2_FIC_501_PV', '2_FIC_501_SP', 
#                          '2_FIC_601_CO', '2_FIC_601_PV', '2_FIC_601_SP', '2_FIT_001_PV', '2_FIT_002_PV', '2_FIT_003_PV', '2_FQ_101_PV', 
#                          '2_FQ_201_PV', '2_FQ_301_PV', '2_FQ_401_PV', '2_FQ_501_PV', '2_FQ_601_PV', '2_LS_101_AH', '2_LS_101_AL', 
#                          '2_LS_201_AH', '2_LS_201_AL', '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', 
#                          '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_LT_001_PV', '2_LT_002_PV', '2_MCV_007_CO', '2_MCV_101_CO', 
#                          '2_MCV_201_CO', '2_MCV_301_CO', '2_MCV_401_CO', '2_MCV_501_CO', '2_MCV_601_CO', '2_MV_001_STATUS', '2_MV_002_STATUS', 
#                          '2_MV_003_STATUS', '2_MV_004_STATUS', '2_MV_005_STATUS', '2_MV_006_STATUS', '2_MV_009_STATUS', '2_MV_101_STATUS', 
#                          '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_SPEED', 
#                          '2_P_003_STATUS', '2_P_004_SPEED', '2_P_004_STATUS', '2_PIC_003_CO', '2_PIC_003_PV', '2_PIC_003_SP', '2_PIT_001_PV', 
#                          '2_PIT_002_PV', '2_PIT_003_PV', '2_SV_101_STATUS', '2_SV_201_STATUS', '2_SV_301_STATUS', '2_SV_401_STATUS', 
#                          '2_SV_501_STATUS', '2_SV_601_STATUS', '2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV', '2A_AIT_004_PV', 
#                          '2B_AIT_001_PV', '2B_AIT_003_PV', '3_AIT_001_PV', '3_AIT_002_PV', '3_AIT_003_PV', '3_AIT_005_PV', '3_FIT_001_PV', 
#                          '3_LS_001_AL', '3_LT_001_PV', '3_MV_001_STATUS', '3_MV_002_STATUS', '3_MV_003_STATUS', '3_P_001_STATUS', 
#                          '3_P_002_STATUS', '3_P_003_STATUS', '3_P_004_STATUS', 'LEAK_DIFF_PRESSURE', 'PLANT_START_STOP_LOG', 
#                          'TOTAL_CONS_REQUIRED_FLOW']
        
#         # Feature groups (arbitrarily grouped; update as needed)
#         self.feature_groups = {
#             'AIT Sensors': [feat for feat in self.features if "AIT" in feat],
#             'Flow Measurements': [feat for feat in self.features if "FIT" in feat or "FQ" in feat],
#             'Tank Levels': [feat for feat in self.features if "LT" in feat or "LS" in feat],
#             'Valves': [feat for feat in self.features if "MV" in feat or "SV" in feat or "MCV" in feat],
#             'Pumps': [feat for feat in self.features if "P_" in feat],
#             'Pressure Monitoring': [feat for feat in self.features if "PIT" in feat or "DPIT" in feat],
#             'System Logs': [feat for feat in self.features if "LOG" in feat],
#         }

#     def format_testbed_data(self, timestamp_data):
#         """
#         Format data for a single timestamp, displaying only the values.
#         """
#         return ", ".join([f"{value:.2f}" for value in timestamp_data])

#     def get_group_summary(self, timestamp_data):
#         """
#         Generate summaries for each feature group.
#         """
#         summaries = []
#         for group, features in self.feature_groups.items():
#             values = [timestamp_data[self.features.index(feat)] for feat in features if feat in self.features]
#             avg_value = sum(values) / len(values) if values else 0
#             summaries.append(f"{group} Group Avg: {avg_value:.2f}")
#         return "\n".join(summaries)

#     def generate_prompt(self, input_data, window_size=10):
#         """
#         Generate professional prompts for anomaly detection tasks in the WADI dataset.
#         Args:
#             input_data: Shape (batch_size, sequence_length, len(features)) data.
#             window_size: Time window size for trend analysis.
#         """
#         prompts = []
        
#         feature_names = ", ".join(self.features)

#         for batch_idx in range(input_data.shape[0]):
#             # Task context and background
#             context = (
#                 "Task: Anomaly Detection in WADI Testbed\n"
#                 "Analyze the following time-series data from critical sensors and actuators to predict the next state and identify potential anomalies.\n\n"
#                 "Background Information:\n"
#                 "- The WADI dataset represents data collected from a water distribution and processing system.\n"
#                 "- Sampling rate: 1-second intervals.\n"
#                 "- Data includes normal operations and simulated attacks.\n\n"
#                 "Normal Behavior Expectations:\n"
#                 "- Tank levels should correlate with pump activity.\n"
#                 "- Flow rates should align with valve operations.\n"
#                 "- Pressure sensors should reflect normal operational ranges.\n"
#                 "- AIT sensors should report stable readings under normal water quality conditions.\n\n"
#                 "Anomaly Indicators:\n"
#                 "- Tank overflows or underflows outside expected ranges.\n"
#                 "- Flow inconsistencies not justified by valve or pump actions.\n"
#                 "- Spikes in pressure or water quality metrics.\n\n"
#                 f"Monitored Variables: {feature_names}\n\n"
#                 "Historical Data:\n"
#             )
            
#             # Add time-series data
#             time_series_data = []
#             for time_step in range(input_data.shape[1]):
#                 timestamp_data = input_data[batch_idx, time_step]
#                 formatted_data = self.format_testbed_data(timestamp_data)
#                 time_series_data.append(f"\nTimestamp {time_step + 1}: {formatted_data}")
                
#                 # Add group summaries at the end of each window
#                 if (time_step + 1) % window_size == 0 or time_step == input_data.shape[1] - 1:
#                     time_series_data.append("\nFeature Group Analysis:")
#                     time_series_data.append(self.get_group_summary(timestamp_data))
            
#             context += "\n".join(time_series_data)
            
#             # Add specific prediction and anomaly detection request
#             request = (
#                 "\n\nAnomaly Detection Request:\n"
#                 "Using the historical data, predict the next timestamp's values for all variables and identify any potential anomalies.\n"
#                 "For anomalies, provide the following details:\n"
#                 "- Which variables are anomalous.\n"
#                 "- The expected normal behavior and how it differs from the observed values.\n"
#                 "<|<end_prompt>|>"
#             )
            
#             prompts.append(context + request)
        
#         return prompts


