import pickle
import sys
sys.path.append("/gyc_data/fray_data/Arbiter")
print(sys.path)
from openbox.utils.config_space import get_one_exchange_neighbourhood, \
    Configuration, ConfigurationSpace
import torch
device = torch.device("cuda")
import re
import torch.nn.functional as F
import ast


def load_cache():
        try:
            with open('BO/train/tpch100.pkl', 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {}  # 如果文件不存在或为空，返回一个空字典
        
cache = load_cache()
print(len(cache))
# def read_config_dicts(log_file_path = '/data2/fray/index_selection_evaluation/train_log/train_tpch_100.log'):

#     # Regular expression to match the arrays
#     array_regex = re.compile(r'INFO:root:Gray code X: (\[\[.*?\])\s+INFO:root:Binary code:')

#     # This list will hold all of the matched arrays
#     matched_arrays = []

#     # Open the log file and read line by line
#     with open(log_file_path, 'r') as file:
#         file_content = file.read()
#         # Find all matches of the regular expression
#         matches = array_regex.findall(file_content)
#         for match in matches:
#             # Append the matched array string, if you need it in list/tuple form, you'll have to parse it further
#             try:
#                 match = ast.literal_eval(match)
#             except ValueError as e:
#                 print(f"Error converting string to array: {e}")
#             matched_arrays.append(match)
    
#     return matched_arrays

# config_dicts = read_config_dicts()[0:1280]
# sample_input = []
# objectives_list = []
# sample_num = 0
# with open('BO/train/tpch100_bak.pkl', 'rb') as f1, open('BO/train/tpch100_label.pkl', 'rb') as f2:
#     while True:
#         try:
#             query_features, data_features, index_features = pickle.load(f1)
#             query_features_device  = []
#             data_features_device = []
             
#             for query_feature in query_features:
#                 query_features_device.append(query_feature.to(device))
#             for data_feature in data_features:
#                 data_features_device.append(data_feature.to(device))
#             index_features_device = index_features.to(device)
#             objectives, constraints = pickle.load(f2)

#             input_list = []
#             input_list.append(query_features_device)
#             input_list.append(data_features_device)
#             input_list.append(index_features_device)
#             input_list.extend(objectives)
#             input_list.extend([0])

#             sample_input.append(input_list)
#             objectives_list.extend(objectives)
    
#             sample_num += 1
#             if sample_num == 1280:
#                 break
#         except EOFError:
#             break
# max_len = max([sample_input[i][1][1].shape[1] for i in range(len(sample_input))])
# for i in range(len(sample_input)):
#     if sample_input[i][1][1].shape[1] < max_len:
#         padded_tensor = F.pad(sample_input[i][1][1], (max_len-sample_input[i][1][1].shape[1], 0))
#         sample_input[i][1][1] = padded_tensor

# print(len(objectives_list))

# decimal_number = 121799282833162979417268800852535250945482380174931699059827269130255299532
# bit_length = 49
# max_decimal = 2**249 - 1

# print(max_decimal >= decimal_number)
# decimal_number = min(decimal_number, max_decimal)

# binary_string = bin(decimal_number)[2:]
# padded_length = 250
# padded_binary_string = binary_string.zfill(padded_length)

# # 反转二进制字符串
# reversed_binary_string = padded_binary_string[::-1]

# res = 0
# power_of_2 = 1
# for i in range(0, padded_length, bit_length):
#     binary_chunk = reversed_binary_string[i:i+bit_length]
#     binary_chunk = binary_chunk[::-1]
#     decimal_chunk = int(binary_chunk, 2)
#     print(decimal_chunk <= 2**bit_length - 1)
#     res += decimal_chunk * power_of_2
#     power_of_2 *= 2**bit_length

# print(res == decimal_number)
