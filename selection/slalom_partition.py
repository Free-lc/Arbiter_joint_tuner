import math
from pathlib import Path
import sys
from scipy import stats
cur_path = Path(__file__).resolve().parent.parent
print(cur_path)
sys.path.append(cur_path.as_posix())
import numpy as np
from scipy.stats import kurtosis
import decimal
from selection.utils import min_records_number


def get_subpartition_num(k, N, sel, partition_max):
    """
    Calculate P(X >= k) using the binomial distribution approximation.
    k: the number of tuples appearing in the result
    N: the total number of tuples in the file
    sel: the selectivity of the first range query
    """
    if sel == 0:
        return 1
    b = math.e / (sel * (1 - sel))
    p = k / N
    m = math.ceil(N * (sel + math.log(1 - sel, b)) / math.log(math.sqrt(2 * math.pi * sel * N )/2, b))
    if m <= 0 :
        m = 1
    if m > partition_max:
        m = partition_max
    return m

# 定义函数来计算数据分布和查询选择性的统计信息
def calculate_statistics(data):
    # data : [record1, record2, ]
    # data = deal_with_none(data)
    numeric_data = []

    # 确保data是一个NumPy数组
    data = np.array(data, dtype=object)

    # 检查每一列是否至少包含一个int或decimal.Decimal
    valid_columns = []
    for i in range(data.shape[1]):
        column = data[:, i]
        if any(isinstance(x, (int, decimal.Decimal, float)) for x in column):
            valid_columns.append(i)

    # 只处理选中的列
    for sublist in data:
        numeric_item = []
        for index in valid_columns:
            item = sublist[index]
            if isinstance(item, (int, decimal.Decimal, float)):
                numeric_item.append(float(item))  # 转换为浮点数以便计算
            else:
                numeric_item.append(np.nan)  # 将None和非数值替换为np.nan
        numeric_data.append(numeric_item)

    numeric_data = np.array(numeric_data, dtype=np.float64)

    # 计算方差、超峰度和唯一元素数量
    variance = np.nanvar(numeric_data, axis=0)
    kurt = kurtosis(numeric_data, axis=0, nan_policy='omit')
    # 某个维度可能为0
    if numeric_data.size > 0 and numeric_data.shape[0] > 0:
        num_unique_elements = np.apply_along_axis(
        lambda x: len(np.unique(x[~np.isnan(x)])), 
        axis=0, 
        arr=numeric_data
        )
    else:
        num_unique_elements = np.array([0]*kurt.size)

    return [list(variance), list(kurt), list(num_unique_elements)]

# 定义函数来决定逻辑分区是否有利于性能
def is_partitioning_beneficial(data, partition_size):
    # 如果数据分布不均匀或平均查询选择性较低，则逻辑分区有利于性能
    sub_arrays = split_list(data, partition_size)
    old = np.array(calculate_statistics(data))
    a = []
    new = [[],[],[]]
    for array in sub_arrays:
        if len(array) != 0:
            statistics_item = calculate_statistics(array)
            a.append(statistics_item)
    try:
        a = np.array(a, dtype=object)
        new = np.nanmean(a, axis=0) 
    except:
        return 1
    stable = 1
    decrease = 0
    for i in range(len(old[0])):
        if (new[:,i] <= old[:,i]).any():
            decrease += 1
        if decrease >= math.ceil(len(data)/2):
            stable = 0
            break
    # print(decrease)
    return stable



# 定义函数来决定一个分区划分出的子分区数量
def decide_partition_size(data:np.ndarray, selectivity, partition_range)->int:
    print(f'the shape of data is: {data.shape}')
    max_stable = 0
    best_partition_size = 1
    mp = {}
    print("calculating partition number")
    left, right = partition_range
    for dataset in data:
        # not consider the table whose records less than min_records_number
        if len(dataset) < min_records_number:
            continue
        # get N and sel from database 
        partition_size = get_subpartition_num(k=1, N = len(dataset), sel = selectivity, partition_max = right - left)
        
        # 如果逻辑分区有利于性能，则将分区划分为子分区
        if partition_size in mp or partition_size == 1:
            continue
        mp[partition_size] = 0
    for key,value in mp.items():
        for dataset in data:
            if len(dataset) < min_records_number:
                continue
            mp[key] += is_partitioning_beneficial(dataset, key)
        if max_stable <= mp[key]:
            max_stable = mp[key]
            best_partition_size = key
    print(f'going to split the partition into {best_partition_size}')
    return best_partition_size
    # return 2

def split_list(data, partition_size):
    # 计算每个分区应有的大致长度
    n = len(data)
    size = (n + partition_size - 1) // partition_size  # 向上取整

    # 分割列表
    return [data[i:i + size] for i in range(0, n, size)]

def deal_with_none(data):
    # 计算每列的众数，仅考虑int和float类型的元素
    data = np.array(data, dtype=object)
    mode = []
    for i in range(data.shape[1]):
        column = data[:, i]
        valid_values = [x for x in column if isinstance(x, (int, decimal.Decimal)) and x is not None]
        if valid_values:
            column_mode = np.atleast_1d(stats.mode(valid_values).mode)[0]
        else:
            column_mode = None
        mode.append(column_mode)
    
    # 对于每个缺省值，如果其类型为int或float，则用相同列的众数填充
    for i in range(data.shape[1]):
        for j in range(data.shape[0]):
            if data[j, i] is None and mode[i] is not None:
                data[j, i] = mode[i]
    
    return data

# # 定义数据集
# data = np.random.randint(0, 100, size=(3, 7,50000))
# # num_partitions = decide_partition_size(data, 0.01)
# print("The number of sub-partitions for each partition is:", is_partitioning_beneficial(data, 10))


