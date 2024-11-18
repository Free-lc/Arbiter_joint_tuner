import logging
import random
import math
import time
from selection.index import Index
from selection.selection_algorithm import DEFAULT_PARAMETER_VALUES, SelectionAlgorithm
from selection.utils import b_to_mb, mb_to_b

from selection.query_generator import QueryGenerator
from selection.workload import Workload

DEFAULT_PARAMETERS = {
    "budget_MB": DEFAULT_PARAMETER_VALUES["budget_MB"],
    "max_index_width": DEFAULT_PARAMETER_VALUES["max_index_width"],
    "min_cost_improvement": 1.003,
}

# This algorithm is a reimplementation of the Salalom
class SlalomAlgorithm(SelectionAlgorithm):
    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(
            self, database_connector, parameters, DEFAULT_PARAMETERS
        )
        logging.getLogger().setLevel(logging.DEBUG)
        self.time_begin = time.time()
        # self.budget = mb_to_b(self.parameters["budget_MB"])
        self.budget = self.parameters["budget_MB"]
        self.max_index_width = self.parameters["max_index_width"]
        self.partition_num = self.parameters["partition_num"]
        self.workload = None
        self.statistic = {}
        random.seed(88)

    def calculate_best_indexes(self, workload):
        logging.info("Calculating best indexes Slalom")
        self.workload = workload
        columns = self.workload.indexable_columns()
        single_attribute_index_candidates = self.workload.potential_indexes()
        # current indexs
        index_combination = []
        index_combination_size = 0
        self.LRU = LRUCache(self.budget)
        # calculate C_i,C_f,C_b
        for candidate in single_attribute_index_candidates:
            # Slalom only refer to single-col index, so we can extract column as follow
            column = candidate.columns[0]
            self.statistic[column] = {}
            self.statistic[column]['index'] = candidate
            # self.cost_evaluation.calculate_cost_slalom(self.workload, index_combination, store_size=True)
            c_f = self.cost_evaluation.calculate_cost_slalom(
                self.workload, index_combination, store_size=True
                )
            c_i = self.cost_evaluation.calculate_cost_slalom(
                self.workload, index_combination+[candidate], store_size=True
            )
            c_b = self.cost_evaluation.get_index_build_cost(candidate)*1000
            if abs(c_f-c_i) < 0.01:
                if c_f < c_i:
                    c_i += 0.01
                else:
                    c_f += 0.01
            self.statistic[column]['c_f'] = c_f
            self.statistic[column]['c_i'] = c_i
            self.statistic[column]['c_b'] = c_b
            self.statistic[column]['threshold'] = c_b / (c_f - c_i)
            self.statistic[column]['access_freq'] = 0
            self.statistic[column]['build'] = False
            self.statistic[column]['T'] = 0
            self.statistic[column]['pj_sum'] = 0
        self.current_size = 0
        # self.LRU = LRUCache(self.budget)
        index_combination = self._dynamic_build_index()
        return index_combination
    
    def calculate_best_indexes_partition(self, total_workload, valid_workloads):
        # return indexes combinations according to partition
        logging.info("Calculating best indexes Slalom")
        self.total_workload = total_workload
        self.valid_workloads = valid_workloads
        columns = self.total_workload.indexable_columns()
        single_attribute_index_candidates = self.total_workload.potential_indexes()
        random.shuffle(single_attribute_index_candidates)
        # current indexs
        index_combination = []
        index_combination_size = 0
        self.LRU = LRUCache(self.budget)
        # calculate C_i,C_f,C_b
        for candidate in single_attribute_index_candidates:
            # Slalom only refer to single-col index, so we can extract column as follow
            column = candidate.columns[0]
            self.statistic[column] = {}
            self.statistic[column]['index'] = candidate
            # self.cost_evaluation.calculate_cost_slalom(self.workload, index_combination, store_size=True)
            c_f = self.cost_evaluation.calculate_cost_multithread_remove_index(
                self.total_workload, index_combination, store_size=True
                )
            c_i = self.cost_evaluation.calculate_cost_multithread_remove_index(
                self.total_workload, index_combination+[candidate], store_size=True
            )
            c_b = self.cost_evaluation.get_index_build_cost(candidate)*1000
            # print(c_f)
            # print(c_i)
            # print(c_b)
            if abs(c_f-c_i) < 0.01:
                if c_f < c_i:
                    c_i += 0.01
                else:
                    c_f += 0.01
            self.statistic[column]['c_f'] = c_f
            self.statistic[column]['c_i'] = c_i
            self.statistic[column]['c_b'] = c_b
            self.statistic[column]['threshold'] = c_b / (c_f - c_i)
            self.statistic[column]['access_freq'] = 0
            self.statistic[column]['build'] = False
            self.statistic[column]['T'] = 0
            self.statistic[column]['pj_sum'] = 0
        logging.info("Finish Slalom statistics")
        self.current_size = 0
        # self.LRU = LRUCache(self.budget)
        index_combination = self._dynamic_build_index()
        return index_combination

    def _dynamic_build_index(self):
        index_combination = []
        p_start = 0.5
        query_ls = []
        for query in self.total_workload.queries:
            query_ls += [query] * query.frequency
        random.shuffle(query_ls)
        
        for query in query_ls:
            for column in query.columns:
                self.statistic[column]['T'] += 1    # 统计每个分区被访问的总次数

        for query_id,query in enumerate(query_ls):
            for column in query.columns:
                self.statistic[column]['access_freq'] += 1
                self.statistic[column]['build_p'] = p = self._calculate_build_p(column, p_start)
                self.statistic[column]['pj_sum'] += p
                if p < 0:
                    p = random.random() * pow(0.5, self.statistic[column]['access_freq'])
                elif p > 1:
                    p = 1
                if not self.statistic[column]['build']:
                    if self._random_unit(p):
                        # print('在第', candidate_row_group.access_freq[column_id], '以概率 ', p, 'build tree')
                        # calculate the index size:
                        del_cols = self.LRU.put(column,self.statistic[column]['index'].estimated_size)
                        index_combination.append(self.statistic[column]['index'])
                        self.statistic[column]['build'] = True
                        if len(del_cols) >0 :
                            for col in del_cols:
                                self.statistic[col]['build'] = False
                                index_combination.remove(self.statistic[col]['index'])
                else:
                    # has established
                    self.LRU.get(column)
            if (query_id <= int(len(query_ls) / 20) and query_id % 5 == 0 ) or query_id % int(len(query_ls) / 20) == 0:
                # Evaluate performance at intervals
                cost = 0
                time_begin_cal = time.time()
                for partition_id in range(self.partition_num):
                    cost += self.cost_evaluation.calculate_cost_multithread(
                        self.valid_workloads[partition_id], index_combination, store_size = False
                    )
                index_size = sum([index.estimated_size for index in index_combination])
                logging.info(f"{time_begin_cal - self.time_begin} s, cost :{cost}, index_size: {index_size}")
                self.time_begin = time.time()

        return index_combination



    
    def _calculate_build_p(self, column, p_start):
        a = random.random()
        c_f = self.statistic[column]['c_f'] 
        c_i = self.statistic[column]['c_i']
        c_b = self.statistic[column]['c_b']
        total = self.statistic[column]['T']
        pj_sum = self.statistic[column]['pj_sum']
        i = self.statistic[column]["access_freq"]
        # if self.statistic[column]['access_freq'] ==1 :
        #     p = p_start
        # else:
        #     if self.statistic[column]['access_freq'] <= math.ceil(self.statistic[column]['threshold']):
        #         p = pow((c_b + c_f - c_i) / c_b, self.statistic[column]['access_freq'] - 2) * (
        #                     (c_f - c_i) / c_b) * (
        #                         p_start + a * c_f / (c_f - c_i))
        #     else:
        #          p = pow((c_b + c_f - c_i) / c_b, self.statistic[column]['access_freq'] - 2) * (
        #                     -(c_f - c_i) / c_b) * (
        #                         1 - p_start + a * c_i / (c_f - c_i))
        # p = min(1, p)
        p = ((c_f - c_i) / (c_b - c_f)) * (total - i) - (1 - pj_sum)
        # print(f'building index, T = {total}, i = {i}, pi = {p}, pj sum = {pj_sum}, c_f = {c_f}, c_i = {c_i}, c_b = {c_b}')
        return p
    
    def _random_unit(self,p):
        assert 0 <= p <= 1, "概率P的值应该处在[0,1]之间！"
        return random.random() < p

        
        

# 输入
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
        self.current_size = 0  # 新增变量跟踪当前缓存值的总和
        
    def get(self, key):
        if key in self.cache:
            # 更新最近访问顺序
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        else:
            return -1
        
    def put(self, key, value):
        if key in self.cache:
            # 更新值和最近访问顺序
            self.current_size += value - self.cache[key]
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
            return []
        else:
            # 插入新的值和最近访问顺序
            self.cache[key] = value
            self.order.append(key)
            self.current_size += value
            del_keys = []
            # 如果超过容量，则移除最旧的项直到总大小不超过容量
            while self.current_size > self.capacity:
                oldest = self.order.pop(0)
                self.current_size -= self.cache[oldest]
                del self.cache[oldest]
                del_keys.append(oldest)
            return del_keys

