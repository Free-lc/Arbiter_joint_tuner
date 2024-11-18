import copy
import json
import logging
import pickle
import sys
import os
from pathlib import Path
cur_path = Path(__file__).resolve().parent.parent
print(cur_path)
sys.path.append(cur_path.as_posix())

import time
import random
import copy
from selection.algorithms.extend_algorithm import ExtendAlgorithm
from selection.algorithms.relaxation_algorithm import RelaxationAlgorithm
from selection.algorithms.auto_admin_algorithm import AutoAdminAlgorithm
from selection.algorithms.slalom_algorithm import SlalomAlgorithm
from selection.algorithms.noindex_algorithm import NoindexAlgorithm
from selection.benchmark import Benchmark
from selection.dbms.hana_dbms import HanaDatabaseConnector
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.query_generator import QueryGenerator
from selection.selection_algorithm import AllIndexesAlgorithm, NoIndexAlgorithm
from selection.table_generator import TableGenerator
from selection.workload import Workload
from selection.utils import min_records_number,BloomFilter
from selection.cost_evaluation import CostEvaluation

DBMSYSTEMS = {"postgres": PostgresDatabaseConnector, "hana": HanaDatabaseConnector}
IndexAlgorithm = {"extend": ExtendAlgorithm, "autoadmin": AutoAdminAlgorithm, "noindex":NoindexAlgorithm, "relaxation":RelaxationAlgorithm}


class BlackBox:
    def __init__(self,test=None, config_file = None):
        logging.debug("Init IndexSelection")
        self.db_connector = None
        self.default_config_file = "example_configs/config_tpch.json"
        self.disable_output_files = False
        self.database_name = None
        self.database_system = None
        self.seed = 20
        if config_file is None:
            config_file = 'benchmark_results/fray_op/config_tpcds.json'
        self.filter_mp = {}
        if not config_file:
            config_file = self.default_config_file
        self.MinMaxMap = {}
        self.BloomFilter = {}
        logging.info("Init Black Box")
        logging.info("Using config file {}".format(config_file))
        with open(config_file) as f:
            config = json.load(f)
        self._setup_config(config,test)
        self.config = config
        self.cost_evaluation = CostEvaluation(self.db_connector)
        

    def _setup_config(self, config, test = None):
        dbms_class = DBMSYSTEMS[config["database_system"]]
        generating_connector = dbms_class(None, autocommit=True)
        self.table_generator_max = TableGenerator(
            config["benchmark_name"], config["scale_factor"],  generating_connector,sql_per_table = config["sql_per_table"], partition_num = config['partition_max'],test=test, seed = self.seed
        )
        self.database_name = self.table_generator_max.database_name()
        self.database_system = config["database_system"]
        self.setup_db_connector(self.database_name, self.database_system)
        self.partition_max = config['partition_max']
        self.skipping_index = config['skipping_index']
        self.budget = config['budget']
        logging.getLogger().setLevel(logging.DEBUG)
        # self.generate_queries()

        if "queries" not in config:
            config["queries"] = None
        self.workloads = []
        self.total_workload = []
        total_queries = []
        if config["scale_factor"] == 100:
            self.seed = 200
        random.seed(self.seed)
        # get all filenames in queries folder and get the max value which is the number of queries
        if "tpch" in self.database_name:
            folder_path = f'./tpch-userdef-kit/queries/{config["scale_factor"]}'
        elif "tpcds" in self.database_name:
            folder_path = f'./tpcds-userdef-kit/queries/{config["scale_factor"]}'
        elif "test" in self.database_name:
            folder_path = f'./tpch-userdef-kit/queries/{config["scale_factor"]}'
        elif "wikimedia" in self.database_name:
            folder_path = f'./wikimedia-kit/queries/{config["scale_factor"]}'
        else:
            assert 0, "undefined benchmarks!!!"
        file_names = os.listdir(folder_path)
        value_ls = [int(filename.replace(".sql","").split("_")[0]) for filename in file_names if filename.endswith(".sql")]
        # self.number_of_queries = max(value_ls)
        if config["scale_factor"] == 10:
            min_sql_number = 15
        elif config["scale_factor"] == 100:
            min_sql_number = 1
        #-----tpch-heavyfilter
        # self.number_of_queries = max(min(max(value_ls)//3, min_sql_number),1)
        # self.queries_id = random.choices(value_ls, k = self.number_of_queries)
        # # if config["scale_factor"] == 100:
        # #     self.queries_id = [89, 46]
        # logging.info(self.queries_id)
        # # self.queries_id = [0, 1, 2, 3, 4]
        # query_frequence = [random.randint(0,config["query_frequence"]) for query_id in range(len(self.queries_id))]
        # logging.info(query_frequence)

        #-----tpch-standard
        # workload_queries = [1,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,21,22]
        workload_queries = [7]
        self.queries_id = [i-1 for i in workload_queries]
        query_frequence = [1 for query_id in range(len(self.queries_id))]

        for partition_id in range(config['partition_max']):
            query_generator = QueryGenerator(
                config["benchmark_name"],
                config["scale_factor"],
                self.db_connector,
                query_frequence,
                self.table_generator_max.columns,
                partition_id,
                queries_id = self.queries_id
            )
            total_queries += query_generator.queries
            self.workload = Workload(query_generator.queries)
            self.workloads.append(self.workload)
        self.total_workload = Workload(total_queries)
        single_query_generator = QueryGenerator(
                config["benchmark_name"],
                config["scale_factor"],
                self.db_connector,
                query_frequence,
                self.table_generator_max.columns,
                queries_id = self.queries_id
            )
        self.single_workload = Workload(single_query_generator.queries)
        # splite data block
        # self.table_generator_max.partition_table(single_query_generator.queries)
        # initialize prefilter parameter
        self.prefiltered_workloads = self.workloads
        for partition_id in range(config['partition_max']):
            for index_id,index in enumerate(self.workloads[partition_id].potential_indexes()):
                self.filter_mp[index] = index_id

    def get_attribute_number(self):
        return len(self.workloads[0].potential_indexes())

    def get_max_partition_number(self):
        return self.partition_max
    
    def get_tuple_number_per_physicalblock(self):
        # TODO: Dont use this
        s = "select * from"
        self.db_connector.exec_fetch(s)
    
    def get_attributes_number(self):
        return [len(self.prefiltered_workloads[i].potential_indexes()) for i in range(self.partition_max)]

    def prefilter_workload(self):
        # prefilter block's workload
        self.filter_mp = {}
        partition_num = self.partition_max   # it is a tupl list
        partition_code = 2 ** partition_num - 1
        self.prefiltered_workloads = []
        candidate_index_number = []
        candidate_index_number_fillered = []
        for partition_id in range(partition_num):
            queries = copy.deepcopy(self.workloads[partition_id].queries)
            valid_queries = []
            for query in queries:
                skip = self._filtering_(query, partition_code)
                if skip:
                    continue
                valid_queries.append(query)
            vaild_workload = Workload(valid_queries)
            total_index_candidates = self.workloads[partition_id].potential_indexes()
            candidate_index_number += total_index_candidates
            prefiltered_index_candidates = vaild_workload.potential_indexes()
            candidate_index_number_fillered += prefiltered_index_candidates
            for index_id,index in enumerate(total_index_candidates):
                if index in prefiltered_index_candidates:
                    self.filter_mp[index] = index_id
            self.prefiltered_workloads.append(vaild_workload)
        candidate_index_number = list(set(candidate_index_number))
        candidate_index_number_fillered = list(set(candidate_index_number_fillered))
        logging.info(f"By Space Compactor the number of candidate indexes from {len(candidate_index_number)} to {len(candidate_index_number_fillered)}")
    
    def recover_workload(self):
        self.prefiltered_workloads = self.workloads
        for partition_id in range(self.partition_max):
            for index_id,index in enumerate(self.workloads[partition_id].potential_indexes()):
                self.filter_mp[index] = index_id

    def run_tupels(self, config_list, partition_code):
        partition_tuples = config_list[0]   # it is a tupl list
        self.index_conbination = config_list[1:]
        partition_num = len(partition_tuples)
        # assert self.code_to_partition_tuples(partition_code, self.partition_max) == partition_tuples, "wrong code!"
        self.db_connector.drop_indexes()
        self.db_connector.create_statistics()
        self.db_connector.commit()
        queried_table = []
        for query in self.prefiltered_workloads[0].queries:
            queried_table += [column.table.name.split("_1_prt_p")[0] for column in query.columns]
        queried_table = list(set(queried_table))
        self.table_generator_max.tuples_partitioning(partition_tuples, queried_table)
        logging.info(f"partitioning the datasets by {partition_tuples}")
        index_combinations = []
        index_combination_size = 0
        # --------1st physical block in a logical partition matters
        # for partition_id in range(partition_num):
        #     index_combination = []
        #     block_id = partition_tuples[partition_id][0]
        #     single_attribute_index_candidates = self.workloads[block_id].potential_indexes()
        #     assert 2**len(single_attribute_index_candidates) >= self.index_conbination[block_id], "the number of indexes is wrong!"
        #     for index_id,index in enumerate(single_attribute_index_candidates):
        #         x_i = copy.deepcopy(self.index_conbination[block_id])
        #         build =  (x_i >> index_id)%2 # build or not build  
        #         if build == 1:
        #             index_combination.append(single_attribute_index_candidates[index_id])
        #     index_combinations.append(index_combination)
         # --------all physical blocks in a logical partition matter
        for partition_id in range(partition_num):
            index_combination = []
            block_id_begin = partition_tuples[partition_id][0]
            block_id_end = partition_tuples[partition_id][1]
            for block_id in range(block_id_begin,block_id_end):
                single_attribute_index_candidates = self.prefiltered_workloads[block_id].potential_indexes()
                assert 2**len(single_attribute_index_candidates)-1 >= self.index_conbination[block_id], f"the number of indexes is wrong! in {block_id}th block, {2**len(single_attribute_index_candidates)-1} VS {self.index_conbination[block_id]}"
                for index_id,index in enumerate(single_attribute_index_candidates):
                    x_i = copy.deepcopy(self.index_conbination[block_id])
                    build =  (x_i >> index_id)%2 # build or not build  
                    if build == 1:
                        #mp: prefiter_index ---------> index_id in total
                        logical_index_id = self.filter_mp[index]
                        logical_index = self.workloads[partition_id].potential_indexes()[logical_index_id]
                        assert logical_index is not None, logging.ERROR(f"{logical_index} dose not exist")
                        # index is total_index_combination
                        index_combination.append(logical_index)
            index_combination = list(set(index_combination))
            index_combinations.append(index_combination)
        skip_start = time.time()
        valid_workloads = self.skip_filter_tuples(partition_num,partition_code)
        skip_end = time.time()
        logging.info(f"skip time {skip_end - skip_start}")
        time_start = time.time()
        cost = 0
        for i in range(partition_num):
            cost += self.cost_evaluation.calculate_cost_multithread(
                valid_workloads[i], index_combinations[i], store_size=False
            )
        time_end = time.time()
        evaluation_time  = time_end - time_start
        logging.info(f"evaluation time {evaluation_time}")
        index_combination_size = sum([index.estimated_size for index_combination in index_combinations for index in index_combination])
        self.cost_evaluation.complete_cost_estimation()
        return index_combination_size,cost

    def run_slalom(self, partition_tuples):
        partition_num = len(partition_tuples)
        self.db_connector.drop_indexes()
        self.db_connector.create_statistics()
        self.db_connector.commit()
        queried_table = []
        for query in self.prefiltered_workloads[0].queries:
            queried_table.append(query.columns[0].table.name.split("_1_prt_p")[0])
        self.table_generator_max.tuples_partitioning(partition_tuples, queried_table)
        logging.info(f"partitioning the datasets by {partition_tuples}")
        index_combinations = []
        index_combination_size = 0
        parameters = {'budget_MB':self.budget, 'partition_num':partition_num}
        slalom_algorithm = SlalomAlgorithm(self.db_connector, parameters)
        logging.info(f"Running algorithm {parameters}")
        partitioned_queries = []
        valid_workloads = self.skip_filter_tuples(partition_num,'slalom')
        for sub_workload in valid_workloads[0: partition_num]:
            partitioned_queries += sub_workload.queries
        partitioned_workload = Workload(partitioned_queries)
        index_combinations = slalom_algorithm.calculate_best_indexes_partition(partitioned_workload, valid_workloads)
        # index_combinations are the list of indexes in all partitions
        self.db_connector.drop_indexes()
        cost = 0
        for i in range(partition_num):
            cost += self.cost_evaluation.calculate_cost_multithread(
                valid_workloads[i], index_combinations, store_size=False
            )
        index_combination_size = sum([index.estimated_size for index in index_combinations])
        self.cost_evaluation.complete_cost_estimation()
        return index_combination_size,cost

    def run_extend(self, partition_tuples, index_algorithm = 'extend'):
        partition_num = len(partition_tuples)
        self.db_connector.drop_indexes()
        self.db_connector.create_statistics()
        self.db_connector.commit()
        self.table_generator_max.tuples_partitioning(partition_tuples)
        logging.info(f"partitioning the datasets by {partition_tuples}")
        index_combinations = []
        index_combination_size = 0
        parameters = {'budget_MB':self.budget, "max_index_width":2, 'partition_num':partition_num}
        extend_algorithm = IndexAlgorithm[index_algorithm](self.db_connector, parameters)
        logging.info(f"Running algorithm {parameters}")

        # partitioned_queries = []
        # for sub_workload in self.workloads[0: partition_num]:
        #     partitioned_queries += sub_workload.queries
        # partitioned_workload = Workload(partitioned_queries)

        partitioned_workload = self.single_workload
        valid_workloads = self.skip_filter_tuples(partition_num, index_algorithm)
        index_combinations = extend_algorithm.calculate_best_indexes(partitioned_workload, valid_workloads)
        # index_combinations are the list of indexes in all partitions
        
        cost = 0
        for i in range(partition_num):
            cost += self.cost_evaluation.calculate_cost_multithread(
                valid_workloads[i], index_combinations, store_size=False
            )
        index_combination_size = sum([index.estimated_size for index in index_combinations])
        self.cost_evaluation.complete_cost_estimation()
        return index_combination_size,cost

    def run(self, config_list):
        logging.getLogger().setLevel(logging.DEBUG)
        partition_num = config_list[0]
        self.index_conbination = config_list[1:]
        self.db_connector.drop_indexes()
        self.db_connector.create_statistics()
        self.db_connector.commit()
        # partitioning the table
        partition_begin = time.time()
        self.table_generator_max.partitioning(partition_num)
        logging.info(f"partitioning the datasets by {partition_num}")
        partition_end = time.time()
        # build index in database
        storage_budget = 0
        logging.info(f"partitioning time {partition_end - partition_begin}")
        index_combinations = []
        index_combination_size = 0
        for partition_id in range(partition_num):
            index_combination = []
            single_attribute_index_candidates = self.workloads[partition_id].potential_indexes()
            assert 2**len(single_attribute_index_candidates) >= self.index_conbination[partition_id], "the number of indexes is wrong!"
            for index_id,index in enumerate(single_attribute_index_candidates):
                x_i = copy.deepcopy(self.index_conbination[partition_id])
                build =  (x_i >> index_id)%2 # build or not build  
                if build == 1:
                    index_combination.append(single_attribute_index_candidates[index_id])
            index_combinations.append(index_combination)
        skip_start = time.time()
        valid_workloads = self.skip_filter(partition_num)
        skip_end = time.time()
        logging.info(f"skip time {skip_end - skip_start}")
        time_start = time.time()
        cost = 0
        for i in range(partition_num):
            cost += self.cost_evaluation.calculate_cost_multithread(
                valid_workloads[i], index_combinations[i], store_size=False
            )
        time_end = time.time()
        evaluation_time  = time_end - time_start
        logging.info(f"evaluation time {evaluation_time}")
        index_combination_size = sum([index.estimated_size for index_combination in index_combinations for index in index_combination])
        self.cost_evaluation.complete_cost_estimation()
        return index_combination_size,cost

    def generate_queries(self):
        logging.info("generate queries")
        self.number_of_queries = 0
        for filename in self.table_generator_max.get_table_files():
            table = filename.replace(".tbl", "").replace(".dat", "")
            if table != 'lineitem':
                continue
            s = f"select count(*) from {table}"
            table_records_num = self.db_connector.exec_fetch(s)[0]
            if table_records_num < min_records_number:
                continue
            self.table_generator_max._write_queries(filename, table)
            logging.info(f"the queries of table {table} hava been generated")
            if "wikimedia" not in self.database_name:
                os.remove(os.path.join(self.table_generator_max.directory, filename))
    
    def setup_db_connector(self, database_name, database_system):
        if self.db_connector:
            logging.info("Create new database connector (closing old)")
            self.db_connector.close()
        self.db_connector = DBMSYSTEMS[database_system](database_name,autocommit = True)
    
    def skip_filter(self, partition_num):
        valid_workloads = []
        for partition_id in range(partition_num):
            workload = self.workloads[partition_id]
            valid_queries = []
            for query in workload.queries:
                skip = self._filtering_(query, partition_num)
                if skip:
                    continue
                valid_queries.append(query)
            valid_workloads.append(Workload(valid_queries))
        total_query_num = sum(query.frequency for partition_id in range(partition_num) for query in self.workloads[partition_id].queries)
        valid_query_num = sum(query.frequency for partition_id in range(partition_num) for query in valid_workloads[partition_id].queries)
        logging.info(f"By data skipping index, the number of queries from {total_query_num} to {valid_query_num}")
        return valid_workloads
    
    def skip_filter_tuple(self, partition_id, partition_tuple):
        workload = self.workloads[partition_id]
        valid_queries = []
        for query in workload.queries:
            skip = self._filtering_(query, partition_tuple)
            if skip:
                continue
            valid_queries.append(query)
        total_query_num =sum([query.frequency for query in workload.queries]) 
        valid_query_num = sum([query.frequency for query in valid_queries])
        logging.info(f"By data skipping index, the number of queries from {total_query_num} to {valid_query_num}")
        return Workload(valid_queries)

    def skip_filter_tuples(self, partition_num, partition_code):
        valid_workloads = []
        for partition_id in range(partition_num):
            workload = self.workloads[partition_id]
            valid_queries = []
            for query in workload.queries:
                skip = self._filtering_(query, partition_code)
                if skip:
                    continue
                valid_queries.append(query)
            valid_workloads.append(Workload(valid_queries))
        total_query_num = sum(query.frequency for partition_id in range(partition_num) for query in self.workloads[partition_id].queries)
        valid_query_num = sum(query.frequency for partition_id in range(partition_num) for query in valid_workloads[partition_id].queries)
        logging.info(f"By data skipping index, the number of queries from {total_query_num} to {valid_query_num}")
        return valid_workloads        
    
    def _filtering_(self, query_temp, partition_code = None):
        # return False
        if partition_code is None:
            partition_code = 2 ** (self.partition_max - 1) -1
        query = copy.deepcopy(query_temp)
        skip = False
        if self.skipping_index == 'MinMax':
            query.text = query.text.split("FROM")[-1]
            table = query.text.split("WHERE")[0].replace(" ", "")
            # print(table)
            filter_content = query.text.split("WHERE")[-1]
            filter_list = list(set(filter_content.split("AND")))
            for filter in filter_list:
                if '=' in filter:
                    column = filter.split("=")[0].replace(" ", "")
                    start = self._convert_to_float(filter.split("=")[-1].replace(" ", "").replace(";",""))
                    end = start
                else:
                    column = filter.split("Between")[0].replace(" ", "")
                    start = self._convert_to_float(filter.split("Between")[-1].split("and")[0].replace(" ", "").replace(";",""))
                    end = self._convert_to_float(filter.split("Between")[-1].split("and")[-1].replace(" ", "").replace(";",""))
                # add min max index
                if partition_code not in self.MinMaxMap:
                    self.MinMaxMap[partition_code] = {}
                if table not in self.MinMaxMap[partition_code]:
                    self.MinMaxMap[partition_code][table] = {}
                if column not in self.MinMaxMap[partition_code][table]:
                    s_max_value = f"select max({column}) from {table}"
                    s_min_value = f"select min({column}) from {table}"
                    max_value = self.db_connector.exec_fetch(s_max_value)[0]
                    min_value = self.db_connector.exec_fetch(s_min_value)[0]
                    self.MinMaxMap[partition_code][table][column] = (min_value, max_value)
                if self.MinMaxMap[partition_code][table][column][0] is None or self.MinMaxMap[partition_code][table][column][1] is None or start > self.MinMaxMap[partition_code][table][column][1] or end < self.MinMaxMap[partition_code][table][column][0]:
                    skip = True
                    return skip
        elif self.skipping_index == 'BloomFilter':
            query.text = query.text.split("FROM")[-1]
            table = query.text.split("WHERE")[0].replace(" ", "")
            # print(table)
            filter_content = query.text.split("WHERE")[-1]
            filter_list = list(set(filter_content.split("AND")))
            if 'between' in query.text.lower():
                raise Exception("Bloom filter does not support range queries.")
            for filter in filter_list:
                if '=' in filter:
                    column = filter.split("=")[0].replace(" ", "")
                    start = self._convert_to_float(filter.split("=")[-1].replace(" ", "").replace(";",""))
                    end = start
                else:
                    column = filter.split("Between")[0].replace(" ", "")
                    start = self._convert_to_float(filter.split("Between")[-1].split("and")[0].replace(" ", "").replace(";",""))
                    end = self._convert_to_float(filter.split("Between")[-1].split("and")[-1].replace(" ", "").replace(";",""))
                # add min max index
                if partition_code not in self.BloomFilter:
                    self.BloomFilter[partition_code] = {}
                if table not in self.BloomFilter[partition_code]:
                    self.BloomFilter[partition_code][table] = {}
                if column not in self.BloomFilter[partition_code][table]:
                    values_s = f"select {column} from {table}"
                    values = [self._convert_to_float(value[0]) for value in self.db_connector.exec_fetch(values_s, False)]
                    self.BloomFilter[partition_code][table][column] = (BloomFilter(len(values)))
                    for value in values:
                        self.BloomFilter[partition_code][table][column].add(value)
                if self.BloomFilter[partition_code][table][column].check(start):
                    skip = True
                    return skip
            return skip
         
    def _convert_to_float(self, s):
        import datetime
        string = copy.deepcopy(s)
        try:
            return float(string)
        except:
            try:
                string = string.replace("\'", "")
                return(datetime.datetime.strptime(string, '%Y-%m-%d').date())
            except:
                try:
                    return (s - datetime.date(1970, 1, 1)).days
                except:
                    return string
            
    def drop_database(self):
        self.db_connector.close()
        self.db_connector.drop_database(self.database_name)

    def get_selectivity(self, partition_tuple):
        print("getting selectivity")
        data, selectivity = self.table_generator_max.get_selectivity(partition_tuple, self.workloads[0].queries , self.partition_max)
        return data, selectivity

    def code_to_partition_tuples(self, code: int, dim_configs: int = 50):
        split_points = []
        partition_tuple_list = [(0,dim_configs)]
        pos = 0
        while code > 0:
            digit = code & 1
            if digit == 1:
                split_points.append(dim_configs - 1 - pos)
            code = code >> 1
            pos += 1
        for pt in split_points:
            new_list = []
            for partition_tuple in partition_tuple_list:
                start, end = partition_tuple
                if start < pt <= end:
                    new_list.append((start, pt))
                    new_list.append((pt, end))
                else:
                    new_list.append(partition_tuple)
            partition_tuple_list = new_list
                    
        return partition_tuple_list
    
    def test_index_disparity(self, partition_code):
        self.db_connector.drop_indexes()
        self.db_connector.create_statistics()
        self.db_connector.commit()
        valid_workloads = self.skip_filter_tuples(self.partition_max,partition_code)
        res = {}
        for block_id in range(self.partition_max):
            res[block_id] = {}
            workload = self.prefiltered_workloads[block_id]
            print(f"{len(workload.potential_indexes())} 个索引")
            valid_workload = valid_workloads[block_id]
            res[block_id]["no index"] = self.cost_evaluation.calculate_cost_multithread(
                valid_workload, [], store_size=False
            )
            for index in workload.potential_indexes():
                cost = self.cost_evaluation.calculate_cost_multithread(
                valid_workload, [index], store_size=False
            )
                res[block_id][index] = cost
        return res


if __name__ == "__main__":
    blackbox = BlackBox(config_file = 'benchmark_results/fray_op/config_tpch.json', test = 1)
    attribute_number = blackbox.get_attribute_number()
    max_partition_number = blackbox.get_max_partition_number()
    # ----------tuples_partition
    for i in range(1):
        random.seed(blackbox.seed)
        config_list = []
        config_list.append([(0, 2), (2, 4), (4, 5), (5, 8), (8, 10), (10, 11), (11, 12), (12, 15), (15, 18), (18, 19), (19, 20), (20, 22), (22, 23), (23, 24), (24, 26), (26, 27), (27, 28), (28, 32), (32, 34), (34, 35), (35, 39), (39, 40), (40, 41), (41, 43), (43, 44), (44, 46), (46, 47), (47, 48), (48, 50), (50, 52), (52, 58), (58, 59), (59, 60), (60, 61), (61, 62), (62, 63), (63, 65), (65, 66), (66, 68), (68, 69), (69, 70), (70, 72), (72, 73), (73, 74), (74, 77), (77, 80), (80, 81), (81, 82), (82, 83), (83, 86), (86, 87), (87, 91), (91, 94), (94, 95), (95, 97), (97, 99), (99, 100)],)
        config_list += [4779, 7981, 4387, 883, 4106, 5513, 967, 4058, 5436, 3683, 1160, 613, 697, 5255, 3531, 1129, 3570, 7717, 1534, 3566, 4479, 4768, 634, 5323, 3729, 1399, 4476, 5567, 446, 2795, 1690, 2109, 4648, 4052, 5118, 1874, 331, 2070, 3218, 7350, 5154, 7842, 3355, 4119, 6187, 4809, 5872, 3782, 2100, 3760, 7821, 4543, 7109, 7093, 7121, 7507, 1708, 3541, 6145, 3670, 5901, 4851, 1601, 5024, 2554, 4326, 5640, 7818, 7522, 2387, 5808, 5388, 3911, 1469, 3007, 1193, 2407, 1852, 4869, 4123, 5598, 7330, 4471, 3027, 1286, 4440, 6686, 5392, 938, 3365, 230, 4635, 3600, 5745, 2412, 7747, 3312, 4152, 3689, 5909]
        index_combination_size,cost = blackbox.run_tupels(config_list, 177027)
        logging.info(f"{i} divide, the index_combination_size is {index_combination_size}, the cost is {cost}")
    # blackbox.drop_database()
   
