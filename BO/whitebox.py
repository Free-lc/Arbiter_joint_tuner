import numpy as np
from typing import List
import sys
import math
sys.path.append('/data2/fray/index_selection_evaluation')
from openbox.utils.config_space import Configuration
import torch.nn as nn
import torch
from scipy.stats import kurtosis
import pickle
import logging
from openbox import space as sp
from BO.white_box_model_modified import WhiteboxModel
from BO.train_loader import TestDataset
import numpy


class WhiteBoxModel:
    def __init__(self, black_box, cache_file=None, train=False):
        logging.debug("Enabling Whitebox")
        self.black_box = black_box
        self.queries = self.black_box.single_workload.queries
        self.cache_file = cache_file
        self.train = train
        self.cache = self.load_cache()  # load from file, cache partition variance & kurtosis & partition selectivities
        logging.info(f"the cache has cached {len(self.cache)} configurations features") 
        if "tpch" in self.black_box.database_name:
            self.pkl_file = f'BO/train/tpch{self.black_box.config["scale_factor"]}'
        elif "tpcds" in self.black_box.database_name:
            self.pkl_file = f'BO/train/tpcds{self.black_box.config["scale_factor"]}'
        elif "wikimedia" in self.black_box.database_name:
            self.pkl_file = f'BO/train/wikimedia{self.black_box.config["scale_factor"]}'
        elif "test" in self.black_box.database_name:
            self.pkl_file = f'BO/train/tpch{self.black_box.config["scale_factor"]}'
        else:
            assert 0, "undefined benchmarks!!!"

    def get_pkl_files(self):
        return self.pkl_file

    def load_cache(self):
        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {}  # 如果文件不存在或为空，返回一个空字典
    
    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            logging.info(f"the cache has cached {len(self.cache)} configurations features") 
            pickle.dump(self.cache, f)

    def save_features(self, configs: List[Configuration]) -> np.ndarray:
        for config in configs:
            query_features, data_features, index_features = self.get_features(config)


    def get_scores(self, configs: List[Configuration]) -> np.ndarray:
        scores = self.evaluate(configs)
        return np.array(scores).reshape(len(configs), 1)
    
    def evaluate(self, configs: List[Configuration]) -> List[float]:
        self.device = torch.device("cuda")
        scores = []
        features = []
        for config in configs:
            query_features, data_features, index_features = self.get_features(config)
            query_features_device  = []
            data_features_device = []
            for query_feature in query_features:
                query_features_device.append(query_feature.to(self.device))
            for data_feature in data_features:
                data_features_device.append(data_feature.to(self.device))
            index_features_device = index_features.to(self.device)
            features.append([query_features_device, data_features_device, index_features_device])
        self.indexable_attributes = len(query_features[0][0])#12
        self.num_partitions = len(index_features)#50
        self.num_query = len(query_feature)   #15
        self.num_data_attribute = len(data_feature[0][0])
        scores = self.get_trust_score(features)
        scores = [float(score[0] > 0.5) for score in scores]
        return scores
    
    def get_features(self, config: Configuration):
        # 1. workload query features
        query_features = []
        query_feature_dim = 3 # attributes, query selectivity, frequency
        indexable_attributes = []
        total_frequency = 0
        for query in self.queries:
            indexable_attributes += [atrribute.name.split(".")[-1] for atrribute in query.columns]
            total_frequency += query.frequency
        indexable_attributes = list(set(indexable_attributes))
        for query in self.queries:
            attribute_feature = [1 if item in [atrribute.name.split(".")[-1] for atrribute in query.columns] else 0 for item in indexable_attributes]
            selectivity_feature = [1 if query.query_class == 'range' else 0.1]
            frequency_feature = [query.frequency / total_frequency]
            query_features.append([attribute_feature, selectivity_feature, frequency_feature])
        attribute_features = torch.tensor([sublist[0] for sublist in query_features], dtype=torch.long)
        selectivity_features = torch.tensor([sublist[1] for sublist in query_features], dtype=torch.float32)
        frequency_features = torch.tensor([sublist[2] for sublist in query_features], dtype=torch.float32)
        query_features = [attribute_features, selectivity_features, frequency_features]
        # 2. data layout features
        data_features = []
        data_feature_dim = 3 # the size of dataset, partitioning vector, partition selectivities & variance & kurtosis
        if self.black_box.get_max_partition_number() <= 50:
            config_p = config['p']
        else:
            config_p = 0
            for i in range(math.ceil((self.black_box.get_max_partition_number() - 1) / 49)):
                config_p += ((2**49)**i) * config[f'p_{i}']
        data_size = torch.tensor([self.black_box.config["scale_factor"] / 100]) # max scale is 100
        p_code = torch.tensor([[int(digit) for digit in str(config_p)]])         # p_code = torch.tensor([config_p / 2 ** (self.black_box.get_max_partition_number() - 1)])
        partition_code = self.black_box.code_to_partition_tuples(config_p, self.black_box.get_max_partition_number())
        for p_id in range(self.black_box.get_max_partition_number()):
            if p_id >= len(partition_code):
                data_features.append(np.zeros_like(data_features[-1]))
                continue
            if (tuple([query.frequency for query in self.queries]), partition_code[p_id]) in self.cache:
                data_features.append(self.cache[(tuple([query.frequency for query in self.queries]), partition_code[p_id])])
            else:
                data, selectivity = self.black_box.get_selectivity(partition_code[p_id])
                variance = np.array([])
                kurt = np.array([])
                for table_data in data:
                    try:
                        var = np.nanvar(table_data, axis=0)
                        kur =  kurtosis(table_data, axis=0)
                    except:
                        table_data = table_data.astype(numpy.float64)
                        var = np.nanvar(table_data, axis=0)
                        kur =  kurtosis(table_data, axis=0)
                    variance = np.append(variance, var)
                    kurt = np.append(kurt, kur)
                assert len(variance) == len(kurt), "bug"
                selectivity = np.array([selectivity]*len(variance))
                data_features.append(np.array([selectivity, variance, kurt]))
                self.cache[(tuple([query.frequency for query in self.queries]), partition_code[p_id])] = np.array([selectivity, variance, kurt])
        data_features = np.array(data_features)
        data_features = [data_size, p_code ,torch.tensor(data_features, dtype=torch.float32)]
        # 3. index feature
        index_feature_dim = 1 # index vector including which attributes are selected
        index_features = self.get_block_index_onehot(config, indexable_attributes)
        index_features = torch.tensor(index_features, dtype=torch.long)

        if self.train:
            # save feature
            with open(f'{self.pkl_file}.pkl', 'ab') as f:
                pickle.dump((query_features, data_features, index_features), f)

        self.save_cache()
        return query_features, data_features, index_features
    
    def get_data_features(self, config_p):
        data_features = []
        p_code = torch.tensor([config_p / 2 ** (self.black_box.get_max_partition_number() - 1)])
        partition_code = self.black_box.code_to_partition_tuples(config_p, self.black_box.get_max_partition_number())
        for p_id in range(self.black_box.get_max_partition_number()):
            if p_id >= len(partition_code):
                data_features.append(np.zeros_like(data_features[-1]))
                continue
            if (tuple([query.frequency for query in self.queries]), partition_code[p_id]) in self.cache:
                data_features.append(self.cache[(tuple([query.frequency for query in self.queries]), partition_code[p_id])])
            else:
                data, selectivity = self.black_box.get_selectivity(partition_code[p_id])
                variance = np.array([])
                kurt = np.array([])
                for table_data in data:
                    variance = np.append(variance, np.nanvar(table_data, axis=0))
                    kurt = np.append(kurt, kurtosis(table_data, axis=0))
                assert len(variance) == len(kurt), "bug"
                selectivity = np.array([selectivity]*len(variance))
                data_features.append(np.array([selectivity, variance, kurt]))
                self.cache[(tuple([query.frequency for query in self.queries]), partition_code[p_id])] = np.array([selectivity, variance, kurt])
        self.save_cache()

    def get_trust_score(self, features) -> float:
        test_dataset = TestDataset(features)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        PATH='./whiteboxmodels_big.pth'
        net = WhiteboxModel(indexable_attributes = self.indexable_attributes, num_partitions = self.num_partitions, num_query = self.num_query, num_data_attribute = self.num_data_attribute)
        net.load_state_dict(torch.load(PATH))
        net = net.to(self.device)
        outputs_batch = []
        for batch_features in test_loader:
            outputs_batch += (net(batch_features)).tolist()
        return outputs_batch


    def get_block_index_onehot(self, config: sp.Configuration, indexable_attributes):
        self.gray = True
        X = []
        OneHot = []
        block_num = self.black_box.get_max_partition_number()
        if self.gray:
            X += [self.gray_to_binary(config[f'x{i}']) for i in range(block_num)]
        else:
            X += [config[f'x{i}'] for i in range(block_num)]
        selected_attributes = []
        for block_id in range(block_num):
            single_attribute_index_candidates = self.black_box.prefiltered_workloads[block_id].potential_indexes()
            selected_attribute = []
            for index_id,index in enumerate(single_attribute_index_candidates):
                x_i = X[block_id]
                build = (x_i >> index_id)%2
                if build == 1:
                        #mp: prefiter_index ---------> index_id in total
                        logical_index_id = self.black_box.filter_mp[index]
                        logical_index = self.black_box.workloads[block_id].potential_indexes()[logical_index_id]
                        # index is total_index_combination
                        # !!! only allow single_column index
                        assert len(logical_index.columns) == 1, 'only allow single_column index'
                        selected_attribute.append(logical_index.columns[0].name.split('.')[-1])
            selected_attributes.append(selected_attribute)
        for block_id in range(block_num):
            OneHot.append([int(item in selected_attributes[block_id]) for item in indexable_attributes])
        return OneHot
                
        

    def gray_to_binary(self, gray):
        mask = gray >> 1
        while mask != 0:
            gray = gray ^ mask
            mask = mask >> 1
        return gray
    
    def end_white_box(self):
        self.save_cache()
    
