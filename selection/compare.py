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
from selection.benchmark import Benchmark
from selection.compare_algorithms.whole_compare import WholeCompare
from selection.dbms.hana_dbms import HanaDatabaseConnector
from selection.dbms.postgres_dbms import PostgresDatabaseConnector
from selection.query_generator import QueryGenerator
from selection.table_generator import TableGenerator
from selection.workload import Workload
from selection.utils import min_records_number
from selection.cost_evaluation import CostEvaluation
from selection.db_openbox import BlackBox
from BO.openbox_clapboard import ClapboardOpenbox
from BO.slalom_tree import PartitionTree
from selection.compare_algorithms.slalom import Slalom
from selection.compare_algorithms.extend import Extend
from selection.compare_algorithms.casper_extend import CasperExtend
from selection.compare_algorithms.casper_slalom import CasperSlalom
from selection.compare_algorithms.slalom_autoadmin import SlalomAutoadmin
from selection.compare_algorithms.slalom_extend import SlalomExtend
from selection.compare_algorithms.fray import Fray,FrayPreFilter,FrayParameterFilter,AllIndexes,NoIndexes,AllIndexesPartition,NoIndexesPartition
DBMSYSTEMS = {"postgres": PostgresDatabaseConnector, "hana": HanaDatabaseConnector}
ALGORITHMS = {
    "FrayParameterFilter":FrayParameterFilter,
    "FrayPreFilter":FrayPreFilter,
    "Fray":Fray,
    "Slalom":Slalom,
    "Extend":Extend,
    "AllIndexes":AllIndexes,
    "NoIndexes":NoIndexes,
    "AllIndexesPartition":AllIndexesPartition,
    "NoIndexesPartition":NoIndexesPartition,
    "CasperExtend":CasperExtend,
    "CasperSlalom":CasperSlalom,
    "SlalomAutoadmin":SlalomAutoadmin,
    "SlalomExtend":SlalomExtend
}
class CompareAlgorithm:
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        logging.debug("Start Comparision")
        self.black_box = BlackBox()
    def compare(self):
        for key,value in ALGORITHMS.items():
            compareAlgorithm = value(black_box)
            index_combination_size,cost = compareAlgorithm.get_optimal_value()
            logging.info(index_combination_size,cost)
    def finish_compare(self):
        self.black_box.drop_database()
        




black_box = BlackBox(config_file = 'benchmark_results/fray_op/config_tpch.json')
# -------baseline----------------
# whole_compare = WholeCompare(black_box)
# whole_compare.get_convergence_curve()

selection_algorithm = ALGORITHMS["Fray"](black_box, train=False)
index_combination_size,cost = selection_algorithm.get_optimal_value()
logging.info(f"index_combination_size : {index_combination_size} , cost : {cost}")


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      