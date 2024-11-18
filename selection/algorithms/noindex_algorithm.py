import logging
import time

from selection.index import Index
from selection.selection_algorithm import DEFAULT_PARAMETER_VALUES, SelectionAlgorithm
from selection.utils import b_to_mb, mb_to_b

# budget_MB: The algorithm can utilize the specified storage budget in MB.
# max_index_width: The number of columns an index can contain at maximum.
# min_cost_improvement: The value of the relative improvement that must be realized by a
#                       new configuration to be selected.
# The algorithm stops if either the budget is exceeded or no further beneficial
# configurations can be found.
DEFAULT_PARAMETERS = {
    "budget_MB": DEFAULT_PARAMETER_VALUES["budget_MB"],
    "max_index_width": DEFAULT_PARAMETER_VALUES["max_index_width"],
    "min_cost_improvement": 1.001,
}


# This algorithm is a reimplementation of the Extend heuristic published by Schlosser,
# Kossmann, and Boissier in 2019.
# Details can be found in the original paper:
# Rainer Schlosser, Jan Kossmann, Martin Boissier: Efficient Scalable
# Multi-attribute Index Selection Using Recursive Strategies. ICDE 2019: 1238-1249
class NoindexAlgorithm(SelectionAlgorithm):
    def __init__(self, database_connector, parameters=None):
        if parameters is None:
            parameters = {}
        SelectionAlgorithm.__init__(
            self, database_connector, parameters, DEFAULT_PARAMETERS
        )
        self.time_begin = time.time()
        self.budget = self.parameters["budget_MB"]
        self.max_index_width = self.parameters["max_index_width"]
        self.partition_num = self.parameters["partition_num"]
        self.workload = None
        self.min_cost_improvement = self.parameters["min_cost_improvement"]

    def _calculate_best_indexes(self, workload, valid_workloads):
        logging.info("Calculating best indexes Extend")
        self.workload = workload
        self.valid_workloads = valid_workloads
        single_attribute_index_candidates = self.workload.potential_indexes()
        extension_attribute_candidates = single_attribute_index_candidates.copy()

        # Current index combination
        index_combination = []

        return index_combination

    