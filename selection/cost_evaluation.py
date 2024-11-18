import logging
import time
import re

from selection.what_if_index_creation import WhatIfIndexCreation


class CostEvaluation:
    def __init__(self, db_connector, cost_estimation="actual_runtimes"):
        logging.debug("Init cost evaluation")
        self.db_connector = db_connector
        self.cost_estimation = cost_estimation
        logging.info("Cost estimation with " + self.cost_estimation)
        self.what_if = WhatIfIndexCreation(db_connector)
        self.current_indexes = set()
        self.cost_requests = 0
        self.cache_hits = 0
        # Cache structure:
        # {(query_object, relevant_indexes): cost}
        self.cache = {}
        self.completed = False
        # It is not necessary to drop hypothetical indexes during __init__().
        # These are only created per connection. Hence, non should be present.

        self.relevant_indexes_cache = {}

    def estimate_size(self, index):
        # TODO: Refactor: It is currently too complicated to compute
        # We must search in current indexes to get an index object with .hypopg_oid
        result = None
        for i in self.current_indexes:
            if index == i:
                result = i
                break
        if result:
            # Index does currently exist and size can be queried
            if not index.estimated_size:
                index.estimated_size = self.what_if.estimate_index_size(result.hypopg_oid)
        else:
            self._simulate_or_create_index(index, store_size=True)

    def which_indexes_utilized_and_cost(self, query, indexes):
        self._prepare_cost_calculation(indexes, store_size=True)

        plan = self.db_connector.get_plan(query)
        cost = plan["Total Cost"]
        plan_str = str(plan)

        recommended_indexes = set()

        # We are iterating over the CostEvalution's indexes and not over `indexes`
        # because it is not guaranteed that hypopg_name is set for all items in
        # `indexes`. This is caused by _prepare_cost_calculation that only creates
        # indexes which are not yet existing. If there is no hypothetical index
        # created for an index object, there is no hypopg_name assigned to it. However,
        # all items in current_indexes must also have an equivalent in `indexes`.
        for index in self.current_indexes:
            assert (
                index in indexes
            ), "Something went wrong with _prepare_cost_calculation."

            if index.index_idx() not in plan_str:
                continue
            recommended_indexes.add(index)

        return recommended_indexes, cost

    def calculate_cost(self, workload, indexes, store_size=False):
        assert (
            self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        self._prepare_cost_calculation(indexes, store_size=store_size)
        total_cost = 0

        # TODO: Make query cost higher for queries which are running often
        for query in workload.queries:
            self.cost_requests += 1
            total_cost += self._request_cache(query, indexes)
        return total_cost
    
    def calculate_cost_multithread(self, workload, indexes, store_size=False, partition_num = 0):
        self._prepare_cost_calculation_actual(indexes, store_size=store_size)
        total_cost = 0
        for query in workload.queries:
            self.cost_requests += 1
            total_cost += self._get_cost(query) * query.frequency
        return total_cost
    
    def calculate_cost_multithread_remove_index(self, workload, indexes, store_size=False, partition_num = 0):
        self._prepare_cost_calculation_actual(indexes, store_size=store_size)
        total_cost = 0
        for query in workload.queries:
            self.cost_requests += 1
            total_cost += self._get_cost(query) * query.frequency
        for index in indexes:
            self._unsimulate_or_drop_index(index)
        return total_cost


    # Creates the current index combination by simulating/creating
    # missing indexes and unsimulating/dropping indexes
    # that exist but are not in the combination.
    def _prepare_cost_calculation(self, indexes, store_size=False):
        for index in set(indexes) - self.current_indexes:
            self._simulate_or_create_index(index, store_size=store_size)
        for index in self.current_indexes - set(indexes):
            self._unsimulate_or_drop_index(index)

        assert self.current_indexes == set(indexes)

    def _simulate_or_create_index(self, index, store_size=False):
        if self.cost_estimation == "whatif":
            self.what_if.simulate_index(index, store_size=store_size)
        elif self.cost_estimation == "actual_runtimes":
            self.db_connector.create_index(index)
        self.current_indexes.add(index)

    def _unsimulate_or_drop_index(self, index):
        if self.cost_estimation == "whatif":
            self.what_if.drop_simulated_index(index)
        elif self.cost_estimation == "actual_runtimes":
            self.db_connector.drop_index(index)
        self.current_indexes.remove(index)

    def _prepare_cost_calculation_actual(self, indexes, store_size=False):
        for index in set(indexes) - self.current_indexes:
            self._create_index(index, store_size=store_size)
        for index in self.current_indexes - set(indexes):
            self._drop_index(index)

        assert self.current_indexes == set(indexes)

    def _create_index(self, index, store_size=False):
        self.db_connector.create_index(index)
        self.current_indexes.add(index)

    def _drop_index(self, index):
        self.db_connector.drop_index(index)
        self.current_indexes.remove(index)

    def _get_cost(self, query):
        if self.cost_estimation == "whatif":
            return self.db_connector.get_cost(query)
        elif self.cost_estimation == "actual_runtimes":
            runtime = self.db_connector.exec_query(query)[0]
            return runtime
        
    def _get_cost_slalom(self, query):
        # if self.cost_estimation == "whatif":
        #     return self.db_connector.get_cost(query)
        # elif self.cost_estimation == "actual_runtimes":
        runtime = self.db_connector.exec_query_slalom(query)[0]
        return runtime


    def complete_cost_estimation(self):
        self.completed = True

        for index in self.current_indexes.copy():
            self._unsimulate_or_drop_index(index)

        assert self.current_indexes == set()

    def _request_cache(self, query, indexes):
        q_i_hash = (query, frozenset(indexes))
        if q_i_hash in self.relevant_indexes_cache:
            relevant_indexes = self.relevant_indexes_cache[q_i_hash]
        else:
            relevant_indexes = self._relevant_indexes(query, indexes)
            self.relevant_indexes_cache[q_i_hash] = relevant_indexes

        # Check if query and corresponding relevant indexes in cache
        if (query, relevant_indexes) in self.cache:
            self.cache_hits += 1
            return self.cache[(query, relevant_indexes)]
        # If no cache hit request cost from database system
        else:
            cost = self._get_cost(query)
            self.cache[(query, relevant_indexes)] = cost
            return cost
        
    def _request_cache_slalom(self, query, indexes):
        q_i_hash = (query, frozenset(indexes))
        if q_i_hash in self.relevant_indexes_cache:
            relevant_indexes = self.relevant_indexes_cache[q_i_hash]
        else:
            relevant_indexes = self._relevant_indexes(query, indexes)
            self.relevant_indexes_cache[q_i_hash] = relevant_indexes

        # Check if query and corresponding relevant indexes in cache
        if (query, relevant_indexes) in self.cache:
            self.cache_hits += 1
            return self.cache[(query, relevant_indexes)]
        # If no cache hit request cost from database system
        else:
            cost = self._get_cost_slalom(query)
            self.cache[(query, relevant_indexes)] = cost
            return cost
    
    # calculate the time of building index
    def get_index_build_cost(self, index, store_size=False):
        assert len(index.columns) == 1,"Slalom only refer to single column index!"
        time_start = time.time()
        self._create_index(index)
        time_end = time.time()
        self._drop_index(index)
        return time_end - time_start

    def calculate_cost_slalom(self, workload, indexes, store_size=False):
        assert len(indexes) <= 1,"There should be a single index!"
        assert (
            self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        self._prepare_cost_calculation(indexes, store_size=store_size)
        total_cost = 0

        # TODO: Make query cost higher for queries which are running often
        for query in workload.queries:
            self.cost_requests += 1
            total_cost += self._request_cache_slalom(query, indexes)
        for index in indexes:
            self._unsimulate_or_drop_index(index)
        return total_cost
        
    
    @staticmethod
    def _relevant_indexes(query, indexes):
        relevant_indexes = [
            x for x in indexes if any(c in query.columns for c in x.columns)
        ]
        return frozenset(relevant_indexes)

    def get_selected_records(self, workload, partition_tuple, page_number):
        print("getting_selected_records")
        data,selectivity = self.db_connector.get_selectivity(partition_tuple, workload.queries, page_number)
        return data, selectivity