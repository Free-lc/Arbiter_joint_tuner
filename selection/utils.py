from selection.workload import Workload
import hashlib
sql_per_table = 50
min_records_number = 1000

# --- Unit conversions ---
# Storage
def b_to_mb(b):
    return b / 1000 / 1000


def mb_to_b(mb):
    return mb * 1000 * 1000


# Time
def s_to_ms(s):
    return s * 1000


# --- Index selection utilities ---


def indexes_by_table(indexes):
    indexes_by_table = {}
    for index in indexes:
        table = index.table()
        if table not in indexes_by_table:
            indexes_by_table[table] = []

        indexes_by_table[table].append(index)

    return indexes_by_table


def get_utilized_indexes(
    workload, indexes_per_query, cost_evaluation, detailed_query_information=False
):
    utilized_indexes_workload = set()
    query_details = {}
    for query, indexes in zip(workload.queries, indexes_per_query):
        (
            utilized_indexes_query,
            cost_with_indexes,
        ) = cost_evaluation.which_indexes_utilized_and_cost(query, indexes)
        utilized_indexes_workload |= utilized_indexes_query

        if detailed_query_information:
            cost_without_indexes = cost_evaluation.calculate_cost_multithread(
                Workload([query]), indexes=[]
            )

            query_details[query] = {
                "cost_without_indexes": cost_without_indexes,
                "cost_with_indexes": cost_with_indexes,
                "utilized_indexes": utilized_indexes_query,
            }

    return utilized_indexes_workload, query_details

tpch_data_types = {
    "customer": {
        "c_custkey": "integer",
        "c_name": "string",
        "c_address": "string",
        "c_nationkey": "integer",
        "c_phone": "string",
        "c_acctbal": "decimal",
        "c_mktsegment": "string",
        "c_comment": "string"
    },
    "lineitem": {
        "l_orderkey": "integer",
        "l_partkey": "integer",
        "l_suppkey": "integer",
        "l_linenumber": "integer",
        "l_quantity": "decimal",
        "l_extendedprice": "decimal",
        "l_discount": "decimal",
        "l_tax": "decimal",
        "l_returnflag": "string",
        "l_linestatus": "string",
        "l_shipdate": "date",
        "l_commitdate": "date",
        "l_receiptdate": "date",
        "l_shipinstruct": "string",
        "l_shipmode": "string",
        "l_comment": "string"
    },
    "nation": {
        "n_nationkey": "integer",
        "n_name": "string",
        "n_regionkey": "integer",
        "n_comment": "string"
    },
    "orders": {
        "o_orderkey": "integer",
        "o_custkey": "integer",
        "o_orderstatus": "string",
        "o_totalprice": "decimal",
        "o_orderdate": "date",
        "o_orderpriority": "string",
        "o_clerk": "string",
        "o_shippriority": "integer",
        "o_comment": "string"
    },
    "part": {
        "p_partkey": "integer",
        "p_name": "string",
        "p_mfgr": "string",
        "p_brand": "string",
        "p_type": "string",
        "p_size": "integer",
        "p_container": "string",
        "p_retailprice": "decimal",
        "p_comment": "string"
    },
    "partsupp": {
        "ps_partkey": "integer",
        "ps_suppkey": "integer",
        "ps_availqty": "integer",
        "ps_supplycost": "decimal",
        "ps_comment": "string"
    },
    "region": {
        "r_regionkey": "integer",
        "r_name": "string",
        "r_comment": "string"
    },
    "supplier": {
        "s_suppkey": "integer",
        "s_name": "string",
        "s_address": "string",
        "s_nationkey": "integer",
        "s_phone": "string",
        "s_acctbal": "decimal",
        "s_comment": "string"
    }
}


import hashlib

class BloomFilter:
    def __init__(self, size):
        self.size = size
        self.bit_array = [0] * size

    def _hash(self, value, seed):
        hash_obj = hashlib.md5()
        # 将不同类型的值统一转换为字符串
        str_value = str(value)
        # 使用统一的编码方式
        hash_obj.update(str_value.encode('utf-8'))
        hash_obj.update(str(seed).encode('utf-8'))
        return int(hash_obj.hexdigest(), 16) % self.size

    def add(self, value):
        for seed in range(3):  # 使用3个不同的哈希函数
            result = self._hash(value, seed)
            self.bit_array[result] = 1

    def check(self, value):
        for seed in range(3):
            result = self._hash(value, seed)
            if self.bit_array[result] == 0:
                return False
        return True
