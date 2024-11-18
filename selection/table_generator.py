import logging
import os
import platform
import re
import subprocess
import random
import math
import numpy as np
import bisect
from datetime import date
import pickle
import copy

from selection.utils import b_to_mb,tpch_data_types
from selection.workload import Column, Table


class TableGenerator:
    def __init__(
        self,
        benchmark_name,
        scale_factor,
        database_connector,
        sql_per_table = None,
        partition_num=None,
        test = None,
        explicit_database_name=None,
        seed = None
    ):
        self.test = test
        self.scale_factor = scale_factor
        self.benchmark_name = benchmark_name
        self.block_num = partition_num # physical block_number
        self.db_connector = database_connector
        self.explicit_database_name = explicit_database_name
        self.seed = seed
        self.query_id = 76
        self.database_names = self.db_connector.database_names()
        self.sql_per_table = sql_per_table
        if sql_per_table is None:
            self.sql_per_table = 50
        self.tables = []
        self.columns = []
        self.total_tables = []
        self.total_columns = []
        self.table_values = {}  # statiscs the value of big table for query generation
        self.current_views = []
        if self.benchmark_name == 'tpcds':
            self.benchmark_path = "tpcds-userdef-kit"
        elif self.benchmark_name == 'tpch_userdef':
            self.benchmark_path = "tpch-userdef-kit"
        elif self.benchmark_name == 'wikimedia':
            self.benchmark_path = "wikimedia-kit"
        else:
            assert 0, "undefined benmarks!!!"
        random.seed(self.seed)
        self._prepare()
        if self.database_name() not in self.database_names:
            self._generate()
            self.create_database()
        else:
            self.reuse_database()
            self.recover_table()
            if self.block_num is not None:
                self.create_subtables(self.block_num) # number of partition
            
            logging.debug("Database with given scale factor already " "existing")
        self._read_column_names()
        print("-----")
    
    def partition_table(self, queries):
        # 分割数据库不在这里处理，改动
        if self.block_num is not None:
            self.create_subtables(self.block_num, queries) # number of partition

    def database_name(self):
        if self.test is not None:
            return "test_database2"
        if self.explicit_database_name:
            return self.explicit_database_name

        name = "indexselection_" + self.benchmark_name + "___"  #改动
        name += str(self.scale_factor).replace(".", "_")
        return name

    def _read_column_names(self):
        # partition_table
        if self.block_num is not None:
            for partition_id in range(self.block_num):
                for key,value in self.tpch_data_types.items():
                    # partition_tables
                    table = Table(key+f"_1_prt_p{partition_id}")
                    self.tables.append(table)
                    for i,name in enumerate(value):
                        # if i< 1:
                        #     continue
                        column_object = Column(name)
                        table.add_column(column_object)
                        self.columns.append(column_object)
                    # father_table
                    if partition_id == 0:
                        table = Table(key)
                        self.total_tables.append(table)
                        for i,name in enumerate(value):
                            # if i< 1:
                            #     continue
                            column_object = Column(name)
                            table.add_column(column_object)
                            self.total_columns.append(column_object) 


        # Read table and column names from 'create table' statements
        # 2. whole table 
        filename = self.directory + "/" + self.create_table_statements_file
        with open(filename, "r") as file:
            data = file.read().lower()
        create_tables = data.split("create table ")[1:]
        for create_table in create_tables:
            splitted = create_table.split("(", 1)
            table = Table(splitted[0].strip())
            self.tables.append(table)
            # TODO regex split? ,[whitespace]\n
            for column in splitted[1].split(",\n"):
                name = column.lstrip().split(" ", 1)[0]
                if name == "primary":
                    continue
                column_object = Column(name)
                table.add_column(column_object)
                self.columns.append(column_object)

    def _generate(self):
        logging.info("Generating {} data".format(self.benchmark_name))
        logging.info("scale factor: {}".format(self.scale_factor))
        if self.benchmark_name == "tpcds" or self.benchmark_name == "tpch_userdef":
            self._run_make()
            self._run_command(self.cmd)
        if self.benchmark_name == "tpcds":
            self._run_command(["bash", "../../scripts/replace_in_dat.sh"])
        logging.info("[Generate command] " + " ".join(self.cmd))
        self._table_files()
        logging.info("Files generated: {}".format(self.table_files))

    def get_table_files(self):
        return self.table_files

    def create_database(self):
        self.db_connector.create_database(self.database_name())
        filename = self.directory + "/" + self.create_table_statements_file
        with open(filename, "r") as file:
            create_statements = file.read()
        # Do not create primary keys
        create_statements = re.sub(r",\s*primary key (.*)", "", create_statements)
        self.db_connector.db_name = self.database_name()
        self.db_connector.create_connection()
        self.create_tables(create_statements)

        self._load_table_data(self.db_connector)
        self._get_type_of_attribtes()
        # 分割数据库不在这里处理，改动
        if self.block_num is not None:
            self.create_subtables(self.block_num) # number of partition
        # self.db_connector.enable_simulation()

    def reuse_database(self):
        self._get_type_of_attribtes()
        self._table_files()


    def _get_type_of_attribtes(self):
        # get type of attribtes
        self.tpch_data_types = {}
        if self.benchmark_name == "tpch_userdef":
            self.tpch_data_types = tpch_data_types
        else:
            get_table_statement = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            tables_name = self.db_connector.exec_fetch(get_table_statement, False)
            for table in tables_name:
                table_name = table[0]
                if table_name == 'dbgen_version':
                    continue
                get_attribute_type_statement = f'''SELECT 
                        attname AS column_name,
                        typname AS data_type
                    FROM 
                        pg_attribute
                    JOIN 
                        pg_type ON pg_attribute.atttypid = pg_type.oid
                    WHERE 
                        attrelid = (SELECT oid FROM pg_class WHERE relname ='{table_name}')
                    AND 
                        attnum > 0 
                    AND 
                        NOT attisdropped 
                    ORDER BY 
                        attnum;'''
                attributes_type = self.db_connector.exec_fetch(get_attribute_type_statement, False)
                for attribute_type in attributes_type:
                    attribute = attribute_type[0]
                    type = attribute_type[1]
                    if 'int' in type:
                        type = 'integer'
                    elif 'numeric' in type:
                        type = 'decimal' 
                    elif 'date' in type:
                        pass
                    elif 'string' in type or 'bpchar' in type or 'char' or 'text' in type:
                        type = 'string'
                    else:
                        assert 0, "undefined type!!"
                    if table_name not in self.tpch_data_types:
                        self.tpch_data_types[table_name] = {}
                    self.tpch_data_types[table_name][attribute] = type

    def create_tables(self, create_statements):
        logging.info("Creating tables")
        for create_statement in create_statements.split(";")[:-1]:
            self.db_connector.exec_only(create_statement)
        self.db_connector.commit()
    
    def create_subtables(self, partition_num=16, queries = None):
        logging.info("Creating partition tables")
        tables = {}
        # if queries is not None:
        #     for query in queries:
        #         if query.join_keys is None: continue
        #         for key,value in query.query_join_key.items():
        #             tables[key] = value
        # get primary key in every single table
        get_table_statement = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
        tables_name = self.db_connector.exec_fetch(get_table_statement, False)
        for table in tables_name:
            table_name = table[0]
            if table_name == 'dbgen_version':
                continue
            if table_name in tables:
                continue
            # default partition key is primary key
            get_primary_key_statement = f'''SELECT 
                column_name
            FROM 
                information_schema.columns
            WHERE 
                table_name = '{table_name}' AND
                table_schema = 'public' 
            ORDER BY 
                ordinal_position;'''
            primary_keys = self.db_connector.exec_fetch(get_primary_key_statement,False)
            for primary_key in primary_keys:
                primary_key = primary_key[0]
                if self.tpch_data_types[table_name][primary_key] != 'string':
                    tables[table_name] = primary_key
                    break
        self.table_partition_key = tables
        # create partition tables
        for key, value in tables.items():
            # partitioning tables
            statements = []
            statements.append(f"ALTER TABLE {key} RENAME TO {key}_bak;")
            statement = f"create table {key} (LIKE {key}_bak) WITH (appendonly=true, orientation=column) DISTRIBUTED BY ({value}) PARTITION BY RANGE({value})("
            s_max_value = f"select max({value}) from {key}"
            s_min_value = f"select min({value}) from {key}"
            max_value = self.db_connector.exec_fetch(s_max_value)[0]
            min_value = self.db_connector.exec_fetch(s_min_value)[0]
            value_range = list(np.linspace(min_value, max_value, partition_num+1))
            value_range = [math.floor(v) for v in value_range]
            # deal with the range cannot be divided into partition_num fields
            value_range = list(set(value_range))
            value_range.sort()
            while len(value_range) < partition_num+1:
                value_range.append(value_range[-1]+1)
            for i in range(len(value_range) - 1):
                if i+1 == len(value_range)-1:
                    op = 'inclusive'
                else:
                    op = 'exclusive,'
                statement += f"PARTITION p{i} start ({value_range[i]})inclusive end ({value_range[i+1]}) {op}"
            statement += ');'
            statements.append(statement)
            for s in statements:
                self.db_connector.exec_only(s)
            # if primary(first) key has null value, change the last partition to default
            statement = f"select count(*) from {key}_bak where {value} IS NULL;"
            num_number = int(self.db_connector.exec_fetch(statement)[0])
            if num_number > 0:
                self.db_connector.exec_only(f"ALTER TABLE {key} DROP PARTITION p{partition_num-1};")
                self.db_connector.exec_only(f"ALTER TABLE {key} ADD DEFAULT PARTITION p{partition_num-1};")
            statement = f"INSERT INTO {key} SELECT * FROM {key}_bak;"
            self.db_connector.exec_only(statement)
        self.db_connector.commit()
    
    def recover_table(self):
        logging.info("Reuse tables")
        self.db_connector.db_name = self.database_name()
        self.db_connector.create_connection()
        # recover table
        get_table_statement = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
        tables_name = self.db_connector.exec_fetch(get_table_statement, False)
        table_bak = []
        for table in tables_name:
            table_name = table[0]
            if not '_bak' in table_name:
                if not '_1_prt_p' in table_name: # only drop whole table
                    self.db_connector.exec_only(f"drop table {table_name};")
            else:
                table_bak.append(table_name)
        for table in table_bak:
            self.db_connector.exec_only(f"ALTER TABLE {table} RENAME TO {table.split('_bak')[0]};")
        # self.db_connector.commit()
        

    def drop_tupel_partitioning(self, db_connector):
        statements = []
        for view in self.current_views:
            statement = "DROP TABLE IF EXISTS " + view +" ;"
            statements.append(statement)
        for s in statements:
            db_connector.exec_only(s)
            db_connector.commit()
        self.current_views.clear()


    def tuples_partitioning(self, partition_tuples = None, query_table = None):
        assert partition_tuples is not None, "You need to specify the partition_tuples"
        partition_num = len(partition_tuples)
        logging.info("Creating partition tables")
        tables = self.table_partition_key
        # create partition tables
        for key, value in tables.items():
            # only partitioning queried table
            if query_table and key not in query_table:
                continue
            # partitioning tables
            statements = []
            logging.info(f"Partitioning {key} table by {value} key.")
            statements.append(f"DROP TABLE IF EXISTS {key};")
            statement = f"create table {key} (LIKE {key}_bak) WITH (appendonly=true, orientation=column) DISTRIBUTED BY ({value}) PARTITION BY RANGE({value})("
            s_max_value = f"select max({value}) from {key}"
            s_min_value = f"select min({value}) from {key}"
            max_value = self.db_connector.exec_fetch(s_max_value)[0]
            min_value = self.db_connector.exec_fetch(s_min_value)[0]
            value_range = list(np.linspace(min_value, max_value, self.block_num+1))
            value_range = [math.floor(v) for v in value_range]
            # deal with the range cannot be divided into partition_num fields
            value_range = list(set(value_range))
            while len(value_range) < self.block_num+1:
                value_range.append(value_range[-1]+1)
            value_range.sort()
            for partition_id,partition_tuple in enumerate(partition_tuples):
                block_start = partition_tuple[0]
                block_end = partition_tuple[1]
                if partition_id == len(partition_tuples)-1:
                    op = 'inclusive'
                else:
                    op = 'exclusive,'
                statement += f"PARTITION p{partition_id} start ({value_range[block_start]})inclusive end ({value_range[block_end]}) {op}"
            statement += ')'
            statements.append(statement)
            for s in statements:
                self.db_connector.exec_only(s)
            # if primary(first) key has null value, change the last partition to default
            statement = f"select count(*) from {key}_bak where {value} IS NULL;"
            num_number = int(self.db_connector.exec_fetch(statement)[0])
            if num_number > 0:
                if partition_num == 1:
                    self.db_connector.exec_only(f"DROP TABLE {key};")
                    self.db_connector.exec_only(f"CREATE TABLE {key} (LIKE {key}_bak) WITH (appendonly=true, orientation=column)DISTRIBUTED BY ({value}) PARTITION BY RANGE({value})(DEFAULT PARTITION p{partition_num-1});")
                else: 
                    self.db_connector.exec_only(f"ALTER TABLE {key} DROP PARTITION p{partition_num-1};")
                    self.db_connector.exec_only(f"ALTER TABLE {key} ADD DEFAULT PARTITION p{partition_num-1};")
            statement = f"INSERT INTO {key} SELECT * FROM {key}_bak;"
            self.db_connector.exec_only(statement)
        self.db_connector.commit()

    def _load_table_data(self, database_connector):
        logging.info("Loading data into the tables")
        for filename in self.table_files:
            logging.debug("    Loading file {}".format(filename))
            table = filename.replace(".tbl", "").replace(".dat", "")
            path = self.directory + "/" + filename
            size = os.path.getsize(path)
            size_string = f"{b_to_mb(size):,.4f} MB"
            logging.debug(f"    Import data of size {size_string}")
            if self.benchmark_name == 'wikimedia':
                delimiter = " " 
            else:
                delimiter = "|"
            database_connector.import_data(table, path, delimiter=delimiter)
            # create user defined tpch workload
            logging.debug("    building user defined tpch workload for table {}".format(filename))
            # self._write_queries(filename, table)
            # ---------------------------
            # os.remove(os.path.join(self.directory, filename))
        database_connector.commit()

    def _run_make(self):
        if "dbgen" not in self._files() and "dsdgen" not in self._files():
            logging.info("Running make in {}".format(self.directory))
            self._run_command(self.make_command)
        else:
            logging.info("No need to run make")

    def _table_files(self):
        self.table_files = [x for x in self._files() if ".tbl" in x or ".dat" in x]

    def _run_command(self, command):
        cmd_out = "[SUBPROCESS OUTPUT] "
        p = subprocess.Popen(
            command,
            cwd=self.directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with p.stdout:
            for line in p.stdout:
                logging.info(cmd_out + line.decode("utf-8").replace("\n", ""))
        p.wait()

    def _files(self):
        return os.listdir(self.directory)

    def _prepare(self):
        if self.benchmark_name == "tpch":
            self.make_command = ["make", "DATABASE=POSTGRESQL"]
            if platform.system() == "Darwin":
                self.make_command.append("MACHINE=MACOS")

            self.directory = "./tpch-kit/dbgen"
            self.create_table_statements_file = "dss.ddl"
            self.cmd = ["./dbgen", "-s", str(self.scale_factor), "-f"]
        elif self.benchmark_name == "tpcds":
            self.make_command = ["make"]
            if platform.system() == "Darwin":
                self.make_command.append("OS=MACOS")

            self.directory = "./tpcds-kit/tools"
            self.create_table_statements_file = "tpcds.sql"
            self.cmd = ["./dsdgen", "-SCALE", str(self.scale_factor), "-FORCE"]

            # 0.001 is allowed for testing
            if (
                int(self.scale_factor) - self.scale_factor != 0
                and self.scale_factor != 0.001
                and self.scale_factor != 0.1
            ):
                raise Exception("Wrong TPCDS scale factor")
        elif self.benchmark_name == "tpch_userdef":
            self.make_command = ["make", "DATABASE=POSTGRESQL"]
            if platform.system() == "Darwin":
                self.make_command.append("MACHINE=MACOS")

            self.directory = "./tpch-kit/dbgen"
            self.create_table_statements_file = "dss.ddl"
            self.cmd = ["./dbgen", "-s", str(self.scale_factor), "-f"]

        elif self.benchmark_name == "wikimedia":
            self.make_command = ["make", "DATABASE=POSTGRESQL"]
            if platform.system() == "Darwin":
                self.make_command.append("MACHINE=MACOS")

            self.directory = "./wikimedia-kit"
            self.create_table_statements_file = "wikimedia.sql"
            self.cmd = ["./dbgen", "-s", str(self.scale_factor), "-f"]

        else:
            raise NotImplementedError("only tpch/ds implemented.")
        
    def _convert_to_float_selectivity(self, s):
        if s is None:
            return np.nan
        import datetime
        string = copy.deepcopy(s)
        try:
            return float(string)
        except:
            try:
                string = string.replace("\'", "")
                return (datetime.datetime.strptime(string, '%Y-%m-%d').date() - datetime.date(1970, 1, 1)).days
            except:
                try:
                    return (s - datetime.date(1970, 1, 1)).days
                except:
                    return string
            
    def _convert_to_float(self, s):
        if s is None:
            return np.nan
        import datetime
        string = copy.deepcopy(s)
        try:
            return float(string)
        except:
            try:
                string = string.replace("\'", "")
                return (datetime.datetime.strptime(string, '%Y-%m-%d').date())
            except:
                try:
                    return (s - datetime.date(1970, 1, 1)).days
                except:
                    return string
            
        
    def generate_random_query(self, file_path, table):
        # select columns related to this table
        s = f"SELECT column_name, data_type, is_nullable, column_default  FROM information_schema.columns WHERE table_name = '{table}'"
        columns = []
        for tup in self.db_connector.exec_fetch(s,False):
            if (tup[1] == 'integer' or tup[1] =='numeric' or tup[1] =='date' or 'int' in tup[1] or tup[0] == 'pagename') and tup[0] in self.table_new[table]:
                columns.append(tup[0])
        col_num = len(columns)
        queries = []
        total_queries = []
        queries_class = []
        queries_columns_range = []
        selectivity = 0
        for j in range(self.sql_per_table):
            select_column_num = random.randint(1, col_num)
            select_columns = random.choices(columns, k=select_column_num)
            select_content = ",".join(select_columns)
            filter_column_num = random.randint(1, 2)
            filter_column_num = min(col_num, filter_column_num) # avoid exceed the number of column
            filter_columns = random.sample(columns, filter_column_num)
            if j <= (2 * self.sql_per_table) // 3:
                queries_class.append('range')
                selectivity = random.uniform(0.005, 0.05)
                filter_contents,query_columns_range = self._generate_filter_content(selectivity, table, filter_columns)
            else:
                queries_class.append('point')
                filter_contents,query_columns_range = self._generate_filter_content(selectivity, table, filter_columns)
            queries_columns_range.append(query_columns_range)
            for filter_content in filter_contents:
                if self.block_num:
                    query = []
                if not self.block_num:
                    query = f"SELECT {select_content} FROM {table} WHERE {filter_content}"
                else:
                    for partition_index in range(self.block_num):
                        query.append(f"SELECT {select_content} FROM {table}_1_prt_p{partition_index} WHERE {filter_content}")
                    total_query = f"SELECT {select_content} FROM {table} WHERE {filter_content}"
                queries.append(query)
                total_queries.append(total_query)
        logging.info(f"the queries of table {table} {j}th hava been generated")
        return queries,total_queries,queries_class,queries_columns_range

    def _write_queries(self, filename, table): 
        self.table_values[table] = {}
        if self.benchmark_name == "tpch_userdef":
            tables_old = {
                'customer': ['c_custkey', 'c_name', 'c_address', 'c_nationkey','c_phone', 'c_acctbal', 'c_mktsegment','c_comment'],
                'lineitem': ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment'],
                'nation': ['n_nationkey', 'n_name', 'n_regionkey', 'n_comment'],
                'orders': ['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority', 'o_clerk', 'o_shippriority', 'o_comment'],
                'part': ['p_partkey', 'p_name', 'p_mfgr', 'p_brand', 'p_type', 'p_size', 'p_container', 'p_retailprice', 'p_comment'],
                'partsupp': ['ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost', 'ps_comment'],
                'region': ['r_regionkey', 'r_name', 'r_comment'],
                'supplier': ['s_suppkey', 's_name', 's_address', 's_nationkey', 's_phone', 's_acctbal', 's_comment']
                }
            self.table_new = tables_new = {
                'customer': ['c_custkey', 'c_nationkey','c_phone', 'c_acctbal', 'c_mktsegment'],
                'lineitem': ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_returnflag', 'l_linestatus', 'l_shipdate', 'l_commitdate', 'l_receiptdate', 'l_shipinstruct', 'l_shipmode', 'l_comment'],
                'nation': ['n_nationkey', 'n_name', 'n_regionkey'],
                'orders': ['o_orderkey', 'o_custkey', 'o_orderstatus', 'o_totalprice', 'o_orderdate', 'o_orderpriority'],
                'part': ['p_partkey', 'p_name', 'p_brand', 'p_type', 'p_size', 'p_container'],
                'partsupp': ['ps_partkey', 'ps_suppkey', 'ps_availqty', 'ps_supplycost'],
                'region': ['r_regionkey', 'r_name'],
                'supplier': ['s_suppkey', 's_name', 's_address', 's_nationkey', 's_acctbal' ]
                }
        else:
            tables_old = {}
            for table_name,attr_type_mp in self.tpch_data_types.items():
                att_ls = []
                for attr,type in attr_type_mp.items():
                    att_ls.append(attr)
                tables_old[table_name] = att_ls
            tables_new = self.table_new = copy.deepcopy(tables_old)
                
        columns = tables_old[table]
        with open(f'{os.path.join(self.directory, filename)}', 'r') as f:
            lines = f.readlines()
            for column in  tables_new[table]:
                col_index = columns.index(column)
                if self.benchmark_name == 'wikimedia':
                    delimiter = " "
                else:
                    delimiter = "|"
                values = [self._convert_to_float(line.split(delimiter)[col_index]) for line in lines]
                if self.tpch_data_types[table][column] == 'integer':
                    values = [math.floor(value) for value in values if value != '\n' and value != '']
                else:
                    values = [value for value in values if value != '\n' and value != '']
                # values.sort()
                self.table_values[table][column] = values
        logging.info(f"the data of table {table} hava been analyzed")
        queries, total_queries, queries_class, queries_columns_range = self.generate_random_query(os.path.join(self.directory, filename), table)
        partition_num = self.block_num
        self.block_num = None
        self.block_num = partition_num
        for query,total_query,query_class,query_columns_range in zip(queries,total_queries, queries_class,queries_columns_range):
            cwd = os.getcwd()
            #partition_queries
            for i in range(len(query)):
                sql_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{self.query_id}_{i}.sql"
                with open(f"{os.path.join(cwd, sql_path)}","w") as f:
                    f.write(query[i])
            # total_queries
            sql_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{self.query_id}.sql"
            with open(f"{os.path.join(cwd, sql_path)}","w") as f:
                f.write(total_query)
            # queries class
            pkl_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{self.query_id}_class.pkl"
            with open(f"{os.path.join(cwd, pkl_path)}","wb") as f:
                pickle.dump(query_class, f)
            # queries_columns_range
            pkl_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{self.query_id}_queries_columns_range.pkl"
            with open(f"{os.path.join(cwd, pkl_path)}","wb") as f:
                pickle.dump(query_columns_range, f)
            self.query_id += 1


    def _generate_filter_content(self, selectivity, table, filter_columns, content_num=1):
        # consider selectivity of mult_columns
        values_list = []
        for filter_column in filter_columns:
            values_list.append(self.table_values[table][filter_column])
        col_num = len(filter_columns)
        value_len = len(values_list[0])
        if selectivity == 0:
            select_len = 1
        else:
            select_len = math.ceil(value_len*selectivity)
        statements = []
        query_columns_range = {}
        for i in range(content_num):
            begin = random.randint(0, value_len - select_len)
            end = begin+select_len-1
            # 并元组
            zipped = zip(*values_list)
            sorted_tuples = sorted(zipped)
            value_list = list(zip(*sorted_tuples))
            statement = ""
            for i in range(len(value_list)):
                value = list(value_list[i])
                value = value[begin:end+1]
                value.sort()
                if self.tpch_data_types[table][filter_columns[i]] == 'string' or self.tpch_data_types[table][filter_columns[i]] == 'date':
                    value_begin = f"'{value[0]}'"
                    value_end = f"'{value[-1]}'"
                else:
                    value_begin = value[0]
                    value_end = value[-1]
                if i == 0:
                    if value_begin == value_end:
                        statement += f"{filter_columns[i]} = {value_begin}"
                    else:
                        statement += f"{filter_columns[i]} Between {value_begin} and {value_end}"
                else:
                    if value_begin == value_end:
                        statement += f" AND {filter_columns[i]} = {value_begin}"
                    else:
                        statement += f" AND {filter_columns[i]} Between {value_begin} and {value_end}"
                query_columns_range[filter_columns[i]] = [value_begin, value_end]
            statement += ";"
            statements.append(statement)
        # get access blocks
        return statements, query_columns_range
    
    def find_interval(self, a, num):
        # 使用bisect_left找到num应该插入的位置
        index = bisect.bisect_left(a, num)

        # 如果index是0，说明num比数列中的最小值还小
        if index == 0:
            return None, 0

        # 如果index等于len(a)，说明num比数列中的最大值还大
        if index == len(a):
            return len(a) - 1, None

        # 否则，num就位于index-1和index之间
        return index - 1, index

    def get_selectivity(self, partition_tuple, selected_queries, page_number = 20):
        assert 0, "disabled"
        statement = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' and table_name LIKE '%\_bak';"
        table_names = [tuple[0] for tuple in self.db_connector.exec_fetch(statement, False)]
        tables = {}
        records = 0
        data_list = []
        start_block = partition_tuple[0]
        end_block = partition_tuple[1]
        table_value = {}
        tables = {}
        for table,key in self.table_partition_key.items():
            tables[table+"_bak"] = key
        # get indexable attributes in queries
        indexable_atrribues = {}
        for query in selected_queries:
            query_text = copy.deepcopy(query.text)
            query_text = self.replace_select_with_star(query_text)
            # only support query related to one table
            table = copy.deepcopy(query.columns[0].table.name).replace("_1_prt_p0","_bak")
            if table in indexable_atrribues:
                indexable_atrribues[table] += [column.name.split(".")[-1] for column in query.columns]
            else:
                indexable_atrribues[table] = [column.name.split(".")[-1] for column in query.columns]
        for table, attributes in indexable_atrribues.items():
            indexable_atrribues[table] = list(set(attributes))
        
        for key, value in tables.items():
            # get statistics from the tables accessed by queries
            if key not in indexable_atrribues:
                continue
            s_max_value = f"select max({value}) from {key}"
            s_min_value = f"select min({value}) from {key}"
            max_value = self.db_connector.exec_fetch(s_max_value)[0]
            min_value = self.db_connector.exec_fetch(s_min_value)[0]
            value_range = list(np.linspace(min_value, max_value, page_number+1))
            value_range = [math.floor(v) for v in value_range]
            # deal with the range cannot be divided into partition_num fields
            value_range = list(set(value_range))
            while len(value_range) < page_number+1:
                value_range.append(value_range[-1]+1)
            value_range.sort()
            value_start = value_range[start_block]
            value_end = value_range[end_block]
            table_value[key] = (value_start, value_end)
            indexable_atrribue = ""
            for column in indexable_atrribues[key]:
                indexable_atrribue += column
                indexable_atrribue += ","
            indexable_atrribue= indexable_atrribue[:-1]
            statement = f"select {indexable_atrribue} from  {key} WHERE {value} between {value_start} and {value_end}"
            # data = [list(tupe) for tupe in self.db_connector.exec_fetch(statement, False)]
            # tupe is a record
            data = []
            column_names = [column[0] for column in self.db_connector.exec_fetch(f"SELECT attname AS column_name,typname AS data_type FROM pg_attribute JOIN pg_type ON pg_attribute.atttypid = pg_type.oid WHERE attrelid = (SELECT oid FROM pg_class WHERE relname ='{key}') AND attnum > 0 AND NOT attisdropped ORDER BY attnum;",False)]
            for tupe in self.db_connector.exec_fetch(statement, False):
                record = np.array([])               
                for itemid, item in enumerate(tupe):
                    item = self._convert_to_float_selectivity(item)
                    if self.tpch_data_types[key.split("_bak")[0]][indexable_atrribues[key][itemid]] == 'integer' and item is not np.nan:
                        item = math.floor(item)
                        ## maybe there is a None
                    record = np.append(record, item)
                data.append(record)
            records += len(data)
            data = np.array(data)
            data_list.append(data)
        selected_records = []
        for query in selected_queries:
            query_text = copy.deepcopy(query.text)
            query_text = self.replace_select_with_star(query_text)
            # only support query related to one table
            table = copy.deepcopy(query.columns[0].table.name).replace("_1_prt_p0","_bak")
            value_start = table_value[table][0]
            value_end = table_value[table][1]
            query_text += f" AND {tables[table]} between {value_start} and {value_end} ;"
            selected_records += [list(tupe) for tupe in self.db_connector.exec_fetch(query_text, False)]
        # the number of tuple appear in both records of queries and records of the partition tuple
        intersection = len(selected_records)
        if records == 0:
            selectivity = 0
        else:
            selectivity = intersection/records
        return np.array(data_list, dtype=object), selectivity
    
    def replace_select_with_star(self, sql):
        # make the content between SELECT and FROM become *
        pattern = r"(?i)(SELECT\s+)(.*?)(\s+FROM)"
        repl = r"\1*\3"
        replaced_sql = re.sub(pattern, repl, sql)
        replaced_sql = replaced_sql.replace("_1_prt_p0","")
        replaced_sql = replaced_sql.replace(";","")
        return replaced_sql
    