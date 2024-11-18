import logging
import os
import platform
import re
import subprocess
import random
import math
import numpy as np
from datetime import date

from selection.utils import b_to_mb,tpch_data_types,sql_per_table
from selection.workload import Column, Table


class TableGenerator:
    def __init__(
        self,
        benchmark_name,
        scale_factor,
        database_connector,
        partition_num=None,
        explicit_database_name=None,
    ):
        self.tpch_data_types = tpch_data_types
        self.scale_factor = scale_factor
        self.benchmark_name = benchmark_name
        self.partition_num = partition_num
        self.db_connector = database_connector
        self.explicit_database_name = explicit_database_name
        self.query_id = 1
        self.database_names = self.db_connector.database_names()
        self.tables = []
        self.columns = []
        self._prepare()
        if self.database_name() not in self.database_names:
            self._generate()
            self.create_database()
        else:
            logging.debug("Database with given scale factor already " "existing")
        self._read_column_names()
        print("-----")

    def database_name(self):
        if self.explicit_database_name:
            return self.explicit_database_name

        name = "indexselection_" + self.benchmark_name + "___"
        name += str(self.scale_factor).replace(".", "_")
        return name

    def _read_column_names(self):
        if self.partition_num is not None:
            for partition_id in range(self.partition_num):
                for key,value in self.tpch_data_types.items():
                    # partition_tables
                    table = Table(key+f"_1_prt_p{partition_id}")
                    self.tables.append(table)
                    for i,name in enumerate(value):
                        if i< 1:
                            continue
                        column_object = Column(name)
                        table.add_column(column_object)
                        self.columns.append(column_object)
                    # father_table
                    if partition_id == 0:
                        table = Table(key)
                        self.total_tables.append(table)
                        for i,name in enumerate(value):
                            if i< 1:
                                continue
                            column_object = Column(name)
                            table.add_column(column_object)
                            self.total_columns.append(column_object) 


        # Read table and column names from 'create table' statements
        else:
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
        self._run_make()
        self._run_command(self.cmd)
        if self.benchmark_name == "tpcds":
            self._run_command(["bash", "../../scripts/replace_in_dat.sh"])
        logging.info("[Generate command] " + " ".join(self.cmd))
        self._table_files()
        logging.info("Files generated: {}".format(self.table_files))

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
        # 分割数据库不在这里处理，改动
        # if self.partition_num is not None:
        #     self.create_subtables(self.partition_num) # number of partition
        # self.db_connector.enable_simulation()

    def create_tables(self, create_statements):
        logging.info("Creating tables")
        for create_statement in create_statements.split(";")[:-1]:
            self.db_connector.exec_only(create_statement)
        self.db_connector.commit()
    
    def create_subtables(self,partition_num=16):
        logging.info("Creating partition tables")
        tables = {
            'customer': 'c_custkey',
            'lineitem': 'l_orderkey',
            'nation': 'n_nationkey',
            'orders': 'o_orderkey',
            'part': 'p_partkey',
            'partsupp': 'ps_partkey',
            'region': 'r_regionkey',
            'supplier': 's_suppkey'
        }
        # create partition tables
        for key, value in tables.items():
            # partitioning tables
            statements = []
            statements.append(f"ALTER TABLE {key} RENAME TO {key}_bak;")
            statement = f"create table {key} (LIKE {key}_bak) WITH (appendonly=true, orientation=column) DISTRIBUTED BY ({value}) PARTITION BY RANGE({value})("
            s_max_value = f"select max({value}) from {key}"
            s_min_value = f"select min({value}) from {key}"
            max_value = self.db_connector.exec_fetch(s_max_value)
            min_value = self.db_connector.exec_fetch(s_min_value)
            value_range = list(np.linspace(min_value, max_value, partition_num+1))
            value_range = [math.ceil(v) for v in value_range]
            # deal with the range cannot be divided into partition_num fields
            value_range = list(set(value_range))
            while len(value_range) < partition_num+1:
                value_range.append(value_range[-1]+1)
            value_range.sort()
            for i in range(len(value_range) - 1):
                if i+1 == len(value_range)-1:
                    op = 'inclusive'
                else:
                    op = 'exclusive,'
                statement += f"PARTITION p{i} start ({value_range[i]})inclusive end ({value_range[i+1]}) {op}"
                # statement = f"create table {key}_{i} as select * from {key} where {value} >= {value_range[i]} and {value} {op} {value_range[i+1]}"
                # self.db_connector.exec_only(s)
            statement += ')'
            statements.append(statement)
            statements.append(f"INSERT INTO {key} SELECT * FROM {key}_bak;")
            for s in statements:
                self.db_connector.exec_only(s)
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
            database_connector.import_data(table, path)
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
            ):
                raise Exception("Wrong TPCDS scale factor")
        elif self.benchmark_name == "tpch_userdef":
            self.make_command = ["make", "DATABASE=POSTGRESQL"]
            if platform.system() == "Darwin":
                self.make_command.append("MACHINE=MACOS")

            self.directory = "./tpch-kit/dbgen"
            self.create_table_statements_file = "dss.ddl"
            self.cmd = ["./dbgen", "-s", str(self.scale_factor), "-f"]

        else:
            raise NotImplementedError("only tpch/ds implemented.")
        
    def _convert_to_float(self, s):
        try:
            return float(s)
        except ValueError:
            return s
        
    def generate_random_query(self,file_path, table):
        # select columns related to this table
        s = f"SELECT column_name, data_type, is_nullable, column_default  FROM information_schema.columns WHERE table_name = '{table}'"
        columns = []
        for tup in self.db_connector.exec_fetch(s,False):
            if tup[1] == 'integer' or tup[1] =='numeric' or tup[1] =='date':
                columns.append(tup[0])
        column1 = random.choice(columns)
        column2 = random.choice(columns)
        while column1 == column2:
            column2 = random.choice(columns)
        col_num = len(columns)
        select_column_num = random.randint(0, col_num-1)
        select_columns = random.choices(columns, k=select_column_num)
        select_content = ",".join(select_columns)
        # Get the minimum and maximum values for the selected columns
        s = f"select {column1} from {table}"
        values = [self._convert_to_float(x[0]) for x in self.db_connector.exec_fetch(s,False)]
        s = f"select {column2} from {table}"
        values2 = [self._convert_to_float(x[0]) for x in self.db_connector.exec_fetch(s,False)]
        min_value = min(values)
        max_value = max(values)
        min_value2 = min(values2)
        max_value2 = max(values2)
        queries = []
        # for i in range(min(int(self.scale_factor*1000),1000)):
        for i in range(sql_per_table):
            if self.partition_num:
                query = []
            # Generate random values between the minimum and maximum values
            if self.tpch_data_types[table][column1] == 'decimal':
                value = random.uniform(min_value, max_value)
                # value2 = random.uniform(min_value, max_value)
                op = random.choice(['>', '<', '=', '>=', '<='])
                if not self.partition_num:
                    query = f"SELECT {select_content} FROM {table} WHERE {column1} {op} {value:.2f}"
                else:
                    for i in range(self.partition_num):
                        query.append(f"SELECT {select_content} FROM {table}_1_prt_p{i} WHERE {column1} {op} {value:.2f}")
            elif self.tpch_data_types[table][column1] == 'integer':
                value = random.randint(min_value, max_value)
                # value2 = random.randint(min_value, max_value)
                op = random.choice(['>', '<', '=', '>=', '<='])
                if not self.partition_num:
                    query = f"SELECT {select_content} FROM {table} WHERE {column1} {op} {value}"
                else:
                    for i in range(self.partition_num):
                        query.append(f"SELECT {select_content} FROM {table}_1_prt_p{i} WHERE {column1} {op} {value}")
            elif self.tpch_data_types[table][column1] == 'date':
                value = random.choice(values)
                # value2 = random.choice(values)
                op = random.choice(['>', '<', '=', '>=', '<='])
                if not self.partition_num:
                    query = f"SELECT {select_content} FROM {table} WHERE {column1} {op} '{value}'"
                else:
                    for i in range(self.partition_num):
                        query.append(f"SELECT {select_content} FROM {table}_1_prt_p{i} WHERE {column1} {op} '{value}'")
            else:
                value = random.choice(values)
                # Format the query with the selected table, columns, and values
                if not self.partition_num:
                    query = f"SELECT {select_content} FROM {table} WHERE {column1} = '{value}'"
                else:
                    for i in range(self.partition_num):
                        query.append(f"SELECT {select_content} FROM {table}_1_prt_p{i} WHERE {column1} = '{value}'")

            # value2
            if self.tpch_data_types[table][column2] == 'decimal':
                value2 = random.uniform(min_value2, max_value2)
                op = random.choice(['>', '<', '=', '>=', '<='])
                if not self.partition_num:
                    query += f" AND {column2} {op} {value2:.2f};"
                else:
                    for i in range(self.partition_num):
                        query[i] += f" AND {column2} {op} {value2:.2f};"

            elif self.tpch_data_types[table][column2] == 'integer':
                value2 = random.randint(min_value2, max_value2)
                op = random.choice(['>', '<', '=', '>=', '<='])
                if not self.partition_num:
                    query += f" AND {column2} {op} {value2};"
                else:
                    for i in range(self.partition_num):
                        query[i] += f" AND {column2} {op} {value2};"

            elif self.tpch_data_types[table][column2] == 'date':
                value2 = random.choice(values2)
                op = random.choice(['>', '<', '=', '>=', '<='])
                if not self.partition_num:
                    query += f" AND {column2} {op} '{value2}';"
                else:
                    for i in range(self.partition_num):
                        query[i] += f" AND {column2} {op} '{value2}';"

            else:
                value2 = random.choice(values2)
                # Format the query with the selected table, columns, and values
                if not self.partition_num:
                    query += f" AND {column2} = '{value2}';"
                else:
                    for i in range(self.partition_num):
                        query[i] += f" AND {column2} = '{value2}';"
            queries.append(query)
        return queries

    def _write_queries(self, filename, table): 
        queries = self.generate_random_query(os.path.join(self.directory, filename), table)
        for query in queries:
            cwd = os.getcwd()
            for i in range(len(query)):
                sql_path = f"tpch-userdef-kit/queries/{self.query_id}_{i}.sql"
                with open(f"{os.path.join(cwd, sql_path)}","w") as f:
                    f.write(query[i])
            self.query_id += 1

        