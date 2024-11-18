import logging
import os
import platform
import re
import subprocess
import random
import pickle 

from selection.workload import Query


class QueryGenerator:
    def __init__(self, benchmark_name, scale_factor, db_connector, query_frequence, columns, partition_id = None, queries_id = None):
        self.scale_factor = scale_factor
        self.benchmark_name = benchmark_name
        self.db_connector = db_connector
        self.queries = []
        self.query_frequence = query_frequence
        # All columns in current database/schema
        self.columns = columns
        self.partition_id = partition_id
        self.queries_id = queries_id
        if self.benchmark_name == 'tpcds':
            self.benchmark_path = "tpcds-userdef-kit"
        elif self.benchmark_name == 'tpch_userdef':
            self.benchmark_path = "tpch-userdef-kit"
        elif self.benchmark_name == 'wikimedia':
            self.benchmark_path = "wikimedia-kit"
        else:
            assert 0, "undefined benchmark !!!"
        self.generate()

    def filter_queries(self, query_ids):
        self.queries = [query for query in self.queries if query.nr in query_ids]

    def add_new_query(self, query_id, query_text, frequency = 1, query_class = None, query_columns_range = None, query_join_key = None):
        if not self.db_connector:
            logging.info("{}:".format(self))
            logging.error("No database connector to validate queries")
            raise Exception("database connector missing")
        query_text = self.db_connector.update_query_text(query_text)
        query = Query(query_id, query_text,frequency = frequency, query_class = query_class,
                                query_columns_range = query_columns_range, query_join_key = query_join_key)
        self._validate_query(query)
        self._store_indexable_columns(query)
        self.queries.append(query)

    def _validate_query(self, query):
        try:
            self.db_connector.get_plan(query)
        except Exception as e:
            self.db_connector.rollback()
            logging.error("{}: {}".format(self, e))

    def _store_indexable_columns(self, query):
        for column in self.columns:
            # **change workload
            # start = query.text.find("FROM") + len("FROM") + 1
            # end = query.text.find("WHERE") - 1
            # result = query.text[start:end]
            # result = result.replace(" ","")
            # query_table = result.split(",")
            # if column.name in query.text and column.table.name in query_table:
            #     query.columns.append(column)
            if column.name in query.text and self._check_substring(column.table.name, query.text):
                if '_1_prt_p' not in column.table.name and '_1_prt_p' in query.text:
                    continue
                query.columns.append(column)

    def _check_substring(self, a, b):
        pattern = re.compile(fr'{a}(?!\d)')
        matches = pattern.finditer(b)
        if any(matches):
            return True
        else:
            return False

    def _generate_tpch(self):
        logging.info("Generating TPC-H Queries")
        self._run_make()
        # Using default parameters (`-d`)
        queries_string = self._run_command(
            ["./qgen", "-c", "-d", "-s", str(self.scale_factor)], return_output=True
        )
        for query in queries_string.split("Query (Q"):
            query_id_and_text = query.split(")\n", 1)
            if len(query_id_and_text) == 2:
                query_id, text = query_id_and_text
                query_id = int(query_id)
                if self.query_ids and query_id not in self.query_ids:
                    continue
                text = text.replace("\t", "")
                self.add_new_query(query_id, text)
        logging.info("Queries generated")

    def _generate_tpch_userdef(self):
        logging.info("Generating TPC-H-User-Define Queries")
        self._run_make()    # build dataset
        # Using default parameters (`-d`)
        queries_string = self._run_command(
            ["./qgen", "-c", "-d", "-s", str(self.scale_factor)], return_output=True
        )
        # according to the config.json, read sql files
        for query_idx,query_id in enumerate(self.queries_id):
            freq = self.query_frequence[query_idx]
            cwd = os.getcwd()
            if self.partition_id is not None:
                sql_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{query_id+1}_{self.partition_id}.sql"
            else:
                sql_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{query_id+1}.sql"
            sql_path = os.path.join(cwd,sql_path)
            with open(sql_path) as f:
                sql_query = f.read()
            pkl_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{query_id+1}_class.pkl"
            if os.path.exists(f"{os.path.join(cwd,pkl_path)}"):
                with open(f"{os.path.join(cwd,pkl_path)}", "rb") as f:
                    query_class = pickle.load(f)
            else:
                query_class = "range"
                logging.info("no query class")
            pkl_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{query_id+1}_queries_columns_range.pkl"
            if os.path.exists(f"{os.path.join(cwd,pkl_path)}"):
                with open(f"{os.path.join(cwd,pkl_path)}", "rb") as f:
                    query_columns_range = pickle.load(f)
            else:
                logging.info("no columns range")
                query_columns_range = None
            pkl_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{query_id+1}_join_key.pkl "
            if os.path.exists(f"{os.path.join(cwd,pkl_path)}"):
                with open(f"{os.path.join(cwd,pkl_path)}", "rb") as f:
                    query_join_key = pickle.load(f)
            else:
                logging.info("no columns range")
                query_join_key = None
            self.add_new_query(query_idx+1, sql_query, frequency = freq, query_class = query_class,
                                query_columns_range = query_columns_range, query_join_key = query_join_key)
        logging.info("Queries generated")

    def _generate_tpcds(self):
        logging.info("Generating TPC-DS Queries")
        self._run_make()
        # dialects: ansi, db2, netezza, oracle, sqlserver
        command = [
            "./dsqgen",
            "-DIRECTORY",
            "../query_templates",
            "-INPUT",
            "../query_templates/templates.lst",
            "-DIALECT",
            "netezza",
            "-QUALIFY",
            "Y",
            "-OUTPUT_DIR",
            "../..",
        ]
        self._run_command(command)
        # according to the config.json, read sql files
        for query_idx,query_id in enumerate(self.queries_id):
            freq = self.query_frequence[query_idx]
            cwd = os.getcwd()
            if self.partition_id is not None:
                sql_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{query_id+1}_{self.partition_id}.sql"
            else:
                sql_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{query_id+1}.sql"
            sql_path = os.path.join(cwd,sql_path)
            with open(sql_path) as f:
                sql_query = f.read()
            pkl_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{query_id+1}_class.pkl"
            with open(f"{os.path.join(cwd,pkl_path)}", "rb") as f:
                query_class = pickle.load(f)
            pkl_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{query_id+1}_queries_columns_range.pkl"
            with open(f"{os.path.join(cwd,pkl_path)}", "rb") as f:
                query_columns_range = pickle.load(f)   
            self.add_new_query(query_idx+1, sql_query, frequency = freq, query_class = query_class,
                                query_columns_range = query_columns_range)
        logging.info("Queries generated")

    def _generate_wikimedia(self):
        logging.info("Generating WIKIMEDIA Queries")
        # according to the config.json, read sql files
        for query_idx,query_id in enumerate(self.queries_id):
            freq = self.query_frequence[query_idx]
            cwd = os.getcwd()
            if self.partition_id is not None:
                sql_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{query_id+1}_{self.partition_id}.sql"
            else:
                sql_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{query_id+1}.sql"
            sql_path = os.path.join(cwd,sql_path)
            with open(sql_path) as f:
                sql_query = f.read()
            pkl_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{query_id+1}_class.pkl"
            with open(f"{os.path.join(cwd,pkl_path)}", "rb") as f:
                query_class = pickle.load(f)
            pkl_path = f"{self.benchmark_path}/queries/{self.scale_factor}/{query_id+1}_queries_columns_range.pkl"
            with open(f"{os.path.join(cwd,pkl_path)}", "rb") as f:
                query_columns_range = pickle.load(f)   
            self.add_new_query(query_idx+1, sql_query, frequency = freq, query_class = query_class,
                                query_columns_range = query_columns_range)
        logging.info("Queries generated")

    # This manipulates TPC-DS specific queries to work in more DBMSs
    def _update_tpcds_query_text(self, query_text):
        query_text = query_text.replace(") returns", ") as returns")
        replaced_string = "case when lochierarchy = 0"
        if replaced_string in query_text:
            new_string = re.search(
                r"grouping\(.*\)\+" r"grouping\(.*\) " r"as lochierarchy", query_text
            ).group(0)
            new_string = new_string.replace(" as lochierarchy", "")
            new_string = "case when " + new_string + " = 0"
            query_text = query_text.replace(replaced_string, new_string)
        return query_text

    def _run_make(self):
        if "qgen" not in self._files() and "dsqgen" not in self._files():
            logging.info("Running make in {}".format(self.directory))
            self._run_command(self.make_command)
        else:
            logging.debug("No need to run make")

    def _run_command(self, command, return_output=False, shell=False):
        env = os.environ.copy()
        env["DSS_QUERY"] = "queries"
        p = subprocess.Popen(
            command,
            cwd=self.directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=shell,
            env=env,
        )
        with p.stdout:
            output_string = p.stdout.read().decode("utf-8")
        p.wait()
        if return_output:
            return output_string
        else:
            logging.debug("[SUBPROCESS OUTPUT] " + output_string)

    def _files(self):
        return os.listdir(self.directory)

    def generate(self):
        if self.benchmark_name == "tpch":
            self.directory = "./tpch-kit/dbgen"
            # DBMS in tpch-kit dbgen Makefile:
            # INFORMIX, DB2, TDAT (Teradata),
            # SQLSERVER, SYBASE, ORACLE, VECTORWISE, POSTGRESQL
            self.make_command = ["make", "DATABASE=POSTGRESQL"]
            if platform.system() == "Darwin":
                self.make_command.append("OS=MACOS")

            self._generate_tpch()
        elif self.benchmark_name == "tpcds":
            self.directory = "./tpcds-kit/tools"
            self.make_command = ["make"]
            if platform.system() == "Darwin":
                self.make_command.append("OS=MACOS")

            self._generate_tpcds()

        elif self.benchmark_name == "tpch_userdef":
            self.directory = "./tpch-kit/dbgen"
            # DBMS in tpch-kit dbgen Makefile:
            # INFORMIX, DB2, TDAT (Teradata),
            # SQLSERVER, SYBASE, ORACLE, VECTORWISE, POSTGRESQL
            self.make_command = ["make", "DATABASE=POSTGRESQL"]
            if platform.system() == "Darwin":
                self.make_command.append("OS=MACOS")

            self._generate_tpch_userdef()
        elif self.benchmark_name == "wikimedia":


            self._generate_wikimedia() 
        else:
            raise NotImplementedError("only tpch/tpcds implemented.")
