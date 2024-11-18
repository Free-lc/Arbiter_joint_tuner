import logging
import re
import math
import numpy as np
import psycopg2
import copy
from io import StringIO

from selection.database_connector import DatabaseConnector


class PostgresDatabaseConnector(DatabaseConnector):
    def __init__(self, db_name, autocommit=False):
        DatabaseConnector.__init__(self, db_name, autocommit=autocommit)
        self.db_system = "postgres"
        self._connection = None

        if not self.db_name:
            self.db_name = "postgres"
        self.create_connection()

        self.set_random_seed()

        logging.debug("Postgres connector created: {}".format(db_name))

    def create_connection(self):
        if self._connection:
            self.close()
        self._connection = psycopg2.connect("dbname={} password=fray".format(self.db_name))
        # self._connection = psycopg2.connect("dbname={}".format(self.db_name))
        self._connection.autocommit = self.autocommit
        self._cursor = self._connection.cursor()

    def enable_simulation(self):
        self.exec_only("create extension hypopg")
        self.commit()

    def database_names(self):
        result = self.exec_fetch("select datname from pg_database", False)
        return [x[0] for x in result]

    # Updates query syntax to work in PostgreSQL
    def update_query_text(self, text):
        text = text.replace(";\nlimit ", " limit ").replace("limit -1", "")
        text = re.sub(r" ([0-9]+) days\)", r" interval '\1 days')", text)
        text = self._add_alias_subquery(text)
        return text

    # PostgreSQL requires an alias for subqueries
    def _add_alias_subquery(self, query_text):
        text = query_text.lower()
        positions = []
        for match in re.finditer(r"((from)|,)[  \n]*\(", text):
            counter = 1
            pos = match.span()[1]
            while counter > 0:
                char = text[pos]
                if char == "(":
                    counter += 1
                elif char == ")":
                    counter -= 1
                pos += 1
            next_word = query_text[pos:].lstrip().split(" ")[0].split("\n")[0]
            if next_word[0] in [")", ","] or next_word in [
                "limit",
                "group",
                "order",
                "where",
            ]:
                positions.append(pos)
        for pos in sorted(positions, reverse=True):
            query_text = query_text[:pos] + " as alias123 " + query_text[pos:]
        return query_text

    def create_database(self, database_name):
        self.exec_only("create database {}".format(database_name))
        logging.info("Database {} created".format(database_name))

    def import_data(self, table, path, delimiter="|"):
        with open(path, "r") as file:
            self._cursor.copy_from(file, table, sep=delimiter, null="")
    
    def import_data_drop_bad(self, table, path, delimiter = "|"):
        with open(path, "r") as file:
            for line_number, line in enumerate(file, 1):
                try:
                    f = StringIO(line)
                    f.seek(0)  # 移动到 StringIO 对象的开头，以便 copy_from 可以从头读取
                    self._cursor.copy_from(f, table, sep=delimiter, null="")  
                    self.commit()
                except psycopg2.Error as e:
                    print(f"Error importing line {line_number}: {line}  {e}")
                    self.rollback()  # 回滚当前事务中的所有操作


    def indexes_size(self):
        # Returns size in bytes
        statement = (
            "select sum(pg_indexes_size(table_name::text)) from "
            "(select table_name from information_schema.tables "
            "where table_schema='public') as all_tables"
        )
        result = self.exec_fetch(statement)
        return result[0]

    def drop_database(self, database_name):
        statement = f"DROP DATABASE {database_name};"
        self.exec_only(statement)

        logging.info(f"Database {database_name} dropped")

    def create_statistics(self):
        logging.info("Postgres: Run `analyze`")
        self.commit()
        self._connection.autocommit = True
        self.exec_only("analyze")
        self._connection.autocommit = self.autocommit

    def set_random_seed(self, value=0.17):
        logging.info(f"Postgres: Set random seed `SELECT setseed({value})`")
        self.exec_only(f"SELECT setseed({value})")

    def supports_index_simulation(self):
        if self.db_system == "postgres":
            return True
        return False

    def _simulate_index(self, index):
        table_name = index.table()
        statement = (
            "select * from hypopg_create_index( "
            f"'create index on {table_name} "
            f"({index.joined_column_names()})')"
        )
        result = self.exec_fetch(statement)
        return result

    def _drop_simulated_index(self, oid):
        statement = f"select * from hypopg_drop_index({oid})"
        result = self.exec_fetch(statement)

        assert result[0] is True, f"Could not drop simulated index with oid = {oid}."

    def create_index(self, index):
        table_name = index.table()
        statement = (
            f"create index {index.index_idx()} "
            f"on {table_name} ({index.joined_column_names()})"
        )
        self.exec_only(statement)
        if "prt" in index.index_idx():
            size = self.exec_fetch(
                f"select relpages from pg_class c " f"where c.relname = '{index.index_idx()}'"
            )
            size_kb = self.exec_fetch(
                f"select pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size FROM pg_indexes WHERE indexname = '{index.index_idx()}'"
            )
            if size is not None and size[0] != 0:
                size = size[0]
                index.estimated_size = size * 8 * 1024
            else:
                if 'kB' in size_kb[0]:
                    size = float(size_kb[0].split('kB')[0].replace(" ",""))
                    index.estimated_size = size / 1024
                elif 'MB' in size_kb[0]:
                    size = float(size_kb[0].split('MB')[0].replace(" ",""))
                    index.estimated_size = size
                else:
                    assert()
        else:
            # 整列索引
            column_list = index._column_names()
            s = f"select pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size FROM pg_indexes WHERE indexname LIKE '%{table_name}%'"
            for column in column_list:
                s += f" AND indexname LIKE '%{column}%'"
            size_ls = self.exec_fetch(
                s, False
            )
            size_ls = sum([float(size[0].split('kB')[0].replace(" ",""))/1024 if 'kB' in size[0] else float(size[0].split('MB')[0].replace(" ","")) for size in size_ls])
            index.estimated_size = size_ls

    def drop_indexes(self):
        logging.info("Dropping indexes")
        stmt = "select indexname from pg_indexes where schemaname='public'"
        indexes = self.exec_fetch(stmt, one=False)
        for index in indexes:
            index_name = index[0]
            drop_stmt = "drop index {}".format(index_name)
            logging.debug("Dropping index {}".format(index_name))
            self.exec_only(drop_stmt)

    # PostgreSQL expects the timeout in milliseconds
    def exec_query(self, query, timeout=None, cost_evaluation=False):
        # Committing to not lose indexes after timeout
        if not cost_evaluation:
            self._connection.commit()
        query_text = self._prepare_query(query)
        if timeout:
            set_timeout = f"set statement_timeout={timeout}"
            self.exec_only(set_timeout)
        statement = f"explain (analyze, buffers, format json) {query_text}"
        try:
            plan = self.exec_fetch(statement, one=True)[0][0]["Plan"]
            result = plan["Actual Total Time"], plan
        except Exception as e:
            logging.error(f"{query.nr}, {e}")
            self._connection.rollback()
            result = None, self._get_plan(query)
        # Disable timeout
        self._cursor.execute("set statement_timeout = 0")
        self._cleanup_query(query)
        return result

    def exec_query_slalom(self, query, timeout=None, cost_evaluation=False):
        # Committing to not lose indexes after timeout
        if not cost_evaluation:
            self._connection.commit()
        query_text = self._prepare_query(query)
        if timeout:
            set_timeout = f"set statement_timeout={timeout}"
            self.exec_only(set_timeout)
        statement = f"explain (analyze, buffers, format json) {query_text}"
        try:
            plan = self.exec_fetch(statement, one=True)[0][0]["Plan"]
            result = plan["Actual Total Time"], plan
        except Exception as e:
            logging.error(f"{query.nr}, {e}")
            self._connection.rollback()
            result = None, self._get_plan(query)
        # Disable timeout
        self._cursor.execute("set statement_timeout = 0")
        self._cleanup_query(query)
        return result

    def _cleanup_query(self, query):
        for query_statement in query.text.split(";"):
            if "drop view" in query_statement:
                self.exec_only(query_statement)
                self.commit()

    def _get_cost(self, query):
        query_plan = self._get_plan(query)
        total_cost = query_plan["Total Cost"]
        return total_cost

    def _get_plan(self, query):
        query_text = self._prepare_query(query)
        statement = f"explain (format json) {query_text}"
        query_plan = self.exec_fetch(statement)[0][0]["Plan"]
        self._cleanup_query(query)
        return query_plan
    
    def _get_cost_slalom(self, query):
        query_plan = self._get_time(query)
        total_cost = query_plan["Total Cost"]
        return total_cost
    
    def _get_time(self,query):
        query_text = self._prepare_query(query)
        statement = f"explain (analyze, format json) {query_text}"
        query_plan = self.exec_fetch(statement)[0][0]["Plan"]
        self._cleanup_query(query)
        return query_plan

    def number_of_indexes(self):
        statement = """select count(*) from pg_indexes
                       where schemaname = 'public'"""
        result = self.exec_fetch(statement)
        return result[0]

    def table_exists(self, table_name):
        statement = f"""SELECT EXISTS (
            SELECT 1
            FROM pg_tables
            WHERE tablename = '{table_name}');"""
        result = self.exec_fetch(statement)
        return result[0]

    def database_exists(self, database_name):
        statement = f"""SELECT EXISTS (
            SELECT 1
            FROM pg_database
            WHERE datname = '{database_name}');"""
        result = self.exec_fetch(statement)
        return result[0]

    def get_index_build_cost(self, index):
        table_name = index.table()
        statement = (
            f"explain create index {index.index_idx()} "
            f"on {table_name} ({index.joined_column_names()})"
        )
        plan = self.exec_fetch(statement)[0][0]["Plan"]
        total_cost = plan["Actual Total Time"]
        return total_cost

    def get_data_from_page_tuple(self, partition_tuple, page_number = 20)->np.ndarray:
        statement = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' and table_name LIKE '%\_bak';"
        table_names = [tuple[0] for tuple in self.exec_fetch(statement, False)]
        tables = {}
        data_list = []
        start_block = partition_tuple[0]
        end_block = partition_tuple[1]
        for table_name in table_names:
            statement = f"SELECT attname AS first_column_name FROM pg_attribute WHERE attrelid = '{table_name}'::regclass AND attnum = 1;"
            primary_key = self.exec_fetch(statement, False)[0][0]
            tables[table_name] = primary_key
        for key, value in tables.items():
            s_max_value = f"select max({value}) from {key}"
            s_min_value = f"select min({value}) from {key}"
            max_value = self.exec_fetch(s_max_value)[0]
            min_value = self.exec_fetch(s_min_value)[0]
            value_range = list(np.linspace(min_value, max_value, page_number+1))
            value_range = [math.floor(v) for v in value_range]
            # deal with the range cannot be divided into partition_num fields
            value_range = list(set(value_range))
            while len(value_range) < page_number+1:
                value_range.append(value_range[-1]+1)
            value_range.sort()
            value_start = value_range[start_block]
            value_end = value_range[end_block]
            statement = f"select * from  {key} WHERE {value} between {value_start} and {value_end}"
            data = [list(tupe) for tupe in self.exec_fetch(statement, False)]
            data_list.append(data)
        row_num = max([len(data) for data in data_list])
        results = []
        for i in range(row_num):
            result = []
            for data in data_list:
                if i < len(data):
                    result += data[i]
            results.append(result)
        return np.array(results)
    
    def get_selectivity(self, partition_tuple, selected_queries, page_number = 20):
        statement = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' and table_name LIKE '%\_bak';"
        table_names = [tuple[0] for tuple in self.exec_fetch(statement, False)]
        tables = {}
        records = []
        data_list = []
        start_block = partition_tuple[0]
        end_block = partition_tuple[1]
        table_value = {}
        for table_name in table_names:
            statement = f"SELECT attname AS first_column_name FROM pg_attribute WHERE attrelid = '{table_name}'::regclass AND attnum = 1;"
            primary_key = self.exec_fetch(statement, False)[0][0]
            tables[table_name] = primary_key
        for key, value in tables.items():
            s_max_value = f"select max({value}) from {key}"
            s_min_value = f"select min({value}) from {key}"
            max_value = self.exec_fetch(s_max_value)[0]
            min_value = self.exec_fetch(s_min_value)[0]
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
            statement = f"select * from  {key} WHERE {value} between {value_start} and {value_end}"
            data = [list(tupe) for tupe in self.exec_fetch(statement, False)]
            records += data
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
            selected_records += [list(tupe) for tupe in self.exec_fetch(query_text, False)]
        # the number of tuple appear in both records of queries and records of the partition tuple
        intersection = len(selected_records)
        return np.array(data_list, dtype=object), intersection/len(records)
    
    def replace_select_with_star(self, sql):
        # make the content between SELECT and FROM become *
        pattern = r"(?i)(SELECT\s+)(.*?)(\s+FROM)"
        repl = r"\1*\3"
        replaced_sql = re.sub(pattern, repl, sql)
        replaced_sql = replaced_sql.replace("_1_prt_p0","")
        replaced_sql = replaced_sql.replace(";","")
        return replaced_sql

        
