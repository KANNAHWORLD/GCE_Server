import psycopg2
from typing import List, Dict
__package__ = 'PostGresQueryGenerator'

class PGQuery:
    def __init__(self):
        self.query = []

    def __str__(self):
        return self.query_string()
    
    def query_string(self):
        return ' '.join(self.query) + ';'

    def login(self, login: Dict[str, str]):
        if login is None:
            return False
        
        self.connection = psycopg2.connect(**login)
        return True

    def execute_nofetch(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute(' '.join(self.query) + ';')
            cursor.close()
        except Exception as e:
            print(e)
            self.connection.rollback()
        
        self.query = []

    def execute_fetch(self):
        try:
            cursor = self.connection.cursor()
            cursor.execute(' '.join(self.query) + ';')
            data = cursor.fetchall()
            cursor.close()
        except Exception as e:
            print(e)
            self.connection.rollback()
        
        self.query = []
        return data

    def commit(self):
        self.connection.commit()
        return self

    def rollback(self):
        self.connection.rollback()
        return self

    def toggleAutoCommit(self):
        self.connection.autocommit = not self.connection.autocommit

    def clear(self):
        self.query = []
        return self

    def SELECT(self, columns: List[str]):
        self.query.append(f"SELECT {', '.join(columns)}")
        return self
    
    def WITH(self, table):
        self.query.append(f"WITH {table}")
        return self

    def LEFT_JOIN(self, table_name: str):
        self.query.append(f"LEFT JOIN {table_name}")
        return self

    def RIGHT_JOIN(self, table_name: str):
        self.query.append(f"RIGHT JOIN {table_name}")
        return self

    def FULL_OUTER_JOIN(self, table_name: str):
        self.query.append(f"FULL OUTER JOIN {table_name}")
        return self

    def HAVING(self, condition: str):
        self.query.append(f"HAVING {condition}")
        return self

    def ON(self, condition: str):
        self.query.append(f"ON {condition}")
        return self

    def NOT(self):
        self.query.append("NOT")
        return self

    def UNION(self):
        self.query.append("UNION")
        return self

    def FROM(self, table_names: List[str]):
        self.query.append(f"FROM {', '.join(table_names)}")
        return self

    def WHERE(self, condition: str):
        self.query.append(f"WHERE {condition}")
        return self

    def AND(self, condition: str):
        self.query.append(f"AND {condition}")
        return self
    
    def OR(self, condition: str):
        self.query.append(f"OR {condition}")
        return self
    
    def INSERT_INTO(self, table_name: str, columns: List[str]):
        self.query.append(f"INSERT INTO {table_name} ({', '.join(columns)})")
        return self

    def VALUES(self, values: List[List[str]]):
        values = [f"({', '.join(value)})" for value in values]
        self.query.append(f"VALUES {', '.join(values)}")
        return self
    
    def ORDER_BY(self, columns: List[str]):
        self.query.append(f"ORDER BY {', '.join(columns)}")
        return self
    
    def GROUP_BY(self, columns: List[str]):
        self.query.append(f"GROUP BY {', '.join(columns)}")
        return self

    def LIMIT(self, limit: int):
        self.query.append(f"LIMIT {limit}")
        return self

    def CREATE_TABLE(self, table_name: str, columns: List[str]):
        self.query.append(f"CREATE TABLE {table_name} ({', '.join(columns)})")
        return self

    def DROP_TABLE(self, table_names: List[str] = []):
        self.query.append(f"DROP TABLE {', '.join(table_names)}")
        return self

    def CREATE_DATABASE(self, database_name: str):
        self.query.append(f"CREATE DATABASE {database_name}")
        return self

    def DROP_DATABASE(self, database_name: str = ''):
        self.query.append(f"DROP DATABASE {database_name}")
        return self

    def IF_EXISTS(self, string: str = ''):
        self.query.append(f"IF EXISTS {string}")
        return self

    def CREATE_EXTENSTION(self, extension: str):
        self.query.append(f"CREATE EXTENSION {extension}")
        return self

    def P(self):
        self.query.append('(')
        return self
    
    def EP(self):
        self.query.append(')')
        return self

    def toVector(list: List[float]):
        return '\'[' + ','.join(map(str, list)) + ']\''

    def toString(string: str):
        return "'{}'".format(string.replace("'", ''))
    
    def toInt(num: int):
        return str(num)
