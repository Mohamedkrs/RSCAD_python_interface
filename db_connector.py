import csv
import os
import sqlite3
from sqlite3 import Error


class DBConnector:
    conn = None

    def __init__(self, database):
        self.DB = database
        self.create_connection()

    def create_connection(self, ):
        """ create a database connection to a SQLite database """
        try:
            self.conn = sqlite3.connect(self.DB)
        except Error as e:
            print(e)

    def create_table(self, create_table_sql):
        """ create a table from the create_table_sql statement
        :param conn: Connection object
        :param create_table_sql: a CREATE TABLE statement
        :return:
        """
        try:
            c = self.conn.cursor()
            c.execute(create_table_sql)
            self.conn.commit()
        except Error as e:
            print(e)

    def add_table(self, table_name, keys: dict = None):
        """ create a table from the create_table_sql statement
            Arguments:
                table_name: name of the table
                keys: row name and type {Pset:FLOAT}
                """
        key_string = ""
        for k, v in keys.items():
            key_string += k + " " + v + ","
        key_string = key_string[:-1]

        sql_create_projects_table = f""" CREATE TABLE IF NOT EXISTS {table_name} ({key_string}); """

        if self.conn is not None:
            # create projects table
            self.create_table(sql_create_projects_table)
        else:
            print("Error! cannot create the database connection.")

    def add_row(self, table_name, row_name, row_type):
        c = self.conn.cursor()
        c.execute(f"ALTER TABLE {table_name} ADD {row_name} {row_type} ")
        self.conn.commit()

    def add_data_to_table(self, table_name, values: list):
        c = self.conn.cursor()
        key_string = ""
        for val in values:
            if isinstance(val, str):
                key_string += f"'{val}',"
            else:
                key_string += f"{val},"
        key_string = key_string[:-1]
        c.execute(f"INSERT INTO {table_name} VALUES ({key_string})")
        self.conn.commit()

    def update_specific_field(self, table_name, key, key_value, filter_key, filter_key_value):
        c = self.conn.cursor()
        if isinstance(key_value, str):
            key_value = f"'{key_value}'"
        if isinstance(filter_key_value, str):
            filter_key_value = f"'{filter_key_value}'"
        c.execute(f"UPDATE '{table_name}' SET {key}={key_value} WHERE {filter_key}={filter_key_value}")
        self.conn.commit()

    def read_data(self, table_name, param=None):
        c = self.conn.cursor()

        if param:
            row = c.execute(f'SELECT * FROM {table_name}')
            return row
        else:
            c.execute(f'SELECT * FROM {table_name}')
            rows = c.fetchall()
            return rows

    def headers(self, table_name):
        c = self.conn.cursor()
        try:
            row = c.execute(f'SELECT * FROM {table_name}')
        except:
            print(f"No sch table {table_name}")
        names = list(map(lambda x: x[0], row.description))
        return names

    def delete_table(self, table_name):
        c = self.conn.cursor()
        c.execute(f"DROP TABLE IF EXISTS {table_name};")
        self.conn.commit()

    def clear_table(self, table_name):
        c = self.conn.cursor()
        c.execute(f"DELETE FROM {table_name};")
        self.conn.commit()

    def to_csv(self, table_name):
        c = self.conn.cursor()
        c.execute(f"select * from {table_name};")
        directory_path = os.getcwd()
        with open(f"{directory_path}\\{table_name}.csv", "w", newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([i[0] for i in c.description])
            csv_writer.writerows(c)