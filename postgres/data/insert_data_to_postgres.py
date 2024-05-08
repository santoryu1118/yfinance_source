import csv
import os
import time
import psycopg2


def connect_to_db(dbname, user, password, host, port):
    """Establishes a connection to the PostgreSQL database and returns the connection and cursor."""
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    cur = conn.cursor()
    return conn, cur


def insert_row(cur, table_name, row):
    """Inserts a single row into the specified table dynamically based on the CSV column headers."""
    # Extracting column names and values from the row
    columns = list(row.keys())
    values = [row[column] for column in columns]

    # Constructing the SQL INSERT statement dynamically
    # Format column names for SQL syntax
    # columns_formatted = ', '.join([f'"{column}"' for column in columns])
    columns_formatted = ', '.join(columns)
    placeholders = ', '.join(['%s'] * len(values))  # Create placeholders for values
    sql = f"INSERT INTO {table_name} ({columns_formatted}) VALUES ({placeholders});"
    print(values)
    cur.execute(sql, values)


def process_csv(csv_file_path, table_name, interval_sec, conn, cur):
    """Reads a CSV file and inserts each row into the specified PostgreSQL table with a delay."""
    with open(csv_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            insert_row(cur, table_name, row)
            conn.commit()  # Commit after each insert
            time.sleep(interval_sec)  # Wait for 10 seconds before the next insert


def main():
    # PostgreSQL connection parameters
    dbname = 'postgres'
    user = 'postgres'
    password = 'postgres'
    host = 'localhost'  # Use the IP address of the Docker host or localhost if it's exposed
    port = '5432'  # The port your PostgreSQL is exposed on
    table_name = 'yfinance'
    interval_sec = 0

    # Construct file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, 'stream_yfdata.csv')

    # Connect to the database
    conn, cur = connect_to_db(dbname, user, password, host, port)

    try:
        # Process the CSV file and insert data into the database
        process_csv(csv_file_path, table_name, interval_sec, conn, cur)
    finally:
        # Close the database connection
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
