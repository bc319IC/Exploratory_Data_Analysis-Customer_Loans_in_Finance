import yaml
from sqlalchemy import create_engine
import psycopg2
import pandas as pd

class RDSDatabaseConnector():

    def __init__(self, creds):
        self.creds = creds

    def initialise_engine(self):
        # Construct database URL for SQLAlchemy
        db_url = f"{'postgresql'}://{self.creds['RDS_USER']}:{self.creds['RDS_PASSWORD']}@{self.creds['RDS_HOST']}:{self.creds['RDS_PORT']}/{self.creds['RDS_DATABASE']}"
        # Create SQLAlchemy engine
        engine = create_engine(db_url)
        return engine
    
    def extract_data_to_dataframe(self, table_name):
        # Initialise SQLAlchemy engine
        engine = self.initialise_engine()
        # Read data from the specified table into a DataFrame
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql(query, engine)
        return df
   
    @staticmethod
    def save_dataframe_to_csv(df, file_path):
        df.to_csv(file_path, index=False)


def load_creds():
    file_path = "credentials.yaml"
    try:
        with open(file_path, 'r') as file:
            creds = yaml.safe_load(file)
        return creds
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error loading YAML file: {e}")
        return None
    
def load_data_to_dataframe(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data to DataFrame: {e}")
        return None
    

if __name__ == '__main__':
    creds = load_creds()
    rds = RDSDatabaseConnector(creds)
    engine = rds.initialise_engine()
    df_temp = rds.extract_data_to_dataframe('loan_payments')
    rds.save_dataframe_to_csv(df_temp, 'loan_payments.csv')