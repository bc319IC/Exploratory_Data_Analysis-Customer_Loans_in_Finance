import yaml
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from scipy import stats
from scipy.special import boxcox1p

class RDSDatabaseConnector():

    def __init__(self, creds):
        '''
        Initialiases the class attributes

        Parameters
        ----------
        creds: dictionary
            a dictionary containing the necessary credentials to initialise the SQLAlchemy engine.
        '''
        self.creds = creds

    def initialise_engine(self):
        '''
        Initialises the SQLAlechemy engine.

        Parameters
        ----------
        None

        Returns
        -------
        engine
        '''
        # Construct a database URL for SQLAlchemy
        db_url = f"{'postgresql'}://{self.creds['RDS_USER']}:{self.creds['RDS_PASSWORD']}@{self.creds['RDS_HOST']}:{self.creds['RDS_PORT']}/{self.creds['RDS_DATABASE']}"
        # Create SQLAlchemy engine
        engine = create_engine(db_url)
        return engine
    
    def extract_data_to_dataframe(self, table_name):
        '''
        Extracts data from the sql table as a dataframe.

        Parameters
        ----------
        table_name

        Returns
        -------
        df
        '''
        # Initialise SQLAlchemy engine
        engine = self.initialise_engine()
        # Read data from the specified table into a dataframe
        query = f"SELECT * FROM {table_name};"
        df = pd.read_sql(query, engine)
        return df
   
    @staticmethod
    def save_dataframe_to_csv(df, file_path):
        '''
        Saves the dataframe as a CSV.

        Parameters
        ----------
        df, file_path

        Returns
        -------
        None
        '''
        df.to_csv(file_path, index=False)


def load_creds():
    '''
        Saves the credentials from a yaml file as a dictionary.

        Parameters
        ----------
        None

        Returns
        -------
        creds
        '''
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
    '''
        Loads data from a CSV file into a dataframe.

        Parameters
        ----------
        file_path

        Returns
        -------
        df
        '''
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading data to DataFrame: {e}")
        return None
    

class DataTransform():

    def __init__(self,df):
        '''
        Initialiases the class attributes

        Parameters
        ----------
        df: dataframe
            dataframe for column types to be updated.
        '''
        self.df = df

    def convert_to_datetime(self, col_name):
        '''
        Converts the column type to datetime.

        Parameters
        ----------
        column_name

        Returns
        -------
        None
        '''
        self.df[col_name] = pd.to_datetime(self.df[col_name], format='%b-%Y')
    
    def convert_to_float(self, col_name):
        '''
        Converts the column type to float.

        Parameters
        ----------
        column_name

        Returns
        -------
        None
        '''
        self.df[col_name] = pd.to_numeric(self.df[col_name], errors='coerce').astype(float)
    
    def convert_to_int(self, col_name):
        '''
        Converts the column type to integer.

        Parameters
        ----------
        column_name

        Returns
        -------
        None
        '''
        self.df[col_name] = pd.to_numeric(self.df[col_name], errors='coerce').astype(int)
    
    def convert_to_category(self, col_name):
        '''
        Converts the column type to category.

        Parameters
        ----------
        column_name

        Returns
        -------
        None
        '''
        self.df[col_name] = self.df[col_name].astype('category')

    def convert_to_bool(self, col_name):
        '''
        Converts the column type to boolean.

        Parameters
        ----------
        column_name

        Returns
        -------
        None
        '''
        self.df[col_name] = self.df[col_name].astype(bool)

    def convert_term_to_numeric(self, col_name):
        self.df[col_name] = self.df[col_name].str.extract('(\d+)')


class DataFrameInfo():

    def __init__(self, df):
        '''
        Initialiases the class attributes

        Parameters
        ----------
        df: dataframe
            dataframe to be analysed.
        '''
        self.df = df
    
    def d_types(self):
        '''
        Finds the data type of all columns.

        Parameters
        ----------
        None

        Returns
        -------
        df.dtypes
        '''
        print("DataFrame dtypes: \n", self.df.dtypes)
    
    def described(self):
        '''
        Describes the dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        df.describe()
        '''
        print("DataFrame described: \n", self.df.describe())
    
    def count_distinct_values(self):
        '''
        Counts the distinct values in the categorical columns.

        Parameters
        ----------
        None

        Returns
        -------
        distinct_vals
        '''
        distinct_vals = {}
        for col in self.df.select_dtypes(include=['category']):
            distinct_vals[col] = self.df[col].nunique()
        print("DataFrame distinct values: \n", distinct_vals)
    
    def shape(self):
        '''
        Returns the shape of the dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        shape
        '''
        print("DataFrame shape: \n", self.df.shape)
    
    def count_null_values(self):
        '''
        Finds the number and percentage of nulls in the dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        null_counts, null_percentage
        '''
        null_counts = self.df.isnull().sum()
        null_percentage = null_counts / len(self.df) * 100
        print("DataFrame nulls: \n", null_counts, "DataFrame nulls as %: \n", null_percentage)
        return null_counts, null_percentage


class DataFrameTransform(DataFrameInfo):

    def __init__(self, df):
        '''
        Initialiases the class attributes

        Parameters
        ----------
        df: dataframe
            dataframe to be transformed.
        '''
        super().__init__(df)

    def drop_null_cols(self, threshold=50):
        '''
        Drops the null columns with more than half null entries.

        Parameters
        ----------
        threshold - 0 to 1

        Returns
        -------
        None
        '''
        null_counts, null_percentage = self.count_null_values()
        columns_to_drop = null_percentage[null_percentage > threshold].index
        self.df.drop(columns_to_drop, axis=1, inplace=True)
    
    def impute_null_cols(self, col_name, strategy):
        '''
        Imputes the null columns with mean or median or mode.

        Parameters
        ----------
        strategy - mean or median or mode

        Returns
        -------
        None
        '''
        if strategy == 'mean':
            mean_value = self.df[col_name].mean()
            self.df[col_name].fillna(mean_value, inplace=True)
        elif strategy == 'median':
            median_value = self.df[col_name].median()
            self.df[col_name].fillna(median_value, inplace=True)
        elif strategy == 'mode':
            mode_value = self.df[col_name].mode()[0]
            self.df[col_name].fillna(mode_value, inplace=True)
        else:
            raise ValueError("Invalid imputation strategy. Please choose 'mean' or 'median' or 'mode'.")
        
    def identify_skewed_columns(self, excluded_cols, threshold=0.5):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
        skewness = self.df[numeric_cols].skew()
        skewed_cols = skewness[abs(skewness) > threshold].index
        print(skewed_cols,'\n', skewness)
        return skewed_cols
    
    def transform_skewed_columns(self, col_name, transform='yeojohnson'):
        if transform == 'log':
            self.df[col_name + '_log'] = self.df[col_name].map(lambda i: np.log(i) if i > 0 else 0)
        elif transform == 'boxcox':
            boxcoxed = self.df[col_name]
            boxcoxed = stats.boxcox(boxcoxed)
            self.df[col_name + '_boxcox'] = pd.Series(boxcoxed[0])
        elif transform == 'yeojohnson':
            yeojohnsoned = self.df[col_name]
            yeojohnsoned = stats.yeojohnson(yeojohnsoned)
            self.df[col_name + '_yeojohnson'] = pd.Series(yeojohnsoned[0])
        else:
            raise ValueError("Invalid transform. Please choose 'log' or 'boxcox' or 'yeojohnson'.")

    def remove_outliers(self, excluded_cols, threshold=3):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
        outlier_indices = set()
        for col in numeric_cols:
            # Calculate z-scores
            z_scores = (self.df[col] - self.df[col].mean()) / self.df[col].std()
            # Find rows with z-scores above threshold
            col_outlier_indices = z_scores[abs(z_scores) > threshold].index
            outlier_indices.update(col_outlier_indices.tolist())
        outlier_indices = list(outlier_indices)
        # Remove rows with z-scores above threshold
        self.df = self.df.drop(index=outlier_indices, axis=0, inplace=True)


class Plotter():

    def __init__(self, df):
        '''
        Initialiases the class attributes

        Parameters
        ----------
        df: dataframe
            dataframe to be visualised.
        '''
        self.df = df

    def visualise_missing_values(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
        plt.title('Nulls in DataFrame')
        plt.show()

    def visualise_skew(self, excluded_cols):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in excluded_cols]

        sns.set_theme(font_scale=0.7)
        f = pd.melt(self.df, value_vars=numeric_cols)
        g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", kde=True)

    def visualise_qq(self, col_name):
        qq_plot = qqplot(self.df[col_name] , scale=1 ,line='q', fit=True)
        plt.show()
    
    def visualise_transformed(self, col_name, cols_w_zero):
            plt.figure(figsize=(12, 4))
            # Log
            plt.subplot(1, 3, 1)
            logged = self.df[col_name].map(lambda i: np.log(i) if i > 0 else 0)
            l = sns.histplot(logged,label="Skewness: %.2f"%(logged.skew()) )
            l.legend()
            plt.title('log ' + col_name)
            # Box-cox
            plt.subplot(1, 3, 2)
            if col_name not in cols_w_zero: #remove columns with zeros
                boxcoxed = self.df[col_name]
                boxcoxed = stats.boxcox(boxcoxed)
                boxcoxed = pd.Series(boxcoxed[0])
                bc=sns.histplot(boxcoxed,label="Skewness: %.2f"%(boxcoxed.skew()) )
                bc.legend()
                plt.title('bc ' + col_name)
            # Yeo-Johnson
            plt.subplot(1, 3, 3)
            yeojohnsoned = self.df[col_name]
            yeojohnsoned = stats.yeojohnson(yeojohnsoned)
            yeojohnsoned = pd.Series(yeojohnsoned[0])
            yj=sns.histplot(yeojohnsoned,label="Skewness: %.2f"%(yeojohnsoned.skew()) )
            yj.legend()
            plt.title('yj ' + col_name)

            plt.tight_layout()
            plt.show()

    def visualise_outliers(self, excluded_cols):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
        for col in numeric_cols:
            plt.figure(figsize=(4, 3))
            sns.boxplot(x=self.df[col])
            plt.title(f'Boxplot of {col}')
            plt.show()

if __name__ == '__main__':
    #Save the CVS file from RDS Database
    """
    creds = load_creds()
    rds = RDSDatabaseConnector(creds)
    engine = rds.initialise_engine()
    df_temp = rds.extract_data_to_dataframe('loan_payments')
    rds.save_dataframe_to_csv(df_temp, 'loan_payments.csv')
    """