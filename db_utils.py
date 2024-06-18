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
        col_name

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
        col_name

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
        col_name

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
        col_name

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
        col_name

        Returns
        -------
        None
        '''
        self.df[col_name] = self.df[col_name].astype(bool)

    def convert_term_to_numeric(self, col_name):
        '''
        Converts the column type to numeric by extracting only the digits part of a string.

        Parameters
        ----------
        col_name

        Returns
        -------
        None
        '''
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
        None
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
        None
        '''
        print("DataFrame described: \n", self.df.describe())
    
    def count_distinct_vals(self):
        '''
        Counts the distinct values in the categorical columns.

        Parameters
        ----------
        None

        Returns
        -------
        None
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
        None
        '''
        print("DataFrame shape: \n", self.df.shape)
    
    def count_null_vals(self):
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
        threshold - 0 to 100

        Returns
        -------
        None
        '''
        null_counts, null_percentage = self.count_null_vals()
        columns_to_drop = null_percentage[null_percentage > threshold].index
        self.df.drop(columns_to_drop, axis=1, inplace=True)
    
    def impute_null_cols(self, col_name, strategy):
        '''
        Imputes the null columns with mean or median or mode.

        Parameters
        ----------
        strategy - 'mean' or 'median' or 'mode'

        Returns
        -------
        None
        '''
        # Mean
        if strategy == 'mean':
            mean_value = self.df[col_name].mean()
            self.df[col_name].fillna(mean_value, inplace=True)
        # Median
        elif strategy == 'median':
            median_value = self.df[col_name].median()
            self.df[col_name].fillna(median_value, inplace=True)
        # Mode
        elif strategy == 'mode':
            mode_value = self.df[col_name].mode()[0]
            self.df[col_name].fillna(mode_value, inplace=True)
        else:
            raise ValueError("Invalid imputation strategy. Please choose 'mean' or 'median' or 'mode'.")
        
    def identify_skewed_cols(self, excluded_cols, threshold=0.5):
        '''
        Identifies which columns have skew above the threshold and to be considered 'skewed' columns.
        Prints the column names and skewness.

        Parameters
        ----------
        excluded_cols, threshold - 0 to 1

        Returns
        -------
        skewed_cols
        '''
        # Select only numerical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
        # Calculate skewness
        skewness = self.df[numeric_cols].skew()
        skewed_cols = skewness[abs(skewness) > threshold].index
        print(skewed_cols,'\n', skewness)
        return skewed_cols
    
    def transform_skewed_cols(self, col_name, transform='yeojohnson'):
        '''
        Transforms the column using the specified transform method.

        Parameters
        ----------
        col_name, transform - 'log' or 'boxcox' or 'yeojohnson'

        Returns
        -------
        None
        '''
        # Log
        if transform == 'log':
            self.df[col_name + '_log'] = self.df[col_name].map(lambda i: np.log(i) if i > 0 else 0)
        # Box-cox
        elif transform == 'boxcox':
            boxcoxed = self.df[col_name]
            boxcoxed = stats.boxcox(boxcoxed)
            self.df[col_name + '_boxcox'] = pd.Series(boxcoxed[0])
        # Yeo-Johnson
        elif transform == 'yeojohnson':
            yeojohnsoned = self.df[col_name]
            yeojohnsoned = stats.yeojohnson(yeojohnsoned)
            self.df[col_name + '_yeojohnson'] = pd.Series(yeojohnsoned[0])
        else:
            raise ValueError("Invalid transform. Please choose 'log' or 'boxcox' or 'yeojohnson'.")

    def remove_outliers(self, excluded_cols, threshold=3):
        '''
        Removes outliers from the dataframe above the z score threshold.

        Parameters
        ----------
        excluded_cols, threshold

        Returns
        -------
        None
        '''
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

    def get_highly_correlated_cols(self, excluded_cols, threshold=0.9):
        '''
        Gets the highly correlated columns.

        Parameters
        ----------
        excluded_cols, threshold - 0 to 1

        Returns
        -------
        to_drop
        '''
        # Select only numerical columns for the correlation matrix
        numerical_df = self.df.select_dtypes(include=[np.number])
        numerical_df = numerical_df.drop(columns=excluded_cols)
        # Compute the correlation matrix
        corr_matrix = numerical_df.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        # Find index of feature columns with correlation greater than the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        print(f"Columns to drop: {to_drop}")
        return to_drop

    def remove_highly_correlated_cols(self, excluded_cols, threshold=0.9):
        '''
        Removes the highly correlated columns.

        Parameters
        ----------
        excluded_cols, threshold - 0 to 1

        Returns
        -------
        None
        '''
        # Drop the columns returned from get_highly_correlated_cols
        to_drop = self.get_highly_correlated_cols(excluded_cols, threshold)
        self.df.drop(columns=to_drop, inplace=True)
        print(f"Removed columns: {to_drop}")

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

    def visualise_missing_vals(self):
        '''
        Visualise the nulls for each column in the dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
        plt.title('Nulls in DataFrame')
        plt.show()

    def visualise_skew(self, excluded_cols):
        '''
        Visualise the skew for each column in the dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in excluded_cols]

        sns.set_theme(font_scale=0.7)
        f = pd.melt(self.df, value_vars=numeric_cols)
        g = sns.FacetGrid(f, col="variable",  col_wrap=3, sharex=False, sharey=False)
        g = g.map(sns.histplot, "value", kde=True)

    def visualise_qq(self, col_name):
        '''
        Visualise the qq plot for a column.

        Parameters
        ----------
        col_name

        Returns
        -------
        None
        '''
        qq_plot = qqplot(self.df[col_name] , scale=1 ,line='q', fit=True)
        plt.show()
    
    def visualise_transformed(self, col_name, cols_w_zero):
        '''
        Visualise the skew for a column that has undergone log, box-cox and yeo-johnson transforms.

        Parameters
        ----------
        col_name, cols_w_zero

        Returns
        -------
        None
        '''
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
        '''
        Visualise the outliers for each column in the dataframe.

        Parameters
        ----------
        excluded_cols

        Returns
        -------
        None
        '''
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in excluded_cols]
        for col in numeric_cols:
            plt.figure(figsize=(4, 3))
            sns.boxplot(x=self.df[col])
            plt.title(f'Boxplot of {col}')
            plt.show()

    def plot_categorical_cols(self):
        '''
        Plots bar charts for each categorical type column in the dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Identify categorical columns
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns
        # Iterate over categorical columns and plot bar plots
        for column in categorical_columns:
            plt.figure(figsize=(4, 3))
            sns.countplot(data=self.df, x=column)
            plt.title(f'Bar plot of {column}')
            plt.xticks(rotation=90)
            plt.show()

    def visualise_correlation_matrix(self):
        '''
        Visualise the correlation matrix of the dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Select only numerical columns for the correlation matrix
        numerical_df = self.df.select_dtypes(include=[np.number])
        # Compute the correlation matrix
        corr_matrix = numerical_df.corr()
        # Plot the heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()


class LoanAnalysis:
    def __init__(self, df):
        '''
        Initialiases the class attributes

        Parameters
        ----------
        df: dataframe
            dataframe to be copied and used for analysis.
        total_funded_amount: float
            sum of the funded_amount column
        total_funded_amount_inv: float
            sum of funded_amount_inv column
        total_payment_received: float
            sum of total_payment column
        total_recoveries: float
            sum of recoveries column
        total_collection_recovery_fee: float
            sum of collection_recovery_fee column
        total_recovered: float
            sum of payments, recoveries, and collection recovery fee
        '''
        self.df = df.copy()
        self.total_funded_amount = self.df['funded_amount'].sum()
        self.total_funded_amount_inv = self.df['funded_amount_inv'].sum()
        self.total_payment_received = self.df['total_payment'].sum()
        self.total_recoveries = self.df['recoveries'].sum()
        self.total_collection_recovery_fee = self.df['collection_recovery_fee'].sum()
        self.total_recovered = self.total_payment_received + self.total_recoveries + self.total_collection_recovery_fee

    def calculate_recovery_percentages(self):
        '''
        Calculate the recovery percentage for the total funded amount and the amount funded by investors.

        Parameters
        ----------
        None

        Returns
        -------
        percent_recovered_total, percent_recovered_inv
        '''
        percent_recovered_total = (self.total_recovered / self.total_funded_amount) * 100
        percent_recovered_inv = (self.total_recovered / self.total_funded_amount_inv) * 100
        return percent_recovered_total, percent_recovered_inv

    def plot_recovery_percentages(self):
        '''
        Visualise the recovery percentages.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Get recovery percentages
        percent_recovered_total, percent_recovered_inv = self.calculate_recovery_percentages()
        # Plot the recovery percentages as a bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Total Funded Amount', 'Investor Funded Amount']
        percentages = [percent_recovered_total, percent_recovered_inv]
        ax.bar(categories, percentages, color=['blue', 'green'])
        ax.set_ylabel('Percentage Recovered (%)')
        ax.set_title('Percentage of Loans Recovered')
        for index, percent in enumerate(percentages):
            ax.text(index, percent, f"{percent:.2f}%", ha='center')

        plt.show()

    def estimate_future_recoveries(self, months=6):
        '''
        Estimate future recoveries in the next however months.

        Parameters
        ----------
        months - default as 6

        Returns
        -------
        cumulative_percentage_recovered
        '''
        # Assuming constant recovery rate
        term_total_months = self.df['term'].sum()
        monthly_recovery_rate = self.total_recovered / term_total_months
        estimated_future_recoveries = [monthly_recovery_rate * i for i in range(1, months + 1)]
        cumulative_recoveries = np.cumsum(estimated_future_recoveries)
        cumulative_percentage_recovered = (cumulative_recoveries / self.total_funded_amount) * 100
        return cumulative_percentage_recovered

    def plot_future_recoveries(self, months=6):
        '''
        Visualise future recoveries.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        cumulative_percentage_recovered = self.estimate_future_recoveries(months)
        # Plot the future recoveries
        fig, ax = plt.subplots(figsize=(10, 6))
        month_labels = [f'Month {i}' for i in range(1, months + 1)]
        ax.plot(month_labels, cumulative_percentage_recovered, marker='o', linestyle='-')
        ax.set_ylabel('Cumulative Percentage Recovered (%)')
        ax.set_title('Projected Cumulative Recoveries Over Next 6 Months')
        for index, percent in enumerate(cumulative_percentage_recovered):
            ax.text(index, percent, f"{percent:.10f}%", ha='center')

        plt.show()

    def calculate_charged_off_statistics(self):
        '''
        Calculate the percentage of charged off loans and the amount paid before charge off.

        Parameters
        ----------
        None

        Returns
        -------
        percent_charged_off, total_paid_before_charged_off
        '''
        charged_off_loans = self.df[self.df['loan_status'] == 'Charged Off']
        total_charged_off_loans = len(charged_off_loans)
        total_loans = len(self.df)
        percent_charged_off = (total_charged_off_loans / total_loans) * 100
        total_paid_before_charged_off = charged_off_loans['total_payment'].sum()
        return percent_charged_off, total_paid_before_charged_off

    def calculate_monthly_payment(self, loan_amount, annual_rate, term):
        '''
        Calculate the monthly payment.

        Parameters
        ----------
        loan_amount, annual_rate, term

        Returns
        -------
        payment
        '''
        monthly_rate = annual_rate / 12 / 100
        payment = loan_amount * monthly_rate * (1 + monthly_rate)**term / ((1 + monthly_rate)**term - 1)
        return payment

    def calculate_remaining_term(self):
        '''
        Add the remaining term column to the dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        # Function to calculate the remaining term based on amortisation schedule
        def remaining_term(row):
            '''
        Calculate the remaining term.

        Parameters
        ----------
        row

        Returns
        -------
        remaining term
        '''
            loan_amount = row['loan_amount']
            annual_rate = row['int_rate']
            term = row['term']
            total_payment_received = row['total_payment']
            # Get the monthly payment amount
            monthly_payment = self.calculate_monthly_payment(loan_amount, annual_rate, term)
            # Calculate the remaining balance after the received payments
            remaining_balance = loan_amount
            months_paid = 0
            while remaining_balance > 0 and months_paid < term:
                interest_payment = remaining_balance * (annual_rate / 12 / 100)
                principal_payment = monthly_payment - interest_payment
                remaining_balance -= principal_payment
                total_payment_received -= monthly_payment
                if total_payment_received < 0:
                    break
                months_paid += 1
            return term - months_paid
        self.df['remaining_term'] = self.df.apply(remaining_term, axis=1)

    def calculate_projected_loss(self, loans):
        '''
        Calculate the projected loss.

        Parameters
        ----------
        loans

        Returns
        -------
        total_principal_loss, total_potential_interest_loss, total_projected_loss
        '''
        loans = loans.copy()  # Create a copy to avoid SettingWithCopyWarning
        loans['remaining_principal'] = loans.apply(
            lambda row: row['loan_amount'] * (row['remaining_term'] / row['term']), axis=1
        )
        # Calculate interest loss
        loans['monthly_interest'] = loans['remaining_principal'] * (loans['int_rate'] / 12 / 100)
        loans['potential_interest_loss'] = loans['monthly_interest'] * loans['remaining_term']
        # Calculate total loss
        total_principal_loss = loans['remaining_principal'].sum()
        total_potential_interest_loss = loans['potential_interest_loss'].sum()
        total_projected_loss = total_principal_loss + total_potential_interest_loss
        return total_principal_loss, total_potential_interest_loss, total_projected_loss
    
    def analyse_charged_off_loans(self):
        '''
        Calculate the projected loss of the charged off loans.

        Parameters
        ----------
        None

        Returns
        -------
        total_principal_loss, total_potential_interest_loss, total_projected_loss
        '''
        self.calculate_remaining_term()
        charged_off_loans = self.df[self.df['loan_status'] == 'Charged Off']
        total_principal_loss, total_potential_interest_loss, total_projected_loss = self.calculate_projected_loss(charged_off_loans)
        return total_principal_loss, total_potential_interest_loss, total_projected_loss

    def plot_projected_charged_off_loss(self):
        '''
        Visualise the loss from the charged off loans.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.calculate_remaining_term()
        charged_off_loans = self.df[self.df['loan_status'] == 'Charged Off'].copy()
        # Calculate remaining principal
        charged_off_loans['remaining_principal'] = charged_off_loans.apply(
            lambda row: row['loan_amount'] * (row['remaining_term'] / row['term']), axis=1
        )
        # Calculate monthly interest and potential interest loss
        charged_off_loans['monthly_interest'] = charged_off_loans['remaining_principal'] * (charged_off_loans['int_rate'] / 12 / 100)
        charged_off_loans['potential_interest_loss'] = charged_off_loans['monthly_interest'] * charged_off_loans['remaining_term']
        # Calculate cumulative losses
        charged_off_loans['cumulative_principal_loss'] = charged_off_loans['remaining_principal'].cumsum()
        charged_off_loans['cumulative_potential_interest_loss'] = charged_off_loans['potential_interest_loss'].cumsum()
        charged_off_loans['cumulative_total_loss'] = charged_off_loans['cumulative_principal_loss'] + charged_off_loans['cumulative_potential_interest_loss']
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(charged_off_loans['remaining_term'], charged_off_loans['cumulative_total_loss'], marker='o', linestyle='-')
        ax.set_xlabel('Remaining Term (Months)')
        ax.set_ylabel('Cumulative Projected Loss ($)')
        ax.set_title('Projected Cumulative Loss Over Remaining Term of Charged Off Loans')
        plt.show()

    def identify_at_risk_loans(self):
        '''
        Identify at risk loans.

        Parameters
        ----------
        None

        Returns
        -------
        at_risk_loans
        '''
        at_risk_statuses = ['Late (31-120 days)', 'Late (16-30 days)', 'In Grace Period', 'Default']
        at_risk_loans = self.df[self.df['loan_status'].isin(at_risk_statuses)]
        return at_risk_loans

    def analyse_at_risk_loans(self):
        '''
        Calculate the percentage of loans at risk and their potential loss.

        Parameters
        ----------
        None

        Returns
        -------
        at_risk_percentage, total_principal_loss, total_potential_interest_loss, total_projected_loss
        '''
        # Percentage at risk
        at_risk_loans = self.identify_at_risk_loans()
        total_loans = len(self.df)
        at_risk_loan_count = len(at_risk_loans)
        at_risk_percentage = (at_risk_loan_count / total_loans) * 100
        # Potential loss
        self.calculate_remaining_term()
        at_risk_loans = self.df[self.df['loan_status'].isin(['Late (31-120 days)', 'Late (16-30 days)', 'In Grace Period', 'Default'])]
        total_principal_loss, total_potential_interest_loss, total_projected_loss = self.calculate_projected_loss(at_risk_loans)
        return at_risk_percentage, total_principal_loss, total_potential_interest_loss, total_projected_loss
    
    def calculate_total_expected_revenue(self):
        '''
        Calculate the total expected revenue from loans.

        Parameters
        ----------
        None

        Returns
        -------
        total_expected_revenue
        '''
        self.calculate_remaining_term()
        self.df['total_expected_payment'] = self.df.apply(
            lambda row: self.calculate_monthly_payment(row['loan_amount'], row['int_rate'], row['term']) * row['term'], axis=1
        )
        total_expected_revenue = self.df['total_expected_payment'].sum()
        return total_expected_revenue
    
    def analyse_combined_loss(self):
        '''
        Calculate the percentage and total loss of the charged off and at risk loans.

        Parameters
        ----------
        None

        Returns
        -------
        combined_loss_percentage, combined_projected_loss
        '''
        at_risk_percentage, at_risk_principal_loss, at_risk_interest_loss, at_risk_projected_loss = self.analyse_at_risk_loans()
        charged_off_loans = self.df[self.df['loan_status'] == 'Charged Off']
        charged_off_principal_loss, charged_off_interest_loss, charged_off_projected_loss = self.calculate_projected_loss(charged_off_loans)
        combined_projected_loss = at_risk_projected_loss + charged_off_projected_loss
        total_expected_revenue = self.calculate_total_expected_revenue()
        combined_loss_percentage = (combined_projected_loss / total_expected_revenue) * 100
        return combined_loss_percentage, combined_projected_loss
    
    def identify_charged_off_loans(self):
        '''
        Identify the charged off loans.

        Parameters
        ----------
        None

        Returns
        -------
        charged_off_loans
        '''
        charged_off_loans = self.df[self.df['loan_status'] == 'Charged Off']
        return charged_off_loans

    def analyse_loan_indicators(self):
        '''
        Visualise potential indicators of laons to become charged off.

        Parameters
        ----------
        None

        Returns
        -------
        summary
        '''
        at_risk_loans = self.identify_at_risk_loans()
        charged_off_loans = self.identify_charged_off_loans()

        # Combine both subsets for comparison
        at_risk_loans = at_risk_loans.copy()
        at_risk_loans.loc[:, 'status'] = 'At Risk'

        charged_off_loans = charged_off_loans.copy()
        charged_off_loans.loc[:, 'status'] = 'Charged Off'

        combined = pd.concat([at_risk_loans, charged_off_loans])

        # Visualize indicators
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18))

        # Grade
        sns.countplot(data=combined, x='grade', hue='status', ax=axes[0])
        axes[0].set_title('Loan Grade Distribution')
        axes[0].set_xlabel('Grade')
        axes[0].set_ylabel('Count')

        # Purpose
        sns.countplot(data=combined, x='purpose', hue='status', ax=axes[1])
        axes[1].set_title('Loan Purpose Distribution')
        axes[1].set_xlabel('Purpose')
        axes[1].set_ylabel('Count')
        axes[1].tick_params(axis='x', rotation=45)

        # Home Ownership
        sns.countplot(data=combined, x='home_ownership', hue='status', ax=axes[2])
        axes[2].set_title('Home Ownership Distribution')
        axes[2].set_xlabel('Home Ownership')
        axes[2].set_ylabel('Count')

        plt.tight_layout()
        plt.show()

        # Summary of findings
        summary = combined.groupby(['status', 'grade', 'purpose', 'home_ownership']).size().unstack(fill_value=0)
        return summary


if __name__ == '__main__':
    #Save the CVS file from RDS Database
    creds = load_creds()
    rds = RDSDatabaseConnector(creds)
    engine = rds.initialise_engine()
    df_temp = rds.extract_data_to_dataframe('loan_payments')
    rds.save_dataframe_to_csv(df_temp, 'loan_payments.csv')
