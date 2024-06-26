import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from scipy import stats
from scipy.special import boxcox1p

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