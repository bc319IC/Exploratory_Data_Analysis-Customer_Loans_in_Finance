# Exploratory Data Analysis (EDA) - Customer Loans in Finance
End-to-End EDA Process of a dataset on customers loans.

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#Installation">Installation</a></li>
    <li><a href="#Usage">Usage</a></li>
    <li>
      <a href="#File-Structure">File Structure</a>
      <ul>
        <li>
          <a href="#customer_loans.ipynb">customer_loans.ipynb</a>
          <ul>
            <li><a href="#db_utils.py">db_utils.py</a></li>
            <li><a href="#db_plotter.py">db_plotter.py</a></li>
            <li><a href="#db_analyse.py">db_analyse.py</a></li>
          </ul>
        </li>
      </ul>
    </li>
    <li><a href="#License">License</a></li>
  </ol>
</details>

## Installation
Clone for local access.
```sh
git clone https://github.com/bc319IC/exploratory-data-analysis---customer-loans-in-finance316.git
```

## Usage
The Jupyter Notebook file `customer_loans.ipynb` has been pre-run so the entire Data Extraction 
and End-to-End EDA Process of the loan_payments dataset can be viewed.

## File Structure

### customer_loans.ipynb <a id="customer_loans.ipynb"></a>
This is the main Juypter Notebook file of this repository. This file will only be able to re-run
if valid credentials are supplied.

The EDA Process consists of converting columns to suitable types, imputing nulls, transforming skewed
columns, handling outliers, removing over-correlated columns, followed by visualisation and analysis
of the cleaned dataframe.

#### db_utils.py <a id="db_utils.py"></a>
Run db_utils.py to load the credentials from the yaml file, initialise and connect to the SQLAlchemy engine, 
extract the loan_repayments data to a database and then convert it to a csv file. Contains the the classes for transforming
the dataframe.

#### db_plotter.py <a id="db_plotter.py"></a>
Contains the class to visualise the results of the transformations carried out.

#### db_analyse.py <a id="db_analyse.py"></a>
Contains the class that visualises and analyses the cleaned dataframe.

## License
This project is licensed under the terms of the MIT license.