import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

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

    # Cumulative version of the plot
    #def plot_projected_charged_off_loss(self):
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
        charged_off_loans = self.identify_charged_off_loans()
        # Calculate the remaining term for each charged off loan
        charged_off_loans = charged_off_loans.copy()  # Ensure it's a copy
        charged_off_loans['remaining_term'] = charged_off_loans['term'] - (
            (pd.to_datetime(charged_off_loans['issue_date']) - pd.to_datetime(charged_off_loans['last_payment_date'])).dt.days // 30
        )
        # Ensure remaining_term is non-negative
        charged_off_loans['remaining_term'] = charged_off_loans['remaining_term'].apply(lambda x: max(x, 0))
        # Generate a DataFrame to plot
        charged_off_loans['remaining_principal'] = charged_off_loans.apply(
            lambda row: row['loan_amount'] * (row['remaining_term'] / row['term']) if row['term'] != 0 else 0, axis=1
        )
        charged_off_loans['monthly_interest'] = charged_off_loans['remaining_principal'] * (charged_off_loans['int_rate'] / 12 / 100)
        charged_off_loans['potential_interest_loss'] = charged_off_loans['monthly_interest'] * charged_off_loans['remaining_term']
        projected_losses = pd.DataFrame({
            'remaining_term': charged_off_loans['remaining_term'],
            'principal_loss': charged_off_loans['remaining_principal'],
            'interest_loss': charged_off_loans['potential_interest_loss']
        })
        projected_losses['total_loss'] = projected_losses['principal_loss'] + projected_losses['interest_loss']
        # Group by remaining_term to get total losses for each period
        loss_by_term = projected_losses.groupby('remaining_term').sum().reset_index()
        # Plot the total loss against the remaining term
        plt.figure(figsize=(12, 6))
        plt.bar(loss_by_term['remaining_term'], loss_by_term['total_loss'], color='red')
        plt.xlabel('Remaining Term (months)')
        plt.ylabel('Total Projected Loss ($)')
        plt.title('Projected Loss for Charged Off Loans Over Remaining Term')
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