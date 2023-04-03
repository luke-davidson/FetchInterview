"""
PROBLEM STATEMENT: At fetch, we are monitoring the number of the scanned receipts to our app on a daily base as one of our KPIs. 
    From business standpoint, we sometimes need to predict the possible number of the scanned receipts for a given future month.

    The following link provides the number of the observed scanned receipts each day for the year 2021. Based on this prior 
        knowledge, please develop an algorithm which can predict the approximate number of the scanned receipts for each month
        of 2022.

author: Luke Davidson
        davidson.luked@gmail.com
        (978) 201-2693
"""

# Import libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Create main class
class Modeler():
    """
    
    """
    def __init__(self):
        """
        
        """
        self.rawdata_df = pd.read_csv(os.path.join(os.getcwd(), "data_daily.csv"))
        self.model_params = np.empty((0, 3))

    def group_data_by_month(self):
        """
        Sum self.rawdata_df.Receipt_Count by month
        """
        self.rawdata_df["# Date"] = pd.to_datetime(self.rawdata_df["# Date"], format="%Y-%m-%d")
        self.rawdata_df['month'] = self.rawdata_df["# Date"].dt.month
        self.month_df = self.rawdata_df.groupby('month').sum()
        
    def fit(self, train, val):
        """
        
        """
        # Fit OLS Linear regression line using Thetas = (X.T @ X)^-1 @ X.T @ Y
        x = np.concatenate((np.ones((10, 1)), train.index.to_numpy().reshape(10, 1)), axis=1)
        y = train.to_numpy()
        t0, t1 = np.linalg.inv(x.T @ x) @ x.T @ y

        # Calc validation error
        val_error = 0
        for i in val.index:
            act = val.Receipt_Count[i]
            pred = t1*i + t0
            val_error += (act - pred)**2

        # Store params and error
        self.model_params = np.concatenate((self.model_params, np.array([t1, t0, val_error]).reshape(1, 3)), axis=0)

    def predict(self, date):
        """
        
        """
        m, b = self.final_model_params
        self.pred = m*date + b
        # print(f"PRED: {self.pred} receipts.")
        self.plot_data(date)

    def plot_data(self, date):
        plt.scatter(self.month_df.index, self.month_df.Receipt_Count, s=10)
        plt.plot([1, date], [self.final_model_params[0] + self.final_model_params[1], self.final_model_params[0]*date + self.final_model_params[1]], c='g')
        plt.scatter(date, self.pred, c='r', marker='x', s=50)
        plt.plot([date, date], [self.pred, min(self.month_df.Receipt_Count)], 'k--')
        plt.xlabel("Month Number (# Months since Jan 2021)")
        plt.ylabel("# Receipts Scanned per Month")
        plt.title(f"PREDICTION: {int(self.pred)} Receipts to be Scanned in {date-12}/2022")
        plt.legend(["Data", "Regression Line", f"Prediction = {int(self.pred)} Receipts"])
        plt.grid()
        plt.show()
    
    def run_kfold_crossval(self, k=6):
        """
        Split data for k-fold cross validation
        """
        # Shuffle
        self.month_df = self.month_df.sample(frac=1)

        # Split
        for t in range(0, 12, int(12/k)):
            all_months = list(range(12))
            val = self.month_df.iloc[[t, t+1], :]
            all_months.pop(t)
            all_months.pop(t)
            train = self.month_df.iloc[all_months, :]
            val = self.month_df.iloc[[t, t+1], :]
            self.fit(train, val)

    def select_model(self):
        """
        Return model params with lowest cross val error as final model params
        """
        min_index = np.argmin(self.model_params[:, 2])
        self.final_model_params = self.model_params[min_index, :-1]

    def __call__(self):
        """
        
        """
        self.group_data_by_month()
        self.run_kfold_crossval()
        self.select_model()