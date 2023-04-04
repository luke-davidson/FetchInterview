"""
Main model script.

author: Luke Davidson
        davidson.luked@gmail.com
        (978) 201-2693
"""

# Import dependencies
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Create main class
class Modeler():
    """
    Class to create an OLS regression model based on 2021 daily receipt scan data.
    """
    def __init__(self):
        """
        Init function.

        Creates:
            self.path ----------> file path; local script path location
            self.rawdata_df ----> pd.DataFrame; dataframe of the raw 2021 csv data
            self.model_params --> np.array; empty matrix to eventually store cross validation parameters and error results
        """
        self.path, _ = os.path.split(__file__)
        self.rawdata_df = pd.read_csv(os.path.join(self.path, "data_daily.csv"))
        self.model_params = np.empty((0, 3))

    def group_data_by_month(self):
        """
        Method to group the raw daily scan data by month

        Creates:
            self.month_df --> pd.DataFrame; DataFrame holding number of monthly receipt scans
        """
        self.rawdata_df["# Date"] = pd.to_datetime(self.rawdata_df["# Date"], format="%Y-%m-%d")
        self.rawdata_df['month'] = self.rawdata_df["# Date"].dt.month
        self.month_df = self.rawdata_df.groupby('month').sum()
        
    def fit(self, train, val):
        """
        Main method to fit training data to an OLS regression curve using the SSE function theoretical solution

        Args:
            train --> pd.DataFrame; holding 10 entries (if 6-fold cross val) of summed month data from self.month_df
            val ----> pd.DataFrame; holding the other 2 entries (if 6-fold cross val) of summed 2021 month data from self.month_df
        Concatenates:
            self.model_params --> np.array; 3 column np.array holding the 2 calculated regression parameters and their corresponding
                                            validation error
        """
        # Fit OLS regression line using [w] = (X.T @ X)^-1 @ X.T @ Y
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
    
    def run_kfold_crossval(self, k=6):
        """
        k-fold cross validation method. Splits the data in to 6 (k) folds, each with 10 training entires and 2 validation entries

        Args:
            k --> int; number of folds in the data
        Calls:
            self.fit() --> Fits a regression model to the training data, calculates the error of that model on the validation data
        """
        # Shuffle data
        self.month_df = self.month_df.sample(frac=1)

        # Split in to training and validation
        for t in range(0, 12, int(12/k)):
            all_months = list(range(12))
            val = self.month_df.iloc[[t, t+1], :]
            all_months.pop(t)
            all_months.pop(t)
            train = self.month_df.iloc[all_months, :]
            val = self.month_df.iloc[[t, t+1], :]

            # Fit a regression model to train, calculate error on val
            self.fit(train, val)

    def select_model(self):
        """
        Returns the model params with lowest cross validation error as the final model params
        """
        min_index = np.argmin(self.model_params[:, 2])
        self.final_model_params = self.model_params[min_index, :-1]

    def calc_offset(self):
        """
        Calculates the average offsets due to the varying number of days in a month.

        Returns:
            self.offsets --> np.array; array holding the average offsets to adjust the predictions
        """
        m, b = self.final_model_params
        self.offsets = np.zeros((3))
        for month in self.month_df.index:
            if month == 2:
                # Feb - 28 days
                self.offsets[0] += self.month_df.Receipt_Count[month] - (m*month + b)
            elif month in [4, 6, 9, 11]:
                # 30 days
                self.offsets[1] += self.month_df.Receipt_Count[month] - (m*month + b)
            else:
                # 31 days
                self.offsets[2] += self.month_df.Receipt_Count[month] - (m*month + b)
        self.offsets[1] /= 4
        self.offsets[2] /= 7

    def predict(self, date):
        """
        Main prediction method. Calculates a prediction based on the final model parameters and adjusts the prediction based on the number of days in the inputted month.

        Args:
            date --> int; Number of months since Jan 2021
        Calls:
            self.render_plot() --> renders a plot of the raw data, regression line, and future prediction
        Returns:
            self.pred --> int; Number of predicted receipts for the given date
        """
        m, b = self.final_model_params
        self.pred = m*date + b
        if date == 14:
            # Feb - 28 days
            self.pred += self.offsets[0]
        elif date in [16, 18, 21, 23]:
            # 30 days
            self.pred += self.offsets[1]
        else:
            # 31 days
            self.pred += self.offsets[2]
        encoded_img = self.render_plot(date)
        return int(self.pred), encoded_img

    def render_plot(self, date):
        """
        Plot rendering method. Renders and encodes a plot including the raw data, regression line, and future prediction
        """
        fig = plt.figure()
        plt.scatter(self.month_df.index, self.month_df.Receipt_Count, s=10)
        plt.plot([1, date], [self.final_model_params[0] + self.final_model_params[1], self.final_model_params[0]*date + self.final_model_params[1]], c='g')
        plt.scatter(date, self.pred, c='r', marker='x', s=50)
        plt.plot([date, date], [self.pred, min(self.month_df.Receipt_Count)], 'k--')
        all = ["Jan 21", "Feb 21", "Mar 21", "Apr 21", "May 21", "Jun 21", "Jul 21", "Aug 21", "Sep 21", "Oct 21", "Nov 21", "Dec 21", 
                "Jan 22", "Feb 22", "Mar 22", "Apr 22", "May 22", "Jun 22", "Jul 22", "Aug 22", "Sep 22", "Oct 22", "Nov 22", "Dec 22"]
        if date == 24:
            plt.xticks(np.arange(1, date+1), all, rotation=45)
        else:
            plt.xticks(np.arange(1, date+2), all[0:date+1], rotation=45)
        plt.ylabel("# Receipts Scanned per Month")
        plt.legend(["2021 Data", "Regression Line", "Prediction"])
        plt.grid()
        tmpfile = BytesIO()
        fig.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        return encoded

    def __call__(self):
        """
        Main call method. Creates the model via cross validation when class is called.
        """
        self.group_data_by_month()
        self.run_kfold_crossval()
        self.select_model()
        self.calc_offset()