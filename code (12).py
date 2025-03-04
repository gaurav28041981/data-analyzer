import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from itertools import product
from warnings import filterwarnings
from scipy.fft import fft
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import pandas as pd
from prophet import Prophet
from sklearn.cluster import KMeans, DBSCAN
from scipy.fft import rfft, rfftfreq
from datetime import datetime
import ephem
import requests
from io import StringIO
from scipy.stats import gaussian_kde

filterwarnings("ignore", category=ConvergenceWarning)

# ---Data Setup---
class DataAnalyzer:
    def __init__(self):
        self.base_url = "https://raw.githubusercontent.com/rahulraz17/data-analyzer/main/"

    def fetch_data_and_analyze(self, dataset_number):
        """Fetches data, analyzes it, and presents results."""
        url = f"{self.base_url}dataset{dataset_number}.txt" # dynamic URL loading
        try:
          response = requests.get(url) #http requsts
          response.raise_for_status()
          data = [int(x.strip()) for x in response.text.split('\n')] # split numbers based on line
        except Exception as e:
          print(f"Exception: Data Load has been failed, please check directory {e}")
          return

        """
        Data Processing and Analysis
        """
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data) #calculate value

        print(f"\nDataset {dataset_number} Statistics:")
        print(f"Mean: {mean:.2f}, Median: {median:.2f}, Std Dev: {std:.2f}")

        kde = gaussian_kde(data)
        x_range = np.linspace(min(data) - 10, max(data) + 10, 200)
        density = kde.evaluate(x_range) #generate density function

        #Extract the basic parameters
        most_likely_index = np.argmax(density) #most likely value
        most_likely_number = x_range[most_likely_index]
        least_likely_index = np.argmin(density) # least likely value
        least_likely_number = x_range[least_likely_index]

        #Make a self learning prediction for each data set
        print(f"Least likely number : {least_likely_number:.2f}")
        print(f"Most likely value  : {most_likely_number:.2f}")
        next_prediction = self.self_learning_prediction(data) #perform self learning
        print(f"Next prediction for {dataset_number}: {next_prediction:.2f}")

    def self_learning_prediction(self, data):
        """Attempts self-learning prediction using patterns in last data points or mean (default)."""
        if len(data) < 3:
            return np.mean(data)  # Not enough points for pattern
        
        diff = np.diff(data[-2:]) #check for pattern, can be updated to other models
        if diff > 0:
            return data[-1] + diff #make prediction based on pattern
        elif diff < 0:
            return data[-1] - np.abs(diff)
        else:
            return np.mean(data)

# ---Main Execution---
if __name__ == "__main__":
    analyzer = DataAnalyzer()
    # Fetch the data sets from URL and try to find patterns and models
    for i in range(6):
        analyzer.fetch_data_and_analyze(i+1)