from FFT import FFTModule
import pandas as pd


# AAPL Stock
train = pd.read_csv("data/stock_data/AAPL.csv", index_col=0, parse_dates=True)
train_ser = train['Close']

"""
[INFO] BACKTEST MODE
    When there is a need to check, how the model worked in the past, you can set 'backtest' argument to 'True'
and then set desired 'testing_interval'. The 'testing_interval' is a float number, that has to be between 0 and 1.
So the value of 0.25 splits data into train and test dataseries with the ratio of 90% to train data and 10% to test.
Also, when there is too much data, you can use 'use_last_data' parameter to use 10% of the latest data,
if the parameter was set to 0.1. You can pass also 'int' value for this parameter.
    If data is not stationary, it is better to use 'preprocessing' parameter
to take into account the trend of data.
"""

model = FFTModule(data=train_ser,
                  preprocessing=True,
                  backtest=True,
                  testing_interval=0.1,
                  use_last_data=0.1
                  )
model.fit()
print("[INFO] \n", model.show_best_parameters())
#     Num cycles  MAPE Error
# 0           1    0.042459
# 1          17    0.046907
# 2          15    0.047330
# 3          18    0.047409
# 4          19    0.047794

# Notice, that if 'backtest' parameter is 'True', the score will be calculated on splitted data
print("[INFO] The MAPE is: ", model.score(n_cycles=17, show_plot=True))
# [INFO] The MAPE is:  0.046906846973413574
