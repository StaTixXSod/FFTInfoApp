import pandas as pd
from FFT import FFTModule

train_data = pd.read_csv('data/climate_TS_data/DailyDelhiClimateTrain.csv', index_col=0, parse_dates=True)
test_data = pd.read_csv('data/climate_TS_data/DailyDelhiClimateTest.csv', index_col=0, parse_dates=True)
train_ser = train_data['meantemp']
test_ser = test_data['meantemp']

"""
[INFO] THE AUTOMATIC METHOD
1. Init model
2. Use 'auto_fit_predict'!
Notice, if you want to check best n_cycles number, you must set 'backtest' parameter to 'True'.
Or you can just set your own number in 'use_n_cycles' parameter.
"""

model = FFTModule(data=train_ser,
                  preprocessing=False,
                  backtest=True,
                  testing_interval=0.25
                  )

model.auto_fit_predict() # Also can use 'model.auto_fit_predict(use_n_cycles=4)'
# [INFO] ERRORS:
#     Num cycles  MAPE Error
# 0           6    0.099403
# 1           4    0.099548
# 2           5    0.099808
# 3           7    0.100497
# 4           3    0.102125
# [INFO] Best N cycles:  6

