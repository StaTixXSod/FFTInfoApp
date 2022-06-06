import pandas as pd
from FFT import FFTModule

train_data = pd.read_csv('data/climate_TS_data/DailyDelhiClimateTrain.csv', index_col=0, parse_dates=True)
test_data = pd.read_csv('data/climate_TS_data/DailyDelhiClimateTest.csv', index_col=0, parse_dates=True)
train_ser = train_data['meantemp']
test_ser = test_data['meantemp']

## [INFO] THE MANUAL METHOD
# 1. Init and Fit model
# 2. Check INFO about cycles (OPTIONAL)
# 3. Predict values

model = FFTModule(train_ser)
model.fit()
model.show_info(n_cycles=5)
# =========== INFO ABOUT STRONGEST CYCLES ===========
#    Cycles (Days/Bars)  Power Spectral Density  PSD Normalize
# 0          365.500000             6836.545301       0.218629
# 1          182.750000             2252.476457       0.072033
# 2          121.833333              454.599665       0.014538
# 3           37.487179              370.364401       0.011844
# 4           91.375000              326.298110       0.010435

model.predict(n_predict=365, n_cycles=3, show_plot=True)
model.score(test_ser, show_plot=True)
