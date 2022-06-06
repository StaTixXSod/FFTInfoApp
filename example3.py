import pandas as pd
from FFT import FFTModule


train_data = pd.read_csv('data/stock_data/AAPL.csv', index_col=0, parse_dates=True)
train_ser = train_data['Close']

"""
[INFO] MAX MEMORY OF DATA
    Not all data has usefull information. Time series data changes over time and if you pass in many data,
there are chances that identified info will be too mixed and useless. To get maximum usefull data range
you can use 'get_max_memory' method.

1. Init model
2. Check timeseries memory
3. Again init module and set max_memory to 'use_last_data' parameter
4. Fit model
5. Predict values
"""
model = FFTModule(data=train_ser)

max_memory = model.get_max_memory(show_plot=True)
print("[INFO] Max memory is: ", max_memory)
# [INFO] Max memory is:  711

model = FFTModule(data=train_ser,
                  use_last_data=max_memory
                  )
model.fit()
model.predict(n_predict=180, show_plot=True)
