# **FFT INFO APP (Fast Fourier Transform Info App)**

## **Basic information**:
This app allows to **predict** timeseries data, based on Fast Fourier Transform method in a simple manner. Also it allows to **check info** about current cycles in dataseries and allows to apply **backtest** mode to check, how good model is at predicting the latest data.

<img src="https://github.com/StaTixXSod/FFTInfoApp/examples_imgs/example1_predict.png" height="400"/>

The app has the following features:
- predict next values
- show info about current strongest cycles
- show timeseries memory data
- in backtest mode show score of the model
- find the best number of strongest cycles to use

### **Available module methods**
* `fit` - fit model on the given data. 
* `predict` - predict N values.
* `show_info` - shows info about current strongest cycles.
* `score` - shows MAPE error to check, how cycles fits the test data
* `show_best_parameters` - finds best count of cycles.
* `get_max_memory` - returns the max memory of the data.
* `auto_fit_predict` - automates the process of splitting, fitting and predicting the data in one line.

----------------------------------------------------------------------
## **1) Initialization (Module parameters)**
* `data`: `pd.Series` - is the timeseries data.
* `preprocessing`: `bool` - if data has trend, it is better to turn this parameter to 'True' to take trend into account.
* `use_last_data`: `float` or `int` - when there is too much data, you can use this parameter to use, for example, 50% of the latest data, if the parameter was set to 0.5. Also you can pass `int` value for this parameter to use certain value of latest data.
* `backtest`: `bool` - parameter, that allows you to split data into train and test series, then fit model on train data and check the score on the test data to decide, if model parameters were good or not and would it be good to trust such a model.
* `testing_interval`: `float` - if `backtest` parameter is 'True', you can pass certain split ratio to this parameter. Value 0.1 means, that the data will be splitted with ratio: 90% train / 10% test data.
* `is_daily`: `bool` - it's supposed, that most of the time the data passed in has 'Daily' time index. If you pass the data with different datetime index, you should turn this parameter to 'False' to use only index values (just the range from 0 to len(data + predicted data)).

Initialize model:
```
model = FFTModule(series)
```
Or
```
model = FFTModule(data=series,
                  preprocessing=True,
                  backtest=True,
                  testing_interval=0.1,
                  use_last_data=0.1
                  )
```

## **2) Fit**
The `fit` method performs Fast Fourier Transform on the given series. It extracts the following data:
- frequencies
- one-dimensional DFT
- Power Spectrum Density

To use methods like 'predict', 'show_info' etc. you should fit the data.
To fit the data just use:
```
model.fit()
```

## **3) Predict**

Predict next N values of given data, based on fitted FFT data. Available after `fit` method.

*Parameters:*

* *param* `n_predict`: `int`. Number of values to predict.
* *param* `show_plot`: `bool`. Use show_plot if you want to visualize prediction.
* *return*: `pd.Series`. Pandas Series with predicted values

To use `predict` method type:
```
model.predict(n_predict=365, show_plot=True)
```

<img src="https://github.com/StaTixXSod/FFTInfoApp/examples_imgs/examples_imgs/example1_predict.png" height="400">

## **4) Show info**
Shows info about strongest cycles. Available after 'fit' method.

*Parameters:*

*param* `n_cycles`: `int`. Show N most strongest cycles. 

*INFO:*

* **Cycles**: Shows the fluctuation period. E.g. the cycle - 365.5 tell us, that the data has 'yearly' seasonality (in case, of course, the data is DAILY). Calculates as (1 / frequency).
* **Power Spectral Density (PSD)**: Tell us, how strong the cycle is. Greater is better.
* **PSD Normalize**: It's just the (PSD value / sum(PSD value)). Normalized value tell us, how strong cycle is in relation to other cycles.

To show info type:
```
model.show_info(n_cycles=5)
```
It will return:

|     |  Cycles (Days/Bars)|  Power Spectral Density | PSD Normalize|
|:---:|      :---:         |        :----------:     |     :----:   |
|0|          365.500000    |           6836.545301   |    0.218629  |
|1|          182.750000    |           2252.476457   |    0.072033  |
|2|          121.833333    |            454.599665   |    0.014538  |
|3|           37.487179    |            370.364401   |    0.011844  |
|4|           91.375000    |            326.298110   |    0.010435  |

## **5) Score**
Calculates MAPE (Mean Absolute Percentage Error) value (less is better).
To calculate the score you should initiate the model with `backtest` parameter = 'True'. But, if you have your own test data, you can pass this data into `test_data` parameter.

*Parameters:*

* *param* `test_data`: `pd.Series` (Optional). Test data for score calculation and model checking.
* *param* `show_plot`: `bool`. Use show_plot if you want to visualize test data and predictions.
* *return*: `float`. Returns error value, based on MAPE.

To calculate the score and show predictions vs actual data type:
```
model.score(show_plot=True)
```
This will return something like that:

<img src="https://github.com/StaTixXSod/FFTInfoApp/examples_imgs/example1_score.png" height="400"/>

## **6) Show best parameters**
Finds the optimal number of cycles to use based on the minimum MAPE error they produce. There is a loop with range from 1 to 'picked number of cycles' and each time the MAPE score will be calculated to find best number of cycles to use.
Actually, it's better to check all good number of cycles and pick the best one manually, because sometimes the best count of cycles not always the best choice.

Returns Pandas DataFrame with number of cycles and corresponded errors.

*Parameters:*

* *param* `cycle_range`: `int`. Max number of cycles, that will be checked.
* *param* `step`: `int`. Range step.
* *return*: `pd.DataFrame`. DataFrame with number of cycles and their corresponding errors.

To show best parameters type: 
```
print(model.show_best_parameters())
```

|  |    Num cycles | MAPE Error |
|-:|:-------------:|:----------:|
|0 |          1    | 0.042459   |
|1 |         17    | 0.046907   |
|2 |         15    | 0.047330   |
|3 |         18    | 0.047409   |
|4 |         19    | 0.047794   |

## **7) Max memory**
The complex time series data, like stock market or something can changing over time. Current year data and data from last year may have different cycles. Therefore, sometimes it's important to get only a usefull chunk of data and fit the model only on them.

The method returns the max memory of the data. Max memory shows the number of samples of data, that can be more usefull, than others.

To find max memory type:
```
max_memory = model.get_max_memory(show_plot=True)
print("Max memory is: ", max_memory)
```
Return: Max memory is:  711

<img src="https://github.com/StaTixXSod/FFTInfoApp/examples_imgs/example3_memory.png" height="400"/>

## **8) Auto Fit Predict**
This method automates previous steps and allows to find max memory of data, fit the model on that data, find optimal number of cycles to use and plot predictions. Use 'backtest=True', when initializing the module, if you want to find best number of cycles 'best_n_cycles' or just set your own number to 'use_n_cycles'.

*Parameters:*

* *param* `n_predict`: `int`. The number of values to predict.
* *param* `use_n_cycle`: `int`. Use N most strongest cycles. 
* *param* `show_plot`: `bool`. Plot predictions

To use `auto_fit_prediction` type:
```
model = FFTModule(data=train_ser,
                  preprocessing=False,
                  backtest=True,
                  testing_interval=0.25
                  )

model.auto_fit_predict()
```
This will return:

<img src="https://github.com/StaTixXSod/FFTInfoApp/examples_imgs/example4_auto_fit_predict.png" height="400"/>


|  |    Num cycles | MAPE Error |
|-:|:-------------:|:----------:|
|0 |          6    | 0.099403   |
|1 |          4    | 0.099548   |
|2 |          5    | 0.099808   |
|3 |          7    | 0.100497   |
|4 |          3    | 0.102125   |

[INFO] Best N cycles:  6

----------------------------------------------------------------------
## Requirements
- numpy
- pandas
- matplotlib

## Installation:
```
- git clone 'repo'
- python -m venv venv
- ./venv/bin/activate
- pip install -r requirements.txt
```

