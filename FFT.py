from typing import Any
from exceptions import *
import numpy as np
import pandas as pd
from numpy import fft
import matplotlib.pyplot as plt
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (8, 5)


class FFTModule:
    def __init__(self,
                 data: pd.Series,
                 preprocessing: bool = False,
                 use_last_data: float = None,
                 backtest: bool = False,
                 testing_interval: float = 0.1,
                 is_daily: bool = True) -> None:
        """
        This module performs FFT and allows to get prediction about the given data.

        Example:
        ________
        1. Pick the best cycle \n
        >>> model = FFTModule(data=data, n_cycles=5, backtest=True)
        >>> model.fit()
        >>> model.show_best_parameters(cycle_range=20)
        >>> model.show_info()

        2. Predict next values, based on the best strongest cycle count. Suppose the best cycle count was 3 \n
        >>> model = FFTModule(data=data, n_cycles=3, backtest=False)
        >>> model.fit()
        >>> model.predict(n_predict=200, show_plot=True)

        Parameters
        __________
        :param data: pd.Series.
                    Time Series Data.
        :param n_cycles: int.
                    Number of cycles to use.
        :param preprocessing: bool.
                    If data is not stationary, it's better to use preprocessing.
        :param backtest: bool.
                    Use to split data into train / test dataset to check model performance.
        :param testing_interval: float.
                    Percent split. Splits data to train/test data to check, how cycle worked in the past.
        :param use_last_data: float.
                    Use last data in percent. If 0.9 - it takes 90% of the latest data.
                    It's possible that older cycles could't work,
                    so sometimes it's better to use latest data to predict cycles.
        :param is_daily: bool.
                    If data is DAILY, then the chart will be indexed with DAYS as the index.
                    Else there will be just indexes.
        """
        self.preprocessing = preprocessing
        self.is_daily = is_daily

        self.full_data = None
        self.train_data = None
        self.test_data = None
        self.frequencies = None
        self.f_hat = None
        self.psd = None
        self.info = None
        self.trend = None

        if use_last_data:
            data = self._cut_off_data(data, split_part=use_last_data)
        if backtest:
            self.full_data, self.train_data, self.test_data = self._train_test_split(data, split_part=testing_interval)
        else:
            self.train_data = data

    def fit(self) -> None:
        """
        Method performs Fast Fourier Transform on the given data.
        """
        nobs = self.train_data.size
        t = np.arange(0, nobs)
        x = self.train_data.values

        if self.preprocessing:
            self.trend = np.polyfit(t, x, 1)  # Find linear trend
            detrended_x = x - self.trend[0] * t  # Get rid of global trend in data
            self.f_hat = fft.fft(detrended_x)

        else:
            self.f_hat = fft.fft(x)

        self.psd = np.abs(self.f_hat)
        self.frequencies = fft.fftfreq(nobs)

        return

    def predict(self, n_predict: int, n_cycles: int = 3, show_plot: bool = False) -> pd.Series:
        """
        Predict next values of given data, based on fitted FFT data.

        Parameters
        ----------
        :param n_predict: int.
                          Number of values to predict.
        :param show_plot: bool.
                          Use show_plot if you want to visualize prediction.
        :return: pd.Series.
                          Pandas Series with predicted values
        """
        if self.psd is None:
            raise NotFittedData(
                "Model is not fitted. Use 'fit' method' firstly."
            )

        indices = self._sort_indices_by_psd()
        t, restored_sig = self._initialize_signal_as_zeros(n_predict)
        restored_sig = self._restore_signal(indices, t, restored_sig, n_cycles)

        if self.preprocessing:
            restored_sig = restored_sig + self.trend[0] * t

        restored_sig = self._apply_datetime_or_numerical_indices(restored_sig, n_predict)

        if show_plot:
            self._plot_prediction(restored_sig)

        return pd.Series(restored_sig)

    def show_info(self, n_cycles: int = 3) -> None:
        """
        Shows info about strongest cycles. Available after 'fit' method

        Parameters
        ----------
        n_cycles: int
                    Show N most strongest cycles. 

        INFO:
        ----------
        Cycles: Shows the fluctuation period.
        E.g. the cycle - 365.5 tell us, that the data has 'yearly' seasonality (in case, of course, the data is DAILY).
        Calculates as (1 / frequency).

        Power Spectral Density (PSD): Tell us, how strong the cycle is. Greater is better.

        PSD Normalize: It's just the (PSD value / sum(PSD value)).
        Normalized value tell us, how strong cycle is in relation to other cycles.
        """

        if self.psd is None:
            raise NotFittedData(
                "Model doesn't fit. Use method 'fit' firstly to get info."
            )

        self.info = self._find_n_peaks(n_cycles=n_cycles)

        print("=========== INFO ABOUT STRONGEST CYCLES ===========")
        print(self.info)

        return

    def score(self, test_data: pd.Series = None, n_cycles: int = 3, show_plot: bool = False) -> float:
        """
        Calculates MAPE value (less is better)

        Parameters
        ----------
        :param test_data: pd.Series. (Optional)
                        Test data for score calculation and model checking.
        :param show_plot: bool.
                        Use show_plot if you want to visualize test data and predictions.
        :return: float.
                        Returns error value, based on MAPE.
        """
        # If self.test_data doesn't exists and test data hasn't been passed
        if self.test_data is None and test_data is None:
            raise MissingTestData(
                "There is no testing data. Nothing to test. Turn 'backest' parameter to True"
            )

        # Conversely, if self.test_data is exists and test data has been passed
        elif self.test_data is not None and test_data is not None:
            print(
                """
[INFO] Test data already exists and score evaluating will be run on backtested data. results will be evaluated based on verified data
If you want check your data, switch 'backtest' parameter to False.
                """
            )

        # If self.test_data doesn't exists and test data was passed
        if test_data is not None and self.test_data is None:
            self.test_data = test_data

        # Get predictions
        continued_data = self.predict(n_predict=self.test_data.size, n_cycles=n_cycles)
        predictions = continued_data[self.train_data.size:]

        error = self._calculate_mape(predictions)

        if show_plot:
            self._plot_with_test_data(
                test_data=self.test_data,
                signal=continued_data
            )

        return error

    def show_best_parameters(self, cycle_range: int = 20, step: int = 1) -> pd.DataFrame:
        """
        Finds the optimal number of cycles to use based on the minimum MAPE error they produce. 
        Actually, it's better to check all good number of cycles and pick the best one
        manually, because sometimes the best count of cycles not always the best choice.
        Returns Pandas DataFrame with number of cycles and corresponded errors.
        
        Parameters
        ----------
        cycle_range: int. Max number of cycles, that will be checked.
        step: int. Range step.

        return: pd.DataFrame. DataFrame with number of cycles and their corresponding errors.
        """
        errors = {
            "cycle": [],
            "error": []
        }

        for cycle in range(1, cycle_range, step):
            error = self.score(n_cycles=cycle)
            errors['cycle'].append(cycle)
            errors['error'].append(error)

        errors = pd.DataFrame(errors)
        errors.columns = ["Num cycles", "MAPE Error"]
        errors = errors\
            .sort_values(by="MAPE Error")\
            .head()\
            .reset_index(drop=True)

        return errors

    def auto_fit_predict(self, n_predict: int = 365, use_n_cycles: int = None, show_plot: bool = True) -> pd.Series:
        """
        Parameters:
        ----------
        n_predict: int
                    The number of values to predict.
        use_n_cycles: int
                    Use N most strongest cycles. 
        show_plot: bool
                    Plot predictions

        INFO:
        ----------
        This method finds the max memory, fits the data and plot predictions automatically.
        Use 'backtest=True', when initializing the module, if you want to find best number of cycles 'best_n_cycles',
        or just set your own number to 'use_n_cycles'.
        """
        # 1. Get max memory
        max_memory = self.get_max_memory(show_plot=False)
        print("[INFO] Max memory: ", max_memory)

        # 2. Cut data with max memory
        self.train_data = self.train_data[-max_memory:]
        self.fit()

        # 3. Find best cycles
        if use_n_cycles is None:
            best_n_cycles = self._get_best_n_cycles()
            print("[INFO] Best N cycles: ", best_n_cycles)
        
        else:
            best_n_cycles = use_n_cycles

        self.train_data = pd.concat((self.train_data, self.test_data))
        self.fit()

        # 4. Predict data and show plot
        predict = self.predict(n_predict=n_predict, n_cycles=best_n_cycles, show_plot=show_plot)
        return predict
        
    def get_max_memory(self, start_days: int = 15, overlap: float = 1.66, filt: int = 100, show_plot=True) -> int:
        """
        Returns the max memory of the data. 
        Max memory shows the number of samples of data, that can be more usefull, than others.
        """
        ser = self._ln_gain(self.full_data if self.full_data is not None else self.train_data)

        max_memory_data ={
            "days": [],
            "f_values": []
        }

        # Find f_value for all passed combinations
        while start_days * overlap <= len(ser):
            first = ser[-start_days:]
            offset = int(start_days * overlap)
            last = ser[-offset:]

            f = self._f_test(first, last)

            max_memory_data["days"].append(start_days)
            max_memory_data["f_values"].append(f)
            start_days += 1

        max_memory_data, max_memory = self._find_best_max_memory(max_memory_data, filt=filt)

        if show_plot:
            self._show_max_memory_plot(
                days=max_memory_data["days"],
                f_list=max_memory_data["f_values"],
                max_memory=max_memory
                )

        return int(max_memory)

    def _get_best_n_cycles(self, cycle_range: int = 20, step: int = 1) -> int:
        errors = self.show_best_parameters(cycle_range=cycle_range, step=step)
        print("[INFO] ERRORS:\n", errors)
        best_n_cycles = errors.iloc[0, 0]
        return best_n_cycles
        
    def _find_n_peaks(self, n_cycles: int = 3) -> pd.DataFrame:
        """Returns strongest n_cycles"""
        peaks = {
            "cycles": [],
            "psd": []
        }

        # Check all PSDs and find peaks
        for i in range(1, len(self.psd) - 2):
            prev_psd = self.psd[i - 1]
            curr_psd = self.psd[i]
            next_psd = self.psd[i + 1]

            # If current PSD is peak
            if prev_psd < curr_psd > next_psd:
                cycle = 1 / self.frequencies[i]
                psd = curr_psd

                peaks["cycles"].append(cycle)
                peaks["psd"].append(psd)

        peaks = pd.DataFrame(data=peaks)
        peaks = peaks[peaks['cycles'] > 0]

        # Normalize PSD
        peaks['norm'] = (peaks["psd"] / peaks["psd"].sum())

        # Rename columns
        peaks.columns = ["Cycles (Days/Bars)", "Power Spectral Density", "PSD Normalize"]

        # Get N strongest cycles
        best_cycles = peaks \
            .sort_values(by="Power Spectral Density", ascending=False) \
            .head(n_cycles) \
            .reset_index(drop=True)

        return best_cycles

    def _sort_indices_by_psd(self) -> list:
        """Sort indices by PSD, higher -> lower"""
        n = self.train_data.size
        indices = list(range(n))
        indices.sort(key=lambda idx: self.psd[idx], reverse=True)

        return indices

    def _initialize_signal_as_zeros(self, n_predict: int) -> np.array:
        """Initialize array for predictions"""
        t = np.arange(0, self.train_data.size + n_predict)
        signal = np.zeros(t.size)

        return t, signal

    def _restore_signal(self, indices:np.array, t:np.array, signal:np.array, n_cycles:int) -> np.array:
        """Returns restored signal"""
        for i in indices[:1 + 2 * n_cycles]:
            amplitude = self.psd[i] / self.train_data.size
            phase = np.angle(self.f_hat[i])
            signal += amplitude * np.cos(2 * np.pi * self.frequencies[i] * t + phase)

        return signal

    def _apply_datetime_or_numerical_indices(self, signal: np.array, n_predict: int) -> pd.Series:
        """
        If data is DAILY, assign DateTime index for time series data.
        If data has another time index, then just assign numerical indices.
        """
        if self.is_daily:
            # Get extended future dates and assign datetime to signal
            dates = self._get_extended_dates(n_predict)
            signal = pd.Series(signal, index=dates)

        else:
            # Get just indexes
            if self.test_data is not None:
                dates = np.arange(0, self.train_data.size + self.test_data.size)
                signal = pd.Series(signal, index=dates)
                self.train_data = self.train_data.values
                self.test_data.index = dates[self.train_data.size:]

            else:
                dates = np.arange(0, self.train_data.size + n_predict)
                signal = pd.Series(signal, index=dates)
                self.train_data = self.train_data.values

        return signal

    def _plot_prediction(self, signal: pd.Series):
        """Plot data and prediction"""
        plt.title("FFT Prediction")
        plt.plot(self.train_data, 'k-', label='original data')
        plt.plot(signal, 'r-', label='continued data', linewidth=2)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_with_test_data(self, test_data, signal):
        """Plot train, test data and prediction"""
        plt.title("FFT Check prediction")
        plt.plot(self.train_data, 'k-', label='original data')
        plt.plot(test_data, 'y-', label='test data')
        plt.plot(signal, 'r-', label='FFT', linewidth=2)
        plt.tight_layout()
        plt.legend()
        plt.show()

    def _get_extended_dates(self, n_predict: int) -> pd.DatetimeIndex:
        """Returns full DateTime Index, extended into the future for 'n_predict' period"""
        past_dates = pd.to_datetime(self.train_data.index)

        # Check, if the dates contains weekends
        weekdays = past_dates.day_name()
        if any(weekdays.isin(["Saturday", "Sunday"])):
            # Date range with all days
            next_dates = pd.date_range(self.train_data.index[-1], periods=n_predict + 1)[1:]
        else:
            # Date range only with business days
            next_dates = pd.bdate_range(self.train_data.index[-1], periods=n_predict + 1)[1:]

        concatenated_dates = past_dates.union(next_dates)

        return pd.DatetimeIndex(concatenated_dates)

    def _cut_off_data(self, data, split_part: Any = 0.1) -> pd.Series:
        """Cut off unwanted part of data"""
        length = data.size
        if type(split_part) == float:
            start_idx = int(length * split_part)
        elif type(split_part) == int:
            start_idx = split_part

        return data[-start_idx:]

    def _train_test_split(self, data: pd.Series, split_part: float = 0.1) -> pd.Series:
        """Split data to train and test"""
        length = data.size
        split_idx = int(length * split_part)
        train_data = data[:-split_idx]
        test_data = data[-split_idx:]
        return data, train_data, test_data

    def _calculate_mape(self, y_pred: np.array) -> float:
        """Returns the MEAN ABSOLUTE PERCENTAGE ERROR (MAPE)"""
        y_true = self.test_data.values
        y_pred = y_pred.values

        epsilon = np.finfo(np.float64).eps
        mape = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), epsilon)
        output_errors = np.average(mape, axis=0)

        return np.average(output_errors)

    def _f_test(self, x1, x2) -> float:
        """Returns F score of 2 passsed data."""
        x1 = np.array(x1)
        x2 = np.array(x2)

        f = np.var(x1, ddof=1) / np.var(x2, ddof=1)

        return f

    def _ln_gain(self, x) -> np.array:
        """Ln gain is another method to get stationary data"""
        return (np.log(x) - np.log(x).shift(1)).dropna()

    def _find_best_max_memory(self, data, filt: int = 100):
        """
        Finds the best max memory.
        filt: int
            'filt' is the filter. Filter is just the number of values (days), that we going to ignore from the beginning.
            The reason to use filter is sometimes algorithm may find max memory to low, e.g. 60 days, which is not enough
            for FFT, so the filter just ignore N values at the beginning.
        """
        data['days'] = np.array(data['days'])
        data['f_values'] = np.array(data['f_values'])

        f_off = data['f_values'][filt:]
        max_memory = np.argmax(f_off) + filt + data['days'][0]
        return data, max_memory

    def _show_max_memory_plot(self, days, f_list, max_memory) -> None:
        """Shows the max memory distribution and the corresponding p value"""
        plt.figure(figsize=(6, 6))
        plt.plot(days, f_list, color='k')
        plt.title("Stochastic cycle (Memory in days)")
        plt.xscale("log")
        plt.vlines(max_memory, 0, np.max(f_list), colors='r', linestyles='dashed',
                    label=f"Max memory: {max_memory} days({round(max_memory / 365, 2)} years"
                          f"/ {round(max_memory / 30.5, 2)} months)")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return

        

