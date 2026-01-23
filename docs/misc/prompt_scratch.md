I've been analyzing this codebase for MOIRAI, a time series transformer model. I have several documents in the docs folder, which were compiled by other agents. Previously I was using Cline and its memory bank to keep track of the current progress and intention of the project. I intend to fine tune a model on historical stock market OHLCV data at the 5 minute time bar resolution. I will be trying to forecast the close prices into the near future. I do not intend to condition on the stock ticker symbols, the model will only see OHLCV, as well as a minutes since market open column. 

The OHLC prices must be normalized collectively with one common mean and std, this will require a new scaler function. Volume will be normalized separately. 

I also want a mid point scaler function, designed for oscillators which may have ranges between 0 and 100. It won't make sense to use Z score which conditions on the mean, instead we'll "compress" the values to a reasonable range between -2 and +2 for example. So like 0 becomes -2, 50 becomes 0, 100 becomes +2. Similarly, for minutes since market open, 0 can become -2, 390 (6.5 hours) becomes +2.  

I have the parquet data files in data/processed_equities. The 5m looks like this:

m5_df = pd.read_parquet('/opt/uni2ts/data/processed_equities/5m/A.parquet')
m5_df


ts	open	high	low	close	volume
0	2000-01-03 14:30:00+00:00	56.3305	56.3305	56.3305	56.3305	146510.0
1	2000-01-03 14:35:00+00:00	56.3305	56.4646	55.7940	56.1069	98559.0
2	2000-01-03 14:40:00+00:00	56.2411	56.2411	55.3022	55.4363	106667.0
3	2000-01-03 14:45:00+00:00	55.4810	55.5705	54.7210	54.8104	79687.0
4	2000-01-03 14:50:00+00:00	54.8552	54.8552	54.0057	54.0057	74653.0
...	...	...	...	...	...	...
499328	2025-07-25 19:35:00+00:00	120.9500	121.0100	120.8700	120.9150	8688.0
499329	2025-07-25 19:40:00+00:00	120.8950	120.9750	120.7500	120.7700	15738.0
499330	2025-07-25 19:45:00+00:00	120.7700	120.7700	120.5900	120.6400	19482.0
499331	2025-07-25 19:50:00+00:00	120.6200	120.6850	120.2000	120.3000	44409.0
499332	2025-07-25 19:55:00+00:00	120.2950	120.3790	120.0900	120.2400	172961.0
499333 rows × 6 columns 

We will be pursuing the dynamic past and future features fine tuning mode. Some features around time will be known in the future, for example minutes since market open. However OHLCV data should obviously not be leaked to the model for predicting into the future.