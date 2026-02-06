# To Do

January 26, 2026

- [x] Implement group-specific mean and standard deviation calculation
- [x] Add unit tests for GroupedPackedStdScaler
- [ ] Implement mid and range scaler. May need to refactor or custom scaler definition, since one scaler may need to be applied to one tensor window at a time. If the mid and range scaler can be applied only to Oscillator type series, then that's good. However we may need to specify all the different normalization strategies for each type of column. 
- [ ] Hybrid scaler which can normalize different types of columns with different strategies.
- [ ] Fine tune model on OHLCV data + time features. Dynamic real features (past and future).
- [ ] Fine tune model on OHLCV + time + technical indicators 
    - [ ] daily SMAs (must be group normalized with OHLC prices)
    - [ ] intraday EMAs (must be group normalized with OHLC prices)
    - [ ] LRSI (must be mid range normalized mid=50, range=25)
    - [ ] Relative Strength against SPY
    - [ ] Relative Volume at a Time (lookback 30 days).
- [ ] Allow the model to learn the identities of the columns.
- [ ] 