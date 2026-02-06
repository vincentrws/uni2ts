"""
OHLCV Data Loader for MOIRAI Fine-Tuning

This module provides functionality to load and prepare OHLCV (Open, High, Low, Close, Volume)
parquet data for MOIRAI model fine-tuning. It handles market gaps, time feature
engineering, and creates HuggingFace datasets compatible with MOIRAI's training pipeline.
"""

from functools import lru_cache
from pathlib import Path
from typing import Generator, Any, Optional

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from datasets import Features, Sequence, Value
from datasets import Dataset as HFDataset


# Module-level cache for time ranges (optimization)
@lru_cache(maxsize=32)
def _get_cached_time_range(
    start: str,
    end: str,
    freq: str
) -> pd.DatetimeIndex:
    """
    Get or create cached time range to avoid recreating timestamps.
    
    Args:
        start: Start timestamp string
        end: End timestamp string
        freq: Frequency string (e.g., '5min')
        
    Returns:
        Cached DatetimeIndex
    """
    return pd.date_range(
        start=pd.Timestamp(start, tz='UTC'),
        end=pd.Timestamp(end, tz='UTC'),
        freq=freq,
        tz='UTC'
    )


class OHLCVLoader(Dataset):
    """
    Load and prepare OHLCV parquet data for MOIRAI fine-tuning.
    
    This loader reads parquet files containing OHLCV data, applies transformations,
    handles market gaps, and creates datasets in HuggingFace format.

    Acts as a PyTorch Dataset for direct integration with training loops.
    
    Data Format:
        Input: Parquet files with columns [ts, open, high, low, close, volume]
            ts: UTC timestamp
            open, high, low, close: Price values
            volume: Trading volume
        
        Output: HuggingFace dataset format with:
            item_id: Stock symbol
            start: Start timestamp
            freq: Data frequency
            target: Close prices [time_steps]
            past_feat_dynamic_real: [open, high, low, volume, time_features] [6, time_steps]
            observed_mask: 1=observed, 0=gap [time_steps]
    
    Gap Handling:
        - Regular gaps: Preserved with observed_mask
        - Weekend/holiday gaps: Filled with volume=0, OHLC=previous close
    
    Time Features:
        - minutes_since_open: 0-390 for trading day (0 at market open)
        - day_of_week: 0-4 for market days (Monday=0, Friday=4), NaN for weekends
    
    Example:
        >>> # Use standard Regular Trading Hours (RTH)
        >>> loader = OHLCVLoader(
        ...     data_path='/opt/uni2ts/data/processed_equities/5m/',
        ...     freq='5min',
        ...     market_open_hour=9,     # 9:30 AM
        ...     market_open_minute=30,
        ...     market_close_hour=16,    # 4:00 PM
        ...     market_close_minute=0
        ... )
        >>> entry = loader.load_single_stock('AAPL', verbose=True)
        >>> dataset = loader.create_hf_dataset(max_stocks=100)
        
        >>> # Or use auto-detected hours (extended)
        >>> loader = OHLCVLoader(
        ...     data_path='/opt/uni2ts/data/processed_equities/5m/',
        ...     freq='5min'
        ... )
    """
    
    def __init__(
        self,
        data_path: str | Path,
        window_size: int = 512,
        stride: int = 512,
        max_stocks: int = None,
        freq: str = '5min',
        timezone: str = 'America/New_York',
        market_open_hour: Optional[int] = None,
        market_open_minute: Optional[int] = None,
        market_close_hour: Optional[int] = None,
        market_close_minute: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize OHLCVLoader.
        
        Args:
            data_path: Path to directory containing parquet files
            window_size: Size of sliding window
            stride: Stride for sliding window
            max_stocks: Limit number of stocks to load
            freq: Frequency of data (e.g., '5min', '1H', '1D')
            timezone: Timezone for market hour calculations
            market_open_hour: Override market open hour (None=auto-detect)
            market_open_minute: Override market open minute (None=auto-detect)
            market_close_hour: Override market close hour (None=auto-detect)
            market_close_minute: Override market close minute (None=auto-detect)
            verbose: Enable verbose logging for all operations
        """
        self.data_path = Path(data_path)
        self.window_size = window_size
        self.stride = stride
        self.freq = freq
        self.timezone = timezone
        self.verbose = verbose
        
        # Custom market hours (overrides auto-detection)
        # If provided, use custom hours; otherwise None (auto-detect from data)
        self.custom_market_hours = {
            'open_hour': market_open_hour,
            'open_minute': market_open_minute,
            'close_hour': market_close_hour,
            'close_minute': market_close_minute,
        }
        
        # Required columns for validation
        self.required_columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"OHLCVLoader Initialized")
            print(f"{'='*60}")
            print(f"  Data path: {self.data_path}")
            print(f"  Window size: {self.window_size}")
            print(f"  Stride: {self.stride}")
            print(f"  Frequency: {self.freq}")
            print(f"  Timezone: {self.timezone}")
            
        self.samples = []
        # Calculate windows immediately
        self._prepare_windows(max_stocks)

    def _prepare_windows(self, max_stocks):
        """Replicates FinetunePatchCrop logic: generates sliding windows"""
        files = sorted(list(self.data_path.glob("*.parquet")))
        if max_stocks: files = files[:max_stocks]
        
        if self.verbose:
            print(f"Indexing windows for {len(files)} files...")
            
        for file_path in files:
            try:
                # Load raw data using existing logic
                entry = self.load_single_stock(file_path.stem, verbose=False)
                
                # Unpack features and target
                # past_feat_dynamic_real: [6, T] -> O, H, L, Vol, Min, Dow
                features = entry['past_feat_dynamic_real']
                # target: [T] -> Close
                target = entry['target']
                
                # 1. PACK FEATURES (Crucial for your scaler!)
                # Combine [Open, High, Low, Close, Volume, Min, Dow] into one array
                # indices: 0:Open, 1:High, 2:Low, 3:Close, 4:Vol, 5:Min, 6:Dow
                full_data = np.stack([
                    features[0], # Open
                    features[1], # High
                    features[2], # Low
                    target,      # Close
                    features[3], # Volume
                    features[4], # Min
                    features[5]  # Dow
                ], axis=0) # Shape: [7, Total_Time]
                
                total_time = full_data.shape[1]
                
                # Sliding Window Logic (stride)
                # Creates indices [0, 512, 1024...]
                for start_idx in range(0, total_time - self.window_size + 1, self.stride):
                    self.samples.append((full_data, start_idx))
                    
            except Exception as e:
                print(f"Error processing {file_path.stem}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        full_data, start_idx = self.samples[idx]
        end_idx = start_idx + self.window_size
        
        # Crop window
        window = full_data[:, start_idx:end_idx] # [Channels, Window_Size]
        
        # Calculate observed_mask based on valid data (not NaN)
        # Transpose to [Time, Channels]
        window_t = window.T
        observed_mask = ~np.isnan(window_t)
        
        # Fill NaNs with 0 to prevent propagation (mask will handle ignoring them)
        window_t = np.nan_to_num(window_t, nan=0.0)
        
        # Convert to Torch
        window_torch = torch.from_numpy(window_t).float()
        observed_mask_torch = torch.from_numpy(observed_mask).bool()
        
        return {
            "target": window_torch,
            "observed_mask": observed_mask_torch,
            "sample_id": torch.full((self.window_size,), idx, dtype=torch.long),
            "variate_id": torch.arange(7).repeat(self.window_size, 1), # 7 Channels
            "prediction_mask": torch.zeros((self.window_size,), dtype=torch.bool)
        }
    
    def _validate_dataframe(self, df: pd.DataFrame, verbose: bool = False) -> None:
        """
        Validate DataFrame structure and data quality.
        
        Args:
            df: DataFrame to validate
            verbose: Print validation details
            
        Raises:
            ValueError: If validation fails
        """
        if verbose:
            print(f"  ✓ Validating DataFrame structure...")
        
        # Check required columns
        if not set(df.columns) == set(self.required_columns):
            raise ValueError(
                f"Expected columns {self.required_columns}, "
                f"but got {list(df.columns)}"
            )
        
        # Check for NaN in required columns (except maybe initial gaps)
        if df.isnull().any().any():
            nan_counts = df.isnull().sum()
            if verbose:
                print(f"    Warning: Found NaN values:")
                for col, count in nan_counts[nan_counts > 0].items():
                    print(f"      {col}: {count} NaN values")
        
        # Validate OHLC relationships
        if not (df['low'] <= df['high']).all():
            raise ValueError("Invalid OHLC: low > high in some rows")
        
        if not (df['low'] <= df['open']).all():
            raise ValueError("Invalid OHLC: open < low in some rows")
        
        if not (df['open'] <= df['high']).all():
            raise ValueError("Invalid OHLC: open > high in some rows")
        
        if not (df['low'] <= df['close']).all():
            raise ValueError("Invalid OHLC: close < low in some rows")
        
        if not (df['close'] <= df['high']).all():
            raise ValueError("Invalid OHLC: close > high in some rows")
        
        # Check volume is non-negative
        if (df['volume'] < 0).any():
            raise ValueError("Volume contains negative values")
        
        # Check for infinite values
        inf_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in inf_cols:
            if np.isinf(df[col].values).any():
                raise ValueError(f"Column '{col}' contains infinite values")
        
        if verbose:
            print(f"    ✓ All validations passed")
            print(f"    ✓ {len(df)} rows validated")
    
    def _detect_market_hours(self, df: pd.DataFrame) -> dict:
        """
        Auto-detect market hours from data distribution.
        
        Analyzes the hour/minute distribution in the data to determine
        typical market opening and closing times.
        
        Args:
            df: DataFrame with ts column (UTC timestamps)
            
        Returns:
            Dictionary with market hour information:
            {
                'open_hour': int,
                'open_minute': int,
                'close_hour': int,
                'close_minute': int,
                'has_extended_hours': bool
            }
        """
        # Convert to local timezone for analysis
        df_local = df.copy()
        df_local['ts_local'] = pd.to_datetime(df_local['ts']).dt.tz_convert(self.timezone)
        df_local['hour'] = df_local['ts_local'].dt.hour
        df_local['minute'] = df_local['ts_local'].dt.minute
        df_local['date'] = df_local['ts_local'].dt.date
        
        # Find most common opening hour/minute
        # Get the earliest hour for each day (likely market open)
        daily_open = df_local.groupby('date').agg({
            'hour': 'min',
            'minute': lambda x: x[x.idxmin()]
        })
        
        # Get the latest hour for each day (likely market close)
        daily_close = df_local.groupby('date').agg({
            'hour': 'max',
            'minute': lambda x: x[x.idxmax()]
        })
        
        # Find mode of opening and closing times
        open_hour = int(daily_open['hour'].mode()[0])
        open_minute = int(daily_open['minute'].mode()[0])
        close_hour = int(daily_close['hour'].mode()[0])
        close_minute = int(daily_close['minute'].mode()[0])
        
        # Calculate if extended hours (more than 6.5 hours)
        has_extended_hours = (close_hour * 60 + close_minute) - (open_hour * 60 + open_minute) > 390
        
        return {
            'open_hour': open_hour,
            'open_minute': open_minute,
            'close_hour': close_hour,
            'close_minute': close_minute,
            'has_extended_hours': has_extended_hours
        }
    
    def _get_market_hours(self, df: pd.DataFrame) -> dict:
        """
        Get market hours (use custom if provided, otherwise auto-detect).
        
        Args:
            df: DataFrame with ts column (UTC timestamps)
            
        Returns:
            Dictionary with market hour information
        """
        # Check if custom market hours are provided (all not None)
        if all(self.custom_market_hours.values()):
            # Use custom market hours
            open_hour = self.custom_market_hours['open_hour']
            open_minute = self.custom_market_hours['open_minute']
            close_hour = self.custom_market_hours['close_hour']
            close_minute = self.custom_market_hours['close_minute']
            has_extended_hours = False  # Custom hours may be RTH or extended
        else:
            # Auto-detect from data
            return self._detect_market_hours(df)
        
        return {
            'open_hour': open_hour,
            'open_minute': open_minute,
            'close_hour': close_hour,
            'close_minute': close_minute,
            'has_extended_hours': has_extended_hours
        }
    
    def _detect_per_day_market_hours(
        self,
        df: pd.DataFrame,
        verbose: bool = False
    ) -> dict:
        """
        Detect market open/close times for each day in the data.
        
        Uses the earliest timestamp of each day as market open and
        the latest timestamp as market close.
        
        Args:
            df: DataFrame with ts column (UTC timestamps)
            verbose: Print detection details
            
        Returns:
            Dictionary mapping date strings to (open_minutes, close_minutes)
            Minutes are from midnight in local timezone
        """
        # Convert to local timezone for analysis
        df_local = df.copy()
        df_local['ts_local'] = pd.to_datetime(df['ts']).dt.tz_convert(self.timezone)
        df_local['date'] = df_local['ts_local'].dt.date
        df_local['time_minutes'] = (
            df_local['ts_local'].dt.hour * 60 + 
            df_local['ts_local'].dt.minute
        )
        
        # Group by date and get min/max time (vectorized - fast)
        daily_times = df_local.groupby('date')['time_minutes'].agg(['min', 'max'])
        
        # Convert to dictionary: date -> (open_minutes, close_minutes)
        per_day_hours = {
            str(date): (int(row['min']), int(row['max'])) 
            for date, row in daily_times.iterrows()
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"PER-DAY MARKET HOURS DETECTION")
            print(f"{'='*70}")
            print(f"  Total days detected: {len(per_day_hours)}")
            print(f"  Sample first 5 days:")
            for i, (date, (open_min, close_min)) in enumerate(list(per_day_hours.items())[:5]):
                open_hr, open_min_only = divmod(open_min, 60)
                close_hr, close_min_only = divmod(close_min, 60)
                duration = close_min - open_min
                print(f"    {date}: {open_hr:02d}:{open_min_only:02d} - {close_hr:02d}:{close_min_only:02d} ({duration} min)")
        
        return per_day_hours
    
    def _prepare_dataset_entry_per_day(
        self,
        df: pd.DataFrame,
        symbol: str,
        per_day_hours: dict,
        gap_fill_strategy: str = 'mask',
        verbose: bool = False
    ) -> dict[str, Any]:
        """
        Prepare dataset entry using per-day market hour detection.
        
        For each day in the data:
        - Uses the earliest timestamp as market open
        - Uses the latest timestamp as market close
        - Timestamps within that day's range are trading periods
        - Timestamps outside that day's range are non-trading periods
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            per_day_hours: Dictionary mapping dates to (open_minutes, close_minutes)
            gap_fill_strategy: 'mask' or 'fill_weekend'
            verbose: Print transformation details
            
        Returns:
            Dictionary with dataset entry
        """
        if verbose:
            print(f"\n🔄 Transformations Applied (Per-Day Market Hours):")
        
        # 1. Create complete time range (OPTIMIZED: use cached time range)
        start_time = df['ts'].min()
        end_time = df['ts'].max()
        
        # Use cached time range (optimization)
        start_str = start_time.strftime('%Y-%m-%d %H:%M:%S%z')
        end_str = end_time.strftime('%Y-%m-%d %H:%M:%S%z')
        
        complete_time_index = _get_cached_time_range(start_str, end_str, self.freq)
        
        if verbose:
            print(f"\n📊 TIME RANGE CREATION")
            print(f"  Total time steps: {len(complete_time_index)}")
            print(f"  From: {complete_time_index[0]} to {complete_time_index[-1]}")
            print(f"  (includes gaps, using cached time range)")
        
        # 2. Create complete DataFrame and merge with actual data (OPTIMIZED: use join)
        complete_df = pd.DataFrame({'ts': complete_time_index})
        complete_df = complete_df.set_index('ts').join(
            df.set_index('ts'),
            how='outer'
        ).reset_index()
        
        gap_count = complete_df['close'].isna().sum()
        if verbose:
            print(f"\n📈 GAP CREATION")
            print(f"  Total gaps from missing source data: {gap_count} ({gap_count/len(complete_df)*100:.2f}%)")
        
        # 3. Add time features and convert to local timezone
        complete_df['ts_local'] = pd.to_datetime(complete_df['ts']).dt.tz_convert(self.timezone)
        complete_df['date'] = complete_df['ts_local'].dt.date
        complete_df['current_minutes'] = (
            complete_df['ts_local'].dt.hour * 60 + 
            complete_df['ts_local'].dt.minute
        )
        complete_df['day_of_week'] = complete_df['ts_local'].dt.dayofweek
        
        # 4. Map date to market hours (vectorized)
        complete_df['open_minutes'] = complete_df['date'].map(
            lambda d: per_day_hours.get(str(d), (None, None))[0]
        )
        complete_df['close_minutes'] = complete_df['date'].map(
            lambda d: per_day_hours.get(str(d), (None, None))[1]
        )
        
        if verbose:
            print(f"\n🕐 MARKET HOURS MAPPING")
            days_with_hours = complete_df['open_minutes'].notna().sum()
            print(f"  Days with detected trading hours: {days_with_hours} ({days_with_hours/len(complete_df)*100:.1f}%)")
            print(f"  Days without data: {complete_df['open_minutes'].isna().sum()}")
        
        # 5. Calculate is_trading_period (within detected range for that day)
        # Use fillna(-1) to handle None/NaN during comparison to avoid TypeError
        complete_df['is_trading'] = (
            (complete_df['open_minutes'].notna()) & 
            (complete_df['close_minutes'].notna()) &
            (complete_df['current_minutes'] >= complete_df['open_minutes'].fillna(-1)) &
            (complete_df['current_minutes'] <= complete_df['close_minutes'].fillna(-1))
        )
        
        if verbose:
            print(f"\n📈 TRADING PERIOD IDENTIFICATION")
            trading_count = complete_df['is_trading'].sum()
            non_trading_count = (~complete_df['is_trading']).sum()
            print(f"  Trading rows: {trading_count} ({trading_count/len(complete_df)*100:.1f}%)")
            print(f"  Non-trading rows: {non_trading_count} ({non_trading_count/len(complete_df)*100:.1f}%)")
        
        # 6. Calculate minutes_since_open (per-day)
        complete_df['minutes_since_open'] = np.where(
            complete_df['is_trading'],
            complete_df['current_minutes'] - complete_df['open_minutes'],
            np.nan
        )
        
        if verbose:
            print(f"\n⏰ TIME FEATURES ADDED")
            print(f"\nDataFrame columns after adding time features:")
            print(f"  {list(complete_df.columns)}")
            
            # Show sample of data with time features
            display_cols = ['ts', 'ts_local', 'date', 'current_minutes', 
                            'open_minutes', 'close_minutes', 'is_trading',
                            'minutes_since_open', 'day_of_week']
            print(f"\n📋 Sample first 10 rows with time features:")
            print(complete_df[display_cols].head(10).to_string(index=False))
            print(f"\n📋 Sample last 10 rows with time features:")
            print(complete_df[display_cols].tail(10).to_string(index=False))
            
            # Show time feature statistics
            print(f"\n📊 TIME FEATURE STATISTICS")
            
            valid_minutes = complete_df['minutes_since_open'][~np.isnan(complete_df['minutes_since_open'])]
            if len(valid_minutes) > 0:
                print(f"\n  minutes_since_open (trading periods only):")
                print(f"    min: {valid_minutes.min():.2f}")
                print(f"    max: {valid_minutes.max():.2f}")
                print(f"    mean: {valid_minutes.mean():.2f}")
                print(f"    std: {valid_minutes.std():.2f}")
                print(f"    valid count: {len(valid_minutes)}")
                print(f"    NaN count: {np.isnan(complete_df['minutes_since_open']).sum()}")
                
                # Show distribution of trading periods by hour
                print(f"\n  Trading periods by hour (local time):")
                trading_by_hour = complete_df[complete_df['is_trading']].groupby(
                    complete_df['ts_local'].dt.hour
                )['minutes_since_open'].agg(['count', 'min', 'max'])
                print(trading_by_hour.to_string())
            
            valid_dow = complete_df['day_of_week'][~np.isnan(complete_df['day_of_week'])]
            print(f"\n  day_of_week:")
            print(f"    min: {valid_dow.min():.2f}")
            print(f"    max: {valid_dow.max():.2f}")
            print(f"    distribution:")
            dow_counts = complete_df['day_of_week'].value_counts().sort_index()
            dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            for dow, count in dow_counts.items():
                print(f"      {dow_names[int(dow)]} ({int(dow)}): {count}")
        
        # 7. Save original NaN state BEFORE filling for observed_mask calculation
        original_close_isna = complete_df['close'].isna().values
        
        # 8. Fill OHLCV for all gaps (ALWAYS do this)
        complete_df['close_filled'] = complete_df['close'].ffill()
        
        ohlc_cols = ['open', 'high', 'low', 'close']
        for col in ohlc_cols:
            complete_df.loc[original_close_isna, col] = complete_df.loc[original_close_isna, 'close_filled']
        
        complete_df.loc[original_close_isna, 'volume'] = 0
        
        filled_count = original_close_isna.sum()
        if verbose:
            print(f"\n🔧 GAP FILLING")
            print(f"  Filled {filled_count} gap periods")
            print(f"  (OHLC=prev_close, volume=0)")
        
        # 9. Prepare target and features
        target = complete_df['close'].values
        past_feat_dynamic_real = np.stack([
            complete_df['open'].values,
            complete_df['high'].values,
            complete_df['low'].values,
            complete_df['volume'].values,
            complete_df['minutes_since_open'].values,
            complete_df['day_of_week'].values,
        ], axis=0)  # Shape: [6, time_steps]
        
        if verbose:
            print(f"\n📦 DATA PREPARATION COMPLETE")
            print(f"  target shape: {target.shape}")
            print(f"  features shape: {past_feat_dynamic_real.shape}")
            print(f"  feature columns: [open, high, low, volume, min_since_open, day_of_week]")
            
            # Check time feature ranges for non-NaN values
            min_since_open_nonan = past_feat_dynamic_real[4][~np.isnan(past_feat_dynamic_real[4])]
            day_of_week_nonan = past_feat_dynamic_real[5][~np.isnan(past_feat_dynamic_real[5])]
            
            print(f"\n  minutes_since_open (non-NaN):")
            print(f"    min: {min_since_open_nonan.min():.2f}")
            print(f"    max: {min_since_open_nonan.max():.2f}")
            print(f"    count: {len(min_since_open_nonan)}")
            print(f"  day_of_week (non-NaN):")
            print(f"    min: {day_of_week_nonan.min():.2f}")
            print(f"    max: {day_of_week_nonan.max():.2f}")
            print(f"    count: {len(day_of_week_nonan)}")
        
        # 10. Create observed mask
        # observed_mask = 1 if original source data exists, 0 if gap
        observed_mask = (~original_close_isna)
        
        if verbose:
            print(f"\n👁️ OBSERVED MASK")
            print(f"  observed: {observed_mask.sum()} points")
            print(f"  gaps: {(~observed_mask).sum()} points")
            print(f"  fill rate: {observed_mask.sum() / len(observed_mask) * 100:.2f}%")
        
        return {
            'item_id': symbol,
            'start': int(start_time.timestamp()),
            'freq': self.freq,
            'target': target.astype(np.float32),
            'past_feat_dynamic_real': past_feat_dynamic_real.astype(np.float32),
            'observed_mask': observed_mask.astype(np.float32),
        }
    
    def _add_time_features(
        self,
        df: pd.DataFrame,
        market_hours: dict
    ) -> pd.DataFrame:
        """
        Add normalized time features to DataFrame.
        
        Args:
            df: DataFrame with ts column (UTC)
            market_hours: Dictionary with market hour info
            
        Returns:
            DataFrame with added time feature columns
        """
        df = df.copy()
        
        # Convert to local timezone
        df['ts_local'] = pd.to_datetime(df['ts']).dt.tz_convert(self.timezone)
        
        # Extract time components
        df['hour'] = df['ts_local'].dt.hour
        df['minute'] = df['ts_local'].dt.minute
        df['day_of_week'] = df['ts_local'].dt.dayofweek  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
        
        # Calculate minutes since market open (RTH: 9:30 AM)
        # This will be 0 at market open, and up to 390 for RTH (9:30 AM - 4:00 PM)
        market_open_minutes = 9 * 60 + 30  # 9:30 AM
        df['minutes_since_open'] = df['hour'] * 60 + df['minute'] - market_open_minutes
        
        # Keep as raw values (not normalized - normalization handled by scaler)
        df['minutes_since_open'] = df['minutes_since_open']
        
        # Keep day_of_week as raw values (0=Monday, 6=Sunday)
        df['day_of_week'] = df['day_of_week']
        
        return df
    
    def _identify_non_trading_periods(
        self,
        df: pd.DataFrame,
        market_hours: dict
    ) -> pd.Series:
        """
        Identify non-trading periods (weekends, holidays, extended hours gaps).
        
        Args:
            df: DataFrame with ts_local column (local timezone)
            market_hours: Dictionary from _detect_market_hours()
            
        Returns:
            Boolean Series where True = non-trading period
        """
        # Add time components if not present
        if 'ts_local' not in df.columns:
            df = df.copy()
            df['ts_local'] = pd.to_datetime(df['ts']).dt.tz_convert(self.timezone)
        
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = df['ts_local'].dt.dayofweek
        
        if 'hour' not in df.columns:
            df['hour'] = df['ts_local'].dt.hour
        
        if 'minute' not in df.columns:
            df['minute'] = df['ts_local'].dt.minute
        
        # Weekends (Saturday=5, Sunday=6)
        is_weekend = df['day_of_week'] >= 5
        
        # Non-trading hours (outside RTH: 9:30 AM - 4:00 PM = 390 minutes)
        market_open_minutes = 9 * 60 + 30  # 9:30 AM = 570 minutes
        market_close_minutes = 16 * 60 + 0  # 4:00 PM = 960 minutes
        
        current_minutes = df['hour'] * 60 + df['minute']
        
        # Outside RTH hours (before 9:30 AM or after 4:00 PM)
        is_outside_hours = (current_minutes < market_open_minutes) | (current_minutes >= market_close_minutes)
        
        # Non-trading periods = weekends OR outside RTH hours
        is_non_trading = is_weekend | is_outside_hours
        
        return is_non_trading
    
    def _fill_non_trading_periods(
        self,
        df: pd.DataFrame,
        is_non_trading: pd.Series
    ) -> pd.DataFrame:
        """
        Fill non-trading periods with volume=0 and OHLC=previous close.
        
        Args:
            df: DataFrame with OHLCV columns
            is_non_trading: Boolean Series marking non-trading periods
            
        Returns:
            DataFrame with filled non-trading periods
        """
        df_filled = df.copy()
        
        # Forward fill close price for non-trading periods
        df_filled['close_filled'] = df_filled['close'].ffill()
        
        # Fill OHLC with previous close for non-trading periods
        ohlc_cols = ['open', 'high', 'low', 'close']
        for col in ohlc_cols:
            df_filled.loc[is_non_trading, col] = df_filled.loc[is_non_trading, 'close_filled']
        
        # Fill volume with 0 for non-trading periods
        df_filled.loc[is_non_trading, 'volume'] = 0
        
        return df_filled
    
    def _prepare_dataset_entry(
        self,
        df: pd.DataFrame,
        symbol: str,
        market_hours: dict,
        gap_fill_strategy: str = 'mask',
        verbose: bool = False
    ) -> dict[str, Any]:
        """
        Prepare dataset entry from DataFrame with optimizations.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            market_hours: Dictionary with market hour info
            gap_fill_strategy: 'mask' or 'fill_weekend'
            verbose: Print transformation details
            
        Returns:
            Dictionary with dataset entry
        """
        if verbose:
            print(f"\n🔄 Transformations Applied:")
        
        # 1. Create complete time range (OPTIMIZED: use cached time range)
        start_time = df['ts'].min()
        end_time = df['ts'].max()
        
        # Use cached time range (optimization)
        start_str = start_time.strftime('%Y-%m-%d %H:%M:%S%z')
        end_str = end_time.strftime('%Y-%m-%d %H:%M:%S%z')
        
        complete_time_index = _get_cached_time_range(start_str, end_str, self.freq)
        
        if verbose:
            print(f"  1. Complete time range: {len(complete_time_index)} points")
            print(f"     ({complete_time_index[0]} to {complete_time_index[-1]})")
            print(f"     (includes gaps, using cached time range)")
        
        # 2. Create complete DataFrame and merge with actual data (OPTIMIZED: use join)
        complete_df = pd.DataFrame({'ts': complete_time_index})
        complete_df = complete_df.set_index('ts').join(
            df.set_index('ts'),
            how='outer'
        ).reset_index()
        
        gap_count = complete_df['close'].isna().sum()
        if verbose:
            print(f"  2. Gap creation: {gap_count} gaps ({gap_count/len(complete_df)*100:.2f}%)")
        
        # 3. Add time features (always needed for observed_mask calculation)
        complete_df = self._add_time_features(complete_df, market_hours)
        
        # 4. Save original NaN state BEFORE filling for observed_mask calculation
        # This captures gaps from missing source data (before join creates gaps)
        original_close_isna = complete_df['close'].isna().values
        
        # 5. Fill OHLCV for all gaps (ALWAYS do this)
        # Forward fill close price, then fill OHLC with previous close
        complete_df['close_filled'] = complete_df['close'].ffill()
        
        # Fill OHLC with previous close for ALL gaps
        ohlc_cols = ['open', 'high', 'low', 'close']
        for col in ohlc_cols:
            complete_df.loc[original_close_isna, col] = complete_df.loc[original_close_isna, 'close_filled']
        
        # Fill volume with 0 for ALL gaps
        complete_df.loc[original_close_isna, 'volume'] = 0
        
        filled_count = original_close_isna.sum()
        if verbose:
            print(f"  3. Filled {filled_count} gap periods")
            print(f"     (OHLC=prev_close, volume=0)")
        
        # 6. Mask time features for non-trading periods
        # Identify weekends (day_of_week >= 5)
        is_weekend = complete_df['day_of_week'] >= 5
        
        # This ensures minutes_since_open is only valid during market hours
        # Mask for weekends AND outside RTH hours
        market_open_minutes = 9 * 60 + 30  # 9:30 AM
        market_close_minutes = 16 * 60 + 0  # 4:00 PM
        current_minutes = complete_df['hour'] * 60 + complete_df['minute']
        is_outside_rth = (current_minutes < market_open_minutes) | (current_minutes >= market_close_minutes)
        
        complete_df.loc[is_weekend, 'minutes_since_open'] = np.nan
        complete_df.loc[is_outside_rth, 'minutes_since_open'] = np.nan
        # Also mask day_of_week for weekends
        complete_df.loc[is_weekend, 'day_of_week'] = np.nan
        
        # 6. Prepare target and features
        target = complete_df['close'].values
        past_feat_dynamic_real = np.stack([
            complete_df['open'].values,
            complete_df['high'].values,
            complete_df['low'].values,
            complete_df['volume'].values,
            complete_df['minutes_since_open'].values,  # NaN for non-trading
            complete_df['day_of_week'].values,  # 0=Monday, 4=Friday (NaN for weekends)
        ], axis=0)  # Shape: [6, time_steps]
        
        if verbose:
            print(f"  4. Data preparation complete:")
            print(f"     target shape: {target.shape}")
            print(f"     features shape: {past_feat_dynamic_real.shape}")
            print(f"     feature columns: [open, high, low, volume, min_since_open, day_of_week]")
            
            # Check time feature ranges for non-NaN values
            min_since_open_nonan = past_feat_dynamic_real[4][~np.isnan(past_feat_dynamic_real[4])]
            day_of_week_nonan = past_feat_dynamic_real[5][~np.isnan(past_feat_dynamic_real[5])]
            
            print(f"     minutes_since_open (non-NaN):")
            print(f"       min: {min_since_open_nonan.min():.2f}")
            print(f"       max: {min_since_open_nonan.max():.2f}")
            print(f"       count: {len(min_since_open_nonan)}")
            print(f"     day_of_week (non-NaN):")
            print(f"       min: {day_of_week_nonan.min():.2f}")
            print(f"       max: {day_of_week_nonan.max():.2f}")
            print(f"       count: {len(day_of_week_nonan)}")
        
        # 7. Create observed mask
        # observed_mask = 1 if original source data exists, 0 if gap
        # Mark weekends and outside RTH as observed since we filled them
        # But actual gaps from missing source data remain as unobserved
        observed_mask = (~original_close_isna)
        
        if verbose:
            print(f"  5. Observed mask:")
            print(f"     observed: {observed_mask.sum()} points")
            print(f"     gaps: {(~observed_mask).sum()} points")
            print(f"     fill rate: {observed_mask.sum() / len(observed_mask) * 100:.2f}%")
        
        return {
            'item_id': symbol,
            'start': int(start_time.timestamp()),  # Convert to Unix timestamp (seconds)
            'freq': self.freq,
            'target': target.astype(np.float32),
            'past_feat_dynamic_real': past_feat_dynamic_real.astype(np.float32),
            'observed_mask': observed_mask.astype(np.float32),
        }
    
    def load_single_stock(
        self,
        symbol: str,
        gap_fill_strategy: str = 'mask',
        verbose: bool = None
    ) -> dict[str, Any]:
        """
        Load a single stock's parquet file and prepare dataset entry.
        
        Args:
            symbol: Stock symbol (filename without .parquet extension)
            gap_fill_strategy: 'mask' (preserve all gaps) or 'fill_weekend'
            verbose: Print detailed information. If None, uses self.verbose
            
        Returns:
            Dictionary with dataset entry
            
        Raises:
            FileNotFoundError: If parquet file doesn't exist
            ValueError: If data validation fails
        """
        # Use instance verbose if not specified
        if verbose is None:
            verbose = self.verbose
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Loading stock: {symbol}")
            print(f"{'='*60}")
        
        # Load parquet file
        file_path = self.data_path / f"{symbol}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {file_path}")
        
        df = pd.read_parquet(file_path)
        
        if verbose:
            print(f"✓ Loaded {len(df)} rows from {file_path}")
            print(f"  Time range: {df['ts'].min()} to {df['ts'].max()}")
            print(f"  Columns: {list(df.columns)}")
        
        # Validate data
        self._validate_dataframe(df, verbose=verbose)
        
        # Detect per-day market hours
        per_day_hours = self._detect_per_day_market_hours(df, verbose=verbose)
        
        # Prepare dataset entry using per-day detection
        entry = self._prepare_dataset_entry_per_day(
            df, symbol, per_day_hours, gap_fill_strategy, verbose=verbose
        )
        
        if verbose:
            print(f"\n✓ Dataset entry created for {symbol}")
            print(f"{'='*60}\n")
        
        return entry
    
    def inspect_dataset_entry(self, entry: dict[str, Any]) -> None:
        """
        Print detailed information about a dataset entry.
        
        Args:
            entry: Dataset entry from load_single_stock
        """
        print(f"\n{'='*60}")
        print(f"DATASET ENTRY INSPECTION")
        print(f"{'='*60}")
        
        print(f"\n📊 Basic Info:")
        print(f"  item_id: {entry['item_id']}")
        print(f"  start: {entry['start']}")
        print(f"  freq: {entry['freq']}")
        print(f"  time_steps: {len(entry['target'])}")
        
        print(f"\n🎯 Target (Close Price):")
        target = entry['target']
        print(f"  shape: {target.shape}")
        print(f"  min: {np.nanmin(target):.4f}")
        print(f"  max: {np.nanmax(target):.4f}")
        print(f"  mean: {np.nanmean(target):.4f}")
        print(f"  std: {np.nanstd(target):.4f}")
        print(f"  nan count: {np.isnan(target).sum()}")
        
        print(f"\n🔧 Features (past_feat_dynamic_real):")
        features = entry['past_feat_dynamic_real']
        feature_names = ['open', 'high', 'low', 'volume', 'min_since_open', 'day_of_week']
        
        print(f"  shape: {features.shape}")
        print(f"  feature names: {feature_names}")
        
        for i, name in enumerate(feature_names):
            feat = features[i]
            print(f"\n  {name}:")
            print(f"    min: {np.nanmin(feat):.4f}")
            print(f"    max: {np.nanmax(feat):.4f}")
            print(f"    mean: {np.nanmean(feat):.4f}")
            print(f"    std: {np.nanstd(feat):.4f}")
            print(f"    nan count: {np.isnan(feat).sum()}")
        
        print(f"\n👁️ Observed Mask:")
        observed_mask = entry['observed_mask']
        print(f"  shape: {observed_mask.shape}")
        print(f"  observed (1.0): {np.sum(observed_mask == 1)}")
        print(f"  gaps (0.0): {np.sum(observed_mask == 0)}")
        print(f"  gap percentage: {np.sum(observed_mask == 0) / len(observed_mask) * 100:.2f}%")
        
        print(f"\n{'='*60}\n")
    
    def _inspect_hf_dataset(self, dataset: HFDataset) -> None:
        """
        Print summary of HuggingFace dataset.
        
        Args:
            dataset: HuggingFace Dataset object
        """
        print(f"\n📈 Dataset Summary:")
        print(f"  Total entries: {len(dataset)}")
        
        # Get sample entry
        sample = dataset[0]
        print(f"  Sample item_id: {sample['item_id']}")
        print(f"  Sample time_steps: {len(sample['target'])}")
        print(f"  Sample features: {sample['past_feat_dynamic_real'].shape}")
        
        # Calculate aggregate statistics
        total_time_steps = sum(len(ds['target']) for ds in dataset)
        total_observed = sum(np.sum(ds['observed_mask']) for ds in dataset)
        
        print(f"\n📊 Aggregate Statistics:")
        print(f"  Total time steps: {total_time_steps:,}")
        print(f"  Total observed: {int(total_observed):,}")
        print(f"  Total gaps: {total_time_steps - int(total_observed):,}")
        print(f"  Overall fill rate: {total_observed/total_time_steps*100:.2f}%")
    
    def create_hf_dataset(
        self,
        symbols: list[str] | None = None,
        max_stocks: int | None = None,
        gap_fill_strategy: str = 'mask',
        verbose: bool = None,
        save_to_disk: bool = True,
        output_path: str | Path | None = None
    ) -> HFDataset:
        """
        Create HuggingFace dataset from OHLCV parquet files.
        
        Args:
            symbols: List of stock symbols to load. If None, load all.
            max_stocks: Maximum number of stocks to load.
            gap_fill_strategy: 'mask' or 'fill_weekend'
            output_path: Path to save dataset. If None, returns in-memory dataset.
            verbose: Print detailed progress information.
            save_to_disk: If True, save to output_path.
            
        Returns:
            HuggingFace Dataset object
        """
        # Use instance verbose if not specified
        if verbose is None:
            verbose = self.verbose
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Creating HuggingFace Dataset")
            print(f"{'='*60}")
        
        # Get list of symbols
        if symbols is None:
            all_files = list(self.data_path.glob('*.parquet'))
            symbols = [f.stem for f in all_files]
        
        if max_stocks:
            symbols = symbols[:max_stocks]
        
        if verbose:
            print(f"📂 Loading {len(symbols)} stocks from {self.data_path}")
            print(f"   Frequency: {self.freq}")
            print(f"   Gap strategy: {gap_fill_strategy}")
            print(f"   Save to disk: {save_to_disk}")
            if output_path:
                print(f"   Output path: {output_path}")
        
        # Generator function
        def gen_func() -> Generator[dict[str, Any], None, None]:
            for i, symbol in enumerate(symbols, 1):
                if verbose:
                    print(f"\n[{i}/{len(symbols)}] Processing {symbol}...")
                
                try:
                    entry = self.load_single_stock(
                        symbol,
                        gap_fill_strategy=gap_fill_strategy,
                        verbose=False  # Suppress individual stock verbose output
                    )
                    yield entry
                    
                    if verbose:
                        print(f"✓ {symbol} loaded successfully")
                        
                except Exception as e:
                    print(f"✗ {symbol} failed: {e}")
                    continue
        
        # Define HuggingFace features schema (following example/prepare_data.ipynb format)
        features = Features({
            'item_id': Value('string'),
            'start': Value('timestamp[s]'),
            'freq': Value('string'),
            'target': Sequence(Value('float32')),
            'past_feat_dynamic_real': Sequence(Sequence(Value('float32'))),
            'observed_mask': Sequence(Value('float32')),
        })
        
        if verbose:
            print(f"\n📊 Creating HuggingFace dataset...")
        
        # Create dataset
        dataset = HFDataset.from_generator(
            gen_func,
            features=features,
            gen_kwargs={},
            num_proc=1  # Can increase for multiprocessing
        )
        
        if verbose:
            print(f"✓ Dataset created with {len(dataset)} entries")
            self._inspect_hf_dataset(dataset)
        
        # Save to disk if requested
        if save_to_disk and output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(output_path))
            
            if verbose:
                print(f"✓ Dataset saved to {output_path}")
        
        if verbose:
            print(f"{'='*60}\n")
    
    def create_train_test_splits(
        self,
        symbol: str,
        output_dir: str | Path,
        split_date: str = '2025-01-01',
        gap_fill_strategy: str = 'mask',
        verbose: bool = True
    ) -> tuple[HFDataset, HFDataset]:
        """
        Create train and test HuggingFace datasets for a single stock with year-based split.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            split_date: Date for split (< split_date = train, >= split_date = test)
            output_dir: Directory to save datasets
            gap_fill_strategy: 'mask' or 'fill_weekend'
            verbose: Print progress
            
        Returns:
            (train_dataset, test_dataset)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Creating Train/Test Splits for {symbol}")
            print(f"  Split date: {split_date}")
            print(f"  Train: < {split_date}")
            print(f"  Test: >= {split_date}")
            print(f"{'='*60}")
        
        # Load full data
        df = pd.read_parquet(self.data_path / f"{symbol}.parquet")
        split_timestamp = pd.Timestamp(split_date, tz='UTC')
        
        # Validate data
        self._validate_dataframe(df, verbose=verbose)
        
        # Detect per-day market hours from full data
        per_day_hours_train = self._detect_per_day_market_hours(
            df[df['ts'] < split_timestamp], verbose=False
        )
        
        if verbose:
            print(f"\n📊 Data split:")
            print(f"  Total rows: {len(df)}")
            print(f"  Train split: {len(df[df['ts'] < split_timestamp])} rows")
            print(f"  Test split: {len(df[df['ts'] >= split_timestamp])} rows")
            print(f"  Train: {len(df[df['ts'] < split_timestamp]) / len(df) * 100:.1f}%")
            print(f"  Test: {len(df[df['ts'] >= split_timestamp]) / len(df) * 100:.1f}%")
        
        # Create TRAIN dataset
        if verbose:
            print(f"\n🚀 Building TRAIN dataset...")
        
        df_train = df[df['ts'] < split_timestamp].copy()
        
        train_entry = self._prepare_dataset_entry_per_day(
            df_train, 
            symbol,
            per_day_hours_train,
            gap_fill_strategy=gap_fill_strategy,
            verbose=False
        )
        
        train_output = Path(output_dir) / f"{symbol}_train"
        # Wrap in list for from_dict (expects batch format)
        train_dataset = HFDataset.from_dict({
            'item_id': [train_entry['item_id']],
            'start': [train_entry['start']],
            'freq': [train_entry['freq']],
            'target': [train_entry['target']],
            'past_feat_dynamic_real': [train_entry['past_feat_dynamic_real']],
            'observed_mask': [train_entry['observed_mask']],
        })
        train_dataset.save_to_disk(str(train_output))
        
        if verbose:
            print(f"  ✓ Train dataset saved: {train_output}")
            print(f"     Time steps: {len(train_entry['target'])}")
            print(f"     Features: {train_entry['past_feat_dynamic_real'].shape}")
        
        # Create TEST dataset
        if verbose:
            print(f"\n🚀 Building TEST dataset...")
        
        df_test = df[df['ts'] >= split_timestamp].copy()
        
        # Use same market hours from training data for consistency
        test_entry = self._prepare_dataset_entry_per_day(
            df_test, 
            symbol,
            per_day_hours_train,  # Use same market hours from train
            gap_fill_strategy=gap_fill_strategy,
            verbose=False
        )
        
        test_output = Path(output_dir) / f"{symbol}_test"
        # Wrap in list for from_dict (expects batch format)
        test_dataset = HFDataset.from_dict({
            'item_id': [test_entry['item_id']],
            'start': [test_entry['start']],
            'freq': [test_entry['freq']],
            'target': [test_entry['target']],
            'past_feat_dynamic_real': [test_entry['past_feat_dynamic_real']],
            'observed_mask': [test_entry['observed_mask']],
        })
        test_dataset.save_to_disk(str(test_output))
        
        if verbose:
            print(f"  ✓ Test dataset saved: {test_output}")
            print(f"     Time steps: {len(test_entry['target'])}")
            print(f"     Features: {test_entry['past_feat_dynamic_real'].shape}")
        
        if verbose:
            print(f"\n✓ Done! Train: {train_output}, Test: {test_output}")
            print(f"  Combined features: [open, high, low, volume, min_since_open, day_of_week]")
        
        return train_dataset, test_dataset