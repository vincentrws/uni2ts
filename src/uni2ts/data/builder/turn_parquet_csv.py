import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import glob

def process_single_file(input_path: Path, output_dir: Path, freq: str = '5min'):
    """Process a single parquet file and save to CSV."""
    stock_name = input_path.stem
    output_path = output_dir / f"{stock_name}.csv"
    
    print(f"Processing {stock_name} from {input_path}...")
    
    # 1. Load Parquet
    df = pd.read_parquet(input_path)
    if 'ts' in df.columns:
        df = df.set_index('ts')
    
    # Ensure DatetimeIndex with UTC
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep='last')]
    
    # 2. Resample (Handle Gaps)
    start_time = df.index.min()
    end_time = df.index.max()
    full_idx = pd.date_range(start=start_time, end=end_time, freq=freq, tz='UTC')
    
    complete_df = pd.DataFrame(index=full_idx)
    complete_df = complete_df.join(df)
    
    # Fix: Ensure complete_df.index is DatetimeIndex
    complete_df.index = pd.DatetimeIndex(complete_df.index)
    
    # 3. Track which rows have real data (for time feature calculation)
    has_data = ~complete_df['close'].isna()
    
    # 4. Calculate Time Features
    complete_df['date'] = complete_df.index.date
    complete_df['minutes_from_midnight'] = complete_df.index.hour * 60 + complete_df.index.minute
    
    # Calculate daily market open/close times from observed data only
    observed_df = complete_df[has_data].copy()
    daily_stats = observed_df.groupby('date')['minutes_from_midnight'].agg(['min', 'max'])
    daily_stats.columns = ['open_min', 'close_min']
    
    # Merge daily stats back to complete_df
    complete_df = complete_df.merge(daily_stats, left_on='date', right_index=True, how='left')
    complete_df[['open_min', 'close_min']] = complete_df[['open_min', 'close_min']].ffill().bfill()
    
    is_trading = (
        (complete_df['minutes_from_midnight'] >= complete_df['open_min']) & 
        (complete_df['minutes_from_midnight'] <= complete_df['close_min'])
    )
    
    complete_df['Minutes after market open'] = np.where(
        is_trading,
        complete_df['minutes_from_midnight'] - complete_df['open_min'],
        np.nan
    )
    
    complete_df['DayOfWeek'] = complete_df.index.dayofweek
    
    # 5. Format Output
    final_df = complete_df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    })
    
    # CRITICAL FIX: Remove observed_mask column
    # The model's pipeline will auto-generate observed_mask from NaN values
    cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Minutes after market open', 'DayOfWeek']
    final_df = final_df[cols]
    
    # Ensure NaN values are properly set for gap periods
    # Where there's no data, all OHLCV should be NaN
    final_df.loc[~has_data, ['Open', 'High', 'Low', 'Close', 'Volume']] = np.nan
    
    final_df.index.name = 'ts'
    final_df = final_df.reset_index()
    
    print(f"  Saving to {output_path} ({final_df.shape[0]} rows)")
    print(f"  Real data points: {has_data.sum()}, Gap points: {(~has_data).sum()}")
    final_df.to_csv(output_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/opt/uni2ts/data/processed_equities/5m', help='Directory containing parquet files')
    parser.add_argument('--output_dir', default='/opt/uni2ts/data/csv', help='Directory to save CSV files')
    parser.add_argument('--limit', type=int, default=5, help='Max files to process (default: 5)')
    parser.add_argument('--freq', default='5min', help='Data frequency')
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted(list(input_dir.glob("*.parquet")))[:args.limit]
    
    if not files:
        print(f"No parquet files found in {input_dir}")
        return

    print(f"Found {len(files)} files. Processing first {args.limit}...")
    
    for file_path in files:
        try:
            process_single_file(file_path, output_dir, args.freq)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    main()
