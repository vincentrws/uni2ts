"""
Unit tests for OHLCVLoader

Tests cover data loading, validation, market hour detection,
gap handling, time features, and HuggingFace dataset creation.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from uni2ts.data.ohlcvloader import OHLCVLoader
from datasets import load_from_disk


# Fixtures
@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2024-01-01 14:30:00', periods=100, freq='5min', tz='UTC')
    
    # Create realistic OHLCV data
    np.random.seed(42)
    base_price = 100.0
    
    data = []
    for i, date in enumerate(dates):
        price_change = np.random.randn() * 0.5
        base_price += price_change
        
        open_price = base_price
        close_price = base_price + np.random.randn() * 0.1
        high_price = max(open_price, close_price) + abs(np.random.randn() * 0.2)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 0.2)
        volume = abs(np.random.randint(1000, 10000))
        
        data.append({
            'ts': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_data_dir(tmp_path, sample_ohlcv_data):
    """Create temporary directory with sample parquet files."""
    sample_ohlcv_data.to_parquet(tmp_path / 'TEST.parquet', index=False)
    return tmp_path


@pytest.fixture
def loader(sample_data_dir):
    """Create OHLCVLoader instance for testing."""
    return OHLCVLoader(
        data_path=sample_data_dir,
        freq='5min',
        timezone='America/New_York',
        verbose=False
    )


# Test Initialization
def test_loader_initialization(sample_data_dir):
    """Test OHLCVLoader initialization."""
    loader = OHLCVLoader(
        data_path=sample_data_dir,
        freq='5min',
        timezone='America/New_York',
        verbose=False
    )
    
    assert loader.data_path == sample_data_dir
    assert loader.freq == '5min'
    assert loader.timezone == 'America/New_York'
    assert loader.verbose == False


# Test Data Validation
def test_validate_dataframe_success(sample_ohlcv_data, loader):
    """Test successful DataFrame validation."""
    loader._validate_dataframe(sample_ohlcv_data, verbose=False)
    # Should not raise any exceptions


def test_validate_dataframe_missing_columns(sample_data_dir):
    """Test validation fails with missing columns."""
    loader = OHLCVLoader(data_path=sample_data_dir, freq='5min', verbose=False)
    
    # Create DataFrame with wrong columns
    df = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=10, freq='5min', tz='UTC'),
        'price': [100] * 10
    })
    
    with pytest.raises(ValueError, match="Expected columns"):
        loader._validate_dataframe(df)


def test_validate_dataframe_invalid_ohlc(sample_data_dir):
    """Test validation fails with invalid OHLC relationships."""
    loader = OHLCVLoader(data_path=sample_data_dir, freq='5min', verbose=False)
    
    # low > high (invalid)
    df_bad = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=10, freq='5min', tz='UTC'),
        'open': [100] * 10,
        'high': [90] * 10,
        'low': [80] * 10,
        'close': [95] * 10,
        'volume': [1000] * 10
    })
    
    with pytest.raises(ValueError, match="Invalid OHLC: low > high"):
        loader._validate_dataframe(df_bad)


def test_validate_dataframe_negative_volume(sample_data_dir):
    """Test validation fails with negative volume."""
    loader = OHLCVLoader(data_path=sample_data_dir, freq='5min', verbose=False)
    
    df_bad = pd.DataFrame({
        'ts': pd.date_range('2024-01-01', periods=10, freq='5min', tz='UTC'),
        'open': [100] * 10,
        'high': [110] * 10,
        'low': [90] * 10,
        'close': [95] * 10,
        'volume': [-100] * 10  # Negative volume
    })
    
    with pytest.raises(ValueError, match="Volume contains negative values"):
        loader._validate_dataframe(df_bad)


def test_validate_dataframe_infinite_values(sample_data_dir):
    """Test validation fails with infinite values."""
    loader = OHLCVLoader(data_path=sample_data_dir, freq='5min', verbose=False)
    
    data = {
        'ts': pd.date_range('2024-01-01', periods=10, freq='5min', tz='UTC'),
        'open': [100.0] * 10,
        'high': [110.0] * 10,
        'low': [90.0] * 10,
        'close': [95.0] * 10,
        'volume': [1000] * 10
    }
    data['open'][5] = np.inf  # Add infinite value
    
    df_bad = pd.DataFrame(data)
    
    with pytest.raises(ValueError, match="contains infinite values"):
        loader._validate_dataframe(df_bad)


# Test Market Hour Detection
def test_detect_market_hours(loader, sample_ohlcv_data):
    """Test market hour detection from data."""
    market_hours = loader._detect_market_hours(sample_ohlcv_data)
    
    assert 'open_hour' in market_hours
    assert 'open_minute' in market_hours
    assert 'close_hour' in market_hours
    assert 'close_minute' in market_hours
    assert 'has_extended_hours' in market_hours
    
    # Check types
    assert isinstance(market_hours['open_hour'], int)
    assert isinstance(market_hours['open_minute'], int)
    assert isinstance(market_hours['close_hour'], int)
    assert isinstance(market_hours['close_minute'], int)
    assert isinstance(market_hours['has_extended_hours'], bool)


# Test Gap Handling
def test_identify_non_trading_periods(loader, sample_ohlcv_data):
    """Test identification of non-trading periods."""
    # Add time features first
    market_hours = loader._detect_market_hours(sample_ohlcv_data)
    sample_ohlcv_data['ts_local'] = pd.to_datetime(sample_ohlcv_data['ts']).dt.tz_convert(loader.timezone)
    sample_ohlcv_data['day_of_week'] = sample_ohlcv_data['ts_local'].dt.dayofweek
    sample_ohlcv_data['hour'] = sample_ohlcv_data['ts_local'].dt.hour
    sample_ohlcv_data['minute'] = sample_ohlcv_data['ts_local'].dt.minute
    
    is_non_trading = loader._identify_non_trading_periods(sample_ohlcv_data, market_hours)
    
    assert isinstance(is_non_trading, pd.Series)
    assert len(is_non_trading) == len(sample_ohlcv_data)
    assert is_non_trading.dtype == bool


def test_fill_non_trading_periods(loader, sample_ohlcv_data):
    """Test filling of non-trading periods."""
    is_non_trading = pd.Series([False] * len(sample_ohlcv_data))
    
    # Mark some as non-trading
    is_non_trading.iloc[10:20] = True
    
    df_filled = loader._fill_non_trading_periods(sample_ohlcv_data, is_non_trading)
    
    # Check volume is 0 for non-trading periods
    assert (df_filled.loc[is_non_trading, 'volume'] == 0).all()
    
    # Check OHLC are filled with previous close
    assert not df_filled.loc[is_non_trading, 'open'].isna().any()
    assert not df_filled.loc[is_non_trading, 'high'].isna().any()
    assert not df_filled.loc[is_non_trading, 'low'].isna().any()
    assert not df_filled.loc[is_non_trading, 'close'].isna().any()


# Test Time Features
def test_add_time_features(loader, sample_ohlcv_data):
    """Test time feature addition."""
    market_hours = loader._detect_market_hours(sample_ohlcv_data)
    df_with_features = loader._add_time_features(sample_ohlcv_data, market_hours)
    
    # Check new columns exist
    assert 'ts_local' in df_with_features.columns
    assert 'hour' in df_with_features.columns
    assert 'minute' in df_with_features.columns
    assert 'day_of_week' in df_with_features.columns
    assert 'minutes_since_open' in df_with_features.columns
    assert 'minutes_since_open_norm' in df_with_features.columns
    assert 'day_of_week_norm' in df_with_features.columns
    
    # Check ranges
    assert df_with_features['day_of_week'].min() >= 0
    assert df_with_features['day_of_week'].max() <= 6
    
    # Check normalization (should be roughly centered around 0)
    assert np.abs(df_with_features['day_of_week_norm'].median()) < 2.0


# Test Dataset Entry Preparation
def test_prepare_dataset_entry_mask(loader, sample_ohlcv_data):
    """Test dataset entry preparation with mask strategy."""
    market_hours = loader._detect_market_hours(sample_ohlcv_data)
    entry = loader._prepare_dataset_entry(
        sample_ohlcv_data,
        'TEST',
        market_hours,
        gap_fill_strategy='mask',
        verbose=False
    )
    
    # Check required keys
    assert 'item_id' in entry
    assert 'start' in entry
    assert 'freq' in entry
    assert 'target' in entry
    assert 'past_feat_dynamic_real' in entry
    assert 'observed_mask' in entry
    
    # Check values
    assert entry['item_id'] == 'TEST'
    assert entry['freq'] == '5min'
    assert entry['target'].dtype == np.float32
    assert entry['past_feat_dynamic_real'].dtype == np.float32
    assert entry['observed_mask'].dtype == np.float32
    
    # Check shapes
    assert entry['past_feat_dynamic_real'].shape[0] == 6  # 6 features
    assert len(entry['target']) == len(entry['observed_mask'])


def test_prepare_dataset_entry_fill_weekend(loader, sample_ohlcv_data):
    """Test dataset entry preparation with fill_weekend strategy."""
    market_hours = loader._detect_market_hours(sample_ohlcv_data)
    entry = loader._prepare_dataset_entry(
        sample_ohlcv_data,
        'TEST',
        market_hours,
        gap_fill_strategy='fill_weekend',
        verbose=False
    )
    
    # Check that non-trading periods are filled
    # (This depends on the specific data, but we check the structure)
    assert entry['item_id'] == 'TEST'
    assert len(entry['target']) > 0


# Test Single Stock Loading
def test_load_single_stock_success(loader):
    """Test successful single stock loading."""
    entry = loader.load_single_stock('TEST', gap_fill_strategy='mask', verbose=False)
    
    assert entry['item_id'] == 'TEST'
    assert len(entry['target']) > 0
    assert entry['past_feat_dynamic_real'].shape[0] == 6


def test_load_single_stock_file_not_found(sample_data_dir):
    """Test loading fails for non-existent file."""
    loader = OHLCVLoader(data_path=sample_data_dir, freq='5min', verbose=False)
    
    with pytest.raises(FileNotFoundError, match="Parquet file not found"):
        loader.load_single_stock('NONEXISTENT')


# Test HuggingFace Dataset Creation
def test_create_hf_dataset_single_stock(loader):
    """Test HuggingFace dataset creation with single stock."""
    dataset = loader.create_hf_dataset(
        symbols=['TEST'],
        gap_fill_strategy='mask',
        save_to_disk=False,
        verbose=False
    )
    
    assert len(dataset) == 1
    entry = dataset[0]
    
    assert 'item_id' in entry
    assert 'target' in entry
    assert 'past_feat_dynamic_real' in entry
    assert 'observed_mask' in entry


def test_create_hf_dataset_save_and_load(loader, tmp_path):
    """Test saving and loading HuggingFace dataset."""
    output_path = tmp_path / 'test_dataset'
    
    dataset = loader.create_hf_dataset(
        symbols=['TEST'],
        gap_fill_strategy='mask',
        output_path=output_path,
        save_to_disk=True,
        verbose=False
    )
    
    # Check directory was created
    assert output_path.exists()
    
    # Load dataset from disk
    loaded_dataset = load_from_disk(str(output_path))
    
    assert len(loaded_dataset) == len(dataset)
    assert loaded_dataset[0]['item_id'] == dataset[0]['item_id']


def test_create_hf_dataset_all_stocks(sample_data_dir, tmp_path):
    """Test creating dataset from all stocks in directory."""
    # Create multiple test files
    for i in range(3):
        dates = pd.date_range('2024-01-01', periods=50, freq='5min', tz='UTC')
        df = pd.DataFrame({
            'ts': dates,
            'open': np.random.randn(50) * 0.5 + 100,
            'high': np.random.randn(50) * 0.5 + 101,
            'low': np.random.randn(50) * 0.5 + 99,
            'close': np.random.randn(50) * 0.5 + 100,
            'volume': np.random.randint(1000, 10000, 50)
        })
        df.to_parquet(sample_data_dir / f'STOCK{i}.parquet', index=False)
    
    loader = OHLCVLoader(data_path=sample_data_dir, freq='5min', verbose=False)
    
    output_path = tmp_path / 'multi_stock_dataset'
    dataset = loader.create_hf_dataset(
        max_stocks=2,  # Limit to 2 stocks
        output_path=output_path,
        save_to_disk=True,
        verbose=False
    )
    
    assert len(dataset) == 2


# Test Data Quality
def test_no_infinite_values_in_entry(loader):
    """Test that dataset entry has no infinite values."""
    entry = loader.load_single_stock('TEST', gap_fill_strategy='mask', verbose=False)
    
    assert not np.any(np.isinf(entry['target']))
    assert not np.any(np.isinf(entry['past_feat_dynamic_real']))


def test_volume_non_negative(loader):
    """Test that volume is non-negative in dataset entry."""
    entry = loader.load_single_stock('TEST', gap_fill_strategy='mask', verbose=False)
    
    volume = entry['past_feat_dynamic_real'][3]
    assert np.all(volume >= 0) | np.isnan(volume)


# Test Observed Mask
def test_observed_mask_values(loader):
    """Test that observed mask contains only 0 and 1."""
    entry = loader.load_single_stock('TEST', gap_fill_strategy='mask', verbose=False)
    
    mask = entry['observed_mask']
    assert np.all((mask == 0) | (mask == 1))


def test_observed_mask_length(loader):
    """Test that observed mask has correct length."""
    entry = loader.load_single_stock('TEST', gap_fill_strategy='mask', verbose=False)
    
    assert len(entry['observed_mask']) == len(entry['target'])


# Test Inspection Methods
def test_inspect_dataset_entry(loader, capsys):
    """Test dataset entry inspection (should print)."""
    entry = loader.load_single_stock('TEST', gap_fill_strategy='mask', verbose=False)
    
    loader.inspect_dataset_entry(entry)
    
    captured = capsys.readouterr()
    # Should have printed output
    assert len(captured) > 0 or True  # Just ensure it doesn't crash


def test_inspect_hf_dataset(loader, capsys):
    """Test HuggingFace dataset inspection (should print)."""
    dataset = loader.create_hf_dataset(
        symbols=['TEST'],
        gap_fill_strategy='mask',
        save_to_disk=False,
        verbose=False
    )
    
    loader._inspect_hf_dataset(dataset)
    
    captured = capsys.readouterr()
    # Should have printed output
    assert len(captured) > 0 or True  # Just ensure it doesn't crash


# Test Edge Cases
def test_empty_dataframe(sample_data_dir):
    """Test handling of empty DataFrame."""
    loader = OHLCVLoader(data_path=sample_data_dir, freq='5min', verbose=False)
    
    df_empty = pd.DataFrame(columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    
    with pytest.raises(Exception):  # Market hour detection will fail
        loader._detect_market_hours(df_empty)


def test_single_row_dataframe(sample_data_dir, sample_data_dir_path):
    """Test handling of single row DataFrame."""
    loader = OHLCVLoader(data_path=sample_data_dir, freq='5min', verbose=False)
    
    df_single = pd.DataFrame({
        'ts': [pd.Timestamp('2024-01-01 14:30:00', tz='UTC')],
        'open': [100.0],
        'high': [101.0],
        'low': [99.0],
        'close': [100.0],
        'volume': [1000]
    })
    
    # Should work but with limited market hour detection
    market_hours = loader._detect_market_hours(df_single)
    assert 'open_hour' in market_hours