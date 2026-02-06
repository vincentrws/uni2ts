#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from typing import Callable, Optional, Dict, List, Union 


import torch
from einops import reduce
from jaxtyping import Bool, Float, Int
from torch import nn

from uni2ts.common.torch_util import safe_div


class PackedScaler(nn.Module):
    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"] = None,
        sample_id: Int[torch.Tensor, "*batch seq_len"] = None,
        variate_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
    ):
        if observed_mask is None:
            observed_mask = torch.ones_like(target, dtype=torch.bool)
        if sample_id is None:
            sample_id = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )
        if variate_id is None:
            variate_id = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )

        loc, scale = self._get_loc_scale(
            target.double(), observed_mask, sample_id, variate_id
        )
        return loc.float(), scale.float()

    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch seq_len #dim"],
        Float[torch.Tensor, "*batch seq_len #dim"],
    ]:
        raise NotImplementedError


class PackedNOPScaler(PackedScaler):
    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch 1 #dim"], Float[torch.Tensor, "*batch 1 #dim"]
    ]:
        loc = torch.zeros_like(target, dtype=target.dtype)
        scale = torch.ones_like(target, dtype=target.dtype)
        return loc, scale


class PackedStdScaler(PackedScaler):
    def __init__(self, correction: int = 1, minimum_scale: float = 1e-5):
        super().__init__()
        self.correction = correction
        self.minimum_scale = minimum_scale

    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch 1 #dim"], Float[torch.Tensor, "*batch 1 #dim"]
    ]:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = reduce(
            id_mask * reduce(target * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = safe_div(loc, tobs)
        var = reduce(
            id_mask
            * reduce(
                ((target - loc) ** 2) * observed_mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        var = safe_div(var, (tobs - self.correction))
        scale = torch.sqrt(var + self.minimum_scale)
        loc[sample_id == 0] = 0
        scale[sample_id == 0] = 1
        return loc, scale


class PackedAbsMeanScaler(PackedScaler):
    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch 1 #dim"], Float[torch.Tensor, "*batch 1 #dim"]
    ]:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        scale = reduce(
            id_mask
            * reduce(target.abs() * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        scale = safe_div(scale, tobs)
        loc = torch.zeros_like(scale)

        loc[sample_id == 0] = 0
        scale[sample_id == 0] = 1
        return loc, scale


class PackedMidRangeScaler(PackedScaler):
    def __init__(self, mid: float = 0.0, range: float = 1.0):
        super().__init__()
        self.mid = mid
        self.range = range

    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch seq_len #dim"],
        Float[torch.Tensor, "*batch seq_len #dim"],
    ]:
        loc = torch.full_like(target, self.mid, dtype=target.dtype)
        scale = torch.full_like(target, self.range, dtype=target.dtype)
        loc[sample_id == 0] = 0
        scale[sample_id == 0] = 1
        return loc, scale


class GroupedPackedStdScaler(PackedScaler):
    """Standardizes data using group-specific moments.

    For a partition of variates into groups, computes collective mean and standard
    deviation across all variates within each group. This allows semantically related
    variates (e.g., multiple price series) to share normalization statistics while
    maintaining separation between conceptually distinct groups.

    Args:
        group_mapping: Tensor mapping variate indices to group IDs, or a callable
            that accepts variate_id tensor and returns group IDs.
        correction: Degrees of freedom correction for variance estimation (default: 1).
        minimum_scale: Minimum scale value to prevent division by zero (default: 1e-5).
    """
    def __init__(
        self,
        group_mapping: Int[torch.Tensor, "#dim"] | Callable[[torch.Tensor], torch.Tensor],
        correction: int = 1,
        minimum_scale: float = 1e-5,
    ):
        super().__init__()
        self.register_buffer('group_mapping', None)
        if isinstance(group_mapping, torch.Tensor):
            self.register_buffer('group_mapping', group_mapping)
        else:
            self.group_mapping_fn = group_mapping
        self.correction = correction
        self.minimum_scale = minimum_scale

    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch 1 #dim"], Float[torch.Tensor, "*batch 1 #dim"]
    ]:
        # Get group IDs for each variate
        if self.group_mapping is not None:
            group_id = self.group_mapping[variate_id]
        else:
            group_id = self.group_mapping_fn(variate_id)

        # Create identity mask across sample and group dimensions
        # This ensures variates in the same group for the same sample get the same normalization
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(group_id.unsqueeze(-1), group_id.unsqueeze(-2)),
        )

        # Compute total observations per group
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )

        # Compute group-wise mean (location parameter)
        loc = reduce(
            id_mask * reduce(target * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = safe_div(loc, tobs)

        # Compute group-wise standard deviation (scale parameter)
        # Using two-pass algorithm for numerical stability
        var = reduce(
            id_mask
            * reduce(
                ((target - loc) ** 2) * observed_mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        var = safe_div(var, (tobs - self.correction))
        scale = torch.sqrt(var + self.minimum_scale)

        # Handle padding samples (sample_id == 0)
        loc[sample_id == 0] = 0
        scale[sample_id == 0] = 1
        return loc, scale



class FlexiblePackedScaler(PackedScaler):
    """Flexible scaler that applies different normalization strategies per variate.

    Supports multiple strategies:
    - 'grouped_std': Collective std scaling across grouped variates
    - 'individual_std': Individual std scaling per variate
    - 'mid_range': Mid-range normalization with specified mid and range

    This allows mixing different normalization approaches for different types of data
    (e.g., OHLC prices with collective std, RSI with mid-range, etc.)

    Args:
        variate_configs: List of config dicts, one per variate
            Each dict must contain:
            - 'strategy': One of 'grouped_std', 'individual_std', 'mid_range'
            - Additional params based on strategy:
                * 'grouped_std': 'group_id' (int)
                * 'individual_std': None
                * 'mid_range': 'mid' (float), 'range' (float)

    Example:
        configs = [
            {'strategy': 'grouped_std', 'group_id': 0},  # Open
            {'strategy': 'grouped_std', 'group_id': 0},  # High
            {'strategy': 'grouped_std', 'group_id': 0},  # Low
            {'strategy': 'grouped_std', 'group_id': 0},  # Close
            {'strategy': 'grouped_std', 'group_id': 1},  # Volume
            {'strategy': 'mid_range', 'mid': 50.0, 'range': 25.0},  # RSI
            {'strategy': 'mid_range', 'mid': 195.0, 'range': 97.5},  # Minutes
            {'strategy': 'mid_range', 'mid': 2, 'range': 1},  # Day of Week
        ]
        scaler = FlexiblePackedScaler(configs)
    """
    def __init__(self, variate_configs: List[Dict]):
        super().__init__()
        self.variate_configs = variate_configs
        self.num_variates = len(variate_configs)

        # Extract configuration
        self.strategies = [cfg['strategy'] for cfg in variate_configs]

        # Build group mapping for grouped_std strategy
        group_ids = []
        for cfg in variate_configs:
            if cfg['strategy'] == 'grouped_std':
                group_ids.append(cfg['group_id'])
            else:
                group_ids.append(-1)  # Not using grouped std
        self.register_buffer('group_mapping', torch.tensor(group_ids))

        # Build mid and range values for mid_range strategy
        mid_values = []
        range_values = []
        for cfg in variate_configs:
            if cfg['strategy'] == 'mid_range':
                mid_values.append(cfg['mid'])
                range_values.append(cfg['range'])
            else:
                mid_values.append(0.0)
                range_values.append(1.0)
        self.register_buffer('mid_values', torch.tensor(mid_values))
        self.register_buffer('range_values', torch.tensor(range_values))

        # Initialize scalers
        self.grouped_std_scaler = GroupedPackedStdScaler(self.group_mapping)
        self.individual_std_scaler = PackedStdScaler()

    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch 1 #dim"],
        Float[torch.Tensor, "*batch 1 #dim"],
    ]:
        # Get loc and scale from each strategy
        grouped_loc, grouped_scale = self.grouped_std_scaler._get_loc_scale(
            target, observed_mask, sample_id, variate_id
        )
        individual_loc, individual_scale = self.individual_std_scaler._get_loc_scale(
            target, observed_mask, sample_id, variate_id
        )

        # Get mid-range loc and scale
        midrange_loc = self.mid_values[variate_id]
        midrange_scale = self.range_values[variate_id]

        # Determine which strategy to use for each variate
        # Strategy mapping: 0=grouped_std, 1=individual_std, 2=mid_range
        strategy_map = {'grouped_std': 0, 'individual_std': 1, 'mid_range': 2}
        strategy_tensor = torch.tensor(
            [strategy_map[s] for s in self.strategies],
            device=target.device
        )
        strategy_per_pos = strategy_tensor[variate_id]

        # Combine based on strategy
        loc = torch.zeros_like(target)
        scale = torch.zeros_like(target)

        # Apply grouped_std
        use_grouped = (strategy_per_pos == 0)
        loc = torch.where(use_grouped.unsqueeze(-1), grouped_loc, loc)
        scale = torch.where(use_grouped.unsqueeze(-1), grouped_scale, scale)

        # Apply individual_std
        use_individual = (strategy_per_pos == 1)
        loc = torch.where(use_individual.unsqueeze(-1), individual_loc, loc)
        scale = torch.where(use_individual.unsqueeze(-1), individual_scale, scale)

        # Apply mid_range
        use_midrange = (strategy_per_pos == 2)
        loc = torch.where(use_midrange.unsqueeze(-1), midrange_loc, loc)
        scale = torch.where(use_midrange.unsqueeze(-1), midrange_scale, scale)

        # Handle padding
        loc[sample_id == 0] = 0
        scale[sample_id == 0] = 1

        return loc, scale


# Helper functions for common normalization strategies


def create_ohlcv_config(
    ohlc_group_id: int = 0,
    volume_group_id: int = 1,
) -> List[Dict]:
    """Create configuration for standard OHLCV data.

    Args:
        ohlc_group_id: Group ID for OHLC collective scaling
        volume_group_id: Group ID for Volume individual scaling

    Returns:
        List of 5 config dicts for [Open, High, Low, Close, Volume]
    """
    return [
        {'strategy': 'grouped_std', 'group_id': ohlc_group_id},  # Open
        {'strategy': 'grouped_std', 'group_id': ohlc_group_id},  # High
        {'strategy': 'grouped_std', 'group_id': ohlc_group_id},  # Low
        {'strategy': 'grouped_std', 'group_id': ohlc_group_id},  # Close
        {'strategy': 'grouped_std', 'group_id': volume_group_id},  # Volume
    ]


def create_mid_range_config(mid: float, range_val: float) -> Dict:
    """Create configuration for mid-range normalization.

    Args:
        mid: Center value for normalization
        range_val: Range for normalization

    Returns:
        Config dict for mid-range strategy
    """
    return {'strategy': 'mid_range', 'mid': mid, 'range': range_val}


def create_individual_std_config() -> Dict:
    """Create configuration for individual std normalization.

    Returns:
        Config dict for individual std strategy
    """
    return {'strategy': 'individual_std'}


def create_grouped_std_config(group_id: int) -> Dict:
    """Create configuration for grouped std normalization.

    Args:
        group_id: Group ID for collective scaling

    Returns:
        Config dict for grouped std strategy
    """
    return {'strategy': 'grouped_std', 'group_id': group_id}


def build_scaler_config(
    ohlc_indices: List[int] = None,
    volume_indices: List[int] = None,
    mid_range_configs: Dict[int, tuple] = None,
    individual_std_indices: List[int] = None,
    grouped_configs: Dict[int, int] = None,
    num_variates: int = None,
) -> List[Dict]:
    """Build scaler configuration from high-level specification.

    This is a convenience function that builds a complete configuration
    from a high-level specification of which columns should use which
    normalization strategy.

    Args:
        ohlc_indices: Indices for OHLC columns (collective std, group 0)
        volume_indices: Indices for Volume columns (individual std, group 1)
        mid_range_configs: Dict mapping {index: (mid, range)}
        individual_std_indices: Indices for individual std scaling
        grouped_configs: Dict mapping {index: group_id}
        num_variates: Total number of variates (required)

    Returns:
        List of config dicts

    Example:
        config = build_scaler_config(
            ohlc_indices=[0, 1, 2, 3],
            volume_indices=[4],
            mid_range_configs={
                5: (50.0, 25.0),   # RSI
                6: (195.0, 97.5),   # Minutes
            },
            individual_std_indices=[7, 8],
            num_variates=9,
        )
    """
    if num_variates is None:
        raise ValueError("num_variates must be specified")

    configs = []
    for i in range(num_variates):
        if ohlc_indices and i in ohlc_indices:
            configs.append(create_grouped_std_config(group_id=0))
        elif volume_indices and i in volume_indices:
            configs.append(create_grouped_std_config(group_id=1))
        elif mid_range_configs and i in mid_range_configs:
            mid, range_val = mid_range_configs[i]
            configs.append(create_mid_range_config(mid=mid, range_val=range_val))
        elif grouped_configs and i in grouped_configs:
            configs.append(create_grouped_std_config(group_id=grouped_configs[i]))
        elif individual_std_indices and i in individual_std_indices:
            configs.append(create_individual_std_config())
        else:
            # Default to individual std
            configs.append(create_individual_std_config())

    return configs


class OHLCVPackedScaler(PackedScaler):
    """Specialized scaler for OHLCV financial time series data.
    
    Designed for data from wide_multivariate CSV with columns:
    [Open, High, Low, Close, Volume, Minutes after market open, DayOfWeek]
    
    After packing, variate_id mapping:
        0: Open → Group 0 (OHLC collective z-score)
        1: High → Group 0 (OHLC collective z-score)
        2: Low → Group 0 (OHLC collective z-score)
        3: Close → Group 0 (OHLC collective z-score)
        4: Volume → Group 1 (individual z-score)
        5: Minutes → Group 2 (individual z-score)
        6: DayOfWeek → Group 3 (individual z-score)
    
    Normalization is computed per window (per sample_id), respecting observed_mask.
    Uses the same einops-based approach as PackedStdScaler for consistency.
    
    Args:
        open_idx: Index for Open column (default: 0)
        high_idx: Index for High column (default: 1)
        low_idx: Index for Low column (default: 2)
        close_idx: Index for Close column (default: 3)
        volume_idx: Index for Volume column (default: 4)
        minutes_idx: Index for Minutes column (default: 5)
        day_of_week_idx: Index for DayOfWeek column (default: 6)
        correction: Degrees of freedom correction for std (default: 1)
        minimum_scale: Minimum scale to prevent division by zero (default: 1e-5)
    """
    
    def __init__(
        self,
        open_idx: int = 0,
        high_idx: int = 1,
        low_idx: int = 2,
        close_idx: int = 3,
        volume_idx: int = 4,
        minutes_idx: int = 5,
        day_of_week_idx: int = 6,
        correction: int = 1,
        minimum_scale: float = 1e-5,
    ):
        super().__init__()
        self.open_idx = open_idx
        self.high_idx = high_idx
        self.low_idx = low_idx
        self.close_idx = close_idx
        self.volume_idx = volume_idx
        self.minutes_idx = minutes_idx
        self.day_of_week_idx = day_of_week_idx
        self.correction = correction
        self.minimum_scale = minimum_scale
    
    def _get_loc_scale(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> tuple[
        Float[torch.Tensor, "*batch seq_len #dim"],
        Float[torch.Tensor, "*batch seq_len #dim"],
    ]:
        """Compute location and scale parameters for OHLCV data.
        
        Uses the same einops-based approach as PackedStdScaler, but with
        custom grouping logic for OHLC collective normalization.
        """
        # Create group_id tensor based on variate_id
        # Group 0: OHLC (indices 0,1,2,3)
        # Group 1: Volume (index 4)
        # Group 2: Minutes (index 5)
        # Group 3: DayOfWeek (index 6)
        group_id = torch.zeros_like(variate_id, dtype=torch.long)
        
        # OHLC group
        ohlc_mask = torch.isin(
            variate_id,
            torch.tensor(
                [self.open_idx, self.high_idx, self.low_idx, self.close_idx],
                device=variate_id.device
            )
        )
        group_id[ohlc_mask] = 0
        
        # Volume group
        group_id[variate_id == self.volume_idx] = 1
        
        # Minutes group
        group_id[variate_id == self.minutes_idx] = 2
        
        # DayOfWeek group
        group_id[variate_id == self.day_of_week_idx] = 3
        
        # Create identity mask using sample_id and group_id
        # This follows the same pattern as PackedStdScaler but groups by group_id instead of variate_id
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(group_id.unsqueeze(-1), group_id.unsqueeze(-2)),
        )
        
        # Compute total observations per group (respecting observed_mask)
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        
        # Compute group-wise mean (location parameter)
        loc = reduce(
            id_mask * reduce(target * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = safe_div(loc, tobs)
        
        # Compute group-wise variance
        var = reduce(
            id_mask
            * reduce(
                ((target - loc) ** 2) * observed_mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        var = safe_div(var, (tobs - self.correction))
        scale = torch.sqrt(var + self.minimum_scale)
        
        # Handle padding samples (sample_id == 0)
        loc[sample_id == 0] = 0
        scale[sample_id == 0] = 1
        
        return loc, scale
