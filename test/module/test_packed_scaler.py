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

from typing import Callable

import pytest
import torch
from einops import rearrange, repeat

from uni2ts.common.torch_util import safe_div
from uni2ts.module.packed_scaler import (
    GroupedPackedStdScaler,
    OHLCVPackedScaler,
    PackedAbsMeanScaler,
    PackedMidRangeScaler,
    PackedNOPScaler,
    PackedScaler,
    PackedStdScaler,
)


def pack_seq(
    xs: list[torch.Tensor],
    max_seq_len: int,
    pad_fn: Callable = torch.zeros,
):
    batch = [[]]
    shape = xs[0].shape[1:]
    for x in xs:
        if sum([b.shape[0] for b in batch[-1]]) + x.shape[0] > max_seq_len:
            batch.append([])
        batch[-1].append(x)

    for i in range(len(batch)):
        curr_len = sum([b.shape[0] for b in batch[i]])
        if curr_len < max_seq_len:
            batch[i].append(
                pad_fn(max_seq_len - curr_len, *shape, dtype=batch[i][0].dtype)
            )

    return torch.stack([torch.cat(x, dim=0) for x in batch], dim=0)


def _test_packed_scaler(
    get_loc_scale_func: Callable[
        [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ],
    packed_scaler: PackedScaler,
    samples: list[tuple[int, int, int, int, int]],
    max_seq_len: int,
    max_patch_size: int,
    observed_pct: float,
):
    data = {
        "target": [],
        "observed_mask": [],
        "sample_id": [],
        "dimension_id": [],
        "loc": [],
        "scale": [],
    }
    sample_id_counter = 1
    packing_counter = 0
    for time, n_target_time, target_dim, cov_dim, patch_size in samples:
        dim = target_dim + cov_dim
        assert time * dim <= max_seq_len

        _target = torch.randn(n_target_time, target_dim, patch_size)
        _cov = torch.randn(time, cov_dim, patch_size)
        _observed = torch.rand(time, dim, patch_size) < observed_pct

        target = torch.randn(time, dim, max_patch_size)
        target[-n_target_time:, :target_dim, :patch_size] = _target
        target[:, target_dim:, :patch_size] = _cov
        target = rearrange(target, "t d p -> (d t) p")

        observed_mask = torch.zeros(time, dim, max_patch_size, dtype=torch.bool)
        observed_mask[-n_target_time:, :target_dim, :patch_size] = True
        observed_mask[:, target_dim:, :patch_size] = True
        observed_mask[:, :, :patch_size] *= _observed
        observed_mask = rearrange(observed_mask, "t d p -> (d t) p")

        if packing_counter + time * dim > max_seq_len:
            sample_id_counter = 1
            packing_counter = 0
        else:
            sample_id_counter += 1
            packing_counter += time * dim
        sample_id = torch.ones(time * dim, dtype=torch.long) * sample_id_counter

        dimension_id = repeat(torch.arange(dim), "d -> (d t)", t=time)

        _target_loc, _target_scale = get_loc_scale_func(
            _target, _observed[-n_target_time:, :target_dim]
        )
        _cov_loc, _cov_scale = get_loc_scale_func(_cov, _observed[:, target_dim:])
        loc = repeat(
            torch.cat([_target_loc, _cov_loc], dim=1), "1 d -> (d t) 1", t=time
        )
        scale = repeat(
            torch.cat([_target_scale, _cov_scale], dim=1), "1 d -> (d t) 1", t=time
        )

        data["target"].append(target)
        data["observed_mask"].append(observed_mask)
        data["sample_id"].append(sample_id)
        data["dimension_id"].append(dimension_id)
        data["loc"].append(loc)
        data["scale"].append(scale)

    target = pack_seq(data["target"], max_seq_len)
    observed_mask = pack_seq(data["observed_mask"], max_seq_len)
    sample_id = pack_seq(data["sample_id"], max_seq_len)
    dimension_id = pack_seq(data["dimension_id"], max_seq_len)
    loc = pack_seq(data["loc"], max_seq_len)
    scale = pack_seq(data["scale"], max_seq_len, pad_fn=torch.ones)

    packed_loc, packed_scale = packed_scaler(
        target,
        observed_mask,
        sample_id,
        dimension_id,
    )

    assert loc.shape[0] == packed_loc.shape[0]
    assert loc.shape[1] == packed_loc.shape[1]
    assert torch.allclose(loc, packed_loc)
    assert scale.shape[0] == packed_scale.shape[0]
    assert scale.shape[1] == packed_scale.shape[1]
    assert torch.allclose(scale, packed_scale, atol=1e-4)


testdata = [
    (
        10,
        2,
        # time, n_target_time, target_dim, cov_dim, patch_size
        [(3, 1, 1, 0, 2), (2, 1, 1, 0, 1)],
    ),
    (
        20,
        3,
        [
            (3, 2, 1, 1, 1),
            (3, 1, 1, 1, 3),
            (3, 3, 1, 1, 2),
        ],
    ),
    (
        100,
        5,
        [
            (10, 5, 1, 0, 3),
            (10, 5, 1, 0, 3),
            (3, 2, 2, 1, 1),
            (3, 1, 2, 1, 3),
            (3, 1, 2, 1, 3),
        ],
    ),
]


@pytest.mark.parametrize("max_seq_len, max_patch_size, samples", testdata)
@pytest.mark.parametrize("observed_pct", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_packed_nop_scaler(
    max_seq_len: int,
    max_patch_size: int,
    samples: list[tuple[int, int, int, int, int]],
    observed_pct: float,
    seed: int,
):
    def get_loc_scale_func(
        _target: torch.Tensor, _observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dim = _target.shape[1]
        return torch.zeros(1, dim), torch.ones(1, dim)

    torch.manual_seed(seed)
    _test_packed_scaler(
        get_loc_scale_func,
        PackedNOPScaler(),
        samples,
        max_seq_len,
        max_patch_size,
        observed_pct,
    )


@pytest.mark.parametrize("max_seq_len, max_patch_size, samples", testdata)
@pytest.mark.parametrize("observed_pct", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_packed_std_scaler(
    max_seq_len: int,
    max_patch_size: int,
    samples: list[tuple[int, int, int, int, int]],
    observed_pct: float,
    seed: int,
):
    def get_loc_scale_func(
        _target: torch.Tensor, _observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _target = _target * _observed
        loc = safe_div(
            _target.sum(dim=(0, -1), keepdim=True),
            _observed.sum(dim=(0, -1), keepdim=True),
        )
        deviation = ((_target - loc) ** 2) * _observed
        var = safe_div(
            deviation.sum(dim=(0, -1), keepdim=True),
            torch.clamp(_observed.sum(dim=(0, -1), keepdim=True) - 1, min=0),
        )
        scale = torch.sqrt(var + 1e-5)
        return loc.squeeze(-1), scale.squeeze(-1)

    torch.manual_seed(seed)
    _test_packed_scaler(
        get_loc_scale_func,
        PackedStdScaler(),
        samples,
        max_seq_len,
        max_patch_size,
        observed_pct,
    )


@pytest.mark.parametrize("max_seq_len, max_patch_size, samples", testdata)
@pytest.mark.parametrize("observed_pct", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_packed_abs_mean_scaler(
    max_seq_len: int,
    max_patch_size: int,
    samples: list[tuple[int, int, int, int, int]],
    observed_pct: float,
    seed: int,
):
    def get_loc_scale_func(
        _target: torch.Tensor, _observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dim = _target.shape[1]
        _target = _target * _observed
        scale = safe_div(
            _target.abs().sum(dim=(0, -1), keepdim=True),
            _observed.sum(dim=(0, -1), keepdim=True),
        )
        return torch.zeros(1, dim), scale.squeeze(-1)

    torch.manual_seed(seed)
    _test_packed_scaler(
        get_loc_scale_func,
        PackedAbsMeanScaler(),
        samples,
        max_seq_len,
        max_patch_size,
        observed_pct,
    )


@pytest.mark.parametrize("max_seq_len, max_patch_size, samples", testdata)
@pytest.mark.parametrize("observed_pct", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("mid,scale", [(0.0, 1.0), (50.0, 25.0), (-10.0, 5.0)])
def test_packed_mid_range_scaler(
    max_seq_len: int,
    max_patch_size: int,
    samples: list[tuple[int, int, int, int, int]],
    observed_pct: float,
    seed: int,
    mid: float,
    scale: float,
):
    def get_loc_scale_func(
        _target: torch.Tensor, _observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dim = _target.shape[1]
        return torch.full((1, dim), mid), torch.full((1, dim), scale)

    torch.manual_seed(seed)
    _test_packed_scaler(
        get_loc_scale_func,
        PackedMidRangeScaler(mid=mid, range=scale),
        samples,
        max_seq_len,
        max_patch_size,
        observed_pct,
    )


def _test_grouped_packed_scaler(
    packed_scaler: GroupedPackedStdScaler,
    samples: list[tuple[int, int, int, int, int]],
    max_seq_len: int,
    max_patch_size: int,
    observed_pct: float,
    group_mapping: torch.Tensor,
):
    """Rigorous test for GroupedPackedStdScaler using explicit loops for reference."""
    data = {
        "target": [],
        "observed_mask": [],
        "sample_id": [],
        "variate_id": [],
    }
    sample_id_counter = 1
    packing_counter = 0

    for time, n_target_time, target_dim, cov_dim, patch_size in samples:
        dim = target_dim + cov_dim
        assert time * dim <= max_seq_len

        # Generate data
        _target = torch.randn(n_target_time, target_dim, patch_size)
        _cov = torch.randn(time, cov_dim, patch_size)
        _observed = torch.rand(time, dim, patch_size) < observed_pct

        # Construct target and observed_mask
        target_full = torch.randn(time, dim, max_patch_size)
        target_full[-n_target_time:, :target_dim, :patch_size] = _target
        target_full[:, target_dim:, :patch_size] = _cov

        obs_mask_full = torch.zeros(time, dim, max_patch_size, dtype=torch.bool)
        obs_mask_full[-n_target_time:, :target_dim, :patch_size] = True
        obs_mask_full[:, target_dim:, :patch_size] = True
        obs_mask_full[:, :, :patch_size] *= _observed

        # Packing
        if packing_counter + time * dim > max_seq_len:
            sample_id_counter = 1
            packing_counter = 0
        else:
            sample_id_counter += 1
            packing_counter += time * dim

        data["target"].append(rearrange(target_full, "t d p -> (d t) p"))
        data["observed_mask"].append(rearrange(obs_mask_full, "t d p -> (d t) p"))
        data["sample_id"].append(torch.ones(time * dim, dtype=torch.long) * sample_id_counter)
        data["variate_id"].append(repeat(torch.arange(dim), "d -> (d t)", t=time))

    target = pack_seq(data["target"], max_seq_len)
    observed_mask = pack_seq(data["observed_mask"], max_seq_len)
    sample_id = pack_seq(data["sample_id"], max_seq_len)
    variate_id = pack_seq(data["variate_id"], max_seq_len)

    # Compute high-performance result
    packed_loc, packed_scale = packed_scaler(target, observed_mask, sample_id, variate_id)

    # Compute reference result using explicit loops
    expected_loc = torch.zeros_like(packed_loc)
    expected_scale = torch.ones_like(packed_scale)

    for b in range(target.shape[0]):
        unique_samples = torch.unique(sample_id[b])
        for s_id in unique_samples:
            if s_id == 0: continue
            sample_mask = (sample_id[b] == s_id)

            # Map variates in this sample to groups
            v_ids = variate_id[b, sample_mask]
            g_ids = group_mapping[v_ids]
            unique_groups = torch.unique(g_ids)

            for g_id in unique_groups:
                # Find all elements in this sample that belong to this group
                group_mask = (g_ids == g_id)
                idxs = sample_mask.nonzero().flatten()[group_mask]

                # Collect all data points in this group-sample pair across time and patch dims
                group_data = target[b, idxs].double()
                group_obs = observed_mask[b, idxs]

                # Compute collective statistics mirroring actual safe_div logic
                obs_count = group_obs.sum()
                mu = safe_div((group_data * group_obs).sum(), obs_count)
                var = safe_div(
                    ((group_data.double() - mu.double())**2 * group_obs).sum(),
                    obs_count - 1
                )
                std = torch.sqrt(var + 1e-5)

                expected_loc[b, idxs, 0] = mu.float()
                expected_scale[b, idxs, 0] = std.float()

    assert torch.allclose(packed_loc, expected_loc, atol=1e-4)
    assert torch.allclose(packed_scale, expected_scale, atol=1e-4)


@pytest.mark.parametrize("max_seq_len, max_patch_size, samples", [
    (10, 2, [(3, 1, 1, 0, 2), (2, 1, 1, 0, 1)]),
    (100, 5, [(10, 5, 2, 3, 3), (10, 5, 1, 2, 3)]),
])
@pytest.mark.parametrize("observed_pct", [0, 0.5, 1])
@pytest.mark.parametrize("group_mapping", [
    torch.tensor([0, 0, 0, 0, 1]), # Typical OHLCV
    torch.tensor([0, 1, 2, 3, 4]), # Identity (standard normalization)
    torch.tensor([0, 0, 0, 0, 0]), # All collective
])
def test_grouped_packed_std_scaler_comprehensive(
    max_seq_len, max_patch_size, samples, observed_pct, group_mapping
):
    scaler = GroupedPackedStdScaler(group_mapping)
    _test_grouped_packed_scaler(
        scaler, samples, max_seq_len, max_patch_size, observed_pct, group_mapping
    )


def test_ohlcv_packed_scaler_basic():
    """Test OHLCVPackedScaler with basic OHLCV data structure."""
    torch.manual_seed(42)
    
    # Create sample OHLCV data matching OHLCVLoader output structure
    time_steps = 10
    num_variates = 6  # [open, high, low, volume, minutes_since_open, day_of_week]
    patch_size = 1
    
    # Generate data
    open_data = torch.tensor([100.0, 104.0, 107.0, 109.0, 111.0, 113.0, 115.0, 117.0, 119.0, 121.0])
    high_data = torch.tensor([105.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0, 120.0, 122.0, 124.0])
    low_data = torch.tensor([99.0, 103.0, 106.0, 108.0, 110.0, 112.0, 114.0, 116.0, 118.0, 120.0])
    volume_data = torch.tensor([1000000, 1200000, 900000, 1100000, 950000, 1050000, 1150000, 1250000, 1350000, 1450000])
    minutes_data = torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0])
    dow_data = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    # Combine all features [time, dim]
    features = torch.stack([open_data, high_data, low_data, volume_data, minutes_data, dow_data], dim=1)
    
    # Add patch dimension [time, dim, patch]
    features = features.unsqueeze(-1)
    
    # Reshape to packed format: [time, dim, patch] -> [(dim * time), patch]
    target_packed = rearrange(features, "t d p -> (d t) p")
    
    # Create sample_id (all same sample)
    sample_id = torch.ones(target_packed.shape[0], dtype=torch.long)
    
    # Create variate_id (0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4, 5,5,2,2,2)
    variate_id = repeat(torch.arange(num_variates), "d -> (d t)", t=time_steps)
    
    # All observed
    observed_mask = torch.ones_like(target_packed, dtype=torch.bool)
    
    # Initialize scaler
    scaler = OHLCVPackedScaler(verbose=False)
    
    # Get loc and scale
    loc, scale = scaler(
        target=target_packed.unsqueeze(0),
        observed_mask=observed_mask.unsqueeze(0),
        sample_id=sample_id.unsqueeze(0),
        variate_id=variate_id.unsqueeze(0),
    )
    
    # Verify shapes
    assert loc.shape == (1, time_steps * num_variates, 1)
    assert scale.shape == (1, time_steps * num_variates, 1)
    
    # Verify OHLC have same statistics (collective normalization)
    open_loc = loc[0, 0:time_steps, 0].unique()
    high_loc = loc[0, time_steps:2*time_steps, 0].unique()
    low_loc = loc[0, 2*time_steps:3*time_steps, 0].unique()
    
    assert len(open_loc) == 1, "Open should have single loc value"
    assert len(high_loc) == 1, "High should have single loc value"
    assert len(low_loc) == 1, "Low should have single loc value"
    
    # All OHLC should have the same mean and std
    assert torch.isclose(open_loc[0], high_loc[0], atol=1e-4), "Open and High should have same loc"
    assert torch.isclose(open_loc[0], low_loc[0], atol=1e-4), "Open and Low should have same loc"
    
    open_scale = scale[0, 0:time_steps, 0].unique()
    high_scale = scale[0, time_steps:2*time_steps, 0].unique()
    low_scale = scale[0, 2*time_steps:3*time_steps, 0].unique()
    
    assert len(open_scale) == 1, "Open should have single scale value"
    assert len(high_scale) == 1, "High should have single scale value"
    assert len(low_scale) == 1, "Low should have single scale value"
    
    assert torch.isclose(open_scale[0], high_scale[0], atol=1e-4), "Open and High should have same scale"
    assert torch.isclose(open_scale[0], low_scale[0], atol=1e-4), "Open and Low should have same scale"
    
    # Verify Volume has independent statistics
    volume_loc = loc[0, 3*time_steps:4*time_steps, 0].unique()
    volume_scale = scale[0, 3*time_steps:4*time_steps, 0].unique()
    assert len(volume_loc) == 1, "Volume should have single loc value"
    assert len(volume_scale) == 1, "Volume should have single scale value"
    assert not torch.isclose(volume_loc[0], open_loc[0], atol=1e-4), "Volume loc should differ from OHLC"
    
    # Verify mid-range for time features
    minutes_loc = loc[0, 4*time_steps:5*time_steps, 0].unique()
    minutes_scale = scale[0, 4*time_steps:5*time_steps, 0].unique()
    
    assert len(minutes_loc) == 1, "Minutes should have single loc value"
    assert torch.isclose(minutes_loc[0], torch.tensor(195.0), atol=1e-4), "Minutes loc should be 195.0"
    assert torch.isclose(minutes_scale[0], torch.tensor(97.5), atol=1e-4), "Minutes scale should be 97.5"
    
    dow_loc = loc[0, 5*time_steps:, 0].unique()
    dow_scale = scale[0, 5*time_steps:, 0].unique()
    
    assert len(dow_loc) == 1, "Day of week should have single loc value"
    assert torch.isclose(dow_loc[0], torch.tensor(2.0), atol=1e-4), "Day of week loc should be 2.0"
    assert torch.isclose(dow_scale[0], torch.tensor(1.0), atol=1e-4), "Day of week scale should be 1.0"


def test_ohlcv_packed_scaler_multiple_windows():
    """Test OHLCVPackedScaler with multiple windows (different sample_ids)."""
    torch.manual_seed(42)
    
    # Create 2 windows
    time_steps = 5
    num_variates = 6
    patch_size = 1
    
    # Window 1 data
    window1_features = torch.randn(time_steps, num_variates) * 10 + 100
    window1_features[:, 4] = torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0])  # minutes
    window1_features[:, 5] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])  # dow
    
    # Window 2 data (different statistics)
    window2_features = torch.randn(time_steps, num_variates) * 20 + 200
    window2_features[:, 4] = torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0])  # minutes
    window2_features[:, 5] = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])  # dow
    
    # Combine windows
    all_features = torch.cat([window1_features, window2_features], dim=0)
    # Add patch dimension
    all_features = all_features.unsqueeze(-1)
    target_packed = rearrange(all_features, "t d p -> (d t) p")
    
    # Create sample_id for each window
    sample_id = torch.cat([
        torch.ones(time_steps * num_variates, dtype=torch.long),  # Window 1
        torch.full((time_steps * num_variates,), 2, dtype=torch.long),  # Window 2
    ])
    
    # Create variate_id
    total_steps = time_steps * 2
    variate_id = repeat(torch.arange(num_variates), "d -> (d t)", t=total_steps)
    
    # All observed
    observed_mask = torch.ones_like(target_packed, dtype=torch.bool)
    
    # Initialize scaler
    scaler = OHLCVPackedScaler(verbose=False)
    
    # Get loc and scale
    loc, scale = scaler(
        target=target_packed.unsqueeze(0),
        observed_mask=observed_mask.unsqueeze(0),
        sample_id=sample_id.unsqueeze(0),
        variate_id=variate_id.unsqueeze(0),
    )
    
    # Verify windows have different statistics for OHLC
    window1_open_loc = loc[0, 0:time_steps, 0].unique()[0]
    window2_open_loc = loc[0, num_variates*time_steps:(num_variates*time_steps + time_steps), 0].unique()[0]
    
    assert not torch.isclose(window1_open_loc, window2_open_loc, atol=1e-4), \
        "Different windows should have different OHLC statistics"
    
    # Verify time features have same statistics across windows (mid-range is fixed)
    window1_minutes_loc = loc[0, 4*time_steps:5*time_steps, 0].unique()[0]
    window2_minutes_loc = loc[0, num_variates*time_steps + 4*time_steps:num_variates*time_steps + 5*time_steps, 0].unique()[0]
    
    assert torch.isclose(window1_minutes_loc, window2_minutes_loc, atol=1e-4), \
        "Minutes should have same mid-range across windows"


def test_ohlcv_packed_scaler_with_observed_mask():
    """Test OHLCVPackedScaler with partial observations."""
    torch.manual_seed(42)
    
    time_steps = 10
    num_variates = 6
    patch_size = 1
    
    # Generate data
    features = torch.randn(time_steps, num_variates) * 10 + 100
    features[:, 4] = torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0])
    features[:, 5] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    # Add patch dimension
    features = features.unsqueeze(-1)
    target_packed = rearrange(features, "t d p -> (d t) p")
    sample_id = torch.ones(target_packed.shape[0], dtype=torch.long)
    variate_id = repeat(torch.arange(num_variates), "d -> (d t)", t=time_steps)
    
    # Partial observations: only first half observed for each variate
    observed_mask = torch.zeros_like(target_packed, dtype=torch.bool)
    for v in range(num_variates):
        observed_mask[v*time_steps:v*time_steps + time_steps//2] = True
    
    # Initialize scaler
    scaler = OHLCVPackedScaler(verbose=False)
    
    # Get loc and scale
    loc, scale = scaler(
        target=target_packed.unsqueeze(0),
        observed_mask=observed_mask.unsqueeze(0),
        sample_id=sample_id.unsqueeze(0),
        variate_id=variate_id.unsqueeze(0),
    )
    
    # Verify statistics were computed from observed data only
    # Manually compute expected OHLC collective statistics
    observed_ohlc_data = features[:time_steps//2, :3].flatten()
    expected_ohlc_mean = observed_ohlc_data.mean()
    expected_ohlc_std = observed_ohlc_data.std()
    
    open_loc = loc[0, 0:time_steps, 0].unique()[0]
    
    assert torch.isclose(open_loc, expected_ohlc_mean, atol=1e-4), \
        "OHLC mean should be computed from observed data only"


def test_ohlcv_packed_scaler_custom_indices():
    """Test OHLCVPackedScaler with custom variate indices."""
    torch.manual_seed(42)
    
    time_steps = 5
    num_variates = 6
    patch_size = 1
    
    features = torch.randn(time_steps, num_variates) * 10 + 100
    features = features.unsqueeze(-1)
    target_packed = rearrange(features, "t d p -> (d t) p")
    sample_id = torch.ones(target_packed.shape[0], dtype=torch.long)
    variate_id = repeat(torch.arange(num_variates), "d -> (d t)", t=time_steps)
    observed_mask = torch.ones_like(target_packed, dtype=torch.bool)
    
    # Create scaler with custom indices
    scaler = OHLCVPackedScaler(
        open_idx=0,
        high_idx=1,
        low_idx=2,
        volume_idx=3,
        minutes_idx=4,
        day_of_week_idx=5,
        verbose=False
    )
    
    # Should work without errors
    loc, scale = scaler(
        target=target_packed.unsqueeze(0),
        observed_mask=observed_mask.unsqueeze(0),
        sample_id=sample_id.unsqueeze(0),
        variate_id=variate_id.unsqueeze(0),
    )
    
    assert loc.shape == (1, time_steps * num_variates, 1)
    assert scale.shape == (1, time_steps * num_variates, 1)


def test_ohlcv_packed_scaler_custom_mid_range():
    """Test OHLCVPackedScaler with custom mid-range values."""
    torch.manual_seed(42)
    
    time_steps = 5
    num_variates = 6
    patch_size = 1
    
    features = torch.randn(time_steps, num_variates) * 10 + 100
    features = features.unsqueeze(-1)
    target_packed = rearrange(features, "t d p -> (d t) p")
    sample_id = torch.ones(target_packed.shape[0], dtype=torch.long)
    variate_id = repeat(torch.arange(num_variates), "d -> (d t)", t=time_steps)
    observed_mask = torch.ones_like(target_packed, dtype=torch.bool)
    
    # Custom mid-range values
    scaler = OHLCVPackedScaler(
        minutes_mid=100.0,
        minutes_range=50.0,
        dow_mid=3.0,
        dow_range=2.0,
        verbose=False
    )
    
    loc, scale = scaler(
        target=target_packed.unsqueeze(0),
        observed_mask=observed_mask.unsqueeze(0),
        sample_id=sample_id.unsqueeze(0),
        variate_id=variate_id.unsqueeze(0),
    )
    
    # Verify custom values are used
    minutes_loc = loc[0, 4*time_steps:5*time_steps, 0].unique()[0]
    minutes_scale = scale[0, 4*time_steps:5*time_steps, 0].unique()[0]
    
    assert torch.isclose(minutes_loc, torch.tensor(100.0), atol=1e-4), "Custom minutes mid should be used"
    assert torch.isclose(minutes_scale, torch.tensor(50.0), atol=1e-4), "Custom minutes range should be used"
    
    dow_loc = loc[0, 5*time_steps:, 0].unique()[0]
    dow_scale = scale[0, 5*time_steps:, 0].unique()[0]
    
    assert torch.isclose(dow_loc, torch.tensor(3.0), atol=1e-4), "Custom dow mid should be used"
    assert torch.isclose(dow_scale, torch.tensor(2.0), atol=1e-4), "Custom dow range should be used"


@pytest.mark.parametrize("verbose", [True, False])
def test_ohlcv_packed_scaler_verbose_mode(verbose):
    """Test OHLCVPackedScaler verbose mode."""
    torch.manual_seed(42)
    
    time_steps = 5
    num_variates = 6
    patch_size = 1
    
    features = torch.randn(time_steps, num_variates) * 10 + 100
    features = features.unsqueeze(-1)
    target_packed = rearrange(features, "t d p -> (d t) p")
    sample_id = torch.ones(target_packed.shape[0], dtype=torch.long)
    variate_id = repeat(torch.arange(num_variates), "d -> (d t)", t=time_steps)
    observed_mask = torch.ones_like(target_packed, dtype=torch.bool)
    
    scaler = OHLCVPackedScaler(verbose=verbose)
    
    # Should work regardless of verbose setting
    loc, scale = scaler(
        target=target_packed.unsqueeze(0),
        observed_mask=observed_mask.unsqueeze(0),
        sample_id=sample_id.unsqueeze(0),
        variate_id=variate_id.unsqueeze(0),
    )
    
    assert loc.shape == (1, time_steps * num_variates, 1)
    assert scale.shape == (1, time_steps * num_variates, 1)
