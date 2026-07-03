"""
Tests for the masking module.
"""
import numpy as np
from aquaculture.masking import (
    select_window_length,
    select_start_month,
    create_s2_mask,
    apply_s2_masking,
    apply_competition_mask,
    validate_masking,
    _validate_probabilities
)
from aquaculture.config import AquacultureConfig


def test_validate_probabilities():
    """Test probability validation function."""
    # Valid probabilities
    _validate_probabilities(np.array([0.5, 0.5]), "test")
    _validate_probabilities(np.array([0.2, 0.3, 0.5]), "test")

    # Invalid probabilities - sum not equal to 1
    try:
        _validate_probabilities(np.array([0.3, 0.3]), "test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must sum to 1.0" in str(e)

    # Invalid probabilities - negative values
    try:
        _validate_probabilities(np.array([0.5, -0.1, 0.6]), "test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must contain non-negative values" in str(e)


def test_select_window_length():
    """Test window length selection."""
    rng = np.random.default_rng(42)

    # Test with fixed probabilities
    result = select_window_length(rng, (1.0, 0.0, 0.0))
    assert result == 4

    result = select_window_length(rng, (0.0, 1.0, 0.0))
    assert result == 5

    result = select_window_length(rng, (0.0, 0.0, 1.0))
    assert result == 6

    # Test with probabilistic selection (should average to expected value over many trials)
    # Using a fixed seed for reproducibility
    rng = np.random.default_rng(123)
    results = [select_window_length(rng, (0.2, 0.5, 0.3)) for _ in range(1000)]
    avg = np.mean(results)
    expected = 4*0.2 + 5*0.5 + 6*0.3  # 5.1
    assert abs(avg - expected) < 0.1  # Should be close to expected value


def test_select_start_month():
    """Test start month selection."""
    rng = np.random.default_rng(42)

    # Test uniform distribution (None)
    result = select_start_month(rng, 4)  # Window length 4
    assert 0 <= result <= 8  # Valid range for window length 4 (0 to 12-4)

    # Test uniform distribution (None) with window length 6
    result = select_start_month(rng, 6)
    assert 0 <= result <= 6  # Valid range for window length 6 (0 to 12-6)

    # Test with custom distribution
    dist = [0.0] * 12
    dist[0] = 1.0  # Always January (0)
    result = select_start_month(rng, 4, dist)
    assert result == 0

    # Test with custom distribution for middle of year
    dist = [0.0] * 12
    dist[6] = 1.0  # Always July (6)
    result = select_start_month(rng, 4, dist)
    assert result == 6

    # Test invalid distribution (wrong length)
    try:
        select_start_month(rng, 4, [0.5, 0.5])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must have length 12" in str(e)

    # Test invalid distribution (doesn't sum to 1)
    try:
        select_start_month(rng, 4, [0.5, 0.4] + [0.0]*10)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must sum to 1.0" in str(e)

    # Test invalid distribution (negative values)
    try:
        select_start_month(rng, 4, [0.5, -0.1, 0.6] + [0.0]*9)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must contain non-negative values" in str(e)


def test_create_s2_mask():
    """Test S2 mask creation."""
    # Test basic functionality
    mask = create_s2_mask(
        n_samples=2,
        n_timesteps=12,
        n_bands=10,
        monthly_dropout=[0.1]*12,
        random_state=42
    )

    assert mask.shape == (2, 12, 10)
    assert mask.dtype == bool

    # With 10% dropout, roughly 90% should be True (kept)
    true_ratio = np.mean(mask)
    assert 0.85 < true_ratio < 0.95  # Allow some variation due to randomness

    # Test with all zeros dropout (no masking)
    mask = create_s2_mask(
        n_samples=2,
        n_timesteps=12,
        n_bands=10,
        monthly_dropout=[0.0]*12,
        random_state=42
    )
    assert np.all(mask == True)  # All should be True (kept)

    # Test with all ones dropout (all masked)
    mask = create_s2_mask(
        n_samples=2,
        n_timesteps=12,
        n_bands=10,
        monthly_dropout=[1.0]*12,
        random_state=42
    )
    assert np.all(mask == False)  # All should be False (masked)

    # Test invalid parameters
    try:
        create_s2_mask(2, 6, 10, [0.1]*12, 42)  # Wrong n_timesteps
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "n_timesteps must be 12" in str(e)

    try:
        create_s2_mask(2, 12, 5, [0.1]*12, 42)  # Wrong n_bands
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "n_bands must be 10" in str(e)

    try:
        create_s2_mask(2, 12, 10, [0.1]*6, 42)  # Wrong dropout length
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "monthly_dropout must have length 12" in str(e)

    try:
        create_s2_mask(2, 12, 10, [1.1]*12, 42)  # Invalid probability > 1
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be between 0 and 1" in str(e)


def test_apply_s2_masking():
    """Test applying S2 mask to data."""
    # Create test data
    np.random.seed(42)
    data = np.random.rand(2, 12, 10).astype(np.float64)

    # Create mask (all True = no masking)
    mask = np.ones((2, 12, 10), dtype=bool)

    # Apply mask
    masked_data = apply_s2_masking(data, mask, -9999.0)

    # Should be unchanged
    np.testing.assert_array_equal(masked_data, data)

    # Create mask (all False = full masking)
    mask = np.zeros((2, 12, 10), dtype=bool)

    # Apply mask
    masked_data = apply_s2_masking(data, mask, -9999.0)

    # Should be all -9999.0
    assert np.all(masked_data == -9999.0)

    # Create mixed mask
    mask = np.ones((2, 12, 10), dtype=bool)
    mask[0, 0, :] = False  # Mask first time step for first sample

    # Apply mask
    masked_data = apply_s2_masking(data, mask, -9999.0)

    # Check that the masked values are set correctly
    assert np.all(masked_data[0, 0, :] == -9999.0)
    # Check that unmasked values are unchanged
    np.testing.assert_array_equal(masked_data[0, 1:, :], data[0, 1:, :])
    np.testing.assert_array_equal(masked_data[1, :, :], data[1, :, :])

    # Test invalid shapes
    try:
        apply_s2_masking(np.random.rand(2, 12, 10), np.ones((2, 12, 9)), -9999.0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Data shape" in str(e) and "must match mask shape" in str(e)


def test_apply_competition_mask():
    """Test competition masking."""
    # Create test data
    np.random.seed(42)
    data = np.random.rand(3, 12, 12).astype(np.float64)

    # Create a simple config
    config = AquacultureConfig(
        simulate_mask=False,  # No random masking for this test
        random_state=42,
        window_length_probs=(1.0, 0.0, 0.0),  # Always 4 months
        start_month_distribution=[1.0] + [0.0]*11  # Always start at month 0 (January)
    )

    # Apply masking
    masked_data, metadata = apply_competition_mask(data, config)

    # Check that we got the expected metadata
    assert np.all(metadata['window_length'] == 4)
    assert np.all(metadata['start_month'] == 0)
    assert np.all(metadata['end_month'] == 3)

    # Check that SAR bands (indices 0,1) are unchanged
    for i in range(3):
        np.testing.assert_array_equal(masked_data[i, :, :2], data[i, :, :2])

    # Check that months 0-3 (January-April) have original S2 values
    for i in range(3):
        np.testing.assert_array_equal(
            masked_data[i, 0:4, 2:],
            data[i, 0:4, 2:]
        )

    # Check that months 4-11 (May-December) are masked (-9999)
    for i in range(3):
        assert np.all(masked_data[i, 4:, 2:] == -9999.0)

    # Test with simulate_mask=True
    config_sim = AquacultureConfig(
        simulate_mask=True,
        random_state=42,
        window_length_probs=(1.0, 0.0, 0.0),  # Always 4 months
        start_month_distribution=[0.5, 0.5] + [0.0]*10  # 50% Jan, 50% Feb
    )

    masked_data2, metadata2 = apply_competition_mask(data, config_sim)

    # With random masking, we should still have the same structure
    assert np.all(metadata2['window_length'] == 4)
    assert np.all((metadata2['start_month'] == 0) | (metadata2['start_month'] == 1))
    assert np.all(metadata2['end_month'] == metadata2['start_month'] + 3)

    # SAR bands should still be unchanged
    for i in range(3):
        np.testing.assert_array_equal(masked_data2[i, :, :2], data[i, :, :2])


def test_validate_masking():
    """Test masking validation."""
    # Create test data
    np.random.seed(42)
    original_data = np.random.rand(2, 12, 12).astype(np.float64)

    # Create a simple config
    config = AquacultureConfig(
        simulate_mask=False,
        random_state=42,
        window_length_probs=(1.0, 0.0, 0.0),  # Always 4 months
        start_month_distribution=None  # Uniform distribution
    )

    # Apply masking
    masked_data, metadata = apply_competition_mask(original_data, config)

    # Validate the masking
    validation = validate_masking(original_data, masked_data, metadata)

    # Check that we got the expected keys
    assert 'sar_unchanged' in validation
    assert 's2_outside_window_masked' in validation
    assert 's2_inside_window_preserved' in validation
    assert 'per_sample_stats' in validation

    # With perfect masking (no cloud simulation), these should all be True
    assert validation['sar_unchanged'] == True
    assert validation['s2_outside_window_masked'] == True
    assert validation['s2_inside_window_preserved'] == True

    # Check per-sample stats
    assert len(validation['per_sample_stats']) == 2  # 2 samples
    for i in range(2):
        stats = validation['per_sample_stats'][i]
        assert 'window_length' in stats
        assert 'start_month' in stats
        assert 'end_month' in stats
        assert 'sar_equal' in stats
        assert 's2_outside_masked_proportion' in stats
        assert 's2_inside_unchanged_proportion' in stats

        # Check specific values for our test case
        assert stats['window_length'] == 4
        assert 0 <= stats['start_month'] <= 8  # Valid range for window length 4
        assert stats['end_month'] == stats['start_month'] + 3
        assert stats['sar_equal'] == True
        assert stats['s2_outside_masked_proportion'] == 1.0  # All should be masked
        assert stats['s2_inside_unchanged_proportion'] == 1.0  # All should be preserved


if __name__ == "__main__":
    test_validate_probabilities()
    test_select_window_length()
    test_select_start_month()
    test_create_s2_mask()
    test_apply_s2_masking()
    test_apply_competition_mask()
    test_validate_masking()
    print("All tests passed!")