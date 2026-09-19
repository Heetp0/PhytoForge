import numpy as np
import pytest
from src.data.multi_energy_fusion import (
    MultiEnergyFusionEngine,
    SpectralFrame,
    fuse_peaks,
)
from src.data.loader import QuerySpectrum


def test_multi_energy_fusion_pooling():
    engine = MultiEnergyFusionEngine(embedding_dim=1024)
    frame_20ev = SpectralFrame(
        collision_energy=20.0,
        peaks=np.array([[100.0, 50.0], [200.0, 100.0]]),
        embedding=np.ones(1024, dtype=np.float32) * 0.2,
    )
    frame_50ev = SpectralFrame(
        collision_energy=50.0,
        peaks=np.array([[50.0, 80.0], [100.0, 40.0]]),
        embedding=np.ones(1024, dtype=np.float32) * 0.8,
    )

    fused_emb = engine.fuse_embeddings([frame_20ev, frame_50ev])
    assert fused_emb.shape == (1024,)
    # Fused vector should be normalized to unit length
    norm = np.linalg.norm(fused_emb)
    assert np.isclose(norm, 1.0, atol=1e-4)
    assert not np.isnan(fused_emb).any()


def test_multi_energy_fusion_weighting_favors_optimal_ce():
    engine = MultiEnergyFusionEngine(embedding_dim=1024)
    emb_a = np.zeros(1024, dtype=np.float32)
    emb_a[:512] = 1.0
    frame_optimal = SpectralFrame(
        collision_energy=35.0,
        peaks=np.array([[100.0, 100.0]]),
        embedding=emb_a,
    )

    emb_b = np.zeros(1024, dtype=np.float32)
    emb_b[512:] = 1.0
    frame_distant = SpectralFrame(
        collision_energy=80.0,
        peaks=np.array([[50.0, 50.0]]),
        embedding=emb_b,
    )

    fused = engine.fuse_embeddings([frame_optimal, frame_distant])
    norm_a_portion = np.linalg.norm(fused[:512])
    norm_b_portion = np.linalg.norm(fused[512:])
    assert norm_a_portion > norm_b_portion * 5.0


def test_multi_energy_fusion_single_frame():
    engine = MultiEnergyFusionEngine(embedding_dim=1024)
    raw_emb = np.arange(1024, dtype=np.float32) + 1.0
    frame = SpectralFrame(
        collision_energy=35.0,
        peaks=np.array([[150.0, 50.0]]),
        embedding=raw_emb,
    )
    fused = engine.fuse_embeddings([frame])
    assert fused.shape == (1024,)
    assert np.isclose(np.linalg.norm(fused), 1.0, atol=1e-5)
    expected = raw_emb / np.linalg.norm(raw_emb)
    assert np.allclose(fused, expected, atol=1e-5)


def test_multi_energy_fusion_empty_raises():
    engine = MultiEnergyFusionEngine(embedding_dim=1024)
    with pytest.raises(ValueError, match="Cannot fuse empty list"):
        engine.fuse_embeddings([])


def test_fuse_peaks_merges_and_deduplicates():
    frame_1 = SpectralFrame(
        collision_energy=20.0,
        peaks=np.array([[100.00, 50.0], [200.00, 100.0]]),
    )
    frame_2 = SpectralFrame(
        collision_energy=50.0,
        peaks=np.array([[50.00, 80.0], [100.01, 40.0]]),
    )

    fused_peaks = fuse_peaks([frame_1, frame_2], tolerance_da=0.02)
    assert isinstance(fused_peaks, np.ndarray)
    assert fused_peaks.shape == (3, 2)

    assert np.all(np.diff(fused_peaks[:, 0]) > 0)
    assert np.isclose(fused_peaks[0, 0], 50.00, atol=1e-4)
    assert np.isclose(fused_peaks[0, 1], 80.0)

    expected_mz = (100.00 * 50.0 + 100.01 * 40.0) / 90.0
    assert np.isclose(fused_peaks[1, 0], expected_mz, atol=1e-4)
    assert np.isclose(fused_peaks[1, 1], 50.0)

    assert np.isclose(fused_peaks[2, 0], 200.00, atol=1e-4)
    assert np.isclose(fused_peaks[2, 1], 100.0)


def test_fuse_peaks_sum_aggregation():
    frame_1 = SpectralFrame(
        collision_energy=20.0,
        peaks=np.array([[100.00, 50.0]]),
    )
    frame_2 = SpectralFrame(
        collision_energy=40.0,
        peaks=np.array([[100.01, 40.0]]),
    )
    fused_sum = fuse_peaks([frame_1, frame_2], tolerance_da=0.02, intensity_aggregation="sum")
    assert fused_sum.shape == (1, 2)
    assert np.isclose(fused_sum[0, 1], 90.0)


def test_fuse_peaks_edge_cases():
    assert fuse_peaks([]).shape == (0, 2)

    empty_frame = SpectralFrame(collision_energy=35.0, peaks=np.zeros((0, 2)))
    assert fuse_peaks([empty_frame]).shape == (0, 2)

    single_peak_frame = SpectralFrame(collision_energy=35.0, peaks=np.array([123.45, 67.8]))
    result = fuse_peaks([single_peak_frame])
    assert result.shape == (1, 2)
    assert np.isclose(result[0, 0], 123.45)
    assert np.isclose(result[0, 1], 67.8)


def test_engine_full_fuse_interface():
    engine = MultiEnergyFusionEngine(embedding_dim=1024)
    frame_1 = SpectralFrame(
        collision_energy=20.0,
        peaks=np.array([[100.0, 50.0]]),
        embedding=np.ones(1024, dtype=np.float32),
    )
    frame_2 = SpectralFrame(
        collision_energy=40.0,
        peaks=np.array([[150.0, 75.0]]),
        embedding=np.ones(1024, dtype=np.float32) * 2.0,
    )

    fused_emb, fused_peaks = engine.fuse([frame_1, frame_2])
    assert fused_emb.shape == (1024,)
    assert np.isclose(np.linalg.norm(fused_emb), 1.0, atol=1e-4)
    assert fused_peaks.shape == (2, 2)


def test_from_query_spectra_conversion():
    engine = MultiEnergyFusionEngine(embedding_dim=1024)
    qs1 = QuerySpectrum(
        molecule_id="MOL001",
        precursor_mz=250.1,
        adduct="[M+H]+",
        polarity="positive",
        collision_energy=20.0,
        mz_array=[100.0, 150.0],
        intensity_array=[30.0, 70.0],
    )
    qs2 = QuerySpectrum(
        molecule_id="MOL001",
        precursor_mz=250.1,
        adduct="[M+H]+",
        polarity="positive",
        collision_energy=40.0,
        mz_array=[100.01, 200.0],
        intensity_array=[25.0, 90.0],
    )

    embs = [np.ones(1024, dtype=np.float32) * 0.5, np.ones(1024, dtype=np.float32) * 1.5]
    frames = engine.from_query_spectra([qs1, qs2], embeddings=embs)
    assert len(frames) == 2
    assert frames[0].collision_energy == 20.0
    assert frames[0].peaks.shape == (2, 2)
    assert frames[1].collision_energy == 40.0
    assert frames[1].peaks.shape == (2, 2)

    fused_emb, fused_peaks = engine.fuse(frames)
    assert fused_emb.shape == (1024,)
    assert fused_peaks.shape == (3, 2)
