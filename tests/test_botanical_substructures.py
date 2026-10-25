"""
Tests for BotanicalSubstructureScorer, Knapsack mixture purging, and pipeline integration.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from rdkit import Chem

from src.reranking.botanical_substructures import (
    BotanicalSubstructureScorer,
    extract_peak_mzs,
)
from src.reranking.fragment_library import KnapsackAssembler, purge_mixtures
from src.pipeline import CASMIOmegaPipeline
from src.data.multi_energy_fusion import SpectralFrame


# Test compound SMILES
QUERCETIN = "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"  # Aglycone (MW ~302.04)
ISOQUERCITRIN = "O=C1C(O[C@@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)=C(c2ccc(O)c(O)c2)Oc2cc(O)cc(O)c12"  # Glucoside (MW ~464.09)
QUERCITRIN = "CC1OC(Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)C(O)C(O)C1O"  # Rhamnoside
QUERCETIN_GLUCURONIDE = "O=C(O)C1OC(Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)C(O)C(O)C1O"  # Glucuronide
EGCG = "O=C(Oc1cc(O)c2c(c1)OC(c1cc(O)c(O)c(O)c1)C(O)C2)c1cc(O)c(O)c(O)c1"  # Gallate ester
CHLOROGENIC = "O=C(O[C@@H]1C[C@](O)(C(=O)O)C[C@@H](O)[C@H]1O)/C=C/c1ccc(O)c(O)c1"  # Caffeoyl ester
PSOROMIC_ACID = "Cc1cc2c(c(O)c1C=O)C(=O)Oc1c(C)cc(O)c(C(=O)O)c1O2"  # Lichen depsidone

# Synthetic distractors
SULFAMETHOXAZOLE = "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1"  # Synthetic sulfonamide
SULFONAMIDE_DISTRACTOR = "Cc1ccc(S(=O)(=O)Nc2ccccc2)cc1"  # Synthetic sulfonamide
CIPROFLOXACIN = "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"  # Synthetic fluorinated quinolone
CHLOROBENZENE_SULFONAMIDE = "Clc1ccc(S(=O)(=O)N)cc1"  # Halogenated sulfonamide


class TestBotanicalSubstructureScorer:
    """Unit tests for BotanicalSubstructureScorer diagnostic logic."""

    def test_initialization(self):
        scorer = BotanicalSubstructureScorer(tolerance_da=0.025)
        assert scorer.tolerance_da == 0.025
        assert "hexose" in scorer._botanical_patterns
        assert "flavonoid" in scorer._botanical_patterns
        assert "depsidone" in scorer._botanical_patterns
        assert "sulfonamide" in scorer._distractor_patterns
        assert "halogenated_ring" in scorer._distractor_patterns

    def test_extract_peak_mzs_formats(self):
        # 2D array
        arr_2d = np.array([[100.5, 50.0], [200.25, 80.0]])
        assert extract_peak_mzs(arr_2d) == [100.5, 200.25]

        # 1D array
        arr_1d = np.array([150.1, 250.2])
        assert extract_peak_mzs(arr_1d) == [150.1, 250.2]

        # List of tuples
        tuples = [(123.4, 10.0), (567.8, 20.0)]
        assert extract_peak_mzs(tuples) == [123.4, 567.8]

        # String
        str_peaks = "100.1:50;200.2:80"
        mzs = extract_peak_mzs(str_peaks)
        assert len(mzs) == 2
        assert abs(mzs[0] - 100.1) < 0.01

        # None / empty
        assert extract_peak_mzs(None) == []
        assert extract_peak_mzs([]) == []

    def test_glycoside_neutral_loss_scoring_quercetin_glucoside_vs_aglycone(self):
        """Verify hexose neutral loss scoring favors quercetin glucoside over aglycone."""
        scorer = BotanicalSubstructureScorer()

        # Query spectrum with precursor at ~464.095 Da and aglycone fragment at 302.042 Da
        # Neutral loss = 464.095 - 302.042 = 162.053 Da (hexose loss)
        fused_peaks = np.array([[302.042, 100.0], [153.019, 45.0]])
        neutral_mass = 464.095

        sig = scorer.detect_spectral_signatures(fused_peaks, neutral_mass)
        assert sig["hexose"] is True
        assert sig["flavonoid"] is True

        score_glucoside = scorer.score_candidate(ISOQUERCITRIN, fused_peaks, neutral_mass)
        score_aglycone = scorer.score_candidate(QUERCETIN, fused_peaks, neutral_mass)

        # Quercetin glucoside possesses the detected hexose moiety; aglycone does not
        assert score_glucoside >= 0.85
        assert score_aglycone <= 0.65
        assert score_glucoside > score_aglycone
        assert (score_glucoside - score_aglycone) >= 0.20

    def test_depsidone_and_flavonoid_scoring_vs_synthetic_sulfonamide_distractor(self):
        """Verify depsidone and flavonoid scoring heavily penalizes synthetic sulfonamides."""
        scorer = BotanicalSubstructureScorer()

        # Spectrum of depsidone (CO2 loss: 43.990 Da, depsidone fragment: 168.040 Da)
        neutral_mass_dep = 358.070
        fused_peaks_dep = np.array([[314.080, 100.0], [168.040, 50.0]])  # NL ~43.99 Da

        sig_dep = scorer.detect_spectral_signatures(fused_peaks_dep, neutral_mass_dep)
        assert sig_dep["depsidone"] is True

        score_depsidone = scorer.score_candidate(PSOROMIC_ACID, fused_peaks_dep, neutral_mass_dep)
        score_sulfonamide = scorer.score_candidate(SULFONAMIDE_DISTRACTOR, fused_peaks_dep, neutral_mass_dep)

        # Depsidone gets high score; synthetic sulfonamide is severely penalized
        assert score_depsidone >= 0.85
        assert score_sulfonamide <= 0.10
        assert (score_depsidone - score_sulfonamide) >= 0.75

        # Also test flavonoid vs synthetic sulfonamide under flavonoid RDA spectrum
        neutral_mass_flav = 302.042
        fused_peaks_flav = np.array([[153.019, 100.0], [137.024, 60.0]])

        sig_flav = scorer.detect_spectral_signatures(fused_peaks_flav, neutral_mass_flav)
        assert sig_flav["flavonoid"] is True

        score_flav = scorer.score_candidate(QUERCETIN, fused_peaks_flav, neutral_mass_flav)
        score_sulfa_flav = scorer.score_candidate(SULFAMETHOXAZOLE, fused_peaks_flav, neutral_mass_flav)

        assert score_flav >= 0.85
        assert score_sulfa_flav <= 0.10
        assert (score_flav - score_sulfa_flav) >= 0.75

    def test_synthetic_halogenated_ring_penalty(self):
        """Verify synthetic halogenated rings are penalized under botanical spectra."""
        scorer = BotanicalSubstructureScorer()
        neutral_mass = 302.042
        fused_peaks = np.array([[153.019, 100.0], [137.024, 50.0]])

        score_natural = scorer.score_candidate(QUERCETIN, fused_peaks, neutral_mass)
        score_halogen = scorer.score_candidate(CIPROFLOXACIN, fused_peaks, neutral_mass)
        score_halo_sulfa = scorer.score_candidate(CHLOROBENZENE_SULFONAMIDE, fused_peaks, neutral_mass)

        assert score_natural >= 0.85
        assert score_halogen <= 0.10
        assert score_halo_sulfa <= 0.05

    def test_rhamnose_deoxyhexose_scoring(self):
        """Verify rhamnose loss (146.058 Da) selectively boosts rhamnosides."""
        scorer = BotanicalSubstructureScorer()
        neutral_mass = 448.100  # Quercitrin neutral mass
        # Peak at 302.042 Da -> NL = 448.100 - 302.042 = 146.058 Da (rhamnose loss)
        fused_peaks = np.array([[302.042, 100.0]])

        sig = scorer.detect_spectral_signatures(fused_peaks, neutral_mass)
        assert sig["rhamnose"] is True

        score_rhamnoside = scorer.score_candidate(QUERCITRIN, fused_peaks, neutral_mass)
        score_aglycone = scorer.score_candidate(QUERCETIN, fused_peaks, neutral_mass)

        assert score_rhamnoside >= 0.85
        assert score_aglycone <= 0.65
        assert score_rhamnoside > score_aglycone

    def test_glucuronide_neutral_loss_scoring(self):
        """Verify glucuronide loss (176.032 Da) selectively boosts glucuronides."""
        scorer = BotanicalSubstructureScorer()
        neutral_mass = 478.075  # Quercetin glucuronide neutral mass
        # Peak at 302.042 Da -> NL = 478.075 - 302.042 = 176.033 Da
        fused_peaks = np.array([[302.042, 100.0]])

        sig = scorer.detect_spectral_signatures(fused_peaks, neutral_mass)
        assert sig["glucuronide"] is True

        score_glucuronide = scorer.score_candidate(QUERCETIN_GLUCURONIDE, fused_peaks, neutral_mass)
        score_aglycone = scorer.score_candidate(QUERCETIN, fused_peaks, neutral_mass)

        assert score_glucuronide >= 0.85
        assert score_aglycone <= 0.65
        assert score_glucuronide > score_aglycone

    def test_galloyl_and_caffeoyl_scoring(self):
        """Verify galloyl (152.011 / 153.019) and caffeoyl (162.032) signature scoring."""
        scorer = BotanicalSubstructureScorer()

        # Galloyl spectrum (peak at 153.019 Da)
        sig_gall = scorer.detect_spectral_signatures(np.array([[153.019, 100.0]]), 458.08)
        assert sig_gall["galloyl"] is True
        score_egcg = scorer.score_candidate(EGCG, np.array([[153.019, 100.0]]), 458.08)
        assert score_egcg >= 0.85

        # Caffeoyl spectrum (NL 162.032 Da)
        # Chlorogenic neutral mass 354.095, peak at 192.063 -> NL = 162.032 Da
        sig_caff = scorer.detect_spectral_signatures(np.array([[192.063, 100.0]]), 354.095)
        assert sig_caff["caffeoyl"] is True
        score_chlorogenic = scorer.score_candidate(CHLOROGENIC, np.array([[192.063, 100.0]]), 354.095)
        assert score_chlorogenic >= 0.85

    def test_neutral_and_empty_spectrum_fallbacks(self):
        """Verify safe fallbacks when spectrum is empty or no botanical signatures detected."""
        scorer = BotanicalSubstructureScorer()

        # Empty spectrum
        assert scorer.score_candidate(QUERCETIN, None, 302.042) == 0.60
        assert scorer.score_candidate(SULFONAMIDE_DISTRACTOR, None, 302.042) == 0.15
        assert scorer.score_candidate(CIPROFLOXACIN, None, 302.042) == 0.25

        # None / invalid mol
        assert scorer.score_candidate(None, np.array([[153.019, 100.0]]), 302.042) == 0.0
        assert scorer.score_candidate("INVALID_SMILES_STRING", np.array([[153.019, 100.0]]), 302.042) == 0.0


class TestKnapsackMixturePurge:
    """Tests verifying disconnected '+' mixtures are strictly purged."""

    def test_knapsack_assembler_filter_mixtures(self):
        assembler = KnapsackAssembler()
        library = {
            200000: ["frag20"],
            300000: ["frag30"],
            500000: ["frag50"],
            1000000: ["frag100_intact"],
        }

        # With filter_mixtures=False (legacy mode), combination contains '+'
        res_legacy = assembler.assemble(100.0, library, max_depth=2, filter_mixtures=False)
        assert any("+" in c for c in res_legacy)

        # With filter_mixtures=True, NO combination contains '+'
        res_purged = assembler.assemble(100.0, library, max_depth=2, filter_mixtures=True)
        assert len(res_purged) > 0
        for c in res_purged:
            assert "+" not in c

    def test_purge_mixtures_helper(self):
        cands = ["c1ccccc1", "c1ccccc1+c2ccccc2", "CC(=O)O", "fragA+fragB+fragC"]
        purged = purge_mixtures(cands)
        assert purged == ["c1ccccc1", "CC(=O)O"]
        assert all("+" not in c for c in purged)


class TestPipelineBotanicalIntegration:
    """Integration tests verifying BotanicalSubstructureScorer in CASMIOmegaPipeline."""

    def test_pipeline_instantiates_botanical_scorer(self):
        pipeline = CASMIOmegaPipeline()
        assert hasattr(pipeline, "botanical_scorer")
        assert isinstance(pipeline.botanical_scorer, BotanicalSubstructureScorer)

    def test_pipeline_purges_knapsack_plus_mixtures(self, tmp_path: Path):
        # Configure pipeline with mock fragment library that would produce '+' mixtures
        mock_library = {
            200000: ["fragA"],
            300000: ["fragB"],
            500000: ["fragC"],
        }
        pipeline = CASMIOmegaPipeline(fragment_library=mock_library)

        test_df = pd.DataFrame({
            "id": ["TEST_SPEC"],
            "precursor_mz": [95.0496],
            "adduct": ["[M+H]+"],
            "polarity": ["positive"],
            "peaks": ["50.0:100.0;94.0418:50.0"],
        })

        out_csv = tmp_path / "sub1.csv"
        sub = pipeline.run(test_df, output_path=out_csv)

        assert len(sub) == 1
        # Check last_aggregated_cands: no entry should have '+'
        for c in pipeline.last_aggregated_cands:
            assert "+" not in str(c.get("smiles", ""))
            assert "+" not in str(c.get("inchikey14", ""))

        # Check candidate row string
        cand_str = sub.iloc[0]["candidates"]
        assert "+" not in cand_str

    def test_pipeline_boosts_botanical_natural_product_over_sulfur_synthetic(self, tmp_path: Path):
        """Verify authentic natural product candidate is boosted to Rank 1 over synthetic distractor."""
        # Create pipeline with two database candidates: Quercetin and a synthetic sulfonamide
        db = {
            "C15H10O7": [
                (QUERCETIN, "IK14QUERCETIN1", np.ones(3, dtype=np.float32)),
                (SULFONAMIDE_DISTRACTOR, "IK14SULFONAMD1", np.ones(3, dtype=np.float32)),
            ]
        }
        pipeline = CASMIOmegaPipeline(db=db)

        # Query spectrum with precursor 303.0496 and flavonoid RDA fragments (153.019 Da)
        test_df = pd.DataFrame({
            "id": ["FLAV_SPEC"],
            "precursor_mz": [303.0496],
            "adduct": ["[M+H]+"],
            "polarity": ["positive"],
            "peaks": ["153.019:100.0;137.024:50.0;303.049:80.0"],
        })

        out_csv = tmp_path / "sub2.csv"
        sub = pipeline.run(test_df, output_path=out_csv)

        assert len(sub) == 1
        top_candidates = sub.iloc[0]["candidates"].split(";")
        # Authentic flavonoid natural product must be Rank 1!
        assert top_candidates[0] == "IK14QUERCETIN1"
        # Synthetic distractor must NOT be Rank 1
        assert top_candidates[0] != "IK14SULFONAMD1"
