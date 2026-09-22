"""Unit tests for exact metric evaluation harness and Bemis-Murcko scaffold splitter."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    calculate_reciprocal_rank,
    evaluate_predictions,
)
from src.evaluation.splitter import (
    BemisMurckoSplitter,
    get_scaffold,
)


class TestCalculateReciprocalRank:
    """Test calculate_reciprocal_rank under all rank positions and boundary conditions."""

    def test_rank_1_match(self):
        pred = ["IK14TARGET0001", "IK14OTHER00002", "IK14OTHER00003"]
        rr = calculate_reciprocal_rank("IK14TARGET0001", pred, k=25)
        assert rr == 1.0

    def test_rank_2_match(self):
        pred = ["IK14OTHER00001", "IK14TARGET0001", "IK14OTHER00003"]
        rr = calculate_reciprocal_rank("IK14TARGET0001", pred, k=25)
        assert rr == 0.5

    def test_rank_3_match(self):
        pred = ["IK14OTHER00001", "IK14OTHER00002", "IK14TARGET0001"]
        rr = calculate_reciprocal_rank("IK14TARGET0001", pred, k=25)
        assert pytest.approx(rr, 1e-6) == 1.0 / 3.0

    def test_rank_k_boundary_exact_hit(self):
        # 5 slots, target is exactly at slot 5 (index 4) with k=5
        pred = [f"DISTRACTOR{i:04d}" for i in range(4)] + ["TARGET_IK14_01"]
        rr = calculate_reciprocal_rank("TARGET_IK14_01", pred, k=5)
        assert pytest.approx(rr, 1e-6) == 1.0 / 5.0

    def test_rank_outside_k(self):
        # Target is at slot 6 (index 5), but cutoff k=5
        pred = [f"DISTRACTOR{i:04d}" for i in range(5)] + ["TARGET_IK14_01"]
        rr = calculate_reciprocal_rank("TARGET_IK14_01", pred, k=5)
        assert rr == 0.0

    def test_target_not_found(self):
        pred = ["IK14OTHER00001", "IK14OTHER00002"]
        rr = calculate_reciprocal_rank("IK14TARGET0001", pred, k=25)
        assert rr == 0.0

    def test_empty_ground_truth(self):
        pred = ["IK14OTHER00001", "IK14TARGET0001"]
        assert calculate_reciprocal_rank("", pred, k=25) == 0.0
        assert calculate_reciprocal_rank(None, pred, k=25) == 0.0  # type: ignore

    def test_empty_predictions(self):
        assert calculate_reciprocal_rank("IK14TARGET0001", [], k=25) == 0.0
        assert calculate_reciprocal_rank("IK14TARGET0001", None, k=25) == 0.0  # type: ignore

    def test_k_zero_or_negative(self):
        pred = ["IK14TARGET0001"]
        assert calculate_reciprocal_rank("IK14TARGET0001", pred, k=0) == 0.0
        assert calculate_reciprocal_rank("IK14TARGET0001", pred, k=-10) == 0.0

    def test_case_insensitivity(self):
        pred = ["ik14target0001", "other"]
        rr = calculate_reciprocal_rank("IK14TARGET0001", pred, k=25)
        assert rr == 1.0

        pred_upper = ["IK14TARGET0001", "other"]
        rr_lower_target = calculate_reciprocal_rank("ik14target0001", pred_upper, k=25)
        assert rr_lower_target == 1.0

    def test_14_char_inchikey_prefix_matching(self):
        # Target is full InChIKey (27 chars), pred has 14-char prefix
        full_ik = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
        prefix_ik14 = "BSYNRYMUTXBXSQ"

        # Case 1: Target full InChIKey, prediction is 14-char prefix
        pred1 = ["DISTRACTOR", prefix_ik14]
        assert calculate_reciprocal_rank(full_ik, pred1, k=25) == 0.5

        # Case 2: Target 14-char prefix, prediction is full InChIKey
        pred2 = ["DISTRACTOR", full_ik]
        assert calculate_reciprocal_rank(prefix_ik14, pred2, k=25) == 0.5

        # Case 3: Target and prediction are both full InChIKeys with matching 14-char skeleton
        stereoisomer_ik = "BSYNRYMUTXBXSQ-VAAAWNOHSA-N"
        pred3 = [stereoisomer_ik]
        assert calculate_reciprocal_rank(full_ik, pred3, k=25) == 1.0

    def test_duplicates_in_predictions_first_occurrence_determines_rank(self):
        # Target appears at rank 2 and again at rank 4
        pred = ["OTHER", "TARGET_IK14", "ANOTHER", "TARGET_IK14", "FINAL"]
        rr = calculate_reciprocal_rank("TARGET_IK14", pred, k=25)
        assert rr == 0.5

    def test_duplicates_of_distractors_in_predictions(self):
        pred = ["DISTRACTOR", "DISTRACTOR", "TARGET_IK14"]
        rr = calculate_reciprocal_rank("TARGET_IK14", pred, k=25)
        assert pytest.approx(rr, 1e-6) == 1.0 / 3.0

    def test_whitespace_tolerance(self):
        pred = ["  IK14TARGET0001 \n"]
        assert calculate_reciprocal_rank(" IK14TARGET0001 ", pred, k=25) == 1.0


class TestEvaluatePredictions:
    """Test evaluate_predictions for aggregate metrics, aliases, and edge cases."""

    def test_evaluate_predictions_standard(self):
        ground_truth = {
            "S1": "IK14TARGET0001",
            "S2": "IK14TARGET0002",
            "S3": "IK14TARGET0003",
        }
        predictions = {
            "S1": ["IK14TARGET0001", "IK14PAD0000001"],  # rank 1 -> RR 1.0
            "S2": ["IK14OTHER00001", "IK14TARGET0002"],  # rank 2 -> RR 0.5
            "S3": ["IK14OTHER00001", "IK14OTHER00002"],  # not in top 2 -> RR 0.0
        }
        metrics = evaluate_predictions(ground_truth, predictions, ks=(1, 5, 25))

        # Expected MRR@25 = (1.0 + 0.5 + 0.0) / 3.0 = 0.5
        assert pytest.approx(metrics["mrr@25"], 1e-4) == 0.5
        # Expected MRR@1 = (1.0 + 0.0 + 0.0) / 3.0 = 1/3
        assert pytest.approx(metrics["mrr@1"], 1e-4) == 1.0 / 3.0
        # Expected Top-1 Accuracy: S1 only (1/3)
        assert pytest.approx(metrics["top1_accuracy"], 1e-4) == 1.0 / 3.0
        assert pytest.approx(metrics["top_1_accuracy"], 1e-4) == 1.0 / 3.0
        # Expected Top-5 Accuracy: S1 and S2 (2/3)
        assert pytest.approx(metrics["top5_accuracy"], 1e-4) == 2.0 / 3.0
        assert pytest.approx(metrics["top_5_accuracy"], 1e-4) == 2.0 / 3.0

    def test_empty_ground_truth(self):
        metrics = evaluate_predictions({}, {"S1": ["IK1401"]}, ks=(1, 5, 10, 25))
        assert isinstance(metrics, dict)
        for k in (1, 5, 10, 25):
            assert metrics[f"mrr@{k}"] == 0.0
            assert metrics[f"top{k}_accuracy"] == 0.0
            assert metrics[f"top_{k}_accuracy"] == 0.0

    def test_missing_queries_in_predictions(self):
        ground_truth = {
            "S1": "TARGET1",
            "S2": "TARGET2",
            "S3": "TARGET3",
        }
        # Only S1 is in predictions; S2 and S3 are missing
        predictions = {
            "S1": ["TARGET1"],
        }
        metrics = evaluate_predictions(ground_truth, predictions, ks=(1, 25))
        assert pytest.approx(metrics["mrr@25"], 1e-4) == 1.0 / 3.0
        assert pytest.approx(metrics["top1_accuracy"], 1e-4) == 1.0 / 3.0
        assert pytest.approx(metrics["top25_accuracy"], 1e-4) == 1.0 / 3.0

    def test_empty_predictions_dict(self):
        ground_truth = {"S1": "TARGET1", "S2": "TARGET2"}
        metrics = evaluate_predictions(ground_truth, {}, ks=(1, 25))
        assert metrics["mrr@1"] == 0.0
        assert metrics["mrr@25"] == 0.0
        assert metrics["top1_accuracy"] == 0.0
        assert metrics["top25_accuracy"] == 0.0

    def test_all_correct_rank1(self):
        ground_truth = {f"S{i}": f"TARGET{i}" for i in range(10)}
        predictions = {f"S{i}": [f"TARGET{i}", "OTHER"] for i in range(10)}
        metrics = evaluate_predictions(ground_truth, predictions, ks=(1, 5, 25))
        assert metrics["mrr@1"] == 1.0
        assert metrics["mrr@25"] == 1.0
        assert metrics["top1_accuracy"] == 1.0
        assert metrics["top25_accuracy"] == 1.0


class TestGetScaffold:
    """Test get_scaffold with various chemistry structures and edge cases."""

    def test_benzene_derivatives_exact(self):
        scaff_phenol = get_scaffold("c1ccccc1O")
        scaff_aniline = get_scaffold("c1ccccc1N")
        assert scaff_phenol == "c1ccccc1"
        assert scaff_aniline == "c1ccccc1"

    def test_benzene_derivatives_generic(self):
        scaff_generic = get_scaffold("c1ccccc1O", generic=True)
        assert scaff_generic == "C1CCCCC1"

    def test_acyclic_molecules_return_empty_string(self):
        assert get_scaffold("CCO") == ""  # ethanol
        assert get_scaffold("CC(=O)O") == ""  # acetic acid
        assert get_scaffold("CCCC") == ""  # butane
        assert get_scaffold("C#N") == ""  # hydrogen cyanide

    def test_invalid_smiles_graceful_degradation(self):
        assert get_scaffold("INVALID_SMILES_123") == "INVALID_SMILES_123"
        assert get_scaffold("") == ""
        assert get_scaffold(None) == ""  # type: ignore

    def test_fused_rings(self):
        # 1-naphthol and 2-naphthol both have naphthalene scaffold
        scaff1 = get_scaffold("Oc1cccc2ccccc12")
        scaff2 = get_scaffold("Oc1ccc2ccccc2c1")
        assert scaff1 == scaff2
        assert "c1ccc2ccccc2c1" in scaff1 or "c1ccccc1" in scaff1


class TestBemisMurckoSplitter:
    """Test BemisMurckoSplitter for zero scaffold leakage, edge cases, and balance."""

    def test_disjoint_scaffolds_strictly_zero_overlap(self):
        df = pd.DataFrame({
            "id": [f"M{i}" for i in range(6)],
            "smiles": [
                "c1ccccc1O", "c1ccccc1N",               # Benzene scaffold
                "c1ccc2ccccc2c1", "c1ccc2cc(O)ccc2c1",  # Naphthalene scaffold
                "C1CCCCC1", "C1CCCC(O)C1",               # Cyclohexane scaffold
            ],
        })
        splitter = BemisMurckoSplitter(n_splits=3, random_state=42)
        splits = list(splitter.split(df, smiles_col="smiles"))
        assert len(splits) == 3

        for train_idx, val_idx in splits:
            # Check indices cover all rows without overlap
            all_idx = np.sort(np.concatenate([train_idx, val_idx]))
            np.testing.assert_array_equal(all_idx, np.arange(len(df)))
            assert len(np.intersect1d(train_idx, val_idx)) == 0

            # MATHEMATICAL GUARANTEE: Strictly zero scaffold overlap
            train_scaffolds = {get_scaffold(df.iloc[i]["smiles"]) for i in train_idx}
            val_scaffolds = {get_scaffold(df.iloc[i]["smiles"]) for i in val_idx}
            assert len(train_scaffolds.intersection(val_scaffolds)) == 0

    def test_five_splits_default_zero_overlap(self):
        # 10 different scaffolds, 3 molecules each = 30 molecules total
        scaffolds = [
            "c1ccccc1",      # benzene
            "c1ccncc1",      # pyridine
            "c1cnccn1",      # pyrazine
            "c1ccc2ccccc2c1",  # naphthalene
            "C1CCCCC1",      # cyclohexane
            "C1CCCC1",       # cyclopentane
            "c1sccc1",       # thiophene
            "c1occc1",       # furan
            "c1[nH]ccc1",    # pyrrole
            "c1c2ccccc2oc1",  # benzofuran
        ]
        smiles_list = []
        for s in scaffolds:
            smiles_list.extend([f"{s}", f"{s}O", f"{s}N"])  # 3 variations each

        df = pd.DataFrame({"smiles": smiles_list})
        splitter = BemisMurckoSplitter(n_splits=5, random_state=42)
        splits = list(splitter.split(df))
        assert len(splits) == 5

        for train_idx, val_idx in splits:
            train_scaffolds = {get_scaffold(df.iloc[i]["smiles"]) for i in train_idx}
            val_scaffolds = {get_scaffold(df.iloc[i]["smiles"]) for i in val_idx}
            assert len(train_scaffolds.intersection(val_scaffolds)) == 0
            assert len(val_idx) > 0
            assert len(train_idx) > 0

    def test_acyclic_molecules_grouped_together(self):
        # Acyclic molecules have scaffold ""
        df = pd.DataFrame({
            "smiles": ["CCO", "CCC", "CCCC", "CC(=O)O", "c1ccccc1O", "c1ccccc1N"]
        })
        splitter = BemisMurckoSplitter(n_splits=2, random_state=42)
        splits = list(splitter.split(df))

        for train_idx, val_idx in splits:
            train_scaffolds = {get_scaffold(df.iloc[i]["smiles"]) for i in train_idx}
            val_scaffolds = {get_scaffold(df.iloc[i]["smiles"]) for i in val_idx}
            assert len(train_scaffolds.intersection(val_scaffolds)) == 0

    def test_single_scaffold_dataset(self):
        # All molecules share the same scaffold
        df = pd.DataFrame({
            "smiles": ["c1ccccc1O", "c1ccccc1N", "c1ccccc1Cl", "c1ccccc1Br"]
        })
        splitter = BemisMurckoSplitter(n_splits=3, random_state=42)
        splits = list(splitter.split(df))
        assert len(splits) == 3

        for train_idx, val_idx in splits:
            train_scaffolds = {get_scaffold(df.iloc[i]["smiles"]) for i in train_idx}
            val_scaffolds = {get_scaffold(df.iloc[i]["smiles"]) for i in val_idx}
            assert len(train_scaffolds.intersection(val_scaffolds)) == 0

    def test_unparseable_smiles_dataset(self):
        df = pd.DataFrame({
            "smiles": [
                "BAD_SMILES_1", "BAD_SMILES_1",
                "BAD_SMILES_2",
                "c1ccccc1O", "c1ccccc1N",
            ]
        })
        splitter = BemisMurckoSplitter(n_splits=2, random_state=42)
        splits = list(splitter.split(df))
        assert len(splits) == 2

        for train_idx, val_idx in splits:
            train_scaffolds = {get_scaffold(df.iloc[i]["smiles"]) for i in train_idx}
            val_scaffolds = {get_scaffold(df.iloc[i]["smiles"]) for i in val_idx}
            assert len(train_scaffolds.intersection(val_scaffolds)) == 0

    def test_generic_scaffold_splitter(self):
        # Phenol and Cyclohexanol share generic scaffold "C1CCCCC1"
        df = pd.DataFrame({
            "smiles": [
                "c1ccccc1O",   # generic: C1CCCCC1
                "C1CCCCC1O",   # generic: C1CCCCC1
                "c1ccc2ccccc2c1",  # generic: bicyclic
            ]
        })
        splitter = BemisMurckoSplitter(n_splits=2, random_state=42, generic=True)
        splits = list(splitter.split(df))

        for train_idx, val_idx in splits:
            train_scaffolds = {get_scaffold(df.iloc[i]["smiles"], generic=True) for i in train_idx}
            val_scaffolds = {get_scaffold(df.iloc[i]["smiles"], generic=True) for i in val_idx}
            assert len(train_scaffolds.intersection(val_scaffolds)) == 0

    def test_non_integer_dataframe_index(self):
        df = pd.DataFrame(
            {"smiles": ["c1ccccc1O", "c1ccc2ccccc2c1"]},
            index=["mol_alpha", "mol_beta"],
        )
        splitter = BemisMurckoSplitter(n_splits=2, random_state=42)
        splits = list(splitter.split(df))
        assert len(splits) == 2

        for train_idx, val_idx in splits:
            # Should be integer arrays suitable for .iloc
            assert isinstance(train_idx, np.ndarray)
            assert np.issubdtype(train_idx.dtype, np.integer)
            assert isinstance(val_idx, np.ndarray)
            assert np.issubdtype(val_idx.dtype, np.integer)
            # iloc access must succeed without error
            _ = df.iloc[train_idx]
            _ = df.iloc[val_idx]

    def test_custom_smiles_col(self):
        df = pd.DataFrame({
            "canonical_smiles": ["c1ccccc1O", "c1ccc2ccccc2c1"],
        })
        splitter = BemisMurckoSplitter(n_splits=2, random_state=42)
        splits = list(splitter.split(df, smiles_col="canonical_smiles"))
        assert len(splits) == 2

    def test_missing_smiles_col_raises_key_error(self):
        df = pd.DataFrame({"other_col": ["c1ccccc1O"]})
        splitter = BemisMurckoSplitter(n_splits=2)
        with pytest.raises(KeyError, match="not found in DataFrame"):
            list(splitter.split(df, smiles_col="smiles"))

    def test_invalid_n_splits_raises_value_error(self):
        with pytest.raises(ValueError, match="n_splits must be at least 2"):
            BemisMurckoSplitter(n_splits=1)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"smiles": []})
        splitter = BemisMurckoSplitter(n_splits=3)
        splits = list(splitter.split(df))
        assert len(splits) == 3
        for train_idx, val_idx in splits:
            assert len(train_idx) == 0
            assert len(val_idx) == 0

    def test_get_n_splits(self):
        splitter = BemisMurckoSplitter(n_splits=5)
        assert splitter.get_n_splits() == 5
