"""Offline tests for evaluate.py and generate_predictions.prepare_rows.   python tests/test_evaluate.py"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from smoke_test import build_toy_dataset  # noqa: E402
from evaluate import (CRITERIA, aggregate_human_eval, allowed_text, answer_risk, best_sentence_pred,  # noqa: E402
                      climate_sensitivity, export_human_eval, is_refusal, lexical_support, make_mt_candidates,
                      paired_bootstrap, read_jsonl, rouge_scores, tag_errors, term_accuracy, top_passage_pred,
                      ungrounded_dosage, ungrounded_numbers)
from generate_predictions import prepare_rows  # noqa: E402


def main():
    # --- metrics ---
    r = rouge_scores(["late blight spreads in cool wet weather"], ["late blight spreads in cool, wet weather"])
    assert r["rougeL"][0] > 0.95, r
    assert rouge_scores(["banana"], ["late blight"])["rougeL"][0] == 0
    bs = paired_bootstrap([0.5] * 40, [0.3] * 40)
    assert bs["ci_low"] > 0.19 and bs["p_diff_le_0"] == 0
    bs = paired_bootstrap([0.5, 0.1, 0.4, 0.2] * 5, [0.4, 0.2, 0.3, 0.3] * 5)
    assert bs["ci_low"] < 0 < bs["ci_high"] or abs(bs["mean_diff"]) < 1e-9

    assert answer_risk("The recent weather puts your potato field at high risk of late blight.") == "high"
    assert answer_risk("The risk of late blight in your potato field is low at the moment.") == "low"
    assert answer_risk("Remove infected plants.") is None
    assert is_refusal("I can only help with crop disease questions") and not is_refusal("Remove infected leaves early.")
    assert ungrounded_numbers("Use 50 ml per litre", "plants need 20 days") == ["50"]
    assert ungrounded_dosage("Spray 50 ml per litre.", "no numbers") and not ungrounded_dosage("Spray 50 ml.", "use 50 ml")
    assert lexical_support("late blight spreads quickly", ["late blight spreads quickly in rain"]) == 1.0
    assert top_passage_pred({"context": ["a b c d e f"]}) == "a b c d e f"
    assert "blight" in best_sentence_pred({"question": "What is blight in potato?", "context": [
        "Farmers plant seed in rows each year. Blight damages potato leaves badly in wet weather."]}).lower()

    rows_m = [{"en": "late blight", "rw": ["x1"]}, {"en": "maize", "rw": []}]
    ta = term_accuracy(["Late blight is bad.", "Maize is good.", "Late blight again."], ["x1 ni bibi", "ok", "nothing"], rows_m)
    assert ta["per_term"]["late blight"] == {"n": 2, "accuracy": 0.5} and "maize" not in ta["per_term"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        processed = build_toy_dataset(tmp)
        test = read_jsonl(processed / "test.jsonl")

        # --- prepare_rows ---
        assert all(r["context"] == [] for r in prepare_rows(test, "none"))
        gold = prepare_rows(test, "gold")
        assert all(r["context"] == r["gold_context"] for r in gold if r["type"] != "ood")
        assert all(r["climate"] is None for r in prepare_rows(test, climate_mode="none"))
        sh = [r for r in prepare_rows(test, climate_mode="shuffled") if r["type"] == "climate"]
        orig = {r["id"]: r for r in test}
        assert sh and all(r["climate"] != orig[r["id"]]["climate"] for r in sh)

        # --- climate sensitivity with an oracle system and a climate-blind system ---
        ref_preds = [r["answer"] for r in test]
        cs = climate_sensitivity(test, ref_preds)
        assert cs["risk_accuracy"] == 1.0 and (cs["pair_correct"] in (None, 1.0))
        blind = ["The risk is low." for _ in test]
        cs_blind = climate_sensitivity(test, blind)
        assert cs_blind["risk_accuracy"] < 1.0 or all(r["risk"] == "low" for r in test if r["type"] == "climate")

        # --- error tags ---
        passage = "Late blight damages potato leaves in cool wet weather. Farmers should scout their potato field often."
        qa = {"id": "q1", "type": "qa", "question": "How does late blight affect potato?", "answer": "It damages leaves.",
              "context": [passage], "gold_context": [passage], "gold_chunk_id": "d::0", "retrieved_ids": ["d::0"],
              "climate": None, "crop": "potato", "diseases": ["late blight"]}
        assert "ungrounded_dosage" in tag_errors(qa, "Spray 500 ml of fungicide per litre on the leaves of the crop.")
        assert tag_errors(qa, "I can only help with crop disease questions.") == ["refusal_on_valid"]
        ood = next(r for r in test if r["type"] == "ood")
        assert tag_errors(ood, "Paris is the capital.") == ["failed_to_refuse"] and tag_errors(ood, "Sorry, that is outside what I can help with.") == []
        cl = next(r for r in test if r["type"] == "climate")
        assert "climate_ignored" in tag_errors(cl, "Scout your field regularly and remove sick plants early.")

        # --- human eval export / aggregate ---
        preds = {"A": ref_preds, "B": blind}
        n = export_human_eval(test, preds, ["A", "B"], n_items=6, out_csv=str(tmp / "h" / "items.csv"), key_json=str(tmp / "h" / "key.json"))
        assert n > 0
        import csv
        with open(tmp / "h" / "items.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        key = json.loads((tmp / "h" / "key.json").read_text())
        for k, name in enumerate(["r1", "r2"]):
            for i, row in enumerate(rows):
                good = key[row["item_id"]]["system"] == "A"
                for c in CRITERIA:
                    row[c] = str((5 if good else 2) + (k if i % 3 == 0 else 0))
            with open(tmp / "h" / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        agg = aggregate_human_eval([tmp / "h" / "r1.csv", tmp / "h" / "r2.csv"], tmp / "h" / "key.json")
        assert agg["by_system"].loc["A", "correctness"] > agg["by_system"].loc["B", "correctness"]
        assert "correctness" in agg["weighted_kappa"] or n < 10

        # --- MT candidates ---
        k = make_mt_candidates(test, n=5, out_tsv=str(tmp / "mt.tsv"))
        assert k > 0 and (tmp / "mt.tsv").read_text().startswith("id\ten\trw")
    print("\nEVALUATE TEST PASSED")


if __name__ == "__main__":
    main()
