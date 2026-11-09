"""Reproduce mathematical checks of the independent review; no data or network."""
import json
import math
from pathlib import Path


def main():
    before = [0.0005, 0.49975, 0.49975]
    after = [0.00001, 0.499995, 0.499995]
    assert abs(sum(before) - 1) < 1e-12
    assert abs(sum(after) - 1) < 1e-12
    drift = max(abs(a - b) for a, b in zip(before, after))
    loss_increase = math.log(before[0] / after[0])
    assert drift <= 0.00049 + 1e-15
    assert loss_increase / 25000 > 0.0001

    # Jensen compares with the mean component loss, not the original alone.
    original_loss, reverse_loss, averaged_loss = map(lambda p: -math.log(p), [0.9, 0.1, 0.5])
    assert averaged_loss <= (original_loss + reverse_loss) / 2
    assert averaged_loss > original_loss

    lower_bound = 2 * 0.83053 - 0.83140
    rounding_low = 2 * (0.83053 - 0.000005) - (0.83140 + 0.000005)
    result = {
        "uses_competition_rows": False,
        "probability_drift_counterexample": {
            "max_probability_difference": drift,
            "true_class_loss_increase": loss_increase,
            "mean_increase_if_one_of_25000_rows": loss_increase / 25000,
            "conclusion": "Absolute probability drift alone cannot bound log-loss drift by 1e-4.",
        },
        "order_averaging_counterexample": {
            "original_loss": original_loss,
            "reverse_loss": reverse_loss,
            "averaged_loss": averaged_loss,
            "conclusion": "Jensen does not guarantee improvement over the original orientation alone.",
        },
        "conditional_two_expert_convexity": {
            "lower_bound_at_gemma_weight_0_6": lower_bound,
            "lower_bound_allowing_score_rounding": rounding_low,
            "rank19_cutoff_snapshot": 0.82915,
            "assumptions": "Identical scored rows, base predictions and postprocessing except mixture weight; losses rounded to five decimals.",
            "applies_to_H60_three_role_path": False,
        },
        "arithmetic_prevalence_algebra": {
            "original_affected_fraction": 73 / 57477,
            "conditional_gain_needed_for_0_00072_overall": 0.00072 / (73 / 57477),
            "conclusion": "Prevalence alone supplies no numerical log-loss ceiling; affected baseline loss is also required. No hidden prevalence or gain is estimated.",
        },
        "official_uncertainty": "Cannot estimate paired standard error from aggregate leaderboard scores alone.",
    }
    path = Path(__file__).resolve().parents[1] / "results/reviews/20261109_claim_checks.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("Review checks passed: two counterexamples and conditional convexity/prevalence calculations.")


if __name__ == "__main__":
    main()
