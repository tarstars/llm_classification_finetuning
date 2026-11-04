"""Probability semantics, portable freezing and exclusion of evaluation labels."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arena.span_cascade import RecipeSet, combine, fit_weight, select_recipes


class CascadeContracts(unittest.TestCase):
    def setUp(self):
        self.config = {"seed": 11, "inner_group_folds": 3,
                       "classifier_temperature_bounds": [.5, 3.],
                       "judge_temperature_bounds": [.25, 8.], "tie_intercept_bounds": [-1., 1.],
                       "route_fractions": [.1, .2], "blend_families": ["vector", "conditional", "vector_tie"]}

    def test_conditional_blend_preserves_tie_mass_and_exact_winner_endpoints(self):
        p = np.array([[.2, .3, .5], [.4, .5, .1]])
        q = np.array([[.8, .1, .1], [.1, .2, .7]])
        np.testing.assert_allclose(combine(p, q, 0, True), p, atol=1e-12)
        out = combine(p, q, 1, True)
        np.testing.assert_allclose(out[:, 2], p[:, 2], atol=1e-12)
        np.testing.assert_allclose(out[:, :2] / out[:, :2].sum(1, keepdims=True),
                                   q[:, :2] / q[:, :2].sum(1, keepdims=True), atol=1e-12)
        np.testing.assert_allclose(combine(p, q, 0), p)
        np.testing.assert_allclose(combine(p, q, 1), q)
        self.assertEqual(fit_weight(p, q, np.array([0, 1]), np.zeros(2, bool)), 0.)

    def test_recipe_roundtrip_swap_equivariance_and_unrouted_rows(self):
        rng = np.random.default_rng(8)
        p, q = rng.dirichlet([2, 2, 2], 180), rng.dirichlet([2, 2, 2], 180)
        y, groups, u = np.arange(180) % 3, np.arange(180) // 2, rng.random(180)
        model = RecipeSet(self.config).fit(p, q, y, groups, u)
        original, masks = model.predict(p, q, u)
        copy = RecipeSet.from_dict(json.loads(json.dumps(model.to_dict())), self.config)
        restored, _ = copy.predict(p, q, u)
        swapped, _ = model.predict(p[:, [1, 0, 2]], q[:, [1, 0, 2]], u)
        for name in original:
            np.testing.assert_array_equal(restored[name], original[name])
            np.testing.assert_allclose(swapped[name][:, [1, 0, 2]], original[name], atol=1e-12)
        for name, mask in masks.items():
            np.testing.assert_allclose(original[name][~mask], original["classifier"][~mask], atol=1e-12)
        np.testing.assert_allclose(original["conditional"][:, 2], original["classifier"][:, 2], atol=1e-12)

    def test_nested_training_selection_cannot_see_external_evaluation_labels(self):
        rng = np.random.default_rng(9)
        p, q = rng.dirichlet([2, 2, 2], 240), rng.dirichlet([2, 2, 2], 240)
        y, groups, u = np.arange(240) % 3, np.arange(240) // 2, rng.random(240)
        changed = y.copy()
        changed[180:] = (changed[180:] + 1) % 3
        values = []
        for labels in [y, changed]:
            selection = select_recipes(p[:180], q[:180], labels[:180], groups[:180], u[:180], self.config)
            model = RecipeSet(self.config).fit(p[:180], q[:180], labels[:180], groups[:180], u[:180])
            values.append((selection, model.to_dict(), model.predict(p[180:], q[180:], u[180:])[0]))
        self.assertEqual(values[0][:2], values[1][:2])
        for name in values[0][2]:
            np.testing.assert_array_equal(values[0][2][name], values[1][2][name])


if __name__ == "__main__":
    unittest.main()
