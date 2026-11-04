"""Grouped nested comparisons of fixed probability recipes and budgeted routing."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .span_calibration import SymmetricCalibration, grouped_folds
from .span_heads import normalize, loss_rows


def entropy(probabilities):
    p = normalize(probabilities)
    return -(p * np.log(p)).sum(1)


def combine(first, second, weight, conditional=False):
    p, q = normalize(first), normalize(second)
    if not conditional:
        return normalize((1 - weight) * p + weight * q)
    direction = ((1 - weight) * p[:, :2] / p[:, :2].sum(1, keepdims=True)
                 + weight * q[:, :2] / q[:, :2].sum(1, keepdims=True))
    return normalize(np.column_stack([(1 - p[:, 2, None]) * direction, p[:, 2]]))


def fit_weight(p, q, y, mask=None, conditional=False):
    if mask is not None:
        p, q, y = p[mask], q[mask], y[mask]
    if len(y) == 0:
        return 0.
    def objective(weight):
        return float(loss_rows(y, combine(p, q, weight, conditional)).mean())
    fit = minimize_scalar(objective, bounds=(0., 1.), method="bounded",
                          options={"xatol": 1e-8})
    assert fit.success
    return min([0., float(fit.x), 1.], key=objective)


class RecipeSet:
    def __init__(self, config):
        self.config = config

    def calibrator(self, index, family=None):
        return SymmetricCalibration(
            family or ("temperature" if index == 0 else "temperature_tie"),
            self.config["classifier_temperature_bounds" if index == 0 else "judge_temperature_bounds"],
            self.config["tie_intercept_bounds"])

    def fit(self, first, second, y, groups, random_scores):
        first, second = normalize(first), normalize(second)
        y = np.asarray(y, int)
        out = [np.full_like(first, np.nan), np.full_like(second, np.nan)]
        for train, val in grouped_folds(y, groups, self.config["inner_group_folds"], self.config["seed"]):
            for index, data in enumerate([first, second]):
                out[index][val] = self.calibrator(index).fit(data[train], y[train]).predict_proba(data[val])
        p, q = out
        assert np.isfinite(p).all() and np.isfinite(q).all()
        self.fits = [self.calibrator(index).fit(data, y)
                     for index, data in enumerate([first, second])]
        vector_weight = fit_weight(p, q, y)
        post = self.calibrator(0, "temperature_tie").fit(combine(p, q, vector_weight), y)
        self.recipes = {
            "vector": {"weight": vector_weight},
            "conditional": {"weight": fit_weight(p, q, y, conditional=True), "conditional": True},
            "vector_tie": {"weight": vector_weight, "post_theta": post.theta.tolist()},
        }
        for kind in ["entropy", "random"]:
            scores = entropy(p) if kind == "entropy" else np.asarray(random_scores)
            for fraction in self.config["route_fractions"]:
                threshold = float(np.quantile(scores, 1 - fraction))
                routed = scores >= threshold
                self.recipes[f"{kind}_{round(100 * fraction)}"] = {
                    "kind": kind, "fraction": fraction, "threshold": threshold,
                    "weight": fit_weight(p, q, y, routed),
                    "training_routed_fraction": float(routed.mean()),
                }
        return self

    def predict(self, first, second, random_scores):
        p, q = [fit.predict_proba(data) for fit, data in zip(self.fits, [first, second])]
        outputs, masks = {"classifier": p, "judge": q}, {}
        for name, recipe in self.recipes.items():
            combined = combine(p, q, recipe["weight"], recipe.get("conditional", False))
            if "post_theta" in recipe:
                post = self.calibrator(0, "temperature_tie")
                post.theta = np.array(recipe["post_theta"])
                combined = post.predict_proba(combined)
            if "kind" in recipe:
                scores = entropy(p) if recipe["kind"] == "entropy" else np.asarray(random_scores)
                mask = scores >= recipe["threshold"]
                combined = np.where(mask[:, None], combined, p)
                masks[name] = mask
            outputs[name] = normalize(combined)
        return outputs, masks

    def to_dict(self):
        return {"component_theta": [fit.theta.tolist() for fit in self.fits], "recipes": self.recipes}

    @classmethod
    def from_dict(cls, values, config):
        model = cls(config)
        model.fits = []
        for index, theta in enumerate(values["component_theta"]):
            fit = model.calibrator(index)
            fit.theta = np.array(theta)
            model.fits.append(fit)
        model.recipes = values["recipes"]
        return model


def candidate_groups(config):
    return {"selected_blend": config["blend_families"],
            **{f"selected_{kind}": [f"{kind}_{round(100 * fraction)}" for fraction in config["route_fractions"]]
               for kind in ["entropy", "random"]}}


def select_recipes(first, second, y, groups, random_scores, config):
    """Each validation prediction excludes its group's calibration, gate and weight labels."""
    inner_outputs = {}
    for train, val in grouped_folds(y, groups, config["inner_group_folds"], config["seed"]):
        model = RecipeSet(config).fit(first[train], second[train], y[train], groups[train], random_scores[train])
        outputs, _ = model.predict(first[val], second[val], random_scores[val])
        for name, p in outputs.items():
            if name not in inner_outputs:
                inner_outputs[name] = np.full_like(first, np.nan)
            inner_outputs[name][val] = p
    scores = {name: float(loss_rows(y, normalize(p)).mean()) for name, p in inner_outputs.items()}
    selected = {name: min(candidates, key=scores.get)
                for name, candidates in candidate_groups(config).items()}
    return selected, scores


def nested_recipes(first, second, y, groups, random_scores, config, progress=None):
    outputs, masks, details = {}, {}, []
    for index, (train, val) in enumerate(grouped_folds(y, groups, config["outer_group_folds"], config["seed"])):
        selected, inner_scores = select_recipes(first[train], second[train], y[train],
                                                groups[train], random_scores[train], config)
        model = RecipeSet(config).fit(first[train], second[train], y[train], groups[train], random_scores[train])
        predicted, routed = model.predict(first[val], second[val], random_scores[val])
        for label, chosen in selected.items():
            predicted[label] = predicted[chosen]
            if chosen in routed:
                routed[label] = routed[chosen]
        for name, p in predicted.items():
            if name not in outputs:
                outputs[name] = np.full_like(first, np.nan)
            outputs[name][val] = p
        for name, mask in routed.items():
            if name not in masks:
                masks[name] = np.zeros(len(y), bool)
            masks[name][val] = mask
        details.append({"fold": index, "train_rows": len(train), "validation_rows": len(val),
                        "selected": selected, "inner_log_loss": inner_scores, **model.to_dict()})
        if progress:
            progress(index, selected)
    assert all(np.isfinite(p).all() for p in outputs.values())
    return outputs, masks, details
