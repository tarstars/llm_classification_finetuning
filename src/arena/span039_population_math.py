"""Stable descriptive standardized differences, including constant columns."""
import math


def standardized_difference(mean_a,sd_a,mean_b,sd_b):
    values=[mean_a,sd_a,mean_b,sd_b]
    if not all(math.isfinite(float(x)) for x in values) or min(sd_a,sd_b)<0:
        raise ValueError('Invalid descriptive moments')
    difference=float(mean_b)-float(mean_a)
    denominator=math.hypot(float(sd_a),float(sd_b))/math.sqrt(2)
    if denominator<1e-12:
        return 0.0 if abs(difference)<1e-12 else None
    return difference/denominator
