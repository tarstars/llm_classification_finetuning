"""Input-only context routing shared by qualification and competition inference."""
import numpy as np


def changed_mask(short_original,short_swapped,long_original,long_swapped):
    assert len(short_original)==len(short_swapped)==len(long_original)==len(long_swapped)
    return np.array([a!=b or c!=d for a,b,c,d in zip(short_original,long_original,short_swapped,long_swapped)],dtype=bool)


def competition_rerun(environment,test_sha256,visible_sha256):
    return environment.get('KAGGLE_IS_COMPETITION_RERUN','').strip().lower() in ['true','1'] or test_sha256!=visible_sha256
