"""One frozen conservative opening-decline detector for H77."""
import re

import numpy as np

from .data import parse_turns
from .span032_refusal import OPENING_CHARACTERS, REQUEST_PATTERNS, FEATURE_NAMES


# No scanning for later quoted speech, generic physical 'we', or arbitrary
# first-person inability after a substantive answer has already started.
PREFACE = r"(?:(?:i(?:'m| am) sorry|i apologize|sorry|unfortunately)\s*[,!.]?\s*(?:but\s+)?|as an? (?:ai|language model|ai language model)[^.!?\n]{0,100}?,\s*)?"
ACTOR = r"i\s+(?:cannot|can't|can’t|won't|will not|am unable to|am not able to)\s+"
ACTIONS = r"(?:provide|assist|help|generate|create|write|fulfil[l]?|comply|support|answer|complete|continue|translate|determine|predict|verify|confirm|access|browse|run|execute|do\s+(?:that|this))\b"
REFUSAL = re.compile(r'\A\s*' + PREFACE + ACTOR + ACTIONS, re.I)


def describe(frame, encoded=True):
    texts = [frame[k].map(parse_turns).tolist() if encoded else frame[k].tolist()
             for k in ['prompt', 'response_a', 'response_b']]
    prompt, a, b = texts
    flags = [np.array([bool(REFUSAL.search(str(s)[:OPENING_CHARACTERS])) for s in values])
             for values in [a, b]]
    direction = flags[0].astype(float) - flags[1].astype(float)
    affected = direction != 0
    contexts = np.array([[bool(p.search(str(s))) for p in REQUEST_PATTERNS] for s in prompt], float)
    length = np.log1p([len(str(s)) for s in a]) - np.log1p([len(str(s)) for s in b])
    code = np.array(['```' in str(s) for s in a], float) - np.array(['```' in str(s) for s in b], float)
    return np.column_stack([direction, direction[:, None] * contexts,
                            affected * length, affected * code]), affected, flags
