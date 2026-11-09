"""Fixed text descriptors for the prospective H16 refusal experiment."""
import re

import numpy as np

from .data import parse_turns


OPENING_CHARACTERS = 600
REFUSAL = re.compile(
    r"\b(?:i|we)\s+(?:(?:am|are)\s+(?:unable|not able)\s+to|cannot|can't|can’t|won't|will not)\b",
    re.IGNORECASE)
REQUEST_PATTERNS = [
    re.compile(r'\b(?:kill|murder|weapon|bomb|hack|steal|suicide|self.harm|porn|illegal|racist)\b', re.I),
    re.compile(r'\b(?:who|when|real|imaginary|cite|source|fact|verify|latest|current|today)\b', re.I),
    re.compile(r'\b(?:internet|browse|access|real.time|up.to.date|execute|image|picture|attachment|file|url)\b|https?://', re.I),
]
FEATURE_NAMES = ['signed_refusal', 'refusal_sensitive_keyword', 'refusal_factual_keyword',
                 'refusal_capability_keyword', 'log_character_difference', 'code_fence_difference']


def describe(frame, encoded=True):
    """Inspect text only. No labels, IDs or unavailable metadata are consulted."""
    texts = [frame[k].map(parse_turns).tolist() if encoded else frame[k].tolist()
             for k in ['prompt', 'response_a', 'response_b']]
    prompt, a, b = texts
    flags = [np.array([bool(REFUSAL.search(str(s)[:OPENING_CHARACTERS])) for s in values])
             for values in [a, b]]
    direction = flags[0].astype(float) - flags[1].astype(float)
    affected = direction != 0
    contexts = np.array([[bool(pattern.search(str(s))) for pattern in REQUEST_PATTERNS]
                         for s in prompt], dtype=float)
    length = np.log1p([len(str(s)) for s in a]) - np.log1p([len(str(s)) for s in b])
    code = np.array(['```' in str(s) for s in a], float) - np.array(['```' in str(s) for s in b], float)
    features = np.column_stack([direction, direction[:, None] * contexts,
                               affected * length, affected * code])
    return features, affected, flags
