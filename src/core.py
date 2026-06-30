"""PcamVLM core: prompts, label maps, logit-level yes/no argmax."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PCAM_QUESTION = (
    "Does the center region of this H&E stained lymph node patch "
    "contain tumor tissue? Answer yes or no."
)

NCT_CRC_QUESTION_TEMPLATE = (
    "This is a histopathology image patch from the NCT-CRC-HE-100K dataset. "
    "The tissue type is one of: {labels}. "
    "Which tissue type is shown? Reply with exactly one label from the list."
)


def nct_crc_question(label_choices: Sequence[str]) -> str:
    return NCT_CRC_QUESTION_TEMPLATE.format(labels=", ".join(label_choices))


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

PCAM_LABEL_NAMES: Dict[int, str] = {0: "no", 1: "yes"}

NCT_CRC_LABEL_NAMES: Dict[int, str] = {
    0: "ADI",   # adipose
    1: "BACK",  # background
    2: "DEB",   # debris
    3: "LYM",   # lymphocytes
    4: "MUC",   # mucus
    5: "MUS",   # smooth muscle
    6: "NORM",  # normal colon mucosa
    7: "STR",   # cancer-associated stroma
    8: "TUM",   # colorectal adenocarcinoma epithelium
}

NCT_CRC_LABEL_CHOICES: List[str] = list(NCT_CRC_LABEL_NAMES.values())


def dataset_question(task: str) -> str:
    """Return the question prompt for a given task name."""
    if task == "pcam":
        return PCAM_QUESTION
    if task == "nct_crc":
        return nct_crc_question(NCT_CRC_LABEL_CHOICES)
    raise ValueError(f"unknown task: {task!r}")


def dataset_answer_for_label(task: str, label: int) -> str:
    """Return the textual answer that should be supervised for `label`."""
    if task == "pcam":
        return PCAM_LABEL_NAMES[int(label)]
    if task == "nct_crc":
        return NCT_CRC_LABEL_NAMES[int(label)]
    raise ValueError(f"unknown task: {task!r}")


def parse_answer(task: str, text: str, label_choices: Iterable[str] | None = None) -> int:
    """Parse a free-form model output to an integer label.

    Robust against extra whitespace, leading punctuation, and capitalization.
    Falls back to 0 if no choice is recognised (caller should treat as miss).
    """
    if text is None:
        return 0
    s = text.strip().lower()
    s = s.lstrip(" .,;:!?\"'`()[]{}")
    if not s:
        return 0
    if task == "pcam":
        if s.startswith("yes"):
            return 1
        if s.startswith("no"):
            return 0
        return 0
    if task == "nct_crc":
        choices = [c.lower() for c in (label_choices or NCT_CRC_LABEL_CHOICES)]
        for i, c in enumerate(choices):
            if s == c or s.startswith(c):
                return i
        # Fallback: substring match
        for i, c in enumerate(choices):
            if c in s:
                return i
        return 0
    raise ValueError(f"unknown task: {task!r}")


# ---------------------------------------------------------------------------
# Logit-level yes/no argmax (more reliable than free-text parsing for PCam)
# ---------------------------------------------------------------------------

def yes_no_token_ids(tokenizer) -> Dict[str, int]:
    """Return the token id for the bare 'yes' and 'no' strings.

    Tries the leading-space variant first because Gemma3 tokenizers prepend a
    space to most word-initial continuations; both are kept as fallback.
    """
    out: Dict[str, int] = {}
    for word in ("yes", "no"):
        ids = tokenizer.encode(word, add_special_tokens=False)
        out[word] = ids[0] if ids else -1
    return out


def yes_no_logit_scores(logits_row, tokenizer) -> Dict[str, float]:
    """Given a 1-D logits row at the final prompt position, return
    softmax probabilities over {yes, no}."""
    import torch
    tid = yes_no_token_ids(tokenizer)
    if tid["yes"] < 0 or tid["no"] < 0:
        return {"yes": 0.0, "no": 1.0}
    pair = torch.tensor([tid["yes"], tid["no"]], dtype=torch.long)
    pair_logits = logits_row.index_select(-1, pair.to(logits_row.device))
    probs = pair_logits.softmax(dim=-1)
    return {"yes": float(probs[0]), "no": float(probs[1])}


__all__ = [
    "PCAM_QUESTION",
    "NCT_CRC_QUESTION_TEMPLATE",
    "nct_crc_question",
    "PCAM_LABEL_NAMES",
    "NCT_CRC_LABEL_NAMES",
    "NCT_CRC_LABEL_CHOICES",
    "dataset_question",
    "dataset_answer_for_label",
    "parse_answer",
    "yes_no_token_ids",
    "yes_no_logit_scores",
]