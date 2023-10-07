"""Set of CLI utilities for analyzing errors and phonetic markers."""
import re


def get_ethnicity(filename):
    dmg_match = re.match(
        r"[A-z]{2,3}\d{1,3}(\w)(\w)\d\w(?:rtn|orig)?", filename.split('_')[0]
    )
    if not dmg_match:
        raise RuntimeError(f"The filename is malformed: {filename}")
    return dmg_match.group(1)