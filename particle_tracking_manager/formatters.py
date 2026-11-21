"""Formatting helper functions."""

from itertools import chain
from typing import Iterable

from pydantic import ValidationError


def format_location(loc: Iterable[int | str]) -> str:
    """Format the validation error field location list."""
    return ".".join(str(l) for l in loc)


def pretty_validation_errors(val_err: ValidationError) -> str:
    """Format validation errors into easier to read lines."""

    errs = val_err.errors(include_url=False)
    missing = (
        f"{format_location(e['loc'])} - Missing required field"
        for e in errs
        if e["type"] == "missing"
    )
    other = (
        f"{format_location(e['loc'])} - {e['msg']} (your value: {e['input']})"
        for e in errs
        if e["type"] != "missing"
    )
    return "\n".join(chain(missing, other))
