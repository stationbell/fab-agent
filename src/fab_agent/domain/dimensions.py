"""Exact imperial dimension parsing without floating-point arithmetic."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from fab_agent.errors import DimensionParseError

_UNICODE_FRACTIONS = {
    "¼": Fraction(1, 4),
    "½": Fraction(1, 2),
    "¾": Fraction(3, 4),
    "⅛": Fraction(1, 8),
    "⅜": Fraction(3, 8),
    "⅝": Fraction(5, 8),
    "⅞": Fraction(7, 8),
    "⅙": Fraction(1, 6),
    "⅚": Fraction(5, 6),
}

_DIMENSION_PUNCTUATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2033": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": "-",
    }
)


@dataclass(frozen=True, slots=True)
class Dimension:
    raw: str
    inches: Fraction

    @property
    def display(self) -> str:
        return format_inches(self.inches)


def _parse_number(value: str) -> Fraction:
    value = value.strip()
    try:
        unicode_value = next((item for item in _UNICODE_FRACTIONS if item in value), None)
        if unicode_value:
            whole = value.replace(unicode_value, "").strip()
            return (
                Fraction(int(whole), 1) + _UNICODE_FRACTIONS[unicode_value]
                if whole
                else _UNICODE_FRACTIONS[unicode_value]
            )
        parts = value.split()
        if len(parts) == 2 and "/" in parts[1]:
            return Fraction(int(parts[0]), 1) + Fraction(parts[1])
        if len(parts) == 1:
            return Fraction(parts[0])
    except (ValueError, ZeroDivisionError) as exc:
        raise DimensionParseError(f"Invalid number: {value!r}") from exc
    raise DimensionParseError(f"Invalid number: {value!r}")


def _normalize_dimension_text(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.strip().lower().translate(_DIMENSION_PUNCTUATION))


def parse_dimension(raw: str) -> Dimension:
    """Parse common feet/inch handwriting forms into an exact Fraction of inches."""

    text = _normalize_dimension_text(raw)
    if not text:
        raise DimensionParseError("Dimension is empty")

    feet_match = re.fullmatch(
        r"(?P<feet>\d+)\s*(?:'|ft|feet|foot)\s*(?:[- ]\s*)?"
        r"(?:(?P<inches>\d+(?:\s+\d+/\d+)?|\d+/\d+|\d*[¼½¾⅛⅜⅝⅞⅙⅚])\s*(?:\"|in|inches?)?)?",
        text,
    )
    if feet_match:
        feet = Fraction(int(feet_match.group("feet")) * 12, 1)
        inches_raw = feet_match.group("inches")
        inches = _parse_number(inches_raw) if inches_raw else Fraction(0)
        return Dimension(raw=raw, inches=feet + inches)

    # Handwritten field dimensions and vision transcripts commonly omit both
    # unit marks while retaining the feet-inches dash (for example,
    # ``4-6 1/4``). Require a whole-inch component after the dash so a nominal
    # size such as ``1-1/4`` is not silently reinterpreted as 1 ft 1/4 in.
    shorthand_match = re.fullmatch(
        r"(?P<feet>\d+)\s*[-\u2013\u2014]\s*"
        r"(?P<inches>\d+(?:\s+\d+/\d+)?|\d+[¼½¾⅛⅜⅝⅞⅙⅚])"
        r'\s*(?:"|in|inches?)?',
        text,
    )
    if shorthand_match:
        inches = _parse_number(shorthand_match.group("inches"))
        if inches >= 12:
            raise DimensionParseError(f"Invalid feet-inches shorthand: {raw!r}")
        feet = Fraction(int(shorthand_match.group("feet")) * 12, 1)
        return Dimension(raw=raw, inches=feet + inches)

    inch_match = re.fullmatch(
        r"(?P<inches>\d+(?:\s+\d+/\d+)?|\d+/\d+|\d*[¼½¾⅛⅜⅝⅞⅙⅚])\s*(?:\"|in|inches?)?",
        text,
    )
    if inch_match:
        return Dimension(raw=raw, inches=_parse_number(inch_match.group("inches")))
    raise DimensionParseError(f"Unsupported dimension: {raw!r}")


_FEET_MARKERS = re.compile(r"['\u2032\u2019]|\bft\b|\bfeet\b|\bfoot\b", re.IGNORECASE)
_HYPHENATED_NOMINAL_SIZE = re.compile(
    r"^(?P<whole>\d+)\s*[-\u2013\u2014]\s*(?P<fraction>\d+/\d+)"
    r'(?P<unit>\s*(?:"|in|inches?)?)$',
    re.IGNORECASE,
)


def parse_nominal_size(raw: str) -> Dimension:
    """Parse a nominal component size, which is always inches and never feet.

    Nominal sizes are conventionally written with a hyphen between the whole
    number and the fraction, as in ``2-1/2`` or ``1-1/4``. That same text is
    genuinely ambiguous for a length, so :func:`parse_dimension` still rejects
    it; only sizes accept the hyphenated form.
    """

    text = _normalize_dimension_text(raw)
    if _FEET_MARKERS.search(text):
        raise DimensionParseError(f"Nominal size must be written in inches: {raw!r}")
    normalized = _HYPHENATED_NOMINAL_SIZE.sub(r"\g<whole> \g<fraction>\g<unit>", text)
    return Dimension(raw=raw, inches=parse_dimension(normalized).inches)


def parse_stated_total(raw: str) -> Dimension:
    """Parse a dimension from a total field while preserving its raw source text."""

    text = raw.strip()
    text = re.sub(r"^total(?:\s+length)?\s*[:=-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[-:]?\s*total(?:\s+length)?$", "", text, flags=re.IGNORECASE)
    parsed = parse_dimension(text)
    return Dimension(raw=raw, inches=parsed.inches)


def format_nominal_size(value: Fraction) -> str:
    """Format a nominal size in inches; a size is never rolled up into feet."""

    whole, numerator = divmod(value.numerator, value.denominator)
    fraction = Fraction(numerator, value.denominator)
    if not fraction:
        return f'{whole}"'
    if not whole:
        return f'{fraction.numerator}/{fraction.denominator}"'
    return f'{whole} {fraction.numerator}/{fraction.denominator}"'


def format_inches(value: Fraction) -> str:
    """Format a length as feet and exact residual inches."""

    sign = "-" if value < 0 else ""
    positive = abs(value)
    feet, remainder_numerator = divmod(positive.numerator, positive.denominator * 12)
    remainder = Fraction(remainder_numerator, positive.denominator)
    whole_inches, fraction_numerator = divmod(remainder.numerator, remainder.denominator)
    fraction = Fraction(fraction_numerator, remainder.denominator)
    inch_text = str(whole_inches)
    if fraction:
        inch_text += f" {fraction.numerator}/{fraction.denominator}"
    if feet:
        return f"{sign}{feet}' {inch_text}\""
    return f'{sign}{inch_text}"'
