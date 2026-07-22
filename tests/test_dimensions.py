from fractions import Fraction

import pytest

from fab_agent.domain.dimensions import format_inches, parse_dimension
from fab_agent.errors import DimensionParseError


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('4"', Fraction(4)),
        ("4 in", Fraction(4)),
        ('1 1/4"', Fraction(5, 4)),
        ('1¼"', Fraction(5, 4)),
        ("3' 7\"", Fraction(43)),
        ("3 ft 7 in", Fraction(43)),
        ("6'", Fraction(72)),
        ("10 ft", Fraction(120)),
        ('1/16"', Fraction(1, 16)),
        ("8'-4 1/2\"", Fraction(100, 1) + Fraction(1, 2)),
        ("4-6 1/4", Fraction(54, 1) + Fraction(1, 4)),
        ('4-6 1/4"', Fraction(54, 1) + Fraction(1, 4)),
        ('6 - 0"', Fraction(72)),
        ("4 \u2013 6¼", Fraction(54, 1) + Fraction(1, 4)),
    ],
)
def test_parse_dimension(raw: str, expected: Fraction) -> None:
    dimension = parse_dimension(raw)
    assert dimension.raw == raw
    assert dimension.inches == expected


def test_format_exact_inches() -> None:
    assert format_inches(Fraction(521, 4)) == "10' 10 1/4\""


@pytest.mark.parametrize(
    "raw",
    ["", "four inches", "1.25 in", "2 meters", "1-1/4", "4-12"],
)
def test_rejects_unsupported_dimensions(raw: str) -> None:
    with pytest.raises(DimensionParseError):
        parse_dimension(raw)
