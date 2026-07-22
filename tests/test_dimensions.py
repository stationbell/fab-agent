from fractions import Fraction

import pytest

from fab_agent.domain.dimensions import (
    format_inches,
    format_nominal_size,
    parse_dimension,
    parse_nominal_size,
    parse_stated_total,
)
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


@pytest.mark.parametrize("raw", ["9'-7\" Total", "Total: 9'-7\"", "TOTAL LENGTH 9'-7\""])
def test_parse_stated_total_allows_only_the_field_label(raw: str) -> None:
    dimension = parse_stated_total(raw)

    assert dimension.raw == raw
    assert dimension.inches == Fraction(115)


def test_normal_dimension_parser_remains_strict_about_field_labels() -> None:
    with pytest.raises(DimensionParseError):
        parse_dimension("9'-7\" Total")


@pytest.mark.parametrize(
    "raw",
    ["", "four inches", "1.25 in", "2 meters", "1-1/4", "4-12"],
)
def test_rejects_unsupported_dimensions(raw: str) -> None:
    with pytest.raises(DimensionParseError):
        parse_dimension(raw)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2", Fraction(2)),
        ('4"', Fraction(4)),
        ("1 1/4", Fraction(5, 4)),
        ('1-1/4"', Fraction(5, 4)),
        ("2-1/2", Fraction(5, 2)),
        ("2 - 1/2", Fraction(5, 2)),
        ("2\u20131/2", Fraction(5, 2)),
        ("1\u20111/4\u2033", Fraction(5, 4)),
        ("2½", Fraction(5, 2)),
        ('3/4"', Fraction(3, 4)),
    ],
)
def test_parse_nominal_size_accepts_the_hyphenated_size_convention(
    raw: str, expected: Fraction
) -> None:
    size = parse_nominal_size(raw)
    assert size.raw == raw
    assert size.inches == expected


@pytest.mark.parametrize("raw", ["4'", "4 ft", "3' 6\"", "1 foot", "", "two inch"])
def test_parse_nominal_size_rejects_feet_and_unreadable_text(raw: str) -> None:
    with pytest.raises(DimensionParseError):
        parse_nominal_size(raw)


def test_invalid_fraction_is_reported_as_a_dimension_error() -> None:
    with pytest.raises(DimensionParseError):
        parse_dimension("1/0")
    with pytest.raises(DimensionParseError):
        parse_nominal_size("2-1/0")


def test_hyphenated_form_stays_ambiguous_for_a_length() -> None:
    """A size may use ``1-1/4``; a length may not, because it could mean 1' 1/4\"."""

    assert parse_nominal_size("1-1/4").inches == Fraction(5, 4)
    with pytest.raises(DimensionParseError):
        parse_dimension("1-1/4")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Fraction(2), '2"'),
        (Fraction(5, 4), '1 1/4"'),
        (Fraction(1, 2), '1/2"'),
        (Fraction(14), '14"'),
    ],
)
def test_format_nominal_size_never_rolls_up_into_feet(value: Fraction, expected: str) -> None:
    assert format_nominal_size(value) == expected


def test_a_large_nominal_size_is_not_formatted_as_a_length() -> None:
    assert format_inches(Fraction(14)) == "1' 2\""
    assert format_nominal_size(Fraction(14)) == '14"'
