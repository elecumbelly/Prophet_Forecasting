import pytest

from prophet_forecasting_tool.utils import (
    duration_to_periods,
    normalize_tz_naive,
    parse_duration,
    shift_timestamp,
    validate_identifier,
)
import pandas as pd


class TestValidateIdentifier:
    def test_accepts_simple(self):
        assert validate_identifier("my_table") == "my_table"

    def test_accepts_underscore_start(self):
        assert validate_identifier("_col") == "_col"

    def test_rejects_dash(self):
        with pytest.raises(ValueError):
            validate_identifier("real-call-metrics")

    def test_rejects_quote(self):
        with pytest.raises(ValueError):
            validate_identifier('drop"table')

    def test_rejects_semicolon(self):
        with pytest.raises(ValueError):
            validate_identifier("foo;drop")

    def test_rejects_leading_digit(self):
        with pytest.raises(ValueError):
            validate_identifier("1table")

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            validate_identifier("")

    def test_rejects_non_string(self):
        with pytest.raises(ValueError):
            validate_identifier(None)  # type: ignore[arg-type]


class TestParseDuration:
    def test_days_compact(self):
        delta = parse_duration("90D")
        assert delta.days == 90

    def test_days_spaced(self):
        assert parse_duration("365 days").days == 365

    def test_years(self):
        assert parse_duration("2Y").years == 2

    def test_months(self):
        assert parse_duration("6M").months == 6

    def test_hours(self):
        delta = parse_duration("24H")
        # relativedelta normalises 24 hours into 1 day — check the total span.
        anchor = pd.Timestamp("2024-01-01")
        assert (anchor + delta) - anchor == pd.Timedelta(days=1)

    def test_minutes(self):
        assert parse_duration("15min").minutes == 15

    def test_weeks(self):
        # relativedelta normalises weeks to days under the hood.
        assert parse_duration("4W").days == 28

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            parse_duration("seven score and ten")


class TestDurationToPeriods:
    def test_daily(self):
        assert duration_to_periods("90D", "D") == 90

    def test_monthly_via_year(self):
        assert duration_to_periods("2Y", "M") == 24

    def test_monthly_via_month_string(self):
        assert duration_to_periods("12M", "M") == 12

    def test_weekly(self):
        # 12 weeks at freq=W → 12 periods
        assert duration_to_periods("12W", "W") == 12


class TestShiftTimestamp:
    def test_add(self):
        ts = pd.Timestamp("2024-01-01")
        assert shift_timestamp(ts, "30D") == pd.Timestamp("2024-01-31")

    def test_subtract(self):
        ts = pd.Timestamp("2024-01-31")
        assert shift_timestamp(ts, "30D", subtract=True) == pd.Timestamp("2024-01-01")

    def test_months_calendar_aware(self):
        ts = pd.Timestamp("2024-03-31")
        # +1 month from Mar 31 should clamp to Apr 30 (calendar-aware).
        result = shift_timestamp(ts, "1M")
        assert result == pd.Timestamp("2024-04-30")


class TestNormalizeTzNaive:
    def test_strips_tz(self):
        ts = pd.Timestamp("2024-01-01 10:00", tz="UTC")
        assert normalize_tz_naive(ts).tzinfo is None

    def test_passes_through(self):
        ts = pd.Timestamp("2024-01-01")
        assert normalize_tz_naive(ts) == ts

    def test_none(self):
        assert normalize_tz_naive(None) is None


class TestFrontendDurationParity:
    """Verify Python parses every duration form the frontend's previewDuration accepts.

    The TypeScript helper ``previewDuration`` in ``frontend/src/lib/api.ts``
    shows users "→ X day(s)" hints while they type. If Python's
    ``parse_duration`` rejected anything the UI accepted, the user would see
    a valid-looking preview followed by a 400 from the API. This test pins
    that contract.
    """

    @pytest.mark.parametrize(
        "value",
        [
            # Compact unit-letter forms.
            "90D", "1D", "365D", "12M", "24M", "1Y", "2Y", "10Y",
            "6H", "12H", "24H", "4W", "8W",
            # Spaced + spelled-out forms.
            "365 days", "1 day", "12 months", "2 years", "4 weeks",
            "24 hours", "30 minutes",
            # Compact min.
            "15min", "60min",
        ],
    )
    def test_parses_without_error(self, value):
        parse_duration(value)
