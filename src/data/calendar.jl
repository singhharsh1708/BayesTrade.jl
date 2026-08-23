"""
A minimal trading calendar.

Weekdays only. Exchange holidays are deliberately not modelled: synthetic data does not need
them, and real data carries its own timestamps, so a hand-maintained holiday list would only
ever be a source of silent disagreement with the vendor.
"""

"""
    BARS_PER_YEAR

Trading days in a year, the convention behind every annualised quantity in this package.
"""
const BARS_PER_YEAR = 252

"""
    ANNUALISER

`sqrt(BARS_PER_YEAR)`, the factor that turns a per-bar volatility into an annual one.
"""
const ANNUALISER = sqrt(BARS_PER_YEAR)

"""
    NSE_CLOSE

15:30 IST expressed in UTC, the close of the Indian equity session.
"""
const NSE_CLOSE = Time(10, 0)

"""
    is_trading_day(day)

Whether the exchange is open. Weekends only; see the module note on holidays.
"""
is_trading_day(day::Date) = dayofweek(day) <= 5

"""
    trading_days(start, count)

The first `count` trading days at or after `start`.
"""
function trading_days(start::Date, count::Integer)
    count >= 0 || throw(ArgumentError("count must be non-negative, got $count"))
    days = Vector{Date}(undef, count)
    day, produced = start, 0
    while produced < count
        if is_trading_day(day)
            produced += 1
            days[produced] = day
        end
        day += Day(1)
    end
    return days
end

"""
    session_close(day, close = NSE_CLOSE)

The timestamp a daily bar for `day` is stamped with.
"""
session_close(day::Date, close::Time = NSE_CLOSE) = DateTime(day) + Hour(hour(close)) +
    Minute(minute(close))

"""
    annualise(bar_volatility)

Per-bar volatility to annual, on the 252-day convention.
"""
annualise(bar_volatility::Real) = bar_volatility * ANNUALISER

"""
    deannualise(annual_volatility)

Annual volatility to per-bar.
"""
deannualise(annual_volatility::Real) = annual_volatility / ANNUALISER
