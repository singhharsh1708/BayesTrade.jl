"""
Trading calendars.

Two of them, and the difference is the point.

The bare `is_trading_day(day)` is weekday arithmetic. Synthetic data needs nothing more, and a
real series carries its own timestamps, so nothing that reads a vendor feed needs to be told
when the exchange was shut.

[`TradingCalendar`](@ref) is exchange data: a list of the days the NSE actually closed. It
exists because gap detection cannot work without it. A five day hole over Diwali is a holiday,
not a feed outage, and a checker that cannot tell them apart either cries wolf every October or
is tuned until it stops noticing real outages.

A calendar states which years it covers, and asking about a year outside that range raises
rather than answering. Silence would be the worse failure: a year with no holidays listed looks
exactly like a year that was never entered, and the arithmetic returns a confident wrong answer
in both cases. Fail closed, as everywhere else here.
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
    CalendarCoverageError

A calendar was asked about a year it does not cover.

Not answerable, and answering anyway is how a backtest runs across a year of holidays that were
silently treated as ordinary Tuesdays.
"""
struct CalendarCoverageError <: Exception
    year::Int
    covered::UnitRange{Int}
end

Base.showerror(io::IO, error::CalendarCoverageError) = print(
    io, "CalendarCoverageError: this calendar covers ", first(error.covered), " to ",
    last(error.covered), " and was asked about ", error.year,
)

"""
    TradingCalendar

The days an exchange was closed, and the days it opened when it should not have.

`holidays` are the dates the normal session did not run. `special_sessions` are the dates a
session ran anyway, which for the NSE means Muhurat: one hour on Diwali, sometimes on a Sunday.
Both lists are needed, and a date can be in both, because Muhurat trades on a day the regular
session is closed.

`covered` is the claim being made. A date outside it is refused rather than guessed at.
"""
struct TradingCalendar
    holidays::Set{Date}
    special_sessions::Set{Date}
    covered::UnitRange{Int}

    function TradingCalendar(
            holidays, special_sessions = Date[];
            covered::Union{UnitRange{Int}, Nothing} = nothing,
        )
        closed = Set{Date}(holidays)
        special = Set{Date}(special_sessions)
        listed = union(closed, special)
        years = if covered !== nothing
            covered
        elseif isempty(listed)
            throw(ArgumentError("an empty calendar must say which years it covers"))
        else
            # Derived only when the caller gave nothing to go on. A year holding no listed
            # holiday is indistinguishable from a year nobody entered, which is exactly why
            # the range is worth stating explicitly.
            minimum(year, listed):maximum(year, listed)
        end
        for date in listed
            year(date) in years || throw(
                ArgumentError(
                    string(date, " lies outside the covered years ", years),
                ),
            )
        end
        return new(closed, special, years)
    end
end

Base.show(io::IO, calendar::TradingCalendar) = print(
    io, "TradingCalendar(", length(calendar.holidays), " holidays, ",
    length(calendar.special_sessions), " special sessions, ",
    first(calendar.covered), "-", last(calendar.covered), ")",
)

"""
    covers(calendar, day)

Whether this calendar can answer for this date.
"""
covers(calendar::TradingCalendar, day::Date) = year(day) in calendar.covered

function require_covered(calendar::TradingCalendar, day::Date)
    covers(calendar, day) ||
        throw(CalendarCoverageError(year(day), calendar.covered))
    return nothing
end

"""
    is_trading_day(calendar, day)

Whether any session ran.

A Muhurat session on a Sunday is a trading day: a bar exists for it. Whether that bar is
comparable to a full day's is a different question, and [`is_full_session`](@ref) answers it.
"""
function is_trading_day(calendar::TradingCalendar, day::Date)
    require_covered(calendar, day)
    day in calendar.special_sessions && return true
    day in calendar.holidays && return false
    return dayofweek(day) <= 5
end

"""
    is_full_session(calendar, day)

Whether the session ran to its normal length.

Muhurat is one hour. Its bar has a day's stamp and an hour's volume, and treating it as an
ordinary observation pulls realised volatility down and volume checks with it.
"""
function is_full_session(calendar::TradingCalendar, day::Date)
    is_trading_day(calendar, day) || return false
    return !(day in calendar.special_sessions)
end

"""
    NSE_HOLIDAYS_2025, NSE_HOLIDAYS_2026

Dates the NSE equity segment did not run its normal session.

Each list was taken from two independent published calendars that agreed on every date. A
holiday list assembled from one source and from memory is the kind of input that produces a
backtest which is wrong in a way nothing downstream can detect.

October 21 2025 and November 8 2026 are Muhurat: the regular session was closed and a one hour
session ran, so they appear in both lists.
"""
const NSE_HOLIDAYS_2025 = Date[
    Date(2025, 2, 26),   # Mahashivratri
    Date(2025, 3, 14),   # Holi
    Date(2025, 3, 31),   # Id-Ul-Fitr
    Date(2025, 4, 10),   # Shri Mahavir Jayanti
    Date(2025, 4, 14),   # Dr Baba Saheb Ambedkar Jayanti
    Date(2025, 4, 18),   # Good Friday
    Date(2025, 5, 1),    # Maharashtra Day
    Date(2025, 8, 15),   # Independence Day
    Date(2025, 8, 27),   # Shri Ganesh Chaturthi
    Date(2025, 10, 2),   # Mahatma Gandhi Jayanti and Dussehra
    Date(2025, 10, 21),  # Diwali Laxmi Pujan, Muhurat session only
    Date(2025, 10, 22),  # Balipratipada
    Date(2025, 11, 5),   # Prakash Gurpurb Sri Guru Nanak Dev
    Date(2025, 12, 25),  # Christmas
]

const NSE_HOLIDAYS_2026 = Date[
    Date(2026, 1, 15),   # Municipal Corporation elections, Maharashtra
    Date(2026, 1, 26),   # Republic Day
    Date(2026, 3, 3),    # Holi
    Date(2026, 3, 26),   # Shri Ram Navami
    Date(2026, 3, 31),   # Shri Mahavir Jayanti
    Date(2026, 4, 3),    # Good Friday
    Date(2026, 4, 14),   # Dr Baba Saheb Ambedkar Jayanti
    Date(2026, 5, 1),    # Maharashtra Day
    Date(2026, 5, 28),   # Bakri Id
    Date(2026, 6, 26),   # Moharram
    Date(2026, 9, 14),   # Ganesh Chaturthi
    Date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    Date(2026, 10, 20),  # Dussehra
    Date(2026, 11, 10),  # Diwali Balipratipada
    Date(2026, 11, 24),  # Prakash Gurpurb Sri Guru Nanak Dev
    Date(2026, 12, 25),  # Christmas
]

const NSE_MUHURAT = Date[Date(2025, 10, 21), Date(2026, 11, 8)]

"""
    nse_calendar()

The NSE equity calendar for the years this package has verified.

Two years, which is a statement about what was checked rather than about what exists. Extend it
with [`load_calendar`](@ref) rather than by guessing at a third.
"""
nse_calendar() = TradingCalendar(
    vcat(NSE_HOLIDAYS_2025, NSE_HOLIDAYS_2026), NSE_MUHURAT; covered = 2025:2026,
)

"""
    load_calendar(path; covered, special_sessions)

Read a calendar from a file: one ISO date per line, `#` starts a comment.

Exchange holidays are published yearly and this package cannot ship a list it has not checked,
so the way to cover another year is to supply it. `covered` must be stated, because a file
listing nothing for a year is not evidence that the year had no holidays.
"""
function load_calendar(
        path::AbstractString; covered::UnitRange{Int},
        special_sessions::AbstractVector{Date} = Date[],
    )
    isfile(path) || throw(ArgumentError("no calendar file at $path"))
    dates = Date[]
    for (number, line) in enumerate(eachline(path))
        text = strip(first(split(line, "#")))
        isempty(text) && continue
        parsed = tryparse(Date, String(text))
        parsed === nothing && throw(
            ArgumentError(string(path, " line ", number, ": \"", text, "\" is not a date")),
        )
        push!(dates, parsed)
    end
    return TradingCalendar(dates, special_sessions; covered = covered)
end

"""
    trading_days(calendar, start, count)

The first `count` days at or after `start` on which a session ran.
"""
function trading_days(calendar::TradingCalendar, start::Date, count::Integer)
    count >= 0 || throw(ArgumentError("count must be non-negative, got $count"))
    days = Vector{Date}(undef, count)
    day, produced = start, 0
    while produced < count
        if is_trading_day(calendar, day)
            produced += 1
            days[produced] = day
        end
        day += Day(1)
    end
    return days
end

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
