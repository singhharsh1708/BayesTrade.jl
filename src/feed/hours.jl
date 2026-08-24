"""
Market hours, and what a timestamp means.

Two things the feed layer needs before it can be pointed at a live market, and neither is
optional.

**Every `DateTime` in this package is exchange-local, with no timezone attached.** That is a
choice, and it is written down here because an unwritten one becomes a bug: Kite sends epoch
seconds, `unix2datetime` yields UTC, and Indian equities trade 09:15 to 15:30 IST. Bucketing a
UTC stamp as though it were local puts every bar five and a half hours out of place and moves
the open to 03:45. The conversion happens once, at the boundary, and is named.

**A bar must not span the overnight gap.** An aggregator that only ever sees the next tick will
happily fold Monday's open into Friday's last bar, because the bucket arithmetic has no opinion
about nights and weekends. The close of the session closes the bar.
"""

"""
    IST_OFFSET

India Standard Time against UTC. No daylight saving, which is the one mercy here.
"""
const IST_OFFSET = Minute(330)   # five and a half hours, as one period

"""
    MarketHours

When an exchange is open, in its own local time.

Defaults are the NSE equity session.
"""
struct MarketHours
    open::Time
    close::Time
    offset::Period

    function MarketHours(;
            open::Time = Time(9, 15), close::Time = Time(15, 30), offset::Period = IST_OFFSET,
        )
        open < close ||
            throw(ArgumentError(string("the open ", open, " must precede the close ", close)))
        return new(open, close, offset)
    end
end

"""
    exchange_time(hours, epoch_seconds)

An exchange timestamp from the seconds a feed sent.

The single place the conversion happens. Doing it at each call site is how half a system ends
up in UTC and the other half in local time, with nothing to show for it but bars in the wrong
buckets.
"""
exchange_time(hours::MarketHours, epoch_seconds::Real) =
    unix2datetime(epoch_seconds) + hours.offset

"""
    is_open(hours, moment)

Whether the exchange is trading at this moment.

Weekends are closed. Holidays are not modelled here: a holiday calendar is exchange data rather
than arithmetic, and pretending to know one would be worse than saying it is absent.
"""
function is_open(hours::MarketHours, moment::DateTime)
    weekday = dayofweek(moment)
    weekday <= 5 || return false
    clock = Time(moment)
    return hours.open <= clock <= hours.close
end

"""
    same_session(hours, left, right)

Whether two moments belong to the same trading session.

What the aggregator needs to know before folding a tick into an open bar: a bucket that spans
a night is not a bar, it is two bars with the gap swallowed.
"""
same_session(hours::MarketHours, left::DateTime, right::DateTime) =
    Date(left) == Date(right)

"""
    session_bounds(hours, day)

The open and close of one day's session.
"""
function session_bounds(hours::MarketHours, day::Date)
    midnight = DateTime(day)
    at(clock) = midnight + Hour(hour(clock)) + Minute(minute(clock)) + Second(second(clock))
    return (at(hours.open), at(hours.close))
end

Base.show(io::IO, hours::MarketHours) =
    print(io, "<MarketHours ", hours.open, "-", hours.close, " UTC+", hours.offset, ">")
