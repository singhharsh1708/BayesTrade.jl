"""
Data quality checks over a bar series.

Bad data does not usually announce itself as bad. An unadjusted split looks like a genuine
fifty percent move, a stale feed looks like a calm market, and a missing week looks like
nothing at all. Each of those produces a confident, wrong posterior, so they are checked for
explicitly and reported before a model ever sees the series.

Nothing here repairs anything. A check that silently fixes its own findings is a check that
can never be audited, and the right repair depends on why the data is wrong.
"""

const MAD_TO_SIGMA = 1.4826
const COMMON_SPLIT_RATIOS = (2.0, 3.0, 4.0, 5.0, 10.0, 1.5, 2.5)

"""
    SPLIT_LOG_TOLERANCE

How far a move may sit from a split ratio and still be called one, in log space.

Eight percent rather than something tight: a split lands on a day that also has its own
ordinary return, so a 1:2 split shows up as a factor near two rather than exactly two. The
common ratios are far enough apart in log space that this cannot make two of them ambiguous.
"""
const SPLIT_LOG_TOLERANCE = 0.08

"""
    Severity

How much a finding should worry you.

`ERROR` means the series should not be traded on until it is understood. `WARNING` means a
model will still fit but its assumptions are strained. `INFO` means it is worth knowing and
probably fine.
"""
@enum Severity INFO WARNING ERROR

"""
    QualityIssue

One finding, anchored to where it was found.
"""
Base.@kwdef struct QualityIssue
    check::Symbol
    severity::Severity
    symbol::String
    detail::String
    at::Union{DateTime, Nothing} = nothing
    observed::Union{Float64, Nothing} = nothing
end

function Base.show(io::IO, issue::QualityIssue)
    where_ = issue.at === nothing ? "" : " at $(issue.at)"
    return print(
        io, '[', lowercase(string(issue.severity)), "] ", issue.symbol, ' ',
        issue.check, where_, ": ", issue.detail,
    )
end

"""
    QualityReport

Everything the checks found in one series.
"""
Base.@kwdef struct QualityReport
    symbol::String
    n_bars::Int
    first::Union{DateTime, Nothing}
    last::Union{DateTime, Nothing}
    issues::Vector{QualityIssue} = QualityIssue[]
end

of_severity(report::QualityReport, severity::Severity) =
    QualityIssue[issue for issue in report.issues if issue.severity === severity]

errors(report::QualityReport) = of_severity(report, ERROR)
warnings(report::QualityReport) = of_severity(report, WARNING)

"""
    is_usable(report)

Whether a model may be fitted to this series without a human looking first.
"""
is_usable(report::QualityReport) = isempty(errors(report))

function summarise(report::QualityReport)
    counts = Dict(severity => length(of_severity(report, severity)) for severity in instances(Severity))
    head = string(
        report.symbol, ": ", report.n_bars, " bars, ",
        counts[ERROR], " errors, ", counts[WARNING], " warnings, ",
        counts[INFO], " notes",
    )
    isempty(report.issues) && return head
    return join([head; ["  " * sprint(show, issue) for issue in report.issues]], '\n')
end

"""
    validate_bars(bars; min_bars, max_gap_bars, stale_run, spike_sigmas)

Run every check over one symbol's bars.

`bars` must all belong to one symbol; a mixed sequence is a programming error and is
reported as one rather than quietly checked as if it were a single series.
"""
function validate_bars(
        bars::AbstractVector{Bar};
        min_bars::Integer = 30,
        max_gap_bars::Integer = 5,
        stale_run::Integer = 5,
        spike_sigmas::Real = 12.0,
    )
    isempty(bars) && return QualityReport(
        symbol = "", n_bars = 0, first = nothing, last = nothing,
    )

    unique_symbols = unique(bar.symbol for bar in bars)
    length(unique_symbols) == 1 ||
        throw(ArgumentError("validate_bars expects one symbol, got $(sort(unique_symbols))"))
    symbol = first(unique_symbols)
    ordered = sort(collect(bars), by = bar -> bar.timestamp)

    issues = QualityIssue[]
    append!(issues, check_ordering(bars, ordered, symbol))
    append!(issues, check_length(ordered, symbol, min_bars))
    append!(issues, check_calendar_gaps(ordered, symbol, max_gap_bars))
    append!(issues, check_volume(ordered, symbol))
    append!(issues, check_stale_prices(ordered, symbol, stale_run))
    append!(issues, check_returns(ordered, symbol, spike_sigmas))

    return QualityReport(
        symbol = symbol, n_bars = length(ordered),
        first = first(ordered).timestamp, last = last(ordered).timestamp,
        issues = issues,
    )
end

function check_ordering(bars::AbstractVector{Bar}, ordered::AbstractVector{Bar}, symbol::AbstractString)
    issues = QualityIssue[]
    stamps = DateTime[bar.timestamp for bar in bars]
    counts = Dict{DateTime, Int}()
    for stamp in stamps
        counts[stamp] = get(counts, stamp, 0) + 1
    end
    for stamp in sort([stamp for (stamp, n) in counts if n > 1])
        push!(
            issues, QualityIssue(
                check = :duplicate_timestamp, severity = ERROR, symbol = symbol, at = stamp,
                detail = "two bars share this timestamp; one of them is wrong",
            ),
        )
    end
    if stamps != DateTime[bar.timestamp for bar in ordered]
        push!(
            issues, QualityIssue(
                check = :unordered_series, severity = INFO, symbol = symbol,
                detail = "bars arrived out of order and were sorted before checking",
            ),
        )
    end
    return issues
end

check_length(bars::AbstractVector{Bar}, symbol::AbstractString, min_bars::Integer) =
    length(bars) >= min_bars ? QualityIssue[] :
    [
        QualityIssue(
            check = :insufficient_history, severity = ERROR, symbol = symbol,
            observed = Float64(length(bars)),
            detail = "$(length(bars)) bars is below the $min_bars needed to fit anything",
        ),
    ]

"""
    check_calendar_gaps(bars, symbol, max_gap_bars)

Report runs of missing trading days.

Holidays are not modelled, so a two or three day gap is ordinary. A gap longer than
`max_gap_bars` is either a trading halt or a hole in the download, and the two need
different responses.
"""
function check_calendar_gaps(bars::AbstractVector{Bar}, symbol::AbstractString, max_gap_bars::Integer)
    issues = QualityIssue[]
    for index in 1:(length(bars) - 1)
        earlier, later = bars[index], bars[index + 1]
        missing_days = trading_days_between(earlier.timestamp, later.timestamp)
        missing_days > max_gap_bars && push!(
            issues, QualityIssue(
                check = :calendar_gap, severity = WARNING, symbol = symbol,
                at = later.timestamp, observed = Float64(missing_days),
                detail = "$missing_days trading days missing since $(Date(earlier.timestamp))",
            ),
        )
    end
    return issues
end

"""
    trading_days_between(earlier, later)

Trading days strictly between two timestamps.

Counted with an explicit loop rather than a stepped date range. The range constructor drags
in overflow handling that does not infer cleanly, and a day-by-day walk over a gap that is
almost always under a week is not the place to be clever.
"""
function trading_days_between(earlier::DateTime, later::DateTime)
    day = Date(earlier) + Day(1)
    final = Date(later) - Day(1)
    counted = 0
    while day <= final
        is_trading_day(day) && (counted += 1)
        day += Day(1)
    end
    return counted
end

function check_volume(bars::AbstractVector{Bar}, symbol::AbstractString)
    zero_volume = Bar[bar for bar in bars if bar.volume == 0]
    isempty(zero_volume) && return QualityIssue[]
    severity = length(zero_volume) > length(bars) ÷ 10 ? ERROR : WARNING
    return [
        QualityIssue(
            check = :zero_volume, severity = severity, symbol = symbol,
            at = first(zero_volume).timestamp, observed = Float64(length(zero_volume)),
            detail = string(
                "$(length(zero_volume)) of $(length(bars)) bars have no volume; ",
                "a liquidity check would pass on a price nobody traded at",
            ),
        ),
    ]
end

"""
    check_stale_prices(bars, symbol, stale_run)

Find runs of identical closes, which usually mean a stopped feed rather than calm.
"""
function check_stale_prices(bars::AbstractVector{Bar}, symbol::AbstractString, stale_run::Integer)
    issues = QualityIssue[]
    run_start, run_length = 1, 1
    for index in 2:length(bars)
        if bars[index].close == bars[index - 1].close
            run_length += 1
            continue
        end
        run_length >= stale_run && push!(issues, stale_issue(bars, symbol, run_start, run_length))
        run_start, run_length = index, 1
    end
    run_length >= stale_run && push!(issues, stale_issue(bars, symbol, run_start, run_length))
    return issues
end

stale_issue(bars::AbstractVector{Bar}, symbol::AbstractString, start::Integer, run_length::Integer) = QualityIssue(
    check = :stale_price, severity = WARNING, symbol = symbol,
    at = bars[start].timestamp, observed = Float64(run_length),
    detail = string(
        "close held at $(bars[start].close) for $run_length bars; ",
        "a stopped feed looks exactly like a calm market to a volatility model",
    ),
)

"""
    check_returns(bars, symbol, spike_sigmas)

Flag outliers against a robust scale, and name the ones that look like splits.

The scale is a median absolute deviation rather than a standard deviation, because a single
unadjusted split inflates a standard deviation enough to hide itself.
"""
function check_returns(bars::AbstractVector{Bar}, symbol::AbstractString, spike_sigmas::Real)
    length(bars) < 3 && return QualityIssue[]
    returns = diff(log.(Float64[bar.close for bar in bars]))
    centre = median(returns)
    scale = MAD_TO_SIGMA * median(abs.(returns .- centre))
    scale <= 0 && return QualityIssue[]

    issues = QualityIssue[]
    for (index, value) in enumerate(returns)
        deviation = abs(value - centre) / scale
        deviation < spike_sigmas && continue
        at = bars[index + 1].timestamp
        ratio = split_ratio(value)
        if ratio === nothing
            push!(
                issues, QualityIssue(
                    check = :price_spike, severity = WARNING, symbol = symbol,
                    at = at, observed = deviation,
                    detail = @sprintf(
                        "log return %+.4f is %.1f robust sigmas from the median",
                        value, deviation
                    ),
                ),
            )
        else
            push!(
                issues, QualityIssue(
                    check = :suspected_split, severity = ERROR, symbol = symbol,
                    at = at, observed = ratio,
                    detail = string(
                        "close moved by a factor of about 1:", ratio,
                        ", which is a common split ratio; the series may be unadjusted",
                    ),
                ),
            )
        end
    end
    return issues
end

"""
    split_ratio(log_return)

The split ratio this move matches, if it matches one.

Matched in log space, where the common ratios are far enough apart that an eight percent
tolerance cannot make two of them ambiguous.
"""
function split_ratio(log_return::Real)
    factor = -log_return
    for ratio in COMMON_SPLIT_RATIOS
        boundary = log(ratio)
        min(abs(factor - boundary), abs(factor + boundary)) < SPLIT_LOG_TOLERANCE &&
            return ratio
    end
    return nothing
end
