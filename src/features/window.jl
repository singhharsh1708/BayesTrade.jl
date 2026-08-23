"""
A window of bars, with the arrays every feature would otherwise recompute.

Twenty features over one window would each build their own close array. Building them once
and sharing them is the difference between a backtest that finishes and one that does not.

The window is the only thing a feature ever sees. It ends at the bar being decided on and
contains nothing after it, which is what makes look-ahead structurally impossible rather
than merely discouraged: a feature has no reference to the store and cannot ask for more.
"""

"""
    BarWindow

Bars up to and including the decision bar, oldest first.
"""
struct BarWindow
    bars::Vector{Bar}
    opens::Vector{Float64}
    highs::Vector{Float64}
    lows::Vector{Float64}
    closes::Vector{Float64}
    volumes::Vector{Float64}
    log_closes::Vector{Float64}
    log_returns::Vector{Float64}

    function BarWindow(bars::AbstractVector{Bar})
        isempty(bars) && throw(ArgumentError("a bar window needs at least one bar"))
        unique_symbols = unique(bar.symbol for bar in bars)
        length(unique_symbols) == 1 ||
            throw(ArgumentError("a window must hold one symbol, got $(sort(unique_symbols))"))
        issorted(bars, by = bar -> bar.timestamp) ||
            throw(ArgumentError("window bars must be ordered oldest first"))

        closes = Float64[bar.close for bar in bars]
        log_closes = log.(closes)
        return new(
            collect(bars),
            Float64[bar.open for bar in bars],
            Float64[bar.high for bar in bars],
            Float64[bar.low for bar in bars],
            closes,
            Float64[bar.volume for bar in bars],
            log_closes,
            diff(log_closes),
        )
    end
end

Base.length(window::BarWindow) = length(window.bars)

window_symbol(window::BarWindow) = first(window.bars).symbol

"""
    current(window)

The bar being decided on. Nothing after it exists in this window.
"""
current(window::BarWindow) = last(window.bars)

"""
    window_as_of(window)

The close of the decision bar.
"""
window_as_of(window::BarWindow) = current(window).timestamp

"""
    turnovers(window)

Traded value per bar, the quantity liquidity checks actually care about.
"""
turnovers(window::BarWindow) = Float64[turnover(bar) for bar in window.bars]

"""
    tail(window, count)

The most recent `count` bars, as a window in its own right.
"""
function tail(window::BarWindow, count::Integer)
    count > 0 || throw(ArgumentError("count must be positive, got $count"))
    return BarWindow(window.bars[max(1, end - count + 1):end])
end
