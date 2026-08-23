"""
Turning a return process into OHLCV bars, keeping the ground truth attached.

The high and low are the extremes of a simulated intraday path, not noise sprinkled around
the body. That distinction matters more than it looks: range-based volatility estimators
such as Parkinson and Garman-Klass are derived from the extremes of a Brownian path, so bars
whose wicks came from some other distribution would make those estimators look biased when
the fault was in the data. Stop-loss simulation has the same requirement, since a stop is
hit by the path rather than by the close.

The path is a Brownian bridge pinned to the open and the close, so the OHLC ordering holds by
construction rather than by rejection sampling.
"""

const PRICE_DECIMALS = 2
const MIN_PRICE = 0.01

"""
    CONTINUITY_CORRECTION

Broadie-Glasserman-Kou constant for discretely monitored extrema.

A path sampled at finitely many points is seen at fewer of its extremes than the continuous
path it came from, so a simulated high is systematically too low. Shifting each extreme
outward by this constant times the per-step standard deviation is what makes a range
estimator recover the volatility that generated the bar instead of one biased down by the
sampling grid. Without it, Parkinson reads 0.263 against a true 0.30 at twenty-six steps;
with it, 0.302.
"""
const CONTINUITY_CORRECTION = 0.5826

"""
    BarShape

How a scalar return becomes a candle.
"""
Base.@kwdef struct BarShape
    gap_fraction::Float64 = 0.25
    intraday_steps::Int = 26
    intraday_volatility_multiple::Float64 = 1.0
    base_volume::Float64 = 1.0e6
    volume_dispersion::Float64 = 0.35
    volume_volatility_beta::Float64 = 0.4

    function BarShape(
            gap_fraction, intraday_steps, intraday_volatility_multiple,
            base_volume, volume_dispersion, volume_volatility_beta,
        )
        0 <= gap_fraction <= 1 ||
            throw(ArgumentError("gap_fraction must lie in [0, 1], got $gap_fraction"))
        intraday_steps >= 1 ||
            throw(ArgumentError("intraday_steps must be at least 1, got $intraday_steps"))
        intraday_volatility_multiple >= 0 ||
            throw(ArgumentError("intraday_volatility_multiple cannot be negative"))
        base_volume > 0 || throw(ArgumentError("base_volume must be positive"))
        volume_dispersion >= 0 || throw(ArgumentError("volume_dispersion cannot be negative"))
        volume_volatility_beta >= 0 ||
            throw(ArgumentError("volume_volatility_beta cannot be negative"))
        return new(
            gap_fraction, intraday_steps, intraday_volatility_multiple,
            base_volume, volume_dispersion, volume_volatility_beta,
        )
    end
end

"""
    SyntheticSeries

Generated bars alongside the latent truth that produced them.
"""
struct SyntheticSeries
    symbol::String
    bars::Vector{Bar}
    path::ProcessPath
    parameters::Dict{String, Any}
    seed::Int
end

Base.length(series::SyntheticSeries) = length(series.bars)

closes(series::SyntheticSeries) = [bar.close for bar in series.bars]
true_log_returns(series::SyntheticSeries) = series.path.log_returns
true_volatility(series::SyntheticSeries) = series.path.volatility
true_states(series::SyntheticSeries) = series.path.states

"""
    realised_log_returns(series)

Log returns recovered from the rounded closes.

Not identical to [`true_log_returns`](@ref): prices are rounded to paise, exactly as an
exchange reports them. A model consumes these, so a parameter-recovery test should use these
too rather than the latent path it never sees.
"""
realised_log_returns(series::SyntheticSeries) = diff(log.(closes(series)))

"""
    bars_until(series, moment)

Every bar that had closed by `moment`. The only safe way to slice history.
"""
bars_until(series::SyntheticSeries, moment::DateTime) =
    [bar for bar in series.bars if bar.timestamp <= moment]

"""
    generate_series(process; kwargs...)

Simulate `n_bars` bars from `process`.

Deterministic in `seed`: the same process, seed and arguments always produce identical bars,
so a failing test is reproducible from its parameters alone.
"""
function generate_series(
        process::ReturnProcess;
        symbol::AbstractString = "SYNTH",
        n_bars::Integer = 500,
        start::Date = Date(2022, 1, 3),
        initial_price::Real = 1000.0,
        seed::Integer = 7,
        shape::BarShape = BarShape(),
        interval::AbstractString = "1d",
    )
    n_bars > 0 || throw(ArgumentError("n_bars must be positive, got $n_bars"))
    initial_price > 0 ||
        throw(ArgumentError("initial_price must be positive, got $initial_price"))

    rng = Xoshiro(seed)
    path = simulate(process, n_bars, rng)

    close_prices = initial_price .* exp.(cumsum(path.log_returns))
    previous_closes = vcat(Float64(initial_price), close_prices[1:(end - 1)])
    open_prices = previous_closes .* exp.(shape.gap_fraction .* path.log_returns)

    high_prices, low_prices =
        simulate_extremes(open_prices, close_prices, path.volatility, shape, rng)
    volumes = simulate_volumes(path, shape, rng)
    timestamps = session_close.(trading_days(start, n_bars))

    # The element type is written out: an unannotated comprehension leaves the collect
    # target abstract, and inference then has to consider growing a dictionary.
    bars = Bar[
        build_bar(
                symbol, timestamps[index], open_prices[index], high_prices[index],
                low_prices[index], close_prices[index], volumes[index], interval,
            ) for index in 1:n_bars
    ]

    parameters = Dict{String, Any}(process_parameters(process))
    parameters["initial_price"] = Float64(initial_price)
    parameters["n_bars"] = Int(n_bars)
    parameters["start"] = string(start)
    parameters["gap_fraction"] = shape.gap_fraction
    parameters["intraday_steps"] = shape.intraday_steps

    return SyntheticSeries(String(symbol), bars, path, parameters, Int(seed))
end

"""
    simulate_extremes(opens, closes, volatility, shape, rng)

The running maximum and minimum of a Brownian bridge from open to close.

The bridge is pinned at both ends, so the simulated path starts exactly at the open and ends
exactly at the close, and its extremes are consistent with the body by construction.

The bridge diffuses at the *session's* volatility, which is the bar's volatility net of the
opening gap: a gap is a jump rather than something the price walked through. That is also
why a range estimator reads lower than a close-to-close one on gappy data, and the tests
check for exactly that rather than treating it as an error.
"""
function simulate_extremes(
        opens::Vector{Float64}, closes::Vector{Float64},
        volatility::Vector{Float64}, shape::BarShape, rng::AbstractRNG,
    )
    steps = shape.intraday_steps
    log_open = log.(opens)
    log_close = log.(closes)
    session = volatility .* (1 - shape.gap_fraction) .* shape.intraday_volatility_multiple
    scale = session ./ sqrt(steps)

    n = length(opens)
    high = Vector{Float64}(undef, n)
    low = Vector{Float64}(undef, n)
    fraction = collect(1:steps) ./ steps
    walk = Vector{Float64}(undef, steps)

    for index in 1:n
        cumulative = 0.0
        for step in 1:steps
            cumulative += scale[index] * randn(rng)
            walk[step] = cumulative
        end
        drift = log_close[index] - log_open[index]
        highest = -Inf
        lowest = Inf
        for step in 1:steps
            level = log_open[index] + drift * fraction[step] +
                (walk[step] - fraction[step] * walk[steps])
            highest = max(highest, level)
            lowest = min(lowest, level)
        end
        correction = CONTINUITY_CORRECTION * scale[index]
        high[index] = exp(max(highest, log_open[index], log_close[index]) + correction)
        low[index] = exp(min(lowest, log_open[index], log_close[index]) - correction)
    end
    return high, low
end

"""
    simulate_volumes(path, shape, rng)

Log-normal volume, lifted by the standardised size of the bar's move.

Volume is not independent noise. A liquidity check calibrated against volume that ignores
volatility would pass in exactly the conditions where it should fail.
"""
function simulate_volumes(path::ProcessPath, shape::BarShape, rng::AbstractRNG)
    standardised = abs.(path.log_returns) ./ path.volatility
    drive = shape.volume_volatility_beta .* (standardised .- sqrt(2 / pi))
    noise = -0.5 * shape.volume_dispersion^2 .+
        shape.volume_dispersion .* randn(rng, length(path))
    return shape.base_volume .* exp.(drive .+ noise)
end

"""
    build_bar(...)

Round to paise the way an exchange reports, then restore the OHLC ordering.

Rounding can push a wick inside the body by a paisa. Clamping afterwards keeps the bar valid
without reintroducing sub-paise precision no real feed would carry.
"""
function build_bar(
        symbol::AbstractString, timestamp::DateTime, open::Float64, high::Float64,
        low::Float64, close::Float64, volume::Float64, interval::AbstractString,
    )
    open_r = max(round(open, digits = PRICE_DECIMALS), MIN_PRICE)
    close_r = max(round(close, digits = PRICE_DECIMALS), MIN_PRICE)
    high_r = max(round(high, digits = PRICE_DECIMALS), open_r, close_r)
    low_r = max(min(round(low, digits = PRICE_DECIMALS), open_r, close_r), MIN_PRICE)
    return Bar(
        symbol, timestamp, open_r, high_r, low_r, close_r,
        round(volume, digits = 2); interval = interval,
    )
end
