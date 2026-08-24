"""
Return-generating processes with known parameters.

Every process here has parameters that a correctly implemented Bayesian model should be able
to recover. That is the point: a model tested only against real market data can never be
shown to be right, because nobody knows what the right answer was. A model tested against a
process whose parameters were chosen five lines earlier can.

All processes work in log-return space and take annualised parameters, converted to per-bar
values with the 252-day convention.
"""

"""
    ProcessPath

One simulated path, with the latent quantities a model would have to infer.

`volatility` is the per-bar conditional standard deviation actually used to draw each
return, and `states` the latent regime index where the process has one. Both are the
answers, kept so a test can grade a model against them rather than against plausibility.
"""
struct ProcessPath
    log_returns::Vector{Float64}
    volatility::Vector{Float64}
    states::Union{Vector{Int}, Nothing}

    function ProcessPath(log_returns, volatility, states = nothing)
        length(log_returns) == length(volatility) ||
            throw(ArgumentError("returns and volatility must have the same length"))
        if states !== nothing && length(states) != length(log_returns)
            throw(ArgumentError("states must have the same length as returns"))
        end
        return new(log_returns, volatility, states)
    end
end

Base.length(path::ProcessPath) = length(path.log_returns)

"""
    ReturnProcess

A generator of log-return paths with recorded ground truth.

Concrete processes define [`simulate`](@ref) and [`process_parameters`](@ref).
"""
abstract type ReturnProcess end

"""
    simulate(process, n_bars, rng)

Draw a path of `n_bars` log returns.
"""
simulate(process::ReturnProcess, n_bars::Integer, rng::AbstractRNG) =
    throw(ArgumentError("$(typeof(process)) must define simulate"))

"""
    process_parameters(process)

The true parameters, in the units a model is expected to report.
"""
process_parameters(process::ReturnProcess) =
    throw(ArgumentError("$(typeof(process)) must define process_parameters"))

"""
    GaussianReturns

Independent normal log returns: the null hypothesis a model must beat.

A momentum model fitted to this data should recover a coefficient indistinguishable from
zero. If it finds signal here, it is fitting noise.
"""
struct GaussianReturns <: ReturnProcess
    annual_drift::Float64
    annual_volatility::Float64

    function GaussianReturns(; annual_drift::Real = 0.1, annual_volatility::Real = 0.25)
        annual_volatility > 0 ||
            throw(ArgumentError("annual_volatility must be positive, got $annual_volatility"))
        return new(Float64(annual_drift), Float64(annual_volatility))
    end
end

function simulate(process::GaussianReturns, n_bars::Integer, rng::AbstractRNG)
    mu = process.annual_drift / BARS_PER_YEAR
    sigma = deannualise(process.annual_volatility)
    return ProcessPath(mu .+ sigma .* randn(rng, n_bars), fill(sigma, n_bars))
end

process_parameters(process::GaussianReturns) = Dict(
    "process" => "gaussian",
    "annual_drift" => process.annual_drift,
    "annual_volatility" => process.annual_volatility,
    "bar_drift" => process.annual_drift / BARS_PER_YEAR,
    "bar_volatility" => deannualise(process.annual_volatility),
)

"""
    AR1Returns

First-order autoregressive returns.

`phi` above zero is momentum, below zero is mean reversion. The innovation scale is set so
the *stationary* volatility matches `annual_volatility`, which makes two processes with
different `phi` comparable on risk rather than only on signal.
"""
struct AR1Returns <: ReturnProcess
    phi::Float64
    annual_drift::Float64
    annual_volatility::Float64

    function AR1Returns(;
            phi::Real = 0.15, annual_drift::Real = 0.1, annual_volatility::Real = 0.25,
        )
        -1 < phi < 1 ||
            throw(ArgumentError("phi must lie in (-1, 1) for stationarity, got $phi"))
        annual_volatility > 0 ||
            throw(ArgumentError("annual_volatility must be positive, got $annual_volatility"))
        return new(Float64(phi), Float64(annual_drift), Float64(annual_volatility))
    end
end

"""
    innovation_scale(process)

Per-bar innovation standard deviation implied by the requested stationary volatility.
"""
innovation_scale(process::AR1Returns) =
    deannualise(process.annual_volatility) * sqrt(1 - process.phi^2)

function simulate(process::AR1Returns, n_bars::Integer, rng::AbstractRNG)
    mu = process.annual_drift / BARS_PER_YEAR
    scale = innovation_scale(process)
    stationary = deannualise(process.annual_volatility)

    deviations = Vector{Float64}(undef, n_bars)
    previous = stationary * randn(rng)
    for index in 1:n_bars
        previous = process.phi * previous + scale * randn(rng)
        deviations[index] = previous
    end
    return ProcessPath(mu .+ deviations, fill(scale, n_bars))
end

process_parameters(process::AR1Returns) = Dict(
    "process" => "ar1",
    "phi" => process.phi,
    "annual_drift" => process.annual_drift,
    "annual_volatility" => process.annual_volatility,
    "bar_drift" => process.annual_drift / BARS_PER_YEAR,
    "innovation_scale" => innovation_scale(process),
)

"""
    RegimeSwitchingReturns

A hidden Markov process over regimes with distinct drift and volatility.

The latent state path is recorded, so a regime model can be graded on how often it
identifies the true state rather than only on whether its output looks plausible.
"""
struct RegimeSwitchingReturns <: ReturnProcess
    labels::Vector{String}
    annual_drifts::Vector{Float64}
    annual_volatilities::Vector{Float64}
    transitions::Matrix{Float64}

    function RegimeSwitchingReturns(;
            labels::Vector{<:AbstractString} = ["bull", "bear", "sideways"],
            annual_drifts::Vector{<:Real} = [0.35, -0.3, 0.02],
            annual_volatilities::Vector{<:Real} = [0.18, 0.45, 0.22],
            transitions::Matrix{<:Real} = [
                0.97 0.01 0.02
                0.02 0.95 0.03
                0.03 0.03 0.94
            ],
        )
        n = length(labels)
        (length(annual_drifts) == n && length(annual_volatilities) == n) ||
            throw(ArgumentError("drifts, volatilities and labels must describe the same regimes"))
        any(<=(0), annual_volatilities) &&
            throw(ArgumentError("every regime needs a positive volatility"))
        size(transitions) == (n, n) ||
            throw(ArgumentError("transition matrix must be $(n)x$(n), got $(size(transitions))"))
        any(<(0), transitions) &&
            throw(ArgumentError("transition probabilities must be non-negative"))
        all(isapprox(1; atol = 1.0e-9), sum(transitions, dims = 2)) ||
            throw(ArgumentError("every transition matrix row must sum to 1"))
        # `convert` rather than a broadcast: broadcasting a type over an abstractly typed
        # container does not infer, and this constructor is on the path every generated
        # series takes.
        return new(
            convert(Vector{String}, labels),
            convert(Vector{Float64}, annual_drifts),
            convert(Vector{Float64}, annual_volatilities),
            convert(Matrix{Float64}, transitions),
        )
    end
end

n_regimes(process::RegimeSwitchingReturns) = length(process.labels)

"""
    stationary_distribution(process)

Long-run regime frequencies.

Solved as a linear system rather than through an eigendecomposition. The stationary vector
satisfies `pi' A = pi'` with `sum(pi) = 1`, which is a square real system once the
normalisation replaces one redundant row. An eigensolver returns complex values that then
need real-part and absolute-value cleanup, and the sign and ordering of its output are
conventions rather than guarantees.

A reducible chain has no unique stationary distribution, and the solve says so rather than
returning whichever eigenvector came back first.
"""
stationary_distribution(process::RegimeSwitchingReturns) =
    stationary_distribution(process.transitions)

function stationary_distribution(transitions::AbstractMatrix{<:Real})
    n = Base.size(transitions, 1)
    Base.size(transitions, 2) == n ||
        throw(ArgumentError(string("a transition matrix must be square, got ", Base.size(transitions))))
    # Built entrywise rather than from `transpose(transitions) - I`, which does not infer
    # once the argument is an abstract matrix.
    system = Matrix{Float64}(undef, n, n)
    for row in 1:n, column in 1:n
        system[row, column] = transitions[column, row] - (row == column ? 1.0 : 0.0)
    end
    for column in 1:n
        system[n, column] = 1.0
    end
    target = zeros(Float64, n)
    target[n] = 1.0
    # An explicit LU rather than a bare backslash. Backslash on a plain matrix goes through
    # a pivoted-QR least-squares path built for possibly rank-deficient systems, which is
    # neither what this is nor what it should silently fall back to: a square chain either
    # has a unique stationary distribution or it does not.
    factorisation = lu(system; check = false)
    issuccess(factorisation) || throw(
        ArgumentError(
            "the transition matrix is reducible, so it has no unique stationary distribution",
        ),
    )
    return factorisation \ target
end

function simulate(process::RegimeSwitchingReturns, n_bars::Integer, rng::AbstractRNG)
    drifts = process.annual_drifts ./ BARS_PER_YEAR
    scales = deannualise.(process.annual_volatilities)

    states = Vector{Int}(undef, n_bars)
    state = sample_index(rng, stationary_distribution(process))
    for index in 1:n_bars
        states[index] = state
        state = sample_index(rng, view(process.transitions, state, :))
    end

    volatility = scales[states]
    returns = drifts[states] .+ volatility .* randn(rng, n_bars)
    return ProcessPath(returns, volatility, states)
end

"""
    sample_index(rng, weights)

Draw an index with probability proportional to `weights`.

Written out rather than reached for from a sampling package so the arithmetic is visible:
this is the one place the latent state path is decided, and the tests grade models against
it.
"""
function sample_index(rng::AbstractRNG, weights)
    threshold = rand(rng) * sum(weights)
    cumulative = 0.0
    for (index, weight) in enumerate(weights)
        cumulative += weight
        cumulative >= threshold && return index
    end
    return length(weights)
end

process_parameters(process::RegimeSwitchingReturns) = Dict(
    "process" => "regime_switching",
    "labels" => process.labels,
    "annual_drifts" => process.annual_drifts,
    "annual_volatilities" => process.annual_volatilities,
    "transitions" => [collect(row) for row in eachrow(process.transitions)],
    "stationary_distribution" => stationary_distribution(process),
)

"""
    StochasticVolatilityReturns

Log-volatility follows a persistent AR(1); returns are conditionally normal.

This produces volatility clustering and fat unconditional tails without either being put in
by hand, which is what a volatility model has to cope with in real data. Conditioning on the
true volatility restores normality, and there is a test asserting exactly that.
"""
struct StochasticVolatilityReturns <: ReturnProcess
    annual_drift::Float64
    annual_volatility::Float64
    persistence::Float64
    volatility_of_volatility::Float64

    function StochasticVolatilityReturns(;
            annual_drift::Real = 0.1,
            annual_volatility::Real = 0.25,
            persistence::Real = 0.95,
            volatility_of_volatility::Real = 0.2,
        )
        -1 < persistence < 1 ||
            throw(ArgumentError("persistence must lie in (-1, 1), got $persistence"))
        annual_volatility > 0 ||
            throw(ArgumentError("annual_volatility must be positive, got $annual_volatility"))
        volatility_of_volatility > 0 ||
            throw(ArgumentError("volatility_of_volatility must be positive"))
        return new(
            Float64(annual_drift), Float64(annual_volatility),
            Float64(persistence), Float64(volatility_of_volatility),
        )
    end
end

"""
    log_volatility_mean(process)

Set so the *median* per-bar volatility equals the requested level.
"""
log_volatility_mean(process::StochasticVolatilityReturns) =
    log(deannualise(process.annual_volatility))

function simulate(process::StochasticVolatilityReturns, n_bars::Integer, rng::AbstractRNG)
    mu = process.annual_drift / BARS_PER_YEAR
    mean_log = log_volatility_mean(process)
    phi = process.persistence
    eta = process.volatility_of_volatility

    log_volatility = Vector{Float64}(undef, n_bars)
    level = mean_log + (eta / sqrt(1 - phi^2)) * randn(rng)
    for index in 1:n_bars
        level = mean_log + phi * (level - mean_log) + eta * randn(rng)
        log_volatility[index] = level
    end

    volatility = exp.(log_volatility)
    return ProcessPath(mu .+ volatility .* randn(rng, n_bars), volatility)
end

process_parameters(process::StochasticVolatilityReturns) = Dict(
    "process" => "stochastic_volatility",
    "annual_drift" => process.annual_drift,
    "annual_volatility" => process.annual_volatility,
    "persistence" => process.persistence,
    "volatility_of_volatility" => process.volatility_of_volatility,
    "log_volatility_mean" => log_volatility_mean(process),
)
