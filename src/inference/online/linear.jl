"""
Bayesian linear regression with a normal-inverse-gamma prior.

The whole model is closed form, which is the point. Everything here runs inside the trading
loop, so nothing may sample, iterate to convergence, or take a variable amount of time.
Conjugacy buys exactly that: absorbing an observation is a rank-one update to a precision
matrix, and predicting is one triangular solve.

The mathematics, with `X` the design matrix and `y` the responses:

    beta | sigma^2 ~ Normal(m0, sigma^2 V0)
    sigma^2        ~ InverseGamma(a0, b0)

    Lambda_n = Lambda_0 + X'X
    m_n      = Lambda_n^-1 (Lambda_0 m0 + X'y)
    a_n      = a0 + n/2
    b_n      = b0 + (m0' Lambda_0 m0 + y'y - m_n' Lambda_n m_n) / 2

and the posterior predictive at a new row `x` is Student-t, not normal:

    y* ~ StudentT(df = 2 a_n, loc = x'm_n, scale^2 = (b_n / a_n)(1 + x' Lambda_n^-1 x))

The heavier tails are not a technicality. They are what an unknown noise variance actually
implies, and using a normal instead would understate the probability of a large loss, which is
the one quantity this system must not understate.

The scale splits cleanly: the `1` is observation noise the data cannot remove, and the
`x' Lambda_n^-1 x` is uncertainty about the coefficients, which shrinks as data arrives. That
split is carried through to the result so a position sizer can respond to the two differently.
"""

"""
    MIN_RATE

Floor on the posterior rate.

The rate is a difference of large terms and can land a hair below zero through rounding alone,
on data where the residuals are genuinely almost zero. Clamping is right there; a negative rate
would produce a complex scale and fail far from the cause.
"""
const MIN_RATE = 1.0e-12

"""
    NormalInverseGammaPrior

Conjugate prior for the coefficients and the noise variance.

Parameterised by precision rather than covariance. A vague prior is then a small number rather
than a large one, which is both better conditioned and easier to reason about: precision zero
is genuinely no information, whereas the covariance that represents it does not exist.
"""
struct NormalInverseGammaPrior
    mean::Vector{Float64}
    precision::Matrix{Float64}
    shape::Float64
    rate::Float64

    function NormalInverseGammaPrior(
            mean::AbstractVector{<:Real}, precision::AbstractMatrix{<:Real},
            shape::Real, rate::Real,
        )
        # Converted to concrete types before anything is checked. Validating an abstractly
        # typed matrix means every comparison below runs through a generic path, which is
        # both slower and harder to reason about than the arithmetic it is checking.
        centre = convert(Vector{Float64}, mean)
        matrix = convert(Matrix{Float64}, precision)
        width = length(centre)

        width >= 1 || throw(ArgumentError("need at least one coefficient"))
        Base.size(matrix) == (width, width) || throw(
            ArgumentError(
                string(
                    "prior precision must be ", width, "x", width,
                    ", got ", Base.size(matrix),
                ),
            ),
        )
        # Checked elementwise rather than through `isapprox` against a transposed view. The
        # norm-based comparison answers a different question — whether the matrices are
        # close overall — where what matters is that no single pair of entries disagrees,
        # and a large well-conditioned block would otherwise mask a small asymmetric one.
        for column in 1:width, row in 1:(column - 1)
            isapprox(matrix[row, column], matrix[column, row]; atol = 1.0e-12) || throw(
                ArgumentError(
                    string(
                        "prior precision must be symmetric, but entry (", row, ", ",
                        column, ") is ", matrix[row, column], " against ",
                        matrix[column, row],
                    ),
                ),
            )
        end

        # Checked by attempting a factorisation rather than by decomposing. A Cholesky
        # succeeds exactly when the matrix is positive definite, and nudging the diagonal
        # first admits the semi-definite case a genuinely vague prior needs, where one
        # direction carries no information at all.
        nudged = Symmetric(matrix + 1.0e-10 * Matrix{Float64}(I, width, width))
        issuccess(cholesky(nudged; check = false)) ||
            throw(ArgumentError("prior precision must be positive semi-definite"))

        shape > 1 || throw(
            ArgumentError(
                string(
                    "prior shape must exceed 1 for the noise variance to have a mean, got ",
                    shape,
                ),
            ),
        )
        rate > 0 || throw(ArgumentError(string("prior rate must be positive, got ", rate)))
        return new(centre, matrix, Float64(shape), Float64(rate))
    end
end

n_features(prior::NormalInverseGammaPrior) = length(prior.mean)

"""
    expected_noise_variance(prior)

`E[sigma^2]` under the prior.
"""
expected_noise_variance(prior::NormalInverseGammaPrior) = prior.rate / (prior.shape - 1)

"""
    weakly_informative_prior(n_features; residual_scale, coefficient_scale, shape)

A prior that expresses a scale rather than a belief about direction.

Coefficients are centred on zero, which for a return model is the honest starting point: no
edge until the data says otherwise. `residual_scale` sets where the noise is expected to sit,
and `coefficient_scale` how far a coefficient may plausibly move, which is what keeps a
near-collinear design from producing an enormous fitted weight on a feature that barely varies.
"""
function weakly_informative_prior(
        n_features::Integer;
        residual_scale::Real,
        coefficient_scale::Real = 1.0,
        shape::Real = 2.0,
    )
    n_features >= 1 ||
        throw(ArgumentError(string("need at least one feature, got ", n_features)))
    residual_scale > 0 ||
        throw(ArgumentError(string("residual_scale must be positive, got ", residual_scale)))
    coefficient_scale > 0 || throw(
        ArgumentError(
            string("coefficient_scale must be positive, got ", coefficient_scale),
        ),
    )
    return NormalInverseGammaPrior(
        zeros(Float64, n_features),
        Matrix{Float64}(I, n_features, n_features) ./ coefficient_scale^2,
        shape,
        residual_scale^2 * (shape - 1),
    )
end

"""
    BayesianLinearModel

Online conjugate linear regression.

Sufficient statistics are kept separately from the prior, so a forgetting factor can discount
the data without also discounting the prior. At `forgetting = 1` this is the exact conjugate
posterior; below one, observations decay geometrically and the model tracks a coefficient that
moves, which is the usual situation in a market and the usual way a model trained on five years
of history is wrong about this month.
"""
mutable struct BayesianLinearModel
    prior::NormalInverseGammaPrior
    forgetting::Float64
    xx::Matrix{Float64}
    xy::Vector{Float64}
    yy::Float64
    weight::Float64
    n_seen::Int

    function BayesianLinearModel(prior::NormalInverseGammaPrior; forgetting::Real = 1.0)
        0 < forgetting <= 1 || throw(
            ArgumentError(string("forgetting must lie in (0, 1], got ", forgetting)),
        )
        size = n_features(prior)
        return new(
            prior, Float64(forgetting), zeros(Float64, size, size),
            zeros(Float64, size), 0.0, 0.0, 0,
        )
    end
end

n_features(model::BayesianLinearModel) = n_features(model.prior)

"""
    n_absorbed(model)

Observations actually absorbed, regardless of how much they still count.
"""
n_absorbed(model::BayesianLinearModel) = model.n_seen

"""
    effective_sample_size(model)

Discounted observation count, which is what the posterior actually reflects.

With forgetting this converges to `1 / (1 - forgetting)` rather than growing, so the posterior
stays responsive instead of hardening around old data.
"""
effective_sample_size(model::BayesianLinearModel) = model.weight

"""
    posterior_precision(model)

Posterior precision of the coefficients, up to the noise scale.
"""
posterior_precision(model::BayesianLinearModel) = model.prior.precision + model.xx

posterior_shape(model::BayesianLinearModel) = model.prior.shape + model.weight / 2

"""
    posterior_rate(model)

Posterior rate.

Computed from the identity `m' Lambda m = m' eta`, where `eta` is the linear term, which
avoids forming the quadratic through the precision matrix a second time.
"""
function posterior_rate(model::BayesianLinearModel)
    linear = model.prior.precision * model.prior.mean + model.xy
    mean = solve_precision(model, linear)
    prior_term = dot(model.prior.mean, model.prior.precision * model.prior.mean)
    value = model.prior.rate + (prior_term + model.yy - dot(mean, linear)) / 2
    return max(value, MIN_RATE)
end

"""
    coefficients(model)

Posterior mean of the coefficients.
"""
coefficients(model::BayesianLinearModel) =
    solve_precision(model, model.prior.precision * model.prior.mean + model.xy)

"""
    noise_variance(model)

`E[sigma^2]` under the posterior.
"""
noise_variance(model::BayesianLinearModel) =
    posterior_rate(model) / (posterior_shape(model) - 1)

residual_scale(model::BayesianLinearModel) = sqrt(noise_variance(model))

"""
    coefficient_covariance(model)

Marginal covariance of the coefficients, with the noise scale integrated out.
"""
coefficient_covariance(model::BayesianLinearModel) =
    noise_variance(model) * inverse_precision(model)

coefficient_std(model::BayesianLinearModel) =
    sqrt.(diag(coefficient_covariance(model)))

"""
    fit!(model, X, y)

Absorb a batch, starting again from the prior.

Computed in one pass rather than by looping the rank-one update. Row `i` of `n` carries weight
`forgetting^(n - i)`, which is exactly what the sequential update leaves behind, so a batch fit
and a sequence of updates over the same rows in the same order agree to floating-point
precision. Walk-forward evaluation refits hundreds of times, and the difference is between
seconds and minutes.
"""
function fit!(model::BayesianLinearModel, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real})
    design, responses = validate_design(model, X, y)
    reset!(model)
    n = Base.size(design, 1)
    n == 0 && return model

    weights = model.forgetting < 1 ?
        Float64[model.forgetting^(n - index) for index in 1:n] : ones(Float64, n)

    weighted = Matrix{Float64}(undef, n, n_features(model))
    for column in 1:n_features(model), row in 1:n
        weighted[row, column] = design[row, column] * weights[row]
    end

    total = 0.0
    for row in 1:n
        total += weights[row] * responses[row]^2
    end

    # `mul!` into a preallocated buffer rather than `*`, which avoids an intermediate
    # allocation and names the method instead of walking a dispatch tree.
    size = n_features(model)
    scatter = Matrix{Float64}(undef, size, size)
    mul!(scatter, transpose(design), weighted)

    # The matrix-vector half is written out. The generic path routes through `gemv!`, which
    # carries a branch for symmetric operands that this call can never take, and the loop is
    # O(n*d) against a design that is at most a few dozen columns wide.
    linear = zeros(Float64, size)
    for column in 1:size, row in 1:n
        linear[column] += weighted[row, column] * responses[row]
    end

    model.xx = scatter
    model.xy = linear
    model.yy = total
    model.weight = sum(weights)
    model.n_seen = n
    return model
end

"""
    update!(model, x, y)

Absorb one observation. Rank-one, and the only operation the hot path needs.
"""
function update!(model::BayesianLinearModel, x::AbstractVector{<:Real}, y::Real)
    row = convert(Vector{Float64}, x)
    length(row) == n_features(model) || throw(
        ArgumentError(
            string("expected ", n_features(model), " features, got ", length(row)),
        ),
    )
    isfinite(y) || throw(ArgumentError(string("response must be finite, got ", y)))
    all(isfinite, row) || throw(ArgumentError("features must be finite"))

    if model.forgetting < 1
        model.xx .*= model.forgetting
        model.xy .*= model.forgetting
        model.yy *= model.forgetting
        model.weight *= model.forgetting
    end
    # Written out rather than broadcast. This is the hot path, the loop allocates nothing,
    # and a broadcast over an abstractly typed accumulator does not infer.
    response = Float64(y)
    size = n_features(model)
    for column in 1:size, index in 1:size
        model.xx[index, column] += row[index] * row[column]
    end
    for index in 1:size
        model.xy[index] += row[index] * response
    end
    model.yy += response^2
    model.weight += 1
    model.n_seen += 1
    return model
end

"""
    reset!(model)

Forget every observation and return to the prior.
"""
function reset!(model::BayesianLinearModel)
    size = n_features(model)
    model.xx = zeros(Float64, size, size)
    model.xy = zeros(Float64, size)
    model.yy = 0.0
    model.weight = 0.0
    model.n_seen = 0
    return model
end

"""
    predict(model, x)

The posterior predictive for one row of features, as a `(distribution, epistemic_variance)`
pair.

The epistemic part is the share of the predictive variance attributable to not knowing the
coefficients. It vanishes as evidence accumulates, unlike the observation noise beside it.
"""
function predict(model::BayesianLinearModel, x::AbstractVector{<:Real})
    row = convert(Vector{Float64}, x)
    length(row) == n_features(model) || throw(
        ArgumentError(
            string("expected ", n_features(model), " features, got ", length(row)),
        ),
    )
    all(isfinite, row) || throw(ArgumentError("features must be finite"))

    leverage = max(0.0, dot(row, solve_precision(model, row)))
    shape = posterior_shape(model)
    scale = sqrt(posterior_rate(model) / shape * (1 + leverage))
    distribution = student_t(dot(row, coefficients(model)), scale, 2 * shape)

    total = var(distribution)
    epistemic = isfinite(total) ? total * leverage / (1 + leverage) : 0.0
    return distribution, epistemic
end

"""
    predict_mean(model, X)

Point predictions for many rows, for scoring and diagnostics.
"""
function predict_mean(model::BayesianLinearModel, X::AbstractMatrix{<:Real})
    design = convert(Matrix{Float64}, X)
    Base.size(design, 2) == n_features(model) || throw(
        ArgumentError(
            string(
                "expected a matrix with ", n_features(model), " columns, got ",
                Base.size(design),
            ),
        ),
    )
    return design * coefficients(model)
end

"""
    parameters(model)

Fitted parameters, in a form that hashes stably and reads in a log.
"""
parameters(model::BayesianLinearModel) = Dict{String, Any}(
    "coefficients" => coefficients(model),
    "coefficient_std" => coefficient_std(model),
    "shape" => posterior_shape(model),
    "rate" => posterior_rate(model),
    "residual_scale" => residual_scale(model),
    "effective_sample_size" => effective_sample_size(model),
    "forgetting" => model.forgetting,
)

"""
    state(model)

The sufficient statistics, enough to reconstruct the posterior exactly.

Not the posterior itself. Storing the mean and covariance would lose the ability to keep
updating, and a saved model that cannot absorb tomorrow's observation is not much of a saved
model.
"""
state(model::BayesianLinearModel) = Dict{String, Any}(
    "xx" => [collect(row) for row in eachrow(model.xx)],
    "xy" => copy(model.xy),
    "yy" => model.yy,
    "weight" => model.weight,
    "n_seen" => model.n_seen,
)

"""
    load_state!(model, state)

Restore statistics produced by [`state`](@ref).
"""
function load_state!(model::BayesianLinearModel, saved::AbstractDict)
    rows = saved["xx"]
    size = n_features(model)
    xx = Matrix{Float64}(undef, size, size)
    length(rows) == size || throw(
        ArgumentError(
            string("state describes ", length(rows), " rows for a ", size, "-feature model"),
        ),
    )
    for (index, row) in enumerate(rows)
        length(row) == size || throw(
            ArgumentError(
                string("state row ", index, " has ", length(row), " entries, expected ", size),
            ),
        )
        xx[index, :] = convert(Vector{Float64}, row)
    end
    # Copied, since `convert` is a no-op on a vector that already has the right type and
    # the model would otherwise share memory with the caller's dictionary.
    xy = copy(convert(Vector{Float64}, saved["xy"]))
    length(xy) == size ||
        throw(ArgumentError(string("state describes ", length(xy), " linear terms")))

    yy = Float64(saved["yy"])
    weight = Float64(saved["weight"])
    n_seen = Int(saved["n_seen"])
    # A sum of squares and a discounted count cannot be negative. Left unchecked these
    # surface much later as a domain error from a square root, naming neither the field nor
    # the file it came from.
    (isfinite(yy) && yy >= 0) ||
        throw(ArgumentError(string("state yy must be finite and non-negative, got ", yy)))
    (isfinite(weight) && weight >= 0) || throw(
        ArgumentError(string("state weight must be finite and non-negative, got ", weight)),
    )
    n_seen >= 0 ||
        throw(ArgumentError(string("state n_seen must be non-negative, got ", n_seen)))

    model.xx = xx
    model.xy = xy
    model.yy = yy
    model.weight = weight
    model.n_seen = n_seen
    return model
end

Base.show(io::IO, model::BayesianLinearModel) = @printf(
    io, "<BayesianLinearModel features=%d n=%d effective=%.1f residual_scale=%.5g>",
    n_features(model), model.n_seen, model.weight, residual_scale(model)
)

"""
    solve_precision(model, target)

Solve `precision * z = target`.

An explicit Cholesky rather than a bare backslash. The precision matrix is symmetric positive
definite by construction, so Cholesky is both the right factorisation and half the work of the
pivoted general solve a backslash would choose. A degenerate design falls back to a
least-norm solve rather than throwing, since a prior with any weight at all keeps the answer
meaningful.
"""
function solve_precision(model::BayesianLinearModel, target::Vector{Float64})
    dense = posterior_precision(model)
    factorisation = cholesky(Symmetric(dense); check = false)
    # `ldiv!` into a copy rather than `\`. The backslash routes through a generic solve,
    # and the argument is typed concretely rather than as an `AbstractVector` so the LAPACK
    # stride check resolves instead of widening to a type it cannot reason about. Every
    # caller in this file already holds a `Vector{Float64}`.
    issuccess(factorisation) && return ldiv!(factorisation, copy(target))

    # Unreachable with any proper prior, since the prior's own precision is already positive
    # definite. A ridge proportional to the scale of the problem is the honest fallback: it
    # says the answer is regularised rather than exact, where a pseudo-inverse would quietly
    # pick the least-norm solution and look like a real posterior.
    ridge = max(tr(dense) / n_features(model), 1.0) * sqrt(eps(Float64))
    nudged = cholesky(Symmetric(dense + ridge * I); check = false)
    issuccess(nudged) || throw(
        ArgumentError(
            "the posterior precision is not positive definite even after regularisation",
        ),
    )
    return ldiv!(nudged, copy(target))
end

function inverse_precision(model::BayesianLinearModel)
    size = n_features(model)
    inverse = Matrix{Float64}(undef, size, size)
    basis = zeros(Float64, size)
    for column in 1:size
        fill!(basis, 0.0)
        basis[column] = 1.0
        inverse[:, column] = solve_precision(model, basis)
    end
    return inverse
end

function validate_design(
        model::BayesianLinearModel, X::AbstractMatrix{<:Real}, y::AbstractVector{<:Real},
    )
    design = convert(Matrix{Float64}, X)
    responses = convert(Vector{Float64}, y)
    Base.size(design, 2) == n_features(model) || throw(
        ArgumentError(
            string(
                "expected ", n_features(model), " features, got ", Base.size(design, 2),
            ),
        ),
    )
    Base.size(design, 1) == length(responses) || throw(
        ArgumentError(
            string(
                Base.size(design, 1), " rows against ", length(responses), " responses",
            ),
        ),
    )
    all(isfinite, design) && all(isfinite, responses) ||
        throw(ArgumentError("design matrix and responses must be finite"))
    return design, responses
end
