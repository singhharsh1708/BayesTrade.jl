"""
Feature standardisation, fitted once and then frozen.

Standardising with statistics computed over the whole sample is one of the quieter ways to
leak the future into a backtest. The mean and scale of a feature over 2022 to 2026 are not
knowable in 2023, and a model given centred features has been told something about the years
ahead of it.

The scaler here is therefore a fitted parameter like any other. It is estimated on the
training window, stored with the model, and applied unchanged to every later row, including
rows that fall outside the range it was fitted on. A feature that drifts out of that range
produces a large standardised value, which is correct: the model is being asked about a
situation it was not trained on, and the predictive interval should widen accordingly.
"""

const MIN_SCALE = 1.0e-12

"""
    FeatureScaler

Centres and scales a design matrix using statistics fixed at fitting time.
"""
struct FeatureScaler
    names::Vector{Symbol}
    centres::Vector{Float64}
    scales::Vector{Float64}

    function FeatureScaler(
            names::AbstractVector{Symbol}, centres::AbstractVector{<:Real},
            scales::AbstractVector{<:Real},
        )
        width = length(names)
        (length(centres) == width && length(scales) == width) || throw(
            ArgumentError(
                string(
                    "scaler for ", width, " features got ", length(centres),
                    " centres and ", length(scales), " scales",
                ),
            ),
        )
        all(>(0), scales) || throw(ArgumentError("every scale must be positive"))
        return new(
            convert(Vector{Symbol}, names),
            convert(Vector{Float64}, centres),
            convert(Vector{Float64}, scales),
        )
    end
end

n_features(scaler::FeatureScaler) = length(scaler.names)

"""
    fit_scaler(names, design)

Estimate centres and scales from the training design matrix.

A feature that does not vary in the training window is given a scale of one rather than zero.
Centring already reduces it to a column of zeros, so it contributes nothing and the prior
keeps its coefficient at zero; dividing by its own zero spread would instead produce
infinities.
"""
function fit_scaler(names::AbstractVector{Symbol}, design::AbstractMatrix{<:Real})
    matrix = convert(Matrix{Float64}, design)
    width = length(names)
    Base.size(matrix, 2) == width || throw(
        ArgumentError(
            string(
                "expected a matrix with ", width, " columns, got ", Base.size(matrix),
            ),
        ),
    )
    Base.size(matrix, 1) >= 2 ||
        throw(ArgumentError("need at least two rows to estimate a scale"))

    centres = Vector{Float64}(undef, width)
    scales = Vector{Float64}(undef, width)
    for column in 1:width
        values = view(matrix, :, column)
        centres[column] = mean(values)
        spread = std(values)
        scales[column] = spread > MIN_SCALE ? spread : 1.0
    end
    return FeatureScaler(names, centres, scales)
end

"""
    transform(scaler, design)

Apply the frozen statistics to a matrix of rows.
"""
function transform(scaler::FeatureScaler, design::AbstractMatrix{<:Real})
    matrix = convert(Matrix{Float64}, design)
    width = n_features(scaler)
    Base.size(matrix, 2) == width || throw(
        ArgumentError(
            string("expected a matrix with ", width, " columns, got ", Base.size(matrix)),
        ),
    )
    scaled = Matrix{Float64}(undef, Base.size(matrix, 1), width)
    for column in 1:width, row in 1:Base.size(matrix, 1)
        scaled[row, column] = (matrix[row, column] - scaler.centres[column]) /
            scaler.scales[column]
    end
    return scaled
end

"""
    transform_row(scaler, row)

Apply the frozen statistics to one row.
"""
function transform_row(scaler::FeatureScaler, row::AbstractVector{<:Real})
    values = convert(Vector{Float64}, row)
    width = n_features(scaler)
    length(values) == width || throw(
        ArgumentError(string("expected ", width, " features, got ", length(values))),
    )
    scaled = Vector{Float64}(undef, width)
    for index in 1:width
        scaled[index] = (values[index] - scaler.centres[index]) / scaler.scales[index]
    end
    return scaled
end

"""
    unscale_coefficients(scaler, coefficients)

Convert standardised coefficients back to the features' own units.

Standardised coefficients are the comparable ones, since they say how much the prediction
moves per typical move in the feature. The raw ones are what a reader checks against a
textbook definition, so both are worth being able to produce.
"""
function unscale_coefficients(scaler::FeatureScaler, coefficients::AbstractVector{<:Real})
    values = convert(Vector{Float64}, coefficients)
    width = n_features(scaler)
    length(values) == width || throw(
        ArgumentError(string("expected ", width, " coefficients, got ", length(values))),
    )
    raw = Vector{Float64}(undef, width)
    for index in 1:width
        raw[index] = values[index] / scaler.scales[index]
    end
    return raw
end

parameters(scaler::FeatureScaler) = Dict{String, Any}(
    "names" => String[string(name) for name in scaler.names],
    "centres" => copy(scaler.centres),
    "scales" => copy(scaler.scales),
)
