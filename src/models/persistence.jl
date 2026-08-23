"""
Saving and loading a fitted model.

A model is stored as its sufficient statistics rather than as its posterior. Storing the mean
and covariance would lose the ability to keep updating, and a saved model that cannot absorb
tomorrow's observation is not much of a saved model.

The bundle carries the parameter hash the model had when it was written, and loading verifies
it. A file that has been truncated, hand-edited, or produced by an incompatible version fails
at load with a clear message rather than silently trading on the wrong coefficients.

JSON rather than Julia serialisation, deliberately. A model file is something a person should
be able to read, diff and check into a repository, and `Serialization` produces a binary blob
tied to the exact package versions that wrote it, which is precisely the wrong property for a
record meant to explain a trade months later.

Unlike the bar files, which are eight fixed columns and hand-written, this uses a JSON
library. The format here is nested and irregular, with string escaping and number parsing to
get wrong, and hand-rolling that would be a liability rather than avoiding one.
"""

const MODEL_SCHEMA_VERSION = 1

"""
    ModelFileError

Raised when a model file cannot be loaded as the model it claims to be.
"""
struct ModelFileError <: Exception
    message::String
end

Base.showerror(io::IO, error::ModelFileError) = print(io, "ModelFileError: ", error.message)

"""
    save_model(model, path)

Write a fitted model to `path` as readable JSON.
"""
function save_model(model::BayesianReturnModel, path::AbstractString)
    is_fitted(model) || throw(ArgumentError("refusing to save an unfitted model"))
    scaler = model.scaler
    scaler === nothing && throw(ArgumentError("refusing to save an unfitted model"))

    version = model_version(model)
    prior = model.regression.prior
    bundle = Dict{String, Any}(
        "schema" => MODEL_SCHEMA_VERSION,
        "model" => Dict{String, Any}(
            "name" => slug(version.name),
            "version" => string(version.version),
            "params_hash" => version.params_hash,
            "fitted_at" => stamp(version.fitted_at),
            "train_start" => stamp(version.train_start),
            "train_end" => stamp(version.train_end),
            "n_observations" => n_observations(model),
        ),
        "config" => Dict{String, Any}(
            "feature_names" => String[string(name) for name in model.feature_names],
            "horizon_bars" => model.horizon_bars,
            "forgetting" => model.regression.forgetting,
            "prior" => Dict{String, Any}(
                "mean" => prior.mean,
                "precision" => [collect(row) for row in eachrow(prior.precision)],
                "shape" => prior.shape,
                "rate" => prior.rate,
            ),
        ),
        "scaler" => parameters(scaler),
        "state" => state(model.regression),
    )

    mkpath(dirname(abspath(path)))
    open(path, "w") do handle
        JSON3.pretty(handle, bundle)
        println(handle)
    end
    return path
end

"""
    load_model(path)

Read a model written by [`save_model`](@ref), verifying it round-tripped.
"""
function load_model(path::AbstractString)
    isfile(path) || throw(ModelFileError(string("no such model file: ", path)))
    # Read lazily rather than into a typed dictionary. The typed read builds the container
    # through a generic path that does not infer, and the lazy object supports the string
    # indexing and `get` this loader uses without materialising anything it will not touch.
    bundle = try
        JSON3.read(read(path, String))
    catch error
        error isa InterruptException && rethrow()
        throw(ModelFileError(string(path, ": ", sprint(showerror, error))))
    end

    schema = get(bundle, "schema", nothing)
    schema == MODEL_SCHEMA_VERSION || throw(
        ModelFileError(
            string(
                path, ": schema ", schema, " cannot be read by this version, which writes ",
                "schema ", MODEL_SCHEMA_VERSION,
            ),
        ),
    )

    model = try
        build_from_bundle(bundle)
    catch error
        (error isa KeyError || error isa ArgumentError || error isa MethodError) || rethrow()
        throw(ModelFileError(string(path, ": ", sprint(showerror, error))))
    end

    expected = get(bundle["model"], "params_hash", nothing)
    if expected !== nothing && params_hash(model) != expected
        throw(
            ModelFileError(
                string(
                    path, ": parameter hash ", params_hash(model),
                    " does not match the stored ", expected,
                    "; the file is corrupt or was written by another version",
                ),
            ),
        )
    end
    return model
end

function build_from_bundle(bundle::AbstractDict)
    config = bundle["config"]
    prior_bundle = config["prior"]
    scaler_bundle = bundle["scaler"]
    stored = bundle["model"]

    precision_rows = prior_bundle["precision"]
    width = length(precision_rows)
    precision = Matrix{Float64}(undef, width, width)
    for (index, row) in enumerate(precision_rows)
        precision[index, :] = convert(Vector{Float64}, row)
    end

    feature_names = Symbol[Symbol(name) for name in config["feature_names"]]
    model = BayesianReturnModel(
        feature_names;
        horizon_bars = Int(config["horizon_bars"]),
        forgetting = Float64(config["forgetting"]),
        prior = NormalInverseGammaPrior(
            convert(Vector{Float64}, prior_bundle["mean"]),
            precision,
            Float64(prior_bundle["shape"]),
            Float64(prior_bundle["rate"]),
        ),
    )

    restore!(
        model;
        scaler = FeatureScaler(
            Symbol[Symbol(name) for name in scaler_bundle["names"]],
            convert(Vector{Float64}, scaler_bundle["centres"]),
            convert(Vector{Float64}, scaler_bundle["scales"]),
        ),
        regression_state = bundle["state"],
        n_observations = Int(stored["n_observations"]),
        fitted_at = unstamp(get(stored, "fitted_at", nothing)),
        train_start = unstamp(get(stored, "train_start", nothing)),
        train_end = unstamp(get(stored, "train_end", nothing)),
    )
    return model
end

stamp(moment::Union{DateTime, Nothing}) = moment === nothing ? nothing : string(moment)
unstamp(::Nothing) = nothing
unstamp(value::AbstractString) = DateTime(value)
