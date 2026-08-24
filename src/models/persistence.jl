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

    prior = model.regression.prior
    bundle = Dict{String, Any}(
        "schema" => MODEL_SCHEMA_VERSION,
        "model" => model_block(model),
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
    bundle["config"]["policy"] = policy_parameters(model.policy)

    return write_bundle(bundle, path)
end

"""
    save_model(model, path)

Write a fitted volatility model to `path`.

The statistics are stored, not the posterior, for the same reason the return model's are: a
saved model that cannot absorb tomorrow's bar is not much of a saved model. The discount grid
and the prior go in the configuration block beside them, since a set of statistics means
nothing without the grid they were accumulated under.
"""
function save_model(model::BayesianVolatilityModel, path::AbstractString)
    is_fitted(model) || throw(ArgumentError("refusing to save an unfitted model"))

    filter = model.filter
    bundle = Dict{String, Any}(
        "schema" => MODEL_SCHEMA_VERSION,
        "model" => model_block(model),
        "config" => Dict{String, Any}(
            "source" => source_name(model.source),
            "columns" => String[string(name) for name in feature_names(model)],
            "horizon_bars" => model.horizon_bars,
            "discounts" => copy(filter.discounts),
            "weight_forgetting" => filter.weight_forgetting,
            "centre" => filter.centre,
            "prior" => Dict{String, Any}(
                "shape" => filter.prior.shape,
                "rate" => filter.prior.rate,
            ),
        ),
        "state" => state(filter),
    )
    return write_bundle(bundle, path)
end

"""
    model_block(model)

The identity block every bundle carries, whatever kind of model wrote it.

`name` has been written since schema 1 and was never read back. Reading it is what lets a
second model type share the format without a schema bump, so every file already on disk keeps
loading.
"""
function model_block(model::ProbabilisticModel)
    version = model_version(model)
    return Dict{String, Any}(
        "name" => slug(version.name),
        "version" => string(version.version),
        "params_hash" => version.params_hash,
        "fitted_at" => stamp(version.fitted_at),
        "train_start" => stamp(version.train_start),
        "train_end" => stamp(version.train_end),
        "n_observations" => n_observations(model),
    )
end

function write_bundle(bundle::Dict{String, Any}, path::AbstractString)
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
    # Read lazily rather than into a typed dictionary. The typed read builds its container
    # through a generic path that does not infer, and the lazy object supports the string
    # indexing this loader uses without materialising anything it will not touch.
    parsed = try
        JSON3.read(read(path, String))
    catch error
        error isa InterruptException && rethrow()
        throw(ModelFileError(string(path, ": ", sprint(showerror, error))))
    end

    # A bare number, string or list is valid JSON, so the top level is narrowed here rather
    # than left to fail somewhere below with a method error naming JSON3 internals instead
    # of the file that was actually wrong.
    parsed isa AbstractDict ||
        throw(ModelFileError(string(path, ": holds a ", typeof(parsed), ", not a bundle")))
    bundle = parsed

    schema = haskey(bundle, "schema") ? bundle["schema"] : nothing
    schema == MODEL_SCHEMA_VERSION || throw(
        ModelFileError(
            string(
                path, ": schema ", schema, " cannot be read by this version, which writes ",
                "schema ", MODEL_SCHEMA_VERSION,
            ),
        ),
    )

    model = try
        stored_name = bundle_text(bundle_object(bundle, "model"), "name")
        if stored_name == "momentum"
            build_return_model(bundle)
        elseif stored_name == "volatility"
            build_volatility_model(bundle)
        else
            throw(ModelFileError(string(path, ": unknown model \"", stored_name, "\"")))
        end
    catch error
        error isa ModelFileError && rethrow()
        (
            error isa KeyError || error isa ArgumentError || error isa MethodError ||
                error isa InexactError || error isa TypeError
        ) || rethrow()
        throw(ModelFileError(string(path, ": ", sprint(showerror, error))))
    end

    # Required, not optional. Treating a missing hash as "nothing to check" means deleting
    # one line from a file turns off the only thing standing between a corrupt bundle and a
    # model that trades on it.
    stored = bundle_object(bundle, "model")
    expected = bundle_text(stored, "params_hash")
    if params_hash(model) != expected
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

"""
Field readers for a parsed bundle.

Below the top level a model file holds whatever a person or another program put there, so
every field is narrowed on the way out. A file carrying a string where a number belongs then
fails naming the field that was wrong, and the constructors below are only ever handed the
types they are declared to take.
"""
function bundle_field(bundle::AbstractDict, name::String)
    haskey(bundle, name) || throw(ModelFileError(string("missing \"", name, "\"")))
    return bundle[name]
end

function bundle_object(bundle::AbstractDict, name::String)
    value = bundle_field(bundle, name)
    value isa AbstractDict || throw(ModelFileError(string("\"", name, "\" is not an object")))
    return value
end

function bundle_number(bundle::AbstractDict, name::String)
    value = bundle_field(bundle, name)
    value isa Real || throw(
        ModelFileError(string("\"", name, "\" holds ", typeof(value), ", expected a number")),
    )
    return Float64(value)
end

function bundle_whole(bundle::AbstractDict, name::String)
    value = bundle_number(bundle, name)
    isinteger(value) ||
        throw(ModelFileError(string("\"", name, "\" is ", value, ", expected a whole number")))
    return Int(value)
end

function bundle_count(bundle::AbstractDict, name::String)
    value = bundle_whole(bundle, name)
    value >= 0 ||
        throw(ModelFileError(string("\"", name, "\" is ", value, ", expected a count")))
    return value
end

function bundle_text(bundle::AbstractDict, name::String)
    value = bundle_field(bundle, name)
    value isa AbstractString || throw(
        ModelFileError(string("\"", name, "\" holds ", typeof(value), ", expected a string")),
    )
    return String(value)
end

function bundle_numbers(bundle::AbstractDict, name::String)
    value = bundle_field(bundle, name)
    value isa AbstractVector || throw(ModelFileError(string("\"", name, "\" is not a list")))
    entries = Float64[]
    for entry in value
        entry isa Real || throw(
            ModelFileError(
                string("\"", name, "\" holds ", typeof(entry), ", expected numbers"),
            ),
        )
        push!(entries, entry)
    end
    return entries
end

function bundle_names(bundle::AbstractDict, name::String)
    value = bundle_field(bundle, name)
    value isa AbstractVector || throw(ModelFileError(string("\"", name, "\" is not a list")))
    entries = Symbol[]
    for entry in value
        entry isa AbstractString || throw(
            ModelFileError(
                string("\"", name, "\" holds ", typeof(entry), ", expected names"),
            ),
        )
        push!(entries, Symbol(entry))
    end
    return entries
end

function bundle_rows(bundle::AbstractDict, name::String)
    value = bundle_field(bundle, name)
    value isa AbstractVector ||
        throw(ModelFileError(string("\"", name, "\" is not a list of rows")))
    rows = Vector{Float64}[]
    for (index, row) in enumerate(value)
        row isa AbstractVector ||
            throw(ModelFileError(string("\"", name, "\" row ", index, " is not a list")))
        entries = Float64[]
        for entry in row
            entry isa Real || throw(
                ModelFileError(
                    string("\"", name, "\" holds ", typeof(entry), ", expected numbers"),
                ),
            )
            push!(entries, entry)
        end
        push!(rows, entries)
    end
    return rows
end

function bundle_matrix(bundle::AbstractDict, name::String)
    rows = bundle_rows(bundle, name)
    width = length(rows)
    entries = Matrix{Float64}(undef, width, width)
    for (index, row) in enumerate(rows)
        length(row) == width || throw(
            ModelFileError(
                string("\"", name, "\" row ", index, " has ", length(row), " entries, expected ", width),
            ),
        )
        entries[index, :] = row
    end
    return entries
end

function bundle_moment(bundle::AbstractDict, name::String)
    haskey(bundle, name) || return nothing
    bundle[name] === nothing && return nothing
    stamped = bundle_text(bundle, name)
    return try
        DateTime(stamped)
    catch error
        error isa InterruptException && rethrow()
        throw(ModelFileError(string("\"", name, "\" is not a timestamp: ", stamped)))
    end
end

function build_return_model(bundle::AbstractDict)
    config = bundle_object(bundle, "config")
    prior_bundle = bundle_object(config, "prior")
    scaler_bundle = bundle_object(bundle, "scaler")
    stored = bundle_object(bundle, "model")
    saved = bundle_object(bundle, "state")

    model = BayesianReturnModel(
        bundle_names(config, "feature_names");
        horizon_bars = bundle_whole(config, "horizon_bars"),
        forgetting = bundle_number(config, "forgetting"),
        policy = build_policy(config),
        prior = NormalInverseGammaPrior(
            bundle_numbers(prior_bundle, "mean"),
            bundle_matrix(prior_bundle, "precision"),
            bundle_number(prior_bundle, "shape"),
            bundle_number(prior_bundle, "rate"),
        ),
    )

    restore!(
        model;
        scaler = FeatureScaler(
            bundle_names(scaler_bundle, "names"),
            bundle_numbers(scaler_bundle, "centres"),
            bundle_numbers(scaler_bundle, "scales"),
        ),
        # The statistics are narrowed here rather than in `load_state!`, which is also reached
        # from callers that already hold real vectors and should not have to know about files.
        regression_state = Dict{String, Any}(
            "xx" => bundle_rows(saved, "xx"),
            "xy" => bundle_numbers(saved, "xy"),
            "yy" => bundle_number(saved, "yy"),
            "weight" => bundle_number(saved, "weight"),
            "n_seen" => bundle_count(saved, "n_seen"),
        ),
        n_observations = bundle_count(stored, "n_observations"),
        fitted_at = bundle_moment(stored, "fitted_at"),
        train_start = bundle_moment(stored, "train_start"),
        train_end = bundle_moment(stored, "train_end"),
    )
    return model
end

"""
    build_policy(config)

The response-scale policy a bundle describes.

A file written before policies existed has no `policy` block, and the model it describes was
by definition unscaled, so its absence means [`ConstantScale`](@ref) rather than an error.
That is what keeps every return-model file already on disk loading unchanged.
"""
function build_policy(config::AbstractDict)
    haskey(config, "policy") || return ConstantScale()
    policy = bundle_object(config, "policy")
    kind = bundle_text(policy, "kind")
    kind == "constant" && return ConstantScale()
    kind == "volatility" && return VolatilityScale(
        Symbol(bundle_text(policy, "column"));
        floor = bundle_number(policy, "floor"),
        annualised = bundle_flag(policy, "annualised"),
    )
    throw(ModelFileError(string("unknown response scale policy \"", kind, "\"")))
end

function bundle_flag(bundle::AbstractDict, name::String)
    value = bundle_field(bundle, name)
    value isa Bool || throw(
        ModelFileError(
            string("\"", name, "\" holds ", typeof(value), ", expected true or false"),
        ),
    )
    return value
end

function build_volatility_model(bundle::AbstractDict)
    config = bundle_object(bundle, "config")
    prior_bundle = bundle_object(config, "prior")
    stored = bundle_object(bundle, "model")
    saved = bundle_object(bundle, "state")

    source_kind = bundle_text(config, "source")
    columns = bundle_names(config, "columns")
    source = if source_kind == "squared_return"
        length(columns) == 1 || throw(
            ModelFileError(
                string("a squared-return source reads one column, got ", length(columns)),
            ),
        )
        SquaredReturnSource(first(columns))
    else
        throw(ModelFileError(string("unknown variance source \"", source_kind, "\"")))
    end

    model = BayesianVolatilityModel(
        source;
        horizon_bars = bundle_whole(config, "horizon_bars"),
        discounts = bundle_numbers(config, "discounts"),
        weight_forgetting = bundle_number(config, "weight_forgetting"),
        centre = bundle_number(config, "centre"),
        prior = InverseGammaPrior(
            bundle_number(prior_bundle, "shape"), bundle_number(prior_bundle, "rate"),
        ),
    )

    restore!(
        model;
        filter_state = Dict{String, Any}(
            "weights" => bundle_numbers(saved, "weights"),
            "squares" => bundle_numbers(saved, "squares"),
            "log_weights" => bundle_numbers(saved, "log_weights"),
            "n_seen" => bundle_count(saved, "n_seen"),
            "n_skipped" => bundle_count(saved, "n_skipped"),
        ),
        n_observations = bundle_count(stored, "n_observations"),
        fitted_at = bundle_moment(stored, "fitted_at"),
        train_start = bundle_moment(stored, "train_start"),
        train_end = bundle_moment(stored, "train_end"),
    )
    return model
end

stamp(moment::Union{DateTime, Nothing}) = moment === nothing ? nothing : string(moment)
