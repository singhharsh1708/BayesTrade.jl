"""
Making a run identifiable.

A backtest that cannot be reconstructed is an anecdote. Six months from now the question will be
"why did the July run say 3.3% and this one says 1.1%", and the only way to answer it is to have
recorded what the July run actually was: which code, which data, which configuration, which seed.

The identifier is a content hash rather than a counter. Two runs with the same identifier are the
same run, wherever and whenever they were executed, and a run that will not reproduce announces
itself as a different identifier rather than as a puzzling number.
"""

const MANIFEST_SCHEMA_VERSION = 1

"""
    RunManifest

Everything needed to reconstruct one run, and a hash of it.

`extra` carries anything a caller wants pinned that this type does not know about, which is what
keeps it from needing a field per experiment.
"""
struct RunManifest
    run_id::String
    label::String
    package_version::String
    dataset::String
    configuration::Dict{String, Any}
    seed::Int
    features::Vector{String}
    models::Vector{String}
    created_at::DateTime
    extra::Dict{String, Any}
end

"""
    run_manifest(; label, dataset, configuration, seed, features, models, created_at, extra)

Build a manifest and the identifier that names it.

`created_at` is recorded but deliberately kept out of the hash. A run repeated tomorrow with the
same code, data and configuration is the same run, and an identifier that changed with the clock
would answer "is this reproducible" with "no" every time.
"""
function run_manifest(;
        label::AbstractString,
        dataset::AbstractString,
        configuration::AbstractDict = Dict{String, Any}(),
        seed::Integer = 0,
        features::AbstractVector = String[],
        models::AbstractVector = String[],
        created_at::DateTime,
        package_version::AbstractString = string(PACKAGE_VERSION),
        extra::AbstractDict = Dict{String, Any}(),
    )
    isempty(label) && throw(ArgumentError("a run needs a label"))
    isempty(dataset) && throw(ArgumentError("a run needs a dataset"))
    feature_names = sort(String[string(name) for name in features])
    model_names = sort(String[string(name) for name in models])

    payload = Dict{String, Any}(
        "schema" => MANIFEST_SCHEMA_VERSION,
        "label" => String(label),
        "package_version" => String(package_version),
        "dataset" => String(dataset),
        "configuration" => Dict{String, Any}(
            string(key) => value for (key, value) in configuration
        ),
        "seed" => Int(seed),
        "features" => feature_names,
        "models" => model_names,
        "extra" => Dict{String, Any}(string(key) => value for (key, value) in extra),
    )
    return RunManifest(
        stable_hash(payload), String(label), String(package_version), String(dataset),
        payload["configuration"], Int(seed), feature_names, model_names, created_at,
        payload["extra"],
    )
end

"""
    manifest_payload(manifest)

The manifest as it should be written beside a result.
"""
manifest_payload(manifest::RunManifest) = Dict{String, Any}(
    "schema" => MANIFEST_SCHEMA_VERSION,
    "run_id" => manifest.run_id,
    "label" => manifest.label,
    "package_version" => manifest.package_version,
    "dataset" => manifest.dataset,
    "configuration" => manifest.configuration,
    "seed" => manifest.seed,
    "features" => manifest.features,
    "models" => manifest.models,
    "created_at" => string(manifest.created_at),
    "extra" => manifest.extra,
)

"""
    run_label(manifest)

A human-readable name, of the shape the brief asks for.

    2026-08-25_BAYES_v0.1.0_ar1-2000_a1b2c3d4
"""
run_label(manifest::RunManifest) = join(
    [
        string(Date(manifest.created_at)),
        "BAYES",
        string("v", manifest.package_version),
        replace(manifest.dataset, r"[^A-Za-z0-9]+" => "-"),
        first(manifest.run_id, 8),
    ], "_",
)

Base.show(io::IO, manifest::RunManifest) =
    print(io, "<RunManifest ", run_label(manifest), ">")
