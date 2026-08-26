#!/usr/bin/env julia
#
#   julia --project=validation validation/mutations.jl [pattern]
#
# Section 21: do not trust the test count.
#
# Each entry breaks the source deliberately and records whether the suite noticed. A mutation the
# suite survives is a claim nothing is checking, whatever the number of assertions says.
#
# The catalogue is the record. Mutations found during this exercise that needed the tests
# strengthened are marked, because "caught" is only interesting alongside "caught after what".

using Printf

const ROOT = dirname(@__DIR__)

"""
    Mutation

One deliberate defect: which file, what to replace, and which test file should notice.
"""
struct Mutation
    id::String
    file::String
    from::String
    to::String
    suite::String
    note::String
end

const CATALOGUE = Mutation[
    Mutation(
        "linear-shape", "src/inference/online/linear.jl",
        "posterior_shape(model::BayesianLinearModel) = model.prior.shape + model.weight / 2",
        "posterior_shape(model::BayesianLinearModel) = model.prior.shape + model.weight",
        "test_model_mathematics.jl", "degrees of freedom grow at twice the rate",
    ),
    Mutation(
        "linear-epistemic", "src/inference/online/linear.jl",
        "    scale = sqrt(posterior_rate(model) / shape * (1 + leverage))",
        "    scale = sqrt(posterior_rate(model) / shape)",
        "test_model_mathematics.jl", "predictive ignores uncertainty about the coefficients",
    ),
    Mutation(
        "linear-clamp", "src/inference/online/linear.jl",
        "    return max(value, MIN_RATE)", "    return value",
        "test_model_mathematics.jl", "negative rate reaches sqrt; needed a deterministic case",
    ),
    Mutation(
        "linear-prior-rate", "src/inference/online/linear.jl",
        "    linear = model.prior.precision * model.prior.mean + model.xy",
        "    linear = model.xy",
        "test_model_mathematics.jl", "prior mean dropped; needed a non-zero-mean prior",
    ),
    Mutation(
        "variance-filtered-shape", "src/inference/online/variance.jl",
        "    filter.prior.shape + filter.weights[component] / 2",
        "    filter.prior.shape * 0.5 + filter.weights[component] / 2",
        "test_filters_fusion.jl", "df can fall to two; needed the filtered shape pinned",
    ),
    Mutation(
        "variance-evolved-shape", "src/inference/online/variance.jl",
        "    filter.prior.shape + filter.discounts[component] * filter.weights[component] / 2",
        "    filter.prior.shape * 0.5 + filter.discounts[component] * filter.weights[component] / 2",
        "test_filters_fusion.jl", "the one-step-ahead shape, read by predictive_df",
    ),
    Mutation(
        "variance-reweight", "src/inference/online/variance.jl",
        "        value = filter.weight_forgetting * filter.log_weights[index] +",
        "        value = 0.0 * filter.log_weights[index] +",
        "test_filters_fusion.jl", "the grid stops learning which decay is right",
    ),
    Mutation(
        "regime-propagate", "src/inference/online/regime.jl",
        "    moved = propagate(filter.belief, filter.transitions)",
        "    moved = copy(filter.belief)",
        "test_filters_fusion.jl", "stickiness becomes absolute; the chain never leaves a state",
    ),
    Mutation(
        "regime-skip", "src/inference/online/regime.jl",
        "    filter.belief = propagate(filter.belief, filter.transitions)\n    filter.n_skipped += 1",
        "    filter.n_skipped += 1",
        "test_filters_fusion.jl", "a gap in the data stops ageing the chain",
    ),
    Mutation(
        "fusion-disagreement", "src/fusion/fuse.jl",
        "        epistemic_variance = within + between,",
        "        epistemic_variance = within,",
        "test_filters_fusion.jl", "conflict between models stops counting as uncertainty",
    ),
    Mutation(
        "fusion-fallback", "src/fusion/fuse.jl",
        "        fill!(masses, 1 / length(masses))", "        masses[1] = 1.0",
        "test_filters_fusion.jl", "unweightable pool collapses onto whichever model is first",
    ),
    Mutation(
        "risk-reducing", "src/risk/engine.jl",
        "        return reducing ? 2 * existing + gap : gap", "        return gap",
        "test_risk_torture.jl", "a position over its limit can never be trimmed",
    ),
    Mutation(
        "risk-ceiling", "src/risk/engine.jl",
        "        allowed = min(allowed, headroom)", "        allowed = headroom",
        "test_risk_torture.jl", "a ceiling becomes a floor; approved exceeds requested",
    ),
    Mutation(
        "risk-early-return", "src/risk/engine.jl",
        "    any(failed, checks) && return ruling(0.0)", "    # mutated",
        "test_risk_torture.jl", "a breached account keeps being offered ceilings",
    ),
    Mutation(
        "risk-liquidity", "src/risk/engine.jl",
        "            daily_turnover >= limits.min_daily_turnover ?",
        "            daily_turnover > limits.min_daily_turnover ?",
        "test_risk_torture.jl", "the liquidity boundary moves by one tick",
    ),
    Mutation(
        "risk-kill-switch", "src/risk/engine.jl",
        "        halted ? fail(:kill_switch, \"trading is halted\", 1.0, 0.0) :",
        "        !halted ? fail(:kill_switch, \"trading is halted\", 1.0, 0.0) :",
        "test_risk_torture.jl", "the kill switch is inverted",
    ),
    Mutation(
        "walk-forward-embargo", "src/inference/offline/walk_forward.jl",
        "        train_stop = index - embargo - 1", "        train_stop = index - 1",
        "test_lookahead_adversarial.jl",
        "training reaches the bar being predicted; invisible to invariance alone",
    ),
    Mutation(
        "features-lookahead", "src/features/engine.jl",
        "    bars = history(engine.store, symbol; as_of = as_of, count = warmup_bars(engine))",
        "    bars = history(engine.store, symbol; as_of = as_of + Day(1), count = warmup_bars(engine))",
        "test_lookahead_adversarial.jl", "a feature reads one bar into the future",
    ),
    Mutation(
        "session-max-drawdown", "src/session/paper.jl",
        "    session.max_drawdown = max(\n        session.max_drawdown,\n        (session.peak_equity - equity(session.broker)) / session.peak_equity,\n    )",
        "    # mutated",
        "test_risk_torture.jl", "the reported drawdown becomes the final bar's again",
    ),
    Mutation(
        "resume-no-install", "src/ops/recovery.jl",
        "    if install_positions && account.consistent",
        "    if false",
        "test_rebuild.jl", "a restart silently starts with an empty book again",
    ),
    Mutation(
        "resume-install-anyway", "src/ops/recovery.jl",
        "    if install_positions && account.consistent",
        "    if install_positions",
        "test_rebuild.jl", "an inconsistent reconstruction is installed regardless",
    ),
    Mutation(
        "rebuild-cross-zero", "src/ops/rebuild.jl",
        "        if iszero(held) || sign(held) != sign(updated) && !iszero(updated)",
        "        if iszero(held)",
        "test_rebuild.jl", "a long is averaged into a short",
    ),
    Mutation(
        "reconcile-fail-open", "src/ops/reconciliation.jl",
        "may_open_new_positions(result::Reconciliation) = reconciled(result)",
        "may_open_new_positions(result::Reconciliation) = result.status !== RECONCILE_MISMATCHED",
        "test_reconciliation.jl", "an unreadable venue stops failing closed",
    ),
    Mutation(
        "reconcile-stale", "src/ops/reconciliation.jl",
        "    if age > tolerances.max_staleness || age < Millisecond(0)",
        "    if false",
        "test_reconciliation.jl", "an old snapshot counts as evidence about now",
    ),
    Mutation(
        "bar-finite", "src/domain/market.jl",
        "        all(isfinite, (open, high, low, close)) ||\n            throw(ArgumentError(\"\$symbol: every price must be finite\"))",
        "        # mutated",
        "test_provenance_stability.jl", "an infinite price enters through the store",
    ),
]

"""
    MARKER

Where an in-progress mutation records which file it broke.

`finally` restores the source on an exception and does nothing at all when the process is
killed, which is what a timeout does. A run that ends that way leaves a deliberate defect sitting
in the working tree looking exactly like code, and it has already been committed once.

The marker survives the kill. The next run reads it, restores the file from git, and says so.
"""
const MARKER = joinpath(@__DIR__, ".mutation-in-progress")

"""
    recover_interrupted!()

Undo a mutation left behind by a run that was killed.
"""
function recover_interrupted!()
    isfile(MARKER) || return nothing
    stranded = strip(read(MARKER, String))
    rm(MARKER; force = true)
    isempty(stranded) && return nothing
    println("A previous run was interrupted while ", stranded, " was mutated.")
    println("Restoring it from git.")
    run(`git -C $ROOT checkout -- $stranded`)
    return nothing
end

"""
    require_clean_tree()

Refuse to start with uncommitted changes under `src/`.

Not fussiness. This script edits source files and puts them back, and it cannot tell a change it
made from one somebody else was working on. Starting anyway risks reverting real work, which has
also already happened once.
"""
function require_clean_tree()
    clean = success(`git -C $ROOT diff --quiet -- src`)
    clean && return true
    println("src/ has uncommitted changes.")
    println("This script mutates source files and restores them from a saved copy, and it")
    println("cannot tell your edits from its own. Commit or stash first.")
    return false
end

run_suite(suite) = success(
    pipeline(
        `julia --project=$ROOT -e "
            using BayesTrade, Dates, Distributions, InteractiveUtils, JSON3, LinearAlgebra,
                Random, SpecialFunctions, Statistics, Test
            include(joinpath(\"$ROOT\", \"test\", \"$suite\"))"`;
        stdout = devnull, stderr = devnull,
    ),
)

function apply!(mutation::Mutation)
    path = joinpath(ROOT, mutation.file)
    original = read(path, String)
    count(mutation.from, original) == 1 || return nothing
    write(MARKER, mutation.file)
    write(path, replace(original, mutation.from => mutation.to))
    return original
end

function restore!(mutation::Mutation, original::AbstractString)
    write(joinpath(ROOT, mutation.file), original)
    rm(MARKER; force = true)
    return nothing
end

function main()
    recover_interrupted!()
    require_clean_tree() || return 2
    pattern = isempty(ARGS) ? "" : ARGS[1]
    selected = isempty(pattern) ? CATALOGUE :
        Mutation[m for m in CATALOGUE if occursin(pattern, m.id)]

    println("="^78)
    println("MUTATION CATALOGUE: ", length(selected), " deliberate defects")
    println("="^78)
    @printf("%-26s %-30s %s\n", "mutation", "suite", "result")

    survivors = String[]
    unanchored = String[]
    for mutation in selected
        original = apply!(mutation)
        if original === nothing
            push!(unanchored, mutation.id)
            @printf("%-26s %-30s %s\n", mutation.id, mutation.suite, "ANCHOR NOT FOUND")
            continue
        end
        caught = try
            !run_suite(mutation.suite)
        finally
            restore!(mutation, original)
        end
        caught || push!(survivors, mutation.id)
        @printf(
            "%-26s %-30s %s\n", mutation.id, mutation.suite,
            caught ? "caught" : "SURVIVED",
        )
    end

    println("="^78)
    @printf(
        "%d caught, %d survived, %d anchors missing\n",
        length(selected) - length(survivors) - length(unanchored),
        length(survivors), length(unanchored),
    )
    # Whatever happened above, the tree has to be as it was found. Checked rather than
    # assumed, because the one time it was not, a deliberate defect reached a commit.
    if !success(`git -C $ROOT diff --quiet -- src`)
        println()
        println("src/ is not as it was found. A mutation may still be applied.")
        println("Run: git -C ", ROOT, " diff -- src")
        return 3
    end
    isempty(survivors) || println("survivors: ", join(survivors, ", "))
    isempty(unanchored) ||
        println("anchors missing (the source moved): ", join(unanchored, ", "))
    return isempty(survivors) && isempty(unanchored) ? 0 : 1
end

exit(main())
