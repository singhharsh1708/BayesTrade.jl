"""
Whether the system is currently entitled to trade.

Everything else in this package answers "what do I believe" and "may I do this". This file
answers a different question: **can the system prove its own state is sound right now**, and if
it cannot, it must do nothing.

The rule is one sentence. **Uncertainty means stop, not trade.** A stale feed, a model that
never fitted, a journal that cannot be written, an account whose history is missing — none of
these are reasons to guess. Each is a reason to sit still, and each has to be visible as a
reason rather than as a silence.

Note the direction of every check below: they are written so that the *absence* of evidence
fails. A check that passes when it cannot measure its input is a check that turns itself off
exactly when something has gone wrong.
"""

"""
    HealthStatus

Whether a single condition is satisfied, and what was seen.
"""
struct HealthStatus
    name::Symbol
    healthy::Bool
    detail::String
end

"""
    SystemHealth

Every condition, and the one verdict that follows from them.
"""
struct SystemHealth
    as_of::DateTime
    checks::Vector{HealthStatus}
    tradeable::Bool
end

healthy(status::HealthStatus) = status.healthy
problems(health::SystemHealth) =
    HealthStatus[status for status in health.checks if !status.healthy]

ok(name::Symbol, detail::AbstractString) = HealthStatus(name, true, String(detail))
bad(name::Symbol, detail::AbstractString) = HealthStatus(name, false, String(detail))

"""
    check_health(session, now)

Everything that must be true before an order may be sent.

`now` is passed in rather than read from the clock, so a health check is a test rather than a
race, and so a replay reaches the same verdict it would have reached live.
"""
function check_health(session::PaperTradingSession, now::DateTime)
    checks = HealthStatus[]

    # The venue's view, if there is a venue. Paper mode has no external account to disagree
    # with, so never having reconciled is not a failure. A comparison that ran and did not
    # agree is, and so is one that could not read the account at all: both mean the position
    # book may be describing an account that does not exist.
    # What a restart reconstructed. A session that could not rebuild its book confidently
    # must not trade on the empty one it started with.
    rebuilt = session.rebuild
    if rebuilt !== nothing
        push!(
            checks,
            rebuilt.consistent ?
                ok(
                    :state_rebuild,
                    string(
                        "rebuilt ", length(rebuilt.positions), " positions from ",
                        rebuilt.n_fills, " fills",
                    ),
                ) :
                bad(:state_rebuild, join(rebuilt.problems, "; ")),
        )
    end

    result = session.reconciliation
    if result !== nothing
        push!(
            checks,
            reconciled(result) ?
                ok(:reconciliation, string("books agreed at ", result.as_of)) :
                bad(
                    :reconciliation,
                    string(
                        lowercase(replace(string(result.status), "RECONCILE_" => "")), ": ",
                        result.detail,
                    ),
                ),
        )
    end

    # The feed. A feed that has gone quiet looks exactly like a still market, and a feed
    # that has never spoken is not a healthy feed waiting to start.
    quiet = silence(session.feed.health, now)
    push!(
        checks,
        if arrived_after_silence(session.feed.health)
            bad(
                :feed,
                string(
                    "this tick arrived after ", session.feed.health.last_gap,
                    " of silence, limit is ", session.feed.health.max_silence,
                ),
            )
        elseif is_stale(session.feed.health, now)
            bad(
                :feed,
                quiet === nothing ? "no tick has ever arrived" :
                    string("silent for ", quiet, ", limit is ", session.feed.health.max_silence),
            )
        else
            ok(:feed, string("last tick ", quiet, " ago"))
        end,
    )

    # The models. An unfitted model predicting is a prior-only guess wearing the costume of
    # a forecast.
    push!(
        checks,
        session.fitted ? ok(:models, string(length(session.models), " fitted")) :
            bad(:models, "no model has been fitted yet"),
    )

    unfitted = [
        string(model_name(model)) for model in session.models if !is_fitted(model)
    ]
    push!(
        checks,
        isempty(unfitted) ? ok(:model_state, "every model is fitted") :
            bad(:model_state, string("unfitted: ", join(unfitted, ", "))),
    )

    # The account. A drawdown limit measured against a peak that was never carried is a
    # limit that silently stops binding.
    push!(
        checks,
        session.peak_equity >= equity(session.broker) ?
            ok(:account, string("equity ", round(equity(session.broker); digits = 2))) :
            bad(:account, "peak equity is below current equity, so history was lost"),
    )
    push!(
        checks,
        session.current_day === nothing ?
            bad(:trading_day, "no trading day has opened") :
            ok(:trading_day, string(session.current_day)),
    )

    # The record. If a decision cannot be written down it cannot be explained afterwards,
    # and an unexplainable trade is one that should not have happened.
    journal = session.journal
    push!(
        checks,
        if journal === nothing
            ok(:journal, "no journal configured")
        elseif session.journal_failed
            bad(:journal, string("the last write to ", journal, " failed"))
        else
            writable = try
                mkpath(dirname(abspath(journal)))
                open(journal, "a") do handle
                end
                true
            catch error
                error isa InterruptException && rethrow()
                false
            end
            writable ? ok(:journal, journal) :
                bad(:journal, string("cannot write ", journal))
        end,
    )

    # Predictions made and never settled pile up when something downstream has stopped.
    push!(
        checks,
        length(session.pending) <= 10 * max(session.horizon_bars, 1) ?
            ok(:settlement, string(length(session.pending), " in flight")) :
            bad(
                :settlement,
                string(length(session.pending), " predictions unsettled, which is a backlog"),
            ),
    )

    return SystemHealth(now, checks, all(healthy, checks))
end

"""
    may_trade(health)

The only question execution should ask. False whenever anything could not be proved.
"""
may_trade(health::SystemHealth) = health.tradeable

"""
    summarise(health)

The state of the system in a form an operator can read at a glance.
"""
function summarise(health::SystemHealth)
    lines = String[
        string(
            health.as_of, "  ", health.tradeable ? "TRADEABLE" : "HALTED",
            health.tradeable ? "" : string(" (", length(problems(health)), " failing)"),
        ),
    ]
    for status in health.checks
        push!(
            lines,
            string("  ", status.healthy ? "ok   " : "FAIL ", rpad(status.name, 14), status.detail),
        )
    end
    return join(lines, "\n")
end

Base.show(io::IO, health::SystemHealth) = @printf(
    io, "<SystemHealth %s %s %d/%d>", health.as_of,
    health.tradeable ? "tradeable" : "halted",
    count(healthy, health.checks), length(health.checks)
)
