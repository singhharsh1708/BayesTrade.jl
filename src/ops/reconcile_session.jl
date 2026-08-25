"""
Reconciling a running session.

Separate from `ops/reconciliation.jl` only because a method signature needs its argument types at
definition time, and the session carries a `Reconciliation` as a field, so the types have to come
first and the session-facing method has to come after.
"""

"""
    reconcile!(session, source_or_snapshot; tolerances, as_of, pending)

Compare the session's book against a venue, record the result on the session, and journal it.

The result is stored rather than returned only, because the thing that has to act on it is the
health check before the next decision, not the caller who happened to run the comparison.

Journalled whatever the outcome. A match is the evidence that the books agreed at a moment, and
tomorrow the useful question is usually "when did they last agree" rather than "when did they
stop".
"""
function reconcile!(
        session::PaperTradingSession, venue;
        tolerances::ReconciliationTolerances = ReconciliationTolerances(),
        as_of::Union{DateTime, Nothing} = nothing,
        pending::AbstractVector{Order} = Order[],
    )
    stamped = session.feed.health.last_tick_at
    moment = as_of !== nothing ? as_of :
        stamped !== nothing ? stamped :
        throw(
            ArgumentError(
                "nothing has been fed and no as_of was given, so there is no moment to " *
                "reconcile at",
            ),
        )
    view = local_account(session.broker, moment; pending = pending)
    result = reconcile(view, venue; tolerances = tolerances, as_of = moment)
    session.reconciliation = result
    record!(session, reconciliation_record(result))
    return result
end
