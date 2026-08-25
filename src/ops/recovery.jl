"""
Restarting without trading the same bar twice.

Weeks of running will not be one process. The journal is the only thing that survives a
restart, so it has to be enough to answer one question before the session does anything:
**which bars has this system already acted on?**

Without that answer a restart re-processes whatever the feed replays, and every re-processed
bar is a second order for a decision that was already taken. In paper mode that corrupts the
record; with a live broker it is a duplicate trade.

The watermark is the defence. A bar at or before the last one in the journal is skipped, and
skipping is recorded so a reader can tell the difference between a bar that was refused and a
bar that had already been handled.
"""

"""
    JournalState

What a journal says about a session that has already run.
"""
struct JournalState
    path::String
    entries::Int
    last_as_of::Union{DateTime, Nothing}
    last_equity::Union{Float64, Nothing}
    fills::Int
    halted::Int
    order_ids::Set{String}
end

Base.isempty(state::JournalState) = state.entries == 0

"""
    read_journal(path)

Read a journal without trusting it.

A journal is a file that a crash may have cut mid-line, so the last line is often incomplete.
That is expected: an unreadable trailing line is skipped rather than treated as corruption of
the whole record, because the alternative is discarding weeks of good history over one
truncated write.
"""
function read_journal(path::AbstractString)
    isfile(path) || return JournalState(String(path), 0, nothing, nothing, 0, 0, Set{String}())

    entries = 0
    last_as_of = nothing
    last_equity = nothing
    fills = 0
    halted = 0
    ids = Set{String}()

    for line in eachline(path)
        isempty(strip(line)) && continue
        entry = try
            JSON3.read(line)
        catch error
            error isa InterruptException && rethrow()
            continue    # a torn final write, not a reason to discard the rest
        end
        entry isa AbstractDict || continue
        entries += 1

        event = get(entry, "event", "")
        stamp = get(entry, "as_of", nothing)
        if stamp isa AbstractString
            moment = tryparse(DateTime, String(stamp))
            if moment !== nothing && (last_as_of === nothing || moment > last_as_of)
                last_as_of = moment
            end
        end
        event == "halted" && (halted += 1)
        if event == "bar"
            value = get(entry, "equity", nothing)
            value isa Real && (last_equity = Float64(value))
            filled = get(entry, "filled", 0)
            filled isa Real && !iszero(filled) && (fills += 1)
            id = get(entry, "order_id", nothing)
            id isa AbstractString && push!(ids, String(id))
        end
    end
    return JournalState(String(path), entries, last_as_of, last_equity, fills, halted, ids)
end

"""
    resume!(session, path)

Point a fresh session at the journal of a previous one.

This does **not** rebuild the position book. It sets the watermark, so the session refuses to
act on any bar the previous process already acted on, and reports what it found. Rebuilding
positions from a journal is a different and harder problem, and claiming to have done it here
would be worse than saying plainly that it is not done.
"""
function resume!(session::PaperTradingSession, path::AbstractString)
    state = read_journal(path)
    session.journal = String(path)
    session.watermark = state.last_as_of
    record!(
        session,
        Dict{String, Any}(
            "event" => "resumed",
            "journal" => state.path,
            "entries" => state.entries,
            "resumed_after" => state.last_as_of === nothing ? nothing :
                string(state.last_as_of),
            "prior_fills" => state.fills,
        ),
    )
    return state
end

"""
    already_handled(session, bar)

Whether this bar was acted on before the restart.
"""
function already_handled(session::PaperTradingSession, bar::Bar)
    mark = session.watermark
    mark === nothing && return false
    return bar.timestamp <= mark
end

Base.show(io::IO, state::JournalState) = @printf(
    io, "<JournalState %d entries through %s fills=%d halted=%d>",
    state.entries, state.last_as_of === nothing ? "nothing" : string(state.last_as_of),
    state.fills, state.halted
)
