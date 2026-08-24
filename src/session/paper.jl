"""
A paper trading session that runs for weeks.

Everything before this file runs on a window of history and stops. This one is meant to be
left alone: it takes ticks as they arrive, builds bars, refits on a schedule, decides, and
writes down what it did. The failures it has to survive are the ones that only appear over
time.

* **The feed goes quiet.** A halted feed looks exactly like a still market, so a bar the feed
  did not deliver is skipped rather than absorbed, and the session records that it skipped.
* **A prediction has not settled yet.** A forecast made now is graded `horizon_bars` from now,
  so nothing is scored or absorbed until its outcome exists. Scoring on arrival would feed the
  models a return that has not happened.
* **The day rolls over.** The daily-loss limit is measured against the equity the day opened
  at, which means somebody has to notice the day changed.
* **It gets restarted.** Weeks of running will not be one process, so every decision is
  appended to a journal as it happens rather than held in memory and written at the end.

It is paper by construction. There is no flag that turns this into a live session; that takes
a different broker.
"""

const SESSION_SCHEMA_VERSION = 1

"""
    SessionCounters

What happened, in aggregate. The numbers a week of running is judged by.
"""
Base.@kwdef mutable struct SessionCounters
    ticks::Int = 0
    bars::Int = 0
    stale_bars::Int = 0
    halted_bars::Int = 0
    predictions::Int = 0
    declined::Int = 0
    approved::Int = 0
    reduced::Int = 0
    vetoed::Int = 0
    fills::Int = 0
    rejected::Int = 0
    refits::Int = 0
    settled::Int = 0
end

"""
    PaperTradingSession

The whole pipeline, wired up and left running.
"""
mutable struct PaperTradingSession{F <: Tuple}
    symbol::String
    horizon_bars::Int
    factories::F
    models::Vector{ProbabilisticModel}
    reliability::ModelReliability
    broker::PaperBroker
    engine::FeatureEngine{InMemoryBarStore}
    feed::FeedSession
    limits::RiskLimits
    sector::String
    warmup::Int
    refit_every::Int
    bars_since_refit::Int
    bar_index::Int
    fitted::Bool
    starting_equity::Float64
    peak_equity::Float64
    day_start_equity::Float64
    current_day::Union{Date, Nothing}
    pending::Vector{Tuple{Int, DateTime, Any}}
    counters::SessionCounters
    journal::Union{String, Nothing}
    journal_failed::Bool

    function PaperTradingSession(
            symbol::AbstractString, factories::F, features::FeatureSet;
            horizon_bars::Integer = 1,
            starting_cash::Real = 1.0e6,
            limits::RiskLimits = RiskLimits(),
            sector::AbstractString = "unknown",
            warmup::Integer = 800,
            refit_every::Integer = 50,
            interval::Period = Minute(1),
            interval_label::AbstractString = "1m",
            max_silence::Period = Minute(5),
            journal::Union{AbstractString, Nothing} = nothing,
        ) where {F <: Tuple}
        isempty(factories) && throw(ArgumentError("a session needs at least one model"))
        horizon_bars >= 1 ||
            throw(ArgumentError(string("horizon_bars must be positive, got ", horizon_bars)))
        warmup >= 1 || throw(ArgumentError("warmup must be positive"))
        refit_every >= 1 || throw(ArgumentError("refit_every must be positive"))

        models = ProbabilisticModel[factory() for factory in factories]
        names = ModelName[model_name(model) for model in models]
        broker = PaperBroker(starting_cash = starting_cash)
        return new{F}(
            String(symbol), Int(horizon_bars), factories, models,
            ModelReliability(names), broker,
            FeatureEngine(InMemoryBarStore(), features),
            FeedSession(
                symbol; interval = interval, label = interval_label,
                max_silence = max_silence,
            ),
            limits, String(sector), Int(warmup), Int(refit_every), 0, 0, false,
            Float64(starting_cash), Float64(starting_cash), Float64(starting_cash), nothing,
            Tuple{Int, DateTime, Any}[], SessionCounters(),
            journal === nothing ? nothing : String(journal), false,
        )
    end
end

function n_bars(session::PaperTradingSession, as_of::DateTime)
    return length(history(session.engine.store, session.symbol; as_of = as_of))
end

"""
    record!(session, entry)

Append one line to the journal.

Appended as it happens rather than held and written at the end, because weeks of running will
not be one process and a crash must not take the record with it.
"""
function record!(session::PaperTradingSession, entry::AbstractDict)
    path = session.journal
    path === nothing && return session
    # A journal that cannot be written must not take the session down with it. The write is
    # attempted, the failure is remembered, and the health check turns that into a refusal
    # to trade: unable to explain a decision is a reason to stop, not a reason to crash.
    try
        mkpath(dirname(abspath(path)))
        open(path, "a") do handle
            JSON3.write(handle, entry)
            println(handle)
        end
        session.journal_failed = false
    catch error
        error isa InterruptException && rethrow()
        session.journal_failed = true
    end
    return session
end

"""
    on_tick!(session, quote)

One tick in. Returns the bar it completed, if it completed one.
"""
function on_tick!(session::PaperTradingSession, price::Quote)
    session.counters.ticks += 1
    verdict, bar = handle_tick!(session.feed, price)
    verdict === :accepted || return nothing
    bar === nothing || on_bar!(session, bar)
    return bar
end

"""
    close_bar!(session)

Force the open bar closed, for the end of a session where no further tick is coming.
"""
function close_bar!(session::PaperTradingSession)
    bar = close_session!(session.feed)
    bar === nothing || on_bar!(session, bar)
    return bar
end

"""
    on_bar!(session, bar)

One completed bar through the whole pipeline.

The order matters and is the same order the replay engine uses: settle what has matured, refit
if due, then predict from the state as it now stands. A prediction is recorded when it is made
and scored only when its outcome exists.
"""
function on_bar!(session::PaperTradingSession, bar::Bar)
    session.counters.bars += 1
    session.bar_index += 1
    upsert!(session.engine.store, [bar])
    roll_day!(session, bar)

    price = Quote(session.symbol, bar.timestamp, bar.close; volume = bar.volume)
    mark_to_market!(session.broker, [price])
    session.peak_equity = max(session.peak_equity, equity(session.broker))

    settle!(session, bar)
    refit!(session, bar)
    session.fitted || return nothing
    return act!(session, bar, price)
end

"""
    roll_day!(session, bar)

Notice the day changed, and reset what the day resets.
"""
function roll_day!(session::PaperTradingSession, bar::Bar)
    today = Date(bar.timestamp)
    session.current_day == today && return session
    session.current_day = today
    session.day_start_equity = equity(session.broker)
    return session
end

"""
    settle!(session, bar)

Score and absorb every prediction whose outcome has now happened.

An outcome is usable only once it exists. At a horizon of one bar the distinction is invisible;
beyond that, scoring on arrival would hand the weights and the models a return from the future.
"""
function settle!(session::PaperTradingSession, bar::Bar)
    while !isempty(session.pending)
        # Counted in bars, not in calendar time. A five-bar horizon is five *trading* bars,
        # and a weekend makes that seven days: measuring the wait in days pops a prediction
        # before its outcome has printed, and it is then dropped rather than scored.
        (index, forecast_at, results) = first(session.pending)
        session.bar_index - index >= session.horizon_bars || break
        popfirst!(session.pending)
        example = training_example(session, forecast_at)
        example === nothing && continue
        score_fusion!(session.reliability, results, example.label.forward_log_return)
        for model in session.models
            try
                update!(model, example)
            catch error
                error isa ArgumentError || rethrow()
            end
        end
        session.counters.settled += 1
    end
    return session
end

"""
    training_example(session, as_of)

The labelled example for one bar, or `nothing` if its outcome has not printed yet.

Built directly rather than by scanning a freshly constructed training set. Rebuilding the whole
set on every settled bar is quadratic in the length of the session, which is invisible over an
afternoon and ruinous over a month.
"""
function training_example(session::PaperTradingSession, as_of::DateTime)
    label = try
        forward_label(
            session.engine.store, session.symbol;
            as_of = as_of, horizon_bars = session.horizon_bars,
        )
    catch error
        error isa ArgumentError || rethrow()
        return nothing
    end
    label === nothing && return nothing
    features = try
        features_at(session.engine, session.symbol, as_of)
    catch error
        error isa ArgumentError || rethrow()
        return nothing
    end
    return TrainingExample(features, label)
end

"""
    refit!(session, bar)

Refit on schedule, on a window that stops short of the present by the horizon.
"""
function refit!(session::PaperTradingSession, bar::Bar)
    session.bars_since_refit += 1
    session.fitted && session.bars_since_refit < session.refit_every && return session

    examples = try
        build_training_set(
            session.engine, session.symbol; horizon_bars = session.horizon_bars,
        )
    catch error
        error isa ArgumentError || rethrow()
        return session
    end
    length(examples) >= session.warmup || return session

    for model in session.models
        try
            fit!(model, examples)
        catch error
            error isa ArgumentError || error isa HorizonMismatchError || rethrow()
            return session
        end
    end
    session.fitted = true
    session.bars_since_refit = 0
    session.counters.refits += 1
    record!(
        session,
        Dict{String, Any}(
            "event" => "refit", "as_of" => string(bar.timestamp),
            "rows" => length(examples), "equity" => equity(session.broker),
        ),
    )
    return session
end

"""
    act!(session, bar, price)

Predict, fuse, decide, rule, and send whatever survives.
"""
function act!(session::PaperTradingSession, bar::Bar, price::Quote)
    vector = try
        features_at(session.engine, session.symbol, bar.timestamp)
    catch error
        error isa ArgumentError || rethrow()
        return nothing
    end

    # Fail closed. Every condition that must hold before an order may be sent is checked
    # here, and any one of them failing stops the bar. A stale feed, an unfitted model, an
    # account whose history was lost, a journal that cannot be written: none of these are
    # reasons to guess, and the refusal is recorded with the condition that caused it.
    health = check_health(session, bar.timestamp)
    if !may_trade(health)
        session.counters.halted_bars += 1
        is_stale(session.feed.health, bar.timestamp) && (session.counters.stale_bars += 1)
        record!(
            session,
            Dict{String, Any}(
                "event" => "halted", "as_of" => string(bar.timestamp),
                "failing" => String[string(status.name) for status in problems(health)],
                "detail" => String[status.detail for status in problems(health)],
            ),
        )
        return nothing
    end

    ready = ProbabilisticModel[
        model for model in session.models if can_predict(model, vector)
    ]
    isempty(ready) && return nothing

    results = Tuple(
        predict(
                model, vector; symbol = session.symbol, as_of = bar.timestamp,
                horizon_bars = session.horizon_bars,
            ) for model in ready
    )
    prediction = fuse(session.reliability, results)
    session.counters.predictions += 1
    push!(session.pending, (session.bar_index, bar.timestamp, results))

    intent = decide(prediction, session.limits)
    book = portfolio(
        session.broker; peak_equity = session.peak_equity,
        day_start_equity = session.day_start_equity, as_of = bar.timestamp,
    )
    ruling = review(
        intent, book, session.limits; sector = session.sector,
        annualised_volatility = annualise(std(prediction)),
        daily_turnover = bar.volume * bar.close,
    )

    is_actionable(intent) || (session.counters.declined += 1)
    if is_actionable(intent) && !approved(ruling)
        session.counters.vetoed += 1
    elseif approved(ruling)
        session.counters.approved += 1
        was_reduced(ruling) && (session.counters.reduced += 1)
    end

    filled = 0.0
    detail = ""
    if approved(ruling)
        order = order_from_ruling(session.broker, ruling, price, book.equity)
        if order !== nothing
            receipt = place_order!(session.broker, order, price)
            detail = receipt.detail
            # Checked on the fill itself rather than on the status. A working limit order
            # is neither filled nor rejected and carries no fill to read.
            fill = receipt.fill
            if fill === nothing
                session.counters.rejected += 1
            else
                session.counters.fills += 1
                filled = fill.quantity * (is_buy(order) ? 1 : -1)
            end
        end
    end

    record!(
        session,
        Dict{String, Any}(
            "event" => "bar", "as_of" => string(bar.timestamp),
            "close" => bar.close, "equity" => equity(session.broker),
            "mean" => mean(prediction), "sd" => std(prediction),
            "probability_up" => probability_positive(prediction),
            "epistemic_share" => epistemic_share(prediction),
            "models" => Int(length(ready)),
            "action" => slug(intent.action),
            "reason" => intent.reason === nothing ? nothing : slug(intent.reason),
            "requested" => ruling.requested_weight,
            "approved" => ruling.approved_weight,
            "failures" => String[string(check.name) for check in failures(ruling)],
            "filled" => filled, "detail" => detail,
        ),
    )
    return ruling
end

"""
    session_report(session)

Where the session has got to.
"""
function session_report(session::PaperTradingSession)
    value = equity(session.broker)
    return Dict{String, Any}(
        "schema" => SESSION_SCHEMA_VERSION,
        "symbol" => session.symbol,
        "fitted" => session.fitted,
        "bars" => session.counters.bars,
        "ticks" => session.counters.ticks,
        "predictions" => session.counters.predictions,
        "declined" => session.counters.declined,
        "vetoed" => session.counters.vetoed,
        "approved" => session.counters.approved,
        "reduced" => session.counters.reduced,
        "fills" => session.counters.fills,
        "rejected" => session.counters.rejected,
        "stale_bars" => session.counters.stale_bars,
        "halted_bars" => session.counters.halted_bars,
        "refits" => session.counters.refits,
        "settled" => session.counters.settled,
        "pending" => length(session.pending),
        "equity" => value,
        "return_pct" => 100 * (value / session.starting_equity - 1),
        "drawdown_pct" => 100 * max(0.0, (session.peak_equity - value) / session.peak_equity),
        "positions" => length(session.broker.positions),
        "reliabilities" => reliabilities(session.reliability),
        "models" => String[slug(name) for name in session.reliability.names],
    )
end

Base.show(io::IO, session::PaperTradingSession) = @printf(
    io, "<PaperTradingSession %s bars=%d fills=%d equity=%.2f %s>",
    session.symbol, session.counters.bars, session.counters.fills,
    equity(session.broker), session.fitted ? "fitted" : "warming up"
)
