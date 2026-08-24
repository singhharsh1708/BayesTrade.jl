"""
The deterministic risk engine.

Plain code with no learned parameters. Every ruling it makes is reproducible from the intent,
the portfolio and the limits alone, so a trade from a year ago can be re-judged without
re-running a single model.

**The risk engine can veto the decision engine, and the decision engine can never override the
risk engine.** That is the invariant the whole architecture rests on, and it is enforced by
shape rather than by discipline: `review` returns the only object execution will accept, and
the size it carries is the size that may be sent, which is never larger than what was asked
for and is often smaller or zero.

The engine consults no model and no prediction. It is handed an intent and it rules on what
that intent would do to the account.
"""

"""
    RiskCheck

One rule, and what it decided.

A check that did not apply is `SKIPPED` rather than `PASS`. The difference matters in a
post-mortem: a limit that was never evaluated is not a limit that was satisfied.
"""
struct RiskCheck
    name::Symbol
    status::RiskCheckStatus
    detail::String
    observed::Float64
    allowed::Float64
end

passed(check::RiskCheck) = check.status === PASS
failed(check::RiskCheck) = check.status === FAIL

"""
    RiskRuling

What execution is allowed to do, and the full record of how that was decided.

`approved_weight` is the only number execution may act on. It is capped by the intent, so the
engine can shrink a trade but never invent one.
"""
struct RiskRuling
    symbol::String
    as_of::DateTime
    action::Action
    requested_weight::Float64
    approved_weight::Float64
    checks::Vector{RiskCheck}
    mode::TradingMode

    function RiskRuling(;
            symbol::AbstractString,
            as_of::DateTime,
            action::Action,
            requested_weight::Real,
            approved_weight::Real,
            checks::Vector{RiskCheck},
            mode::TradingMode,
        )
        requested = Float64(requested_weight)
        approved = Float64(approved_weight)
        (isfinite(requested) && requested >= 0) ||
            throw(ArgumentError("requested_weight must be finite and non-negative"))
        (isfinite(approved) && approved >= 0) ||
            throw(ArgumentError("approved_weight must be finite and non-negative"))
        # The engine may shrink a trade. It may never grow one, and this is the line that
        # makes "risk outranks decision" a property of the type rather than a convention.
        approved <= requested || throw(
            ArgumentError(
                string(
                    "approved ", approved, " exceeds the requested ", requested,
                    "; the risk engine may only reduce",
                ),
            ),
        )
        approved > 0 && any(failed, checks) &&
            throw(ArgumentError("a ruling with a failed check cannot approve a trade"))
        return new(
            String(symbol), as_of, action, requested, approved, checks, mode,
        )
    end
end

"""
    approved(ruling)

Whether anything at all may be sent.
"""
approved(ruling::RiskRuling) = ruling.approved_weight > 0

"""
    failures(ruling)

Every check that vetoed, for the log and the post-mortem.
"""
failures(ruling::RiskRuling) = RiskCheck[check for check in ruling.checks if failed(check)]

"""
    was_reduced(ruling)

Whether the engine allowed the trade but not at the size that was asked for.
"""
was_reduced(ruling::RiskRuling) =
    approved(ruling) && ruling.approved_weight < ruling.requested_weight

pass(name::Symbol, detail::AbstractString, observed::Real, allowed::Real) =
    RiskCheck(name, PASS, String(detail), Float64(observed), Float64(allowed))
fail(name::Symbol, detail::AbstractString, observed::Real, allowed::Real) =
    RiskCheck(name, FAIL, String(detail), Float64(observed), Float64(allowed))
skip(name::Symbol, detail::AbstractString) =
    RiskCheck(name, SKIPPED, String(detail), NaN, NaN)

"""
    gate(name, observed, allowed, description)

A ceiling check, written once so every limit reads the same way in a log.
"""
gate(name::Symbol, observed::Real, allowed::Real, description::AbstractString) =
    observed <= allowed ?
    pass(name, string(description, " within limit"), observed, allowed) :
    fail(
        name,
        string(description, " ", round(observed; digits = 4), " exceeds ", allowed),
        observed, allowed,
    )

"""
    review(intent, portfolio, limits; mode, sector, halted, annualised_volatility, daily_turnover)

Rule on one intent.

Account-level limits are checked first and symbol-level ones after, because an account that has
breached its drawdown must stop trading everything rather than stop trading one name.

A trade that would breach a position or sector ceiling is **reduced to the headroom** rather
than refused outright, since a smaller trade is genuinely within the limit. A ceiling with no
headroom left fails instead, so a refusal always names something. Account-level breaches and
the kill switch refuse outright: there is no smaller version of a halted account.

`sector` is not defaulted. A missing sector is `SKIPPED`, never silently passed, because a
default value would compare the trade against the exposure of a sector nothing is held in and
report a limit as satisfied that was never evaluated.
"""
function review(
        intent::TradeIntent, portfolio::Portfolio, limits::RiskLimits;
        mode::TradingMode = PAPER,
        sector::Union{AbstractString, Nothing} = nothing,
        halted::Bool = false,
        annualised_volatility::Union{Real, Nothing} = nothing,
        daily_turnover::Union{Real, Nothing} = nothing,
    )
    checks = RiskCheck[]
    requested = intent.target_weight

    ruling(weight) = RiskRuling(
        symbol = intent.symbol, as_of = intent.as_of, action = intent.action,
        requested_weight = requested, approved_weight = weight, checks = checks,
        mode = mode,
    )

    if !is_actionable(intent)
        push!(
            checks,
            skip(
                :actionable,
                string(
                    "nothing to rule on: ",
                    intent.reason === nothing ? slug(intent.action) : slug(intent.reason),
                ),
            ),
        )
        return ruling(0.0)
    end

    push!(
        checks,
        halted ? fail(:kill_switch, "trading is halted", 1.0, 0.0) :
            pass(:kill_switch, "trading is live", 0.0, 0.0),
    )
    push!(checks, gate(:daily_loss, daily_loss(portfolio), limits.max_daily_loss, "daily loss"))
    push!(checks, gate(:drawdown, drawdown(portfolio), limits.max_drawdown, "drawdown"))

    existing = position_weight(portfolio, intent.symbol)
    opening = !haskey(portfolio.positions, intent.symbol)
    push!(
        checks,
        opening ?
            gate(
                :open_positions, n_positions(portfolio) + 1, limits.max_open_positions,
                "open positions",
            ) : skip(:open_positions, "already holding this symbol"),
    )

    if annualised_volatility === nothing
        push!(checks, skip(:volatility, "no volatility supplied"))
    else
        push!(
            checks,
            gate(
                :volatility, annualised_volatility, limits.max_annualised_volatility,
                "annualised volatility",
            ),
        )
    end

    if daily_turnover === nothing
        push!(checks, skip(:liquidity, "no turnover supplied"))
    else
        push!(
            checks,
            daily_turnover >= limits.min_daily_turnover ?
                pass(:liquidity, "turnover sufficient", daily_turnover, limits.min_daily_turnover) :
                fail(
                    :liquidity,
                    string("turnover ", daily_turnover, " below ", limits.min_daily_turnover),
                    daily_turnover, limits.min_daily_turnover,
                ),
        )
    end

    # Any account-level failure ends it here. A smaller trade does not fix a halted account or
    # a breached drawdown, so there is nothing to reduce to.
    any(failed, checks) && return ruling(0.0)

    # Ceilings reduce rather than refuse, because the headroom under a limit is a trade that
    # genuinely satisfies it. No headroom at all is a different thing and is recorded as a
    # failure: a refusal whose every check says PASS names nothing as its cause, and this is
    # the one component whose whole purpose is being auditable afterwards.
    allowed = requested

    function ceiling!(name::Symbol, used::Float64, cap::Float64, description::AbstractString)
        headroom = max(0.0, cap - used)
        if headroom <= 0
            push!(
                checks,
                fail(
                    name,
                    string(description, " ", round(used; digits = 4), " leaves no headroom under ", cap),
                    used, cap,
                ),
            )
        else
            push!(
                checks,
                pass(
                    name, string(description, " within limit"),
                    used + min(allowed, headroom), cap,
                ),
            )
        end
        allowed = min(allowed, headroom)
        return nothing
    end

    ceiling!(:position_weight, existing, limits.max_position_weight, "position weight")
    if sector === nothing
        push!(checks, skip(:sector_exposure, "no sector supplied"))
    else
        ceiling!(
            :sector_exposure, sector_exposure(portfolio, sector),
            limits.max_sector_exposure, "sector exposure",
        )
    end
    ceiling!(
        :portfolio_exposure, portfolio_exposure(portfolio),
        limits.max_portfolio_exposure, "portfolio exposure",
    )

    return ruling(any(failed, checks) ? 0.0 : max(0.0, allowed))
end

"""
    summarise(ruling)

The ruling in a form a person can read, which is what a post-mortem actually needs.
"""
function summarise(ruling::RiskRuling)
    verdict = approved(ruling) ?
        (
            was_reduced(ruling) ?
            @sprintf(
                "REDUCED to %.3f%% of %.3f%% requested",
                100 * ruling.approved_weight, 100 * ruling.requested_weight
            ) : @sprintf("APPROVED at %.3f%%", 100 * ruling.approved_weight)
        ) : "REFUSED"
    lines = String[
        string(
            ruling.symbol, " ", slug(ruling.action), " ", ruling.as_of, " [",
            slug(ruling.mode), "]: ", verdict,
        ),
    ]
    for check in ruling.checks
        push!(lines, string("  ", rpad(slug(check.status), 8), rpad(check.name, 22), check.detail))
    end
    return join(lines, "\n")
end

Base.show(io::IO, ruling::RiskRuling) = @printf(
    io, "<RiskRuling %s %s %s %.3f%% of %.3f%%>",
    ruling.symbol, slug(ruling.action), approved(ruling) ? "approved" : "refused",
    100 * ruling.approved_weight, 100 * ruling.requested_weight
)
