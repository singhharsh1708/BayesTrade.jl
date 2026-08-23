"""
Resolved runtime configuration.

Live trading takes two independent keys, not one. `BAYESTRADE_TRADING_MODE=live` says where
orders should go; `BAYESTRADE_ALLOW_LIVE_TRADING=true` says a human meant it. A single
environment variable is too easy to inherit from a stale shell, a copied compose file or a
CI secret, and the failure mode is real money.

Settings are resolved by an explicit call and the environment is injectable, so a test
constructs whatever configuration it needs without mutating the process. Nothing here is
read at load time.
"""

"""
    Secret

A value that must never appear in a log, a stack trace or an error message.

`show` is overridden rather than trusted to callers, because the leak this prevents happens
by accident: a struct containing a credential printed at a REPL, or interpolated into an
exception by code that had no idea what it was holding.
"""
struct Secret
    value::String
end

Base.show(io::IO, ::Secret) = print(io, "Secret(********)")
Base.show(io::IO, ::MIME"text/plain", secret::Secret) = show(io, secret)
Base.string(::Secret) = "********"

"""
    reveal(secret)

The underlying value. Every call site is a place a credential can escape, so there are few
of them and they are all in the broker adapter.
"""
reveal(secret::Secret) = secret.value
reveal(::Nothing) = nothing

"""
    BrokerSettings

Broker credentials and request behaviour.
"""
Base.@kwdef struct BrokerSettings
    kite_api_key::Union{Secret, Nothing} = nothing
    kite_api_secret::Union{Secret, Nothing} = nothing
    kite_access_token::Union{Secret, Nothing} = nothing
    order_timeout_seconds::Float64 = 10.0
    max_retries::Int = 3
end

"""
    has_credentials(broker)

Whether an api key and secret are present. Enough to start a login, not to trade.
"""
has_credentials(broker::BrokerSettings) =
    broker.kite_api_key !== nothing && broker.kite_api_secret !== nothing

"""
    has_session(broker)

Whether a usable trading session exists: credentials plus a live access token.
"""
has_session(broker::BrokerSettings) =
    has_credentials(broker) && broker.kite_access_token !== nothing

Base.@kwdef struct DataSettings
    database_url::Union{Secret, Nothing} = nothing
    redis_url::Union{Secret, Nothing} = nothing
    cache_dir::String = joinpath("data", "cache")
    default_interval::String = "1d"
end

"""
    ExecutionSettings

Simulated execution costs, in basis points.

The defaults are non-zero on purpose. Optimism here is the cheapest way to fake a good
backtest, and a system that only looks profitable at zero cost is not profitable.
"""
Base.@kwdef struct ExecutionSettings
    slippage_bps::Float64 = 5.0
    brokerage_bps::Float64 = 3.0
    taxes_bps::Float64 = 12.0

    function ExecutionSettings(slippage_bps, brokerage_bps, taxes_bps)
        all(>=(0), (slippage_bps, brokerage_bps, taxes_bps)) ||
            throw(ArgumentError("execution costs cannot be negative"))
        return new(slippage_bps, brokerage_bps, taxes_bps)
    end
end

total_cost_bps(costs::ExecutionSettings) =
    costs.slippage_bps + costs.brokerage_bps + costs.taxes_bps

"""
    round_trip_cost(costs)

Cost of entering and exiting once, as a fraction of notional.
"""
round_trip_cost(costs::ExecutionSettings) = 2 * total_cost_bps(costs) / 1.0e4

"""
    Settings

Everything the system needs to run.
"""
Base.@kwdef struct Settings
    trading_mode::TradingMode = PAPER
    allow_live_trading::Bool = false

    starting_equity::Float64 = 1.0e6
    base_currency::String = "INR"
    universe::Vector{String} = String[]
    seed::Int = 7
    log_level::String = "INFO"

    risk::RiskLimits = RiskLimits()
    broker::BrokerSettings = BrokerSettings()
    data::DataSettings = DataSettings()
    execution::ExecutionSettings = ExecutionSettings()

    function Settings(
            trading_mode, allow_live_trading, starting_equity, base_currency,
            universe, seed, log_level, risk, broker, data, execution,
        )
        starting_equity > 0 ||
            throw(ArgumentError("starting_equity must be positive, got $starting_equity"))
        if trading_mode === LIVE
            allow_live_trading || throw(
                ArgumentError(
                    string(
                        "trading_mode is live but allow_live_trading is false. ",
                        "Live trading requires both BAYESTRADE_TRADING_MODE=live and ",
                        "BAYESTRADE_ALLOW_LIVE_TRADING=true, so no single stale ",
                        "variable can arm it.",
                    ),
                ),
            )
            has_session(broker) || throw(
                ArgumentError(
                    "live trading requires a broker api key, api secret and access token",
                ),
            )
        end
        return new(
            trading_mode, allow_live_trading, starting_equity, base_currency,
            universe, seed, log_level, risk, broker, data, execution,
        )
    end
end

is_live(settings::Settings) = settings.trading_mode === LIVE
is_simulated(settings::Settings) = is_simulated(settings.trading_mode)

"""
    load_settings(; env = ENV)

Resolve settings from environment variables prefixed `BAYESTRADE_`.

Nested settings use a double underscore, so `BAYESTRADE_RISK__MAX_DAILY_LOSS` reaches
`settings.risk.max_daily_loss`. The environment is a parameter rather than a global read,
which is what lets a test cover the live-trading guard without arming anything.
"""
function load_settings(; env::AbstractDict{<:AbstractString, <:AbstractString} = ENV)
    risk_defaults = RiskLimits()
    risk = RiskLimits(
        max_position_weight = envget(env, "RISK__MAX_POSITION_WEIGHT", risk_defaults.max_position_weight),
        max_sector_exposure = envget(env, "RISK__MAX_SECTOR_EXPOSURE", risk_defaults.max_sector_exposure),
        max_portfolio_exposure = envget(env, "RISK__MAX_PORTFOLIO_EXPOSURE", risk_defaults.max_portfolio_exposure),
        max_open_positions = envget(env, "RISK__MAX_OPEN_POSITIONS", risk_defaults.max_open_positions),
        max_daily_loss = envget(env, "RISK__MAX_DAILY_LOSS", risk_defaults.max_daily_loss),
        max_drawdown = envget(env, "RISK__MAX_DRAWDOWN", risk_defaults.max_drawdown),
        risk_budget_per_trade = envget(env, "RISK__RISK_BUDGET_PER_TRADE", risk_defaults.risk_budget_per_trade),
        min_probability_positive = envget(env, "RISK__MIN_PROBABILITY_POSITIVE", risk_defaults.min_probability_positive),
        max_probability_large_loss = envget(env, "RISK__MAX_PROBABILITY_LARGE_LOSS", risk_defaults.max_probability_large_loss),
        min_daily_turnover = envget(env, "RISK__MIN_DAILY_TURNOVER", risk_defaults.min_daily_turnover),
    )
    broker = BrokerSettings(
        kite_api_key = envsecret(env, "BROKER__KITE_API_KEY"),
        kite_api_secret = envsecret(env, "BROKER__KITE_API_SECRET"),
        kite_access_token = envsecret(env, "BROKER__KITE_ACCESS_TOKEN"),
        order_timeout_seconds = envget(env, "BROKER__ORDER_TIMEOUT_SECONDS", 10.0),
        max_retries = envget(env, "BROKER__MAX_RETRIES", 3),
    )
    data = DataSettings(
        database_url = envsecret(env, "DATA__DATABASE_URL"),
        redis_url = envsecret(env, "DATA__REDIS_URL"),
        cache_dir = envget(env, "DATA__CACHE_DIR", joinpath("data", "cache")),
        default_interval = envget(env, "DATA__DEFAULT_INTERVAL", "1d"),
    )
    execution = ExecutionSettings(
        slippage_bps = envget(env, "EXECUTION__SLIPPAGE_BPS", 5.0),
        brokerage_bps = envget(env, "EXECUTION__BROKERAGE_BPS", 3.0),
        taxes_bps = envget(env, "EXECUTION__TAXES_BPS", 12.0),
    )
    return Settings(
        trading_mode = envmode(env),
        allow_live_trading = envget(env, "ALLOW_LIVE_TRADING", false),
        starting_equity = envget(env, "STARTING_EQUITY", 1.0e6),
        base_currency = envget(env, "BASE_CURRENCY", "INR"),
        universe = envlist(env, "UNIVERSE"),
        seed = envget(env, "SEED", 7),
        log_level = envget(env, "LOG_LEVEL", "INFO"),
        risk = risk,
        broker = broker,
        data = data,
        execution = execution,
    )
end

const ENV_PREFIX = "BAYESTRADE_"

"""
    envget(env, name, default)

Read one setting, parsed to the type of `default`.

A present but unparsable value is an error rather than a fallback to the default. Silently
ignoring `MAX_DAILY_LOSS=2%` would run the system on limits nobody chose.
"""
function envget(env, name::AbstractString, default::T) where {T}
    raw = get(env, ENV_PREFIX * name, nothing)
    raw === nothing && return default
    parsed = tryparsevalue(T, raw)
    parsed === nothing && throw(
        ArgumentError("$(ENV_PREFIX * name)=$(raw) cannot be read as $T"),
    )
    return parsed
end

tryparsevalue(::Type{String}, raw::AbstractString) = String(raw)
tryparsevalue(::Type{Bool}, raw::AbstractString) =
    lowercase(strip(raw)) in ("1", "true", "yes", "on") ? true :
    lowercase(strip(raw)) in ("0", "false", "no", "off") ? false : nothing
tryparsevalue(::Type{T}, raw::AbstractString) where {T <: Number} = tryparse(T, strip(raw))

function envsecret(env, name::AbstractString)
    raw = get(env, ENV_PREFIX * name, nothing)
    (raw === nothing || isempty(strip(raw))) && return nothing
    return Secret(String(strip(raw)))
end

function envlist(env, name::AbstractString)
    raw = get(env, ENV_PREFIX * name, nothing)
    raw === nothing && return String[]
    return [String(strip(part)) for part in split(raw, ',') if !isempty(strip(part))]
end

function envmode(env)
    raw = get(env, ENV_PREFIX * "TRADING_MODE", nothing)
    raw === nothing && return PAPER
    wanted = lowercase(strip(raw))
    for mode in instances(TradingMode)
        slug(mode) == wanted && return mode
    end
    valid = join(slug.(collect(instances(TradingMode))), ", ")
    throw(ArgumentError("$(ENV_PREFIX)TRADING_MODE=$raw is not one of: $valid"))
end

"""
    describe(settings)

A redacted, human-readable summary. Never contains a secret value.
"""
function describe(settings::Settings)
    risk = settings.risk
    return [
        "trading_mode" => slug(settings.trading_mode),
        "live_armed" => string(settings.allow_live_trading),
        "starting_equity" => @sprintf("%.2f %s", settings.starting_equity, settings.base_currency),
        "universe" => isempty(settings.universe) ? "(unset)" : join(settings.universe, ", "),
        "seed" => string(settings.seed),
        "risk_limits_version" => string(risk.version),
        "max_position_weight" => @sprintf("%.2f%%", 100 * risk.max_position_weight),
        "max_portfolio_exposure" => @sprintf("%.2f%%", 100 * risk.max_portfolio_exposure),
        "max_sector_exposure" => @sprintf("%.2f%%", 100 * risk.max_sector_exposure),
        "max_daily_loss" => @sprintf("%.2f%%", 100 * risk.max_daily_loss),
        "max_drawdown" => @sprintf("%.2f%%", 100 * risk.max_drawdown),
        "max_open_positions" => string(risk.max_open_positions),
        "min_probability_positive" => @sprintf("%.2f%%", 100 * risk.min_probability_positive),
        "max_probability_large_loss" => @sprintf("%.2f%%", 100 * risk.max_probability_large_loss),
        "round_trip_cost" => @sprintf("%.4f%%", 100 * round_trip_cost(settings.execution)),
        "broker_credentials" => has_credentials(settings.broker) ? "present" : "absent",
        "broker_session" => has_session(settings.broker) ? "present" : "absent",
        "database" => settings.data.database_url === nothing ? "unset" : "configured",
        "redis" => settings.data.redis_url === nothing ? "unset" : "configured",
    ]
end

"""
    doctor(settings; io = stdout)

Print the resolved configuration and state plainly whether live trading is armed.
"""
function doctor(settings::Settings = load_settings(); io::IO = stdout)
    entries = describe(settings)
    pad = maximum(length(first(entry)) for entry in entries)
    println(io, "BayesTrade configuration\n")
    for (name, value) in entries
        println(io, "  ", rpad(name, pad), "  ", value)
    end
    println(io)
    if settings.trading_mode === LIVE
        println(io, "  LIVE TRADING IS ARMED. Orders will reach a real broker.")
    elseif settings.allow_live_trading
        println(
            io,
            string(
                "  Live trading is permitted but not selected; orders are simulated. ",
                "Set BAYESTRADE_TRADING_MODE=live to arm it.",
            ),
        )
    else
        println(io, "  Orders are simulated. Live trading is not armed.")
    end
    return nothing
end
