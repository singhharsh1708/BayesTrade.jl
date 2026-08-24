"""
Talking to Kite.

Zerodha ships no Julia SDK, so this is written against the documented HTTP API. Two decisions
shape the file.

**The transport is injected.** Nothing here opens a socket. A `KiteSession` holds a function
that takes a request and returns a response, so the whole client is exercised against a stub in
the tests and the real one is wired in at the edge by whoever has credentials. That keeps an
HTTP library out of the package's dependencies and, more usefully, means every request this
code would send can be asserted on without a network.

**Live is not a flag on a paper object.** Reaching a real venue takes a different type, and
that type refuses to exist unless both independent keys agree, which is the guard
`config/settings.jl` already describes. One environment variable set by accident is not enough
to send an order.
"""

const KITE_API_ROOT = "https://api.kite.trade"
const KITE_LOGIN_ROOT = "https://kite.zerodha.com/connect/login"

"""
    KiteRequest

One call, as it would be sent.
"""
struct KiteRequest
    method::Symbol
    path::String
    query::Dict{String, String}
    body::Dict{String, String}
    headers::Dict{String, String}
end

"""
    KiteResponse

One call, as it came back.
"""
struct KiteResponse
    status::Int
    body::String
end

"""
    KiteError

Raised when Kite refuses a call, or answers with something this client cannot read.
"""
struct KiteError <: Exception
    status::Int
    message::String
end

Base.showerror(io::IO, error::KiteError) =
    print(io, "KiteError(", error.status, "): ", error.message)

"""
    KiteCredentials

What identifies the account.

The secret is carried but never printed: `show` redacts it, because a credential that reaches a
log has already leaked and no amount of care afterwards puts it back.
"""
struct KiteCredentials
    api_key::String
    api_secret::String
    user_id::String

    function KiteCredentials(; api_key::AbstractString, api_secret::AbstractString, user_id::AbstractString = "")
        isempty(api_key) && throw(ArgumentError("api_key is required"))
        isempty(api_secret) && throw(ArgumentError("api_secret is required"))
        return new(String(api_key), String(api_secret), String(user_id))
    end
end

Base.show(io::IO, credentials::KiteCredentials) =
    print(io, "KiteCredentials(api_key=", credentials.api_key, ", api_secret=<redacted>)")

"""
    credentials_from_env()

Read credentials from the environment, or `nothing` when they are not set.

Absence is a normal state. Every test in this package runs without credentials, and so does
every backtest.
"""
function credentials_from_env()
    key = get(ENV, "KITE_API_KEY", "")
    secret = get(ENV, "KITE_API_SECRET", "")
    (isempty(key) || isempty(secret)) && return nothing
    return KiteCredentials(
        api_key = key, api_secret = secret, user_id = get(ENV, "KITE_USER_ID", ""),
    )
end

"""
    login_url(credentials)

Where a human goes to authorise this application.

The flow needs a browser and a person, which is deliberate on Zerodha's part: an access token
cannot be minted from the secret alone.
"""
login_url(credentials::KiteCredentials) =
    string(KITE_LOGIN_ROOT, "?v=3&api_key=", credentials.api_key)

"""
    session_checksum(credentials, request_token)

`sha256(api_key + request_token + api_secret)`, which is what Kite checks in exchange for an
access token.
"""
session_checksum(credentials::KiteCredentials, request_token::AbstractString) =
    bytes2hex(sha256(string(credentials.api_key, request_token, credentials.api_secret)))

"""
    KiteSession

An authenticated client, or one waiting to be.

`transport` is any callable taking a [`KiteRequest`](@ref) and returning a
[`KiteResponse`](@ref).
"""
mutable struct KiteSession{T}
    credentials::KiteCredentials
    transport::T
    access_token::Union{String, Nothing}
    mode::TradingMode

    function KiteSession(
            credentials::KiteCredentials, transport::T;
            access_token::Union{AbstractString, Nothing} = nothing,
            mode::TradingMode = PAPER,
        ) where {T}
        if mode === LIVE
            # The same two-key rule the settings enforce, restated at the only object that
            # can actually reach a venue. A single variable set by accident is not consent.
            get(ENV, "BAYESTRADE_ALLOW_LIVE_TRADING", "false") == "true" || throw(
                ArgumentError(
                    "a live Kite session needs BAYESTRADE_ALLOW_LIVE_TRADING=true; " *
                        "one variable is not enough to send an order",
                ),
            )
        end
        return new{T}(
            credentials, transport,
            access_token === nothing ? nothing : String(access_token), mode,
        )
    end
end

is_authenticated(session::KiteSession) = session.access_token !== nothing
broker_mode(session::KiteSession) = session.mode

"""
    authorisation(session)

The `Authorization` header Kite expects.
"""
function authorisation(session::KiteSession)
    token = session.access_token
    token === nothing && throw(KiteError(401, "no access token; call authenticate! first"))
    return string("token ", session.credentials.api_key, ":", token)
end

"""
    build_request(session, method, path; query, body, authenticated)

Assemble a request without sending it.

Separated from sending on purpose: this is the part with the mistakes in it, and it is the part
that can be asserted on with no network and no account.
"""
function build_request(
        session::KiteSession, method::Symbol, path::AbstractString;
        query::Dict{String, String} = Dict{String, String}(),
        body::Dict{String, String} = Dict{String, String}(),
        authenticated::Bool = true,
    )
    headers = Dict{String, String}("X-Kite-Version" => "3")
    authenticated && (headers["Authorization"] = authorisation(session))
    return KiteRequest(method, String(path), query, body, headers)
end

"""
    kite_call(session, request)

Send a request and read the reply, raising when Kite says no.
"""
function kite_call(session::KiteSession, request::KiteRequest)
    response = session.transport(request)
    response isa KiteResponse ||
        throw(KiteError(0, "transport returned a $(typeof(response)), not a KiteResponse"))

    # Read lazily rather than into a typed dictionary: the typed read builds its container
    # through a path that does not infer. Any JSON value is a valid document, including a
    # bare number, so the top level is narrowed here rather than left to fail deeper down.
    parsed = try
        JSON3.read(response.body)
    catch error
        error isa InterruptException && rethrow()
        throw(KiteError(response.status, string("unreadable reply: ", response.body)))
    end
    parsed isa AbstractDict || throw(
        KiteError(
            response.status,
            string("the reply is a ", typeof(parsed), ", not an object: ", response.body),
        ),
    )

    if response.status >= 400
        throw(
            KiteError(
                response.status,
                string(get(parsed, "message", "no message"), " [", get(parsed, "error_type", "unknown"), "]"),
            ),
        )
    end
    get(parsed, "status", "") == "success" ||
        throw(KiteError(response.status, string("status was ", get(parsed, "status", "absent"))))

    # `data` is narrowed here rather than at every call site. Kite sends an object; a reply
    # whose data is a number or a list is one this client cannot read, and saying so once is
    # better than each caller discovering it separately.
    data = get(parsed, "data", nothing)
    data === nothing && return Dict{String, Any}()
    data isa AbstractDict || throw(
        KiteError(response.status, string("data is a ", typeof(data), ", not an object")),
    )
    return data
end

"""
    authenticate!(session, request_token)

Exchange the request token a human just authorised for an access token.
"""
function authenticate!(session::KiteSession, request_token::AbstractString)
    data = kite_call(
        session,
        build_request(
            session, :POST, "/session/token";
            body = Dict{String, String}(
                "api_key" => session.credentials.api_key,
                "request_token" => String(request_token),
                "checksum" => session_checksum(session.credentials, request_token),
            ),
            authenticated = false,
        ),
    )
    token = get(data, "access_token", nothing)
    token isa AbstractString ||
        throw(KiteError(200, "the reply carried no access_token"))
    session.access_token = String(token)
    return session
end

"""
    kite_quote(session, exchange, symbol)

The last traded price for one instrument, over REST rather than the socket.
"""
function kite_quote(session::KiteSession, exchange::AbstractString, symbol::AbstractString, at::DateTime)
    key = string(exchange, ":", symbol)
    data = kite_call(
        session,
        build_request(session, :GET, "/quote/ltp"; query = Dict{String, String}("i" => key)),
    )
    entry = get(data, key, nothing)
    entry isa AbstractDict ||
        throw(KiteError(200, string("no quote for ", key, " in the reply")))
    price = get(entry, "last_price", nothing)
    price isa Real ||
        throw(KiteError(200, string("no last_price for ", key)))
    return Quote(symbol, at, Float64(price))
end

"""
    order_params(order, product, variety)

An order as the form fields Kite expects.

Quantity is sent as an integer count of shares, so it is rounded here and the rounding is
visible rather than happening inside a formatter.
"""
function order_params(
        order::Order; product::AbstractString = "CNC", exchange::AbstractString = "NSE",
    )
    quantity = round(Int, order.quantity)
    quantity > 0 || throw(ArgumentError("an order rounding to zero shares cannot be sent"))
    params = Dict{String, String}(
        "tradingsymbol" => order.symbol,
        "exchange" => String(exchange),
        "transaction_type" => order.side === BUY_SIDE ? "BUY" : "SELL",
        "quantity" => string(quantity),
        "product" => String(product),
        "order_type" => order.order_type === LIMIT ? "LIMIT" : "MARKET",
        "validity" => "DAY",
    )
    if order.order_type === LIMIT
        limit = order.limit_price
        limit === nothing && throw(ArgumentError("a limit order needs a limit price"))
        params["price"] = string(round(limit; digits = 2))
    end
    return params
end

Base.show(io::IO, session::KiteSession) = print(
    io, "<KiteSession ", slug(session.mode), " ",
    is_authenticated(session) ? "authenticated" : "not authenticated", ">",
)
