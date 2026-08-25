"""
Talking to Groww, for history only.

Groww publishes a free HTTP API. What this package wants from it is the historical candle
endpoint and nothing else, so that is all this client can reach.

**There is no order path here, and it is not a flag.** `build_request` refuses any path
outside [`GROWW_READ_PATHS`](@ref), so a mistake, a refactor or a copied snippet cannot turn
this into something that places an order. The same reasoning as `execution/broker.jl`, where
`PaperBroker` is the only subtype: a capability that does not exist is a stronger guarantee
than one that is switched off.

**The transport is injected**, as with Kite. Nothing here opens a socket. A `GrowwSession`
holds a function taking a request and returning a response, which keeps an HTTP library out of
the dependency list and, more usefully, lets every request this code would send be asserted on
without a network or an account.

**Credentials are read from the environment at run time.** Never a file in the repository,
never an argument in a saved script. A secret that reaches a commit has leaked, and rotating it
afterwards is the only remedy.
"""

const GROWW_API_ROOT = "https://api.groww.in/v1"

"""
    GROWW_READ_PATHS

Every path this client is permitted to build.

Two entries: mint a token, and read candles. Order placement, order modification, position and
holding endpoints all exist in Groww's API and none of them are here. The whitelist is checked
in [`build_request`](@ref), so the restriction is enforced rather than documented.
"""
const GROWW_READ_PATHS = Set{String}(["/token/api/access", "/historical/candles"])

"""
    GrowwRequest

One call, as it would be sent.
"""
struct GrowwRequest
    method::Symbol
    path::String
    query::Dict{String, String}
    body::Dict{String, Any}
    headers::Dict{String, String}
end

"""
    GrowwResponse

One call, as it came back.
"""
struct GrowwResponse
    status::Int
    body::String
end

"""
    GrowwError

Groww refused a call, or answered with something this client cannot read.

`code` is Groww's own error code when it sent one, and empty otherwise.
"""
struct GrowwError <: Exception
    status::Int
    code::String
    message::String
end

GrowwError(status::Integer, message::AbstractString) =
    GrowwError(Int(status), "", String(message))

Base.showerror(io::IO, error::GrowwError) = print(
    io, "GrowwError(", error.status,
    isempty(error.code) ? "" : string(", ", error.code), "): ", error.message,
)

"""
    GrowwCredentials

What identifies the account.

The secret is carried but never printed: `show` redacts it, because a credential that reaches
a log has already leaked and no amount of care afterwards puts it back.

Groww offers two ways to mint a token. An *approval* key is paired with a secret and signed
with a checksum; a *TOTP* key is paired with a rotating code and has no stored secret at all.
Both are supported, and a credential with no secret is a normal object rather than an invalid
one.
"""
struct GrowwCredentials
    api_key::String
    api_secret::String

    function GrowwCredentials(; api_key::AbstractString, api_secret::AbstractString = "")
        isempty(api_key) && throw(ArgumentError("api_key is required"))
        return new(String(api_key), String(api_secret))
    end
end

has_secret(credentials::GrowwCredentials) = !isempty(credentials.api_secret)

Base.show(io::IO, credentials::GrowwCredentials) = print(
    io, "GrowwCredentials(api_key=<redacted>, api_secret=",
    has_secret(credentials) ? "<redacted>" : "absent", ")",
)

"""
    groww_credentials_from_env()

Read credentials from the environment, or `nothing` when they are not set.

Absence is a normal state. Every test in this package runs without credentials, and so does
every backtest against a stored or synthetic series.
"""
function groww_credentials_from_env()
    key = get(ENV, "GROWW_API_KEY", "")
    isempty(key) && return nothing
    return GrowwCredentials(api_key = key, api_secret = get(ENV, "GROWW_API_SECRET", ""))
end

"""
    access_checksum(credentials, timestamp)

`sha256(api_secret + timestamp)`, which is what Groww checks in exchange for an access token.

The timestamp is passed in rather than read from the clock, so the signature this client would
send is reproducible in a test.
"""
access_checksum(credentials::GrowwCredentials, timestamp::Integer) =
    bytes2hex(sha256(string(credentials.api_secret, timestamp)))

"""
    GrowwSession

An authenticated client, or one waiting to be.

`transport` is any callable taking a [`GrowwRequest`](@ref) and returning a
[`GrowwResponse`](@ref).

The token Groww issues expires daily, so `expiry` is kept beside it. A client that does not
know when its token dies discovers the fact halfway through a fetch, as an authentication
failure that looks like a bug.
"""
mutable struct GrowwSession{T}
    credentials::GrowwCredentials
    transport::T
    access_token::Union{String, Nothing}
    expiry::Union{DateTime, Nothing}

    function GrowwSession(
            credentials::GrowwCredentials, transport::T;
            access_token::Union{AbstractString, Nothing} = nothing,
            expiry::Union{DateTime, Nothing} = nothing,
        ) where {T}
        return new{T}(
            credentials, transport,
            access_token === nothing ? nothing : String(access_token), expiry,
        )
    end
end

is_authenticated(session::GrowwSession) = session.access_token !== nothing

Base.show(io::IO, session::GrowwSession) = print(
    io, "GrowwSession(", is_authenticated(session) ? "authenticated" : "no token",
    session.expiry === nothing ? "" : string(", expires ", session.expiry), ")",
)

"""
    token_expired(session, moment)

Whether the token is known to be dead at this moment.

An unknown expiry is not treated as expired. Refusing to send a request because the expiry was
never recorded would fail a session that is perfectly usable.
"""
function token_expired(session::GrowwSession, moment::DateTime)
    expiry = session.expiry
    expiry === nothing && return false
    return moment >= expiry
end

"""
    build_request(session, method, path; query, body, authenticated)

Assemble a request without sending it.

Separated from sending on purpose: this is the part with the mistakes in it, and it is the part
that can be asserted on with no network and no account.

A path outside [`GROWW_READ_PATHS`](@ref) is refused here. That is the whole reason this
function exists rather than the caller building a request directly.
"""
function build_request(
        session::GrowwSession, method::Symbol, path::AbstractString;
        query::Dict{String, String} = Dict{String, String}(),
        body::Dict{String, Any} = Dict{String, Any}(),
        authenticated::Bool = true,
    )
    String(path) in GROWW_READ_PATHS || throw(
        ArgumentError(
            string(
                "\"", path, "\" is not a readable path; this client reaches ",
                join(sort(collect(GROWW_READ_PATHS)), " and "),
                " and nothing else, so it cannot place an order",
            ),
        ),
    )

    headers = Dict{String, String}(
        "Accept" => "application/json",
        "Content-Type" => "application/json",
        "x-api-version" => "1.0",
    )
    if authenticated
        token = session.access_token
        token === nothing &&
            throw(GrowwError(401, "no access token; call authenticate! first"))
        headers["Authorization"] = string("Bearer ", token)
    else
        # Minting a token authenticates with the API key itself, not with a token.
        headers["Authorization"] = string("Bearer ", session.credentials.api_key)
    end
    return GrowwRequest(method, String(path), query, body, headers)
end

"""
    groww_call(session, request)

Send a request and read the reply, raising when Groww says no.

Groww wraps a success as `{"status": "SUCCESS", "payload": {...}}` and a failure as
`{"status": "FAILURE", "error": {"code", "message"}}`, and the token endpoint answers with a
bare object carrying `token`. All three shapes are handled here so no caller has to.
"""
function groww_call(session::GrowwSession, request::GrowwRequest)
    response = session.transport(request)
    response isa GrowwResponse || throw(
        GrowwError(0, "transport returned a $(typeof(response)), not a GrowwResponse"),
    )

    # Read lazily rather than into a typed dictionary: the typed read builds its container
    # through a path that does not infer. Any JSON value is a valid document, including a
    # bare number, so the top level is narrowed here rather than left to fail deeper down.
    parsed = try
        JSON3.read(response.body)
    catch error
        error isa InterruptException && rethrow()
        throw(GrowwError(response.status, string("unreadable reply: ", response.body)))
    end
    parsed isa AbstractDict || throw(
        GrowwError(
            response.status,
            string("the reply is a ", typeof(parsed), ", not an object: ", response.body),
        ),
    )

    # The envelope is checked before the status code. Groww answers some refusals with 200 and
    # a FAILURE body, and reading the code alone would take one of those for a success.
    if get(parsed, "status", "") == "FAILURE"
        detail = get(parsed, "error", nothing)
        code, message = if detail isa AbstractDict
            String(string(get(detail, "code", ""))),
                String(string(get(detail, "message", "no message")))
        else
            "", string(detail)
        end
        throw(GrowwError(response.status, code, message))
    end
    response.status < 400 || throw(
        GrowwError(response.status, string("the request failed: ", response.body)),
    )

    payload = get(parsed, "payload", nothing)
    payload === nothing && return parsed        # the token endpoint answers unwrapped
    payload isa AbstractDict || throw(
        GrowwError(response.status, string("payload is a ", typeof(payload), ", not an object")),
    )
    return payload
end

"""
    authenticate!(session; timestamp, totp)

Exchange the API key for an access token.

With a secret, the key is an *approval* key and the request carries `sha256(secret + timestamp)`.
With a `totp`, the key is a TOTP key and the request carries the current code instead. Sending
both is refused rather than resolved: guessing which one the caller meant is how a request goes
out signed the wrong way and fails with an error that says nothing useful.
"""
function authenticate!(
        session::GrowwSession;
        timestamp::Union{Integer, Nothing} = nothing,
        totp::Union{AbstractString, Nothing} = nothing,
    )
    credentials = session.credentials
    body = if totp !== nothing
        has_secret(credentials) && throw(
            ArgumentError(
                "these credentials carry a secret, so they are an approval key; " *
                    "pass a totp only with a TOTP key",
            ),
        )
        isempty(strip(String(totp))) && throw(ArgumentError("the totp is empty"))
        Dict{String, Any}("key_type" => "totp", "totp" => strip(String(totp)))
    else
        has_secret(credentials) || throw(
            ArgumentError(
                "no api_secret and no totp; set GROWW_API_SECRET or pass totp = ...",
            ),
        )
        stamp = timestamp === nothing ? round(Int, datetime2unix(now(UTC))) : Int(timestamp)
        Dict{String, Any}(
            "key_type" => "approval",
            "checksum" => access_checksum(credentials, stamp),
            "timestamp" => string(stamp),
        )
    end

    reply = groww_call(
        session,
        build_request(
            session, :POST, "/token/api/access"; body = body, authenticated = false,
        ),
    )
    token = get(reply, "token", nothing)
    token isa AbstractString || throw(GrowwError(200, "the reply carried no token"))
    session.access_token = String(token)

    expiry = get(reply, "expiry", nothing)
    session.expiry = expiry isa AbstractString ?
        tryparse(DateTime, replace(String(expiry), " " => "T")) : nothing
    return session
end
