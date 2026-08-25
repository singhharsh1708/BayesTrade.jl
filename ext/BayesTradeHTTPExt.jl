"""
The network edge.

The only file in this package that opens a socket, and it is an extension rather than part of
the core on purpose. `BayesTrade` resolves, precompiles, backtests and paper trades with no HTTP
library present; loading one is a decision the caller makes when they actually want to reach a
vendor.

Nothing here interprets a reply. It turns a `GrowwRequest` into bytes on a wire and the answer
back into a `GrowwResponse`, and every judgement about what that response means stays in
`src/groww/session.jl` where it can be tested without a network.
"""
module BayesTradeHTTPExt

using BayesTrade
using Dates
using HTTP
using JSON3

import BayesTrade: GROWW_API_ROOT, GrowwRequest, GrowwResponse, GrowwSession,
    GrowwCredentials, GrowwSource, RetryingSource, MarketHours, IST_OFFSET,
    authenticate!, connect_groww, dashboard_page, groww_credentials_from_env,
    groww_transport, serve_dashboard

"""
    groww_transport(; timeout, root)

A callable that sends one request and returns one response.

Two settings are deliberate and both are about not hiding failures.

`status_exception = false`, because a 429 is data. Groww answers a rate limit with a status
code, `translate_error` sorts it into a transient failure, and `RetryingSource` backs off. Let
HTTP raise instead and the classification never happens.

`retry = false`, because HTTP.jl retrying underneath a retry wrapper spends the budget twice and
makes the rate limit worse at exactly the moment it should be easing off.

A socket that never opened is not a refusal by Groww, so a connection failure comes back as
status zero rather than as an exception. `translate_error` reads zero as transient, which is
what a refused connection is.
"""
function groww_transport(;
        timeout::Real = 30, root::AbstractString = GROWW_API_ROOT,
    )
    return function (request::GrowwRequest)
        url = string(root, request.path)
        headers = [String(name) => String(value) for (name, value) in request.headers]
        try
            response = if request.method === :GET
                HTTP.get(
                    url, headers; query = request.query, status_exception = false,
                    retry = false, readtimeout = timeout, connect_timeout = timeout,
                )
            else
                HTTP.request(
                    string(request.method), url, headers, JSON3.write(request.body);
                    status_exception = false, retry = false,
                    readtimeout = timeout, connect_timeout = timeout,
                )
            end
            return GrowwResponse(response.status, String(response.body))
        catch error
            error isa InterruptException && rethrow()
            return GrowwResponse(0, sprint(showerror, error))
        end
    end
    """
        serve_dashboard(producer; port, host, title, refresh_seconds, open_browser)

    Serve the dashboard on this machine, rebuilding it on every request.

    `producer` is called per request rather than once at startup, so a session that is still running
    shows what it is doing now. A `Dict` is accepted too and is served unchanged.

    **It binds to the loopback interface and there is no keyword to change that.** The page carries
    positions, limits and every decision the system took. Binding to `0.0.0.0` would put that on
    whatever network the machine is attached to, with no authentication, and there is no version of
    "just for a minute" that makes it a good idea.

    Returns the server. `close(server)` stops it.
    """
    function serve_dashboard(
            producer;
            port::Integer = 8787,
            title::AbstractString = "",
            refresh_seconds::Real = 0,
            open_browser::Bool = true,
        )
        payload_of() = producer isa AbstractDict ? producer : producer()

        server = HTTP.serve!("127.0.0.1", port; verbose = -1) do request
            path = HTTP.URI(request.target).path
            try
                if path == "/payload.json"
                    return HTTP.Response(
                        200, ["Content-Type" => "application/json"];
                        body = JSON3.write(payload_of()),
                    )
                elseif path == "/" || path == "/index.html"
                    page = dashboard_page(
                        payload_of(); title = title, refresh_seconds = refresh_seconds,
                    )
                    return HTTP.Response(
                        200, ["Content-Type" => "text/html; charset=utf-8"]; body = page,
                    )
                end
                return HTTP.Response(404, ["Content-Type" => "text/plain"]; body = "not here")
            catch error
                error isa InterruptException && rethrow()
                # A producer that raises is a bug worth reading, and a dead browser tab is a
                # worse way to report it than a page that says what went wrong.
                return HTTP.Response(
                    500, ["Content-Type" => "text/plain"];
                    body = sprint(showerror, error),
                )
            end
        end

        url = "http://127.0.0.1:$port/"
        @info "dashboard listening" url
        open_browser && open_in_browser(url)
        return server
    end

    """
        open_in_browser(url)

    Ask the desktop to open a URL, and shrug if it will not.

    Failing to open a browser is not a reason to take down a server that is running correctly.
    """
    function open_in_browser(url::AbstractString)
        command = if Sys.isapple()
            `open $url`
        elseif Sys.iswindows()
            `cmd /c start $url`
        else
            `xdg-open $url`
        end
        try
            run(command; wait = false)
        catch error
            error isa InterruptException && rethrow()
            @warn "could not open a browser; open it yourself" url
        end
        return nothing
    end

end

"""
    connect_groww(; symbols, exchange, segment, hours, totp, attempts, timeout, root, strict, pause_seconds)

Authenticate against Groww and hand back a bar source that is ready to fetch.

Credentials come from `GROWW_API_KEY` and `GROWW_API_SECRET` in the environment, and from
nowhere else. There is no keyword to pass a secret in, because an argument is a thing that ends
up in a script, a shell history and a stack trace. A TOTP code is accepted as an argument since
it is worthless thirty seconds later.

Returns a [`RetryingSource`](@ref) wrapping a [`GrowwSource`](@ref). The session is reachable as
`source.inner.session` for its token expiry.
"""
function connect_groww(;
        exchange::AbstractString = "NSE",
        segment::AbstractString = "CASH",
        symbols::Dict{String, String} = Dict{String, String}(),
        hours::MarketHours = MarketHours(),
        totp::Union{AbstractString, Nothing} = nothing,
        attempts::Integer = 3,
        timeout::Real = 30,
        root::AbstractString = GROWW_API_ROOT,
        strict::Bool = true,
        pause_seconds::Real = 0.2,
    )
    credentials = groww_credentials_from_env()
    credentials === nothing && throw(
        ArgumentError(
            "no GROWW_API_KEY in the environment; export it and GROWW_API_SECRET " *
                "(or pass totp = ... for a TOTP key) rather than writing them into a file",
        ),
    )

    session = GrowwSession(credentials, groww_transport(timeout = timeout, root = root))
    authenticate!(session; totp = totp)
    source = GrowwSource(
        session; exchange = exchange, segment = segment, symbols = symbols,
        hours = hours, strict = strict, pause_seconds = pause_seconds,
    )
    return RetryingSource(source; attempts = attempts)
    """
        serve_dashboard(producer; port, host, title, refresh_seconds, open_browser)

    Serve the dashboard on this machine, rebuilding it on every request.

    `producer` is called per request rather than once at startup, so a session that is still running
    shows what it is doing now. A `Dict` is accepted too and is served unchanged.

    **It binds to the loopback interface and there is no keyword to change that.** The page carries
    positions, limits and every decision the system took. Binding to `0.0.0.0` would put that on
    whatever network the machine is attached to, with no authentication, and there is no version of
    "just for a minute" that makes it a good idea.

    Returns the server. `close(server)` stops it.
    """
    function serve_dashboard(
            producer;
            port::Integer = 8787,
            title::AbstractString = "",
            refresh_seconds::Real = 0,
            open_browser::Bool = true,
        )
        payload_of() = producer isa AbstractDict ? producer : producer()

        server = HTTP.serve!("127.0.0.1", port; verbose = -1) do request
            path = HTTP.URI(request.target).path
            try
                if path == "/payload.json"
                    return HTTP.Response(
                        200, ["Content-Type" => "application/json"];
                        body = JSON3.write(payload_of()),
                    )
                elseif path == "/" || path == "/index.html"
                    page = dashboard_page(
                        payload_of(); title = title, refresh_seconds = refresh_seconds,
                    )
                    return HTTP.Response(
                        200, ["Content-Type" => "text/html; charset=utf-8"]; body = page,
                    )
                end
                return HTTP.Response(404, ["Content-Type" => "text/plain"]; body = "not here")
            catch error
                error isa InterruptException && rethrow()
                # A producer that raises is a bug worth reading, and a dead browser tab is a
                # worse way to report it than a page that says what went wrong.
                return HTTP.Response(
                    500, ["Content-Type" => "text/plain"];
                    body = sprint(showerror, error),
                )
            end
        end

        url = "http://127.0.0.1:$port/"
        @info "dashboard listening" url
        open_browser && open_in_browser(url)
        return server
    end

    """
        open_in_browser(url)

    Ask the desktop to open a URL, and shrug if it will not.

    Failing to open a browser is not a reason to take down a server that is running correctly.
    """
    function open_in_browser(url::AbstractString)
        command = if Sys.isapple()
            `open $url`
        elseif Sys.iswindows()
            `cmd /c start $url`
        else
            `xdg-open $url`
        end
        try
            run(command; wait = false)
        catch error
            error isa InterruptException && rethrow()
            @warn "could not open a browser; open it yourself" url
        end
        return nothing
    end

end

"""
    serve_dashboard(producer; port, host, title, refresh_seconds, open_browser)

Serve the dashboard on this machine, rebuilding it on every request.

`producer` is called per request rather than once at startup, so a session that is still running
shows what it is doing now. A `Dict` is accepted too and is served unchanged.

**It binds to the loopback interface and there is no keyword to change that.** The page carries
positions, limits and every decision the system took. Binding to `0.0.0.0` would put that on
whatever network the machine is attached to, with no authentication, and there is no version of
"just for a minute" that makes it a good idea.

Returns the server. `close(server)` stops it.
"""
function serve_dashboard(
        producer;
        port::Integer = 8787,
        title::AbstractString = "",
        refresh_seconds::Real = 0,
        open_browser::Bool = true,
    )
    payload_of() = producer isa AbstractDict ? producer : producer()

    server = HTTP.serve!("127.0.0.1", port; verbose = -1) do request
        path = HTTP.URI(request.target).path
        try
            if path == "/payload.json"
                return HTTP.Response(
                    200, ["Content-Type" => "application/json"];
                    body = JSON3.write(payload_of()),
                )
            elseif path == "/" || path == "/index.html"
                page = dashboard_page(
                    payload_of(); title = title, refresh_seconds = refresh_seconds,
                )
                return HTTP.Response(
                    200, ["Content-Type" => "text/html; charset=utf-8"]; body = page,
                )
            end
            return HTTP.Response(404, ["Content-Type" => "text/plain"]; body = "not here")
        catch error
            error isa InterruptException && rethrow()
            # A producer that raises is a bug worth reading, and a dead browser tab is a
            # worse way to report it than a page that says what went wrong.
            return HTTP.Response(
                500, ["Content-Type" => "text/plain"];
                body = sprint(showerror, error),
            )
        end
    end

    url = "http://127.0.0.1:$port/"
    @info "dashboard listening" url
    open_browser && open_in_browser(url)
    return server
end

"""
    open_in_browser(url)

Ask the desktop to open a URL, and shrug if it will not.

Failing to open a browser is not a reason to take down a server that is running correctly.
"""
function open_in_browser(url::AbstractString)
    command = if Sys.isapple()
        `open $url`
    elseif Sys.iswindows()
        `cmd /c start $url`
    else
        `xdg-open $url`
    end
    try
        run(command; wait = false)
    catch error
        error isa InterruptException && rethrow()
        @warn "could not open a browser; open it yourself" url
    end
    return nothing
end

end
