"""
Kite's binary tick protocol.

The feed does not send JSON. It sends packed big-endian binary, and the whole reason this file
exists is that a misread offset produces a *plausible* number rather than an error: a price
read four bytes late is still a price, and it will be wrong in a way nothing downstream can
detect. Every field is therefore read by name from a documented offset, and the tests assert
against frames whose bytes were laid out by hand.

Wire format:

    [2 bytes] number of packets
    for each packet:
        [2 bytes] packet length
        [n bytes] packet

Packet length is what identifies the mode, since the payload carries no tag:

| bytes | mode |
| --- | --- |
| 8 | last price only |
| 28 | index quote |
| 32 | index full |
| 44 | quote |
| 184 | full, with market depth |

Prices arrive as integers in the instrument's minor unit. Equities are in paise, so a hundredth
of a rupee, and currency derivatives are not, which is the kind of detail that silently
multiplies a position by ten thousand if it is guessed rather than looked up.
"""

const KITE_LTP_BYTES = 8
const KITE_INDEX_QUOTE_BYTES = 28
const KITE_INDEX_FULL_BYTES = 32
const KITE_QUOTE_BYTES = 44
const KITE_FULL_BYTES = 184
const KITE_DEPTH_ENTRY_BYTES = 12

"""
    KiteSegment

Which exchange segment an instrument trades in, taken from the low byte of its token.

The segment decides the price divisor, and getting it wrong is not a rounding error. A currency
future read at the equity divisor is out by a factor of a hundred thousand.
"""
@enum KiteSegment NSE_CM BSE_CM NSE_FO CDS BSE_CDS MCX_FO OTHER_SEGMENT

"""
    segment_of(token)

The segment encoded in an instrument token.
"""
function segment_of(token::Integer)
    code = token & 0xFF
    code == 1 && return NSE_CM
    code == 2 && return NSE_FO
    code == 3 && return CDS
    code == 4 && return BSE_CM
    code == 6 && return BSE_CDS
    code == 5 && return MCX_FO
    return OTHER_SEGMENT
end

"""
    price_divisor(segment)

What to divide a wire price by to get the instrument's own units.
"""
function price_divisor(segment::KiteSegment)
    segment === CDS && return 10_000_000.0
    segment === BSE_CDS && return 10_000.0
    return 100.0
end

"""
    DepthEntry

One rung of the order book.
"""
struct DepthEntry
    quantity::Int
    price::Float64
    orders::Int
end

"""
    KiteTick

One instrument's state, as far as this packet described it.

Everything beyond the last price is `nothing` when the mode did not carry it. A zero would be
indistinguishable from a genuine zero, and a genuine zero volume is a real thing on a thin
instrument.
"""
struct KiteTick
    token::Int
    segment::KiteSegment
    last_price::Float64
    last_quantity::Union{Int, Nothing}
    average_price::Union{Float64, Nothing}
    volume::Union{Int, Nothing}
    buy_quantity::Union{Int, Nothing}
    sell_quantity::Union{Int, Nothing}
    open::Union{Float64, Nothing}
    high::Union{Float64, Nothing}
    low::Union{Float64, Nothing}
    close::Union{Float64, Nothing}
    exchange_timestamp::Union{DateTime, Nothing}
    open_interest::Union{Int, Nothing}
    bids::Vector{DepthEntry}
    asks::Vector{DepthEntry}
    mode::Symbol
end

"""
    KiteProtocolError

Raised when a frame cannot be read as the protocol describes.
"""
struct KiteProtocolError <: Exception
    message::String
end

Base.showerror(io::IO, error::KiteProtocolError) =
    print(io, "KiteProtocolError: ", error.message)

be_int32(bytes::AbstractVector{UInt8}, offset::Int) =
    Int(reinterpret(Int32, [bytes[offset + 4], bytes[offset + 3], bytes[offset + 2], bytes[offset + 1]])[1])

be_uint32(bytes::AbstractVector{UInt8}, offset::Int) =
    Int(
    UInt32(bytes[offset + 1]) << 24 | UInt32(bytes[offset + 2]) << 16 |
        UInt32(bytes[offset + 3]) << 8 | UInt32(bytes[offset + 4])
)

be_uint16(bytes::AbstractVector{UInt8}, offset::Int) =
    Int(UInt16(bytes[offset + 1]) << 8 | UInt16(bytes[offset + 2]))

"""
    parse_depth(bytes, offset, divisor)

Ten rungs: five bids then five asks, each twelve bytes.
"""
function parse_depth(bytes::AbstractVector{UInt8}, offset::Int, divisor::Float64)
    bids = DepthEntry[]
    asks = DepthEntry[]
    for index in 0:9
        base = offset + index * KITE_DEPTH_ENTRY_BYTES
        entry = DepthEntry(
            be_uint32(bytes, base),
            be_uint32(bytes, base + 4) / divisor,
            be_uint16(bytes, base + 8),
        )
        index < 5 ? push!(bids, entry) : push!(asks, entry)
    end
    return bids, asks
end

"""
    parse_packet(bytes)

One packet into a [`KiteTick`](@ref).

The length is the only thing that says which mode this is, so an unrecognised length is an
error rather than a best guess. Reading a 44-byte quote as though it were a 184-byte full
packet would produce depth rungs out of whatever followed it in the buffer.
"""
function parse_packet(bytes::AbstractVector{UInt8})
    n = length(bytes)
    n >= KITE_LTP_BYTES || throw(
        KiteProtocolError(string("a packet of ", n, " bytes is shorter than any mode")),
    )

    token = be_uint32(bytes, 0)
    segment = segment_of(token)
    divisor = price_divisor(segment)
    price(offset) = be_int32(bytes, offset) / divisor

    n == KITE_LTP_BYTES && return KiteTick(
        token, segment, price(4), nothing, nothing, nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing,
        DepthEntry[], DepthEntry[], :ltp,
    )

    if n == KITE_INDEX_QUOTE_BYTES || n == KITE_INDEX_FULL_BYTES
        stamp = n == KITE_INDEX_FULL_BYTES ?
            unix2datetime(be_uint32(bytes, 28)) : nothing
        return KiteTick(
            token, segment, price(4), nothing, nothing, nothing, nothing, nothing,
            price(8), price(12), price(16), price(20), stamp, nothing,
            DepthEntry[], DepthEntry[], n == KITE_INDEX_FULL_BYTES ? :index_full : :index_quote,
        )
    end

    (n == KITE_QUOTE_BYTES || n == KITE_FULL_BYTES) || throw(
        KiteProtocolError(string("no mode has a packet length of ", n, " bytes")),
    )

    full = n == KITE_FULL_BYTES
    bids, asks = full ? parse_depth(bytes, 64, divisor) : (DepthEntry[], DepthEntry[])
    return KiteTick(
        token, segment, price(4),
        be_uint32(bytes, 8), price(12), be_uint32(bytes, 16),
        be_uint32(bytes, 20), be_uint32(bytes, 24),
        price(28), price(32), price(36), price(40),
        full ? unix2datetime(be_uint32(bytes, 60)) : nothing,
        full ? be_uint32(bytes, 48) : nothing,
        bids, asks, full ? :full : :quote,
    )
end

"""
    parse_frame(bytes)

A whole websocket frame into its ticks.

An empty frame is a heartbeat, which is a normal thing to receive and not an error. A frame
whose declared lengths run past its end is truncated, and that *is* an error: reading it as far
as it goes would hand the system a tick assembled partly from whatever followed in memory.
"""
function parse_frame(bytes::AbstractVector{UInt8})
    length(bytes) < 2 && return KiteTick[]
    count = be_uint16(bytes, 0)
    count == 0 && return KiteTick[]

    ticks = Vector{KiteTick}(undef, count)
    offset = 2
    for index in 1:count
        offset + 2 <= length(bytes) || throw(
            KiteProtocolError(
                string("frame ends before the length of packet ", index),
            ),
        )
        size = be_uint16(bytes, offset)
        offset += 2
        offset + size <= length(bytes) || throw(
            KiteProtocolError(
                string(
                    "packet ", index, " declares ", size, " bytes but only ",
                    length(bytes) - offset, " remain",
                ),
            ),
        )
        ticks[index] = parse_packet(view(bytes, (offset + 1):(offset + size)))
        offset += size
    end
    return ticks
end

"""
    to_quote(tick, symbol)

A [`Quote`](@ref) for the rest of the system, from a tick that carried enough to make one.
"""
function to_quote(tick::KiteTick, symbol::AbstractString, at::DateTime)
    best_bid = isempty(tick.bids) ? nothing : first(tick.bids).price
    best_ask = isempty(tick.asks) ? nothing : first(tick.asks).price
    return Quote(
        symbol, tick.exchange_timestamp === nothing ? at : tick.exchange_timestamp,
        tick.last_price;
        bid = best_bid, ask = best_ask,
        bid_quantity = isempty(tick.bids) ? 0 : first(tick.bids).quantity,
        ask_quantity = isempty(tick.asks) ? 0 : first(tick.asks).quantity,
        volume = tick.volume === nothing ? 0.0 : Float64(tick.volume),
    )
end

Base.show(io::IO, tick::KiteTick) = @printf(
    io, "<KiteTick %d %s %.2f%s>", tick.token, String(tick.mode), tick.last_price,
    tick.volume === nothing ? "" : @sprintf(" vol=%d", tick.volume)
)
