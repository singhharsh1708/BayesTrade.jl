#!/usr/bin/env julia
#
# Fold a payload into the template to produce the page.
#
#   julia build.jl [payload.json] [index.html]
#
# The template and the data are kept apart on purpose. An 800 kB page with the
# numbers baked into it is not a thing anyone can edit; a template plus a
# payload is two things that are each editable on their own.

const HERE = @__DIR__
const PAYLOAD = length(ARGS) >= 1 ? ARGS[1] : joinpath(HERE, "payload.json")
const OUT = length(ARGS) >= 2 ? ARGS[2] : joinpath(HERE, "index.html")
const MARKER = "/*__DATA__*/null"

isfile(PAYLOAD) || error("no payload at $PAYLOAD; run generate.jl first")
template = read(joinpath(HERE, "template.html"), String)
occursin(MARKER, template) || error("template.html has no $MARKER to fill")

data = strip(read(PAYLOAD, String))
isempty(data) && error("$PAYLOAD is empty")
startswith(data, "{") || error("$PAYLOAD does not look like a JSON object")

write(OUT, replace(template, MARKER => data))
println("wrote ", OUT, "  ", filesize(OUT), " bytes")
