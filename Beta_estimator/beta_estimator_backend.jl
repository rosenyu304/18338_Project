# beta_backend.jl
using Optim, SpecialFunctions, Statistics

"""
    beta_mle(spacings)

Estimate Dyson–Mehta β via Wigner surmise:
P(s; β) = aβ * s^β * exp(-bβ * s^2)
"""
function beta_mle(spacings::Vector{Float64})
    s = spacings ./ mean(spacings)  # normalize ⟨s⟩ = 1
    loglike(β) = begin
        bβ = (gamma((β + 2)/2) / gamma((β + 1)/2))^2
        aβ = 2 * bβ^((β + 1)/2) / gamma((β + 1)/2)
        sum(log.(aβ .* s.^β .* exp.(-bβ .* s.^2)))
    end
    result = optimize(β -> -loglike(β), 0.1, 5.0)
    return Optim.minimizer(result)
end

"""
Read a plain-text file with numbers (commas/space/newlines OK),
print β to stdout.
"""
function compute_beta_file(path::AbstractString)
    raw = read(path, String)
    clean = replace(raw, ',' => ' ')
    toks = split(clean)
    if isempty(toks)
        error("No numbers parsed from input file.")
    end
    spacings = parse.(Float64, toks)
    β̂ = beta_mle(spacings)
    println(β̂)  # stdout
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 1
        error("Usage: julia beta_backend.jl <data_file>")
    end
    compute_beta_file(ARGS[1])
end
