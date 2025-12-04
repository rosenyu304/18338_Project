###############################
#  β-Hermite Naive Histogram  #
###############################

using Random
using PyPlot
using Distributions
using LinearAlgebra   # for eigen(Hermitian(T))

# ------------------------------
# β-Hermite scaled tridiagonal sampler
# ------------------------------
function sample_beta_hermite_scaled(n::Int, β::Float64, rng::AbstractRNG)
    diag = randn(rng, n)
    k = (n-1):-1:1
    dof = β .* k
    off = sqrt.(rand.(Ref(rng), Chisq.(dof))) ./ sqrt(2.0)
    s = 1.0 / sqrt(2.0 * n * β)
    return diag .* s, off .* s
end

# ------------------------------
# Dense naive histogram
# ------------------------------
function naive_histogram(a, b, bin_edges)
    n = length(a)
    T = diagm(0 => a, 1 => b, -1 => b)        # dense tridiagonal
    eigs = eigen(Hermitian(T)).values        # identical to np.linalg.eigvalsh
    hist = zeros(Int, length(bin_edges)-1)

    # manual histogram (equivalent to numpy)
    @inbounds for λ in eigs
        bin = searchsortedlast(bin_edges, λ)
        if 1 ≤ bin ≤ length(hist)
            hist[bin] += 1
        end
    end
    return hist
end

# ------------------------------
# Main (Plotting)
# ------------------------------
# n = 2000
# β = 2.0
# rng = MersenneTwister(0)

# a, b = sample_beta_hermite_scaled(n, β, rng)
# bin_edges = range(-1.25, 1.25; length=201)

# hist = naive_histogram(a, b, bin_edges)
# centers = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2
# width = step(bin_edges)
# density = hist ./ (sum(hist) * width)

# # Semicircle law
# R = 1.0
# semicircle = (2/pi) .* sqrt.(max.(0.0, R^2 .- centers.^2)) ./ R^2

# # Plot
# figure(figsize=(6,4))
# bar(centers, density; width=width, color="gray", edgecolor="black",
#     alpha=0.7, label="Naive histogram")
# plot(centers, semicircle; lw=2, linestyle="--", label="Semicircle law")

# title("β-Hermite Naive Histogram (β=$β, n=$n)")
# xlabel("Eigenvalue λ")
# ylabel("Density (normalized)")
# legend()

# savefig("histogram_naive_julia.png")


using CSV
using DataFrames

# ------------------------------
#  Timing experiment (naive dense eigen-decomposition)
# ------------------------------


β = 2.0
filename = "timing_results_naive_julia.csv"

df = DataFrame(n=Int[], bins_n=Int[], time_ms=Float64[])

println("Running timing experiment...")

for n in [100, 200, 400, 600, 800, 1000, 1500, 2000]
    for bins_n in [20, 40, 60, 80, 100, 150, 200]
        total_ms = 0.0

        for trial in 1:20
            rng = MersenneTwister(0)

            t0 = time_ns()

            a, b = sample_beta_hermite_scaled(n, β, rng)
            bin_edges = range(-1.25, 1.25; length=201)

            hist = naive_histogram(a, b, bin_edges)

            t1 = time_ns()
            total_ms += (t1 - t0) / 1e6     # convert ns → ms
        end

        avg_ms = total_ms / 20
        push!(df, (n=n, bins_n=bins_n, time_ms=avg_ms))

        println("n=$n, bins=$bins_n → $(round(avg_ms, digits=3)) ms")
    end
end

CSV.write(filename, df)
println("Saved timing results to $filename")
