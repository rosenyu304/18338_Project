###############################
#  β-Hermite Sturm Histogram  #
###############################

using Random
using PyPlot
using Distributions

# ------------------------------
# Sturm negative-eigenvalue count
# ------------------------------
function count_neg_eigs_ratio(a::Vector{Float64}, b::Vector{Float64}, λ::Float64)
    n = length(a)
    neg = 0
    r = a[1] - λ
    if r < 0
        neg += 1
    end
    tiny = 1e-300

    @inbounds for i in 2:n
        denom = abs(r) > tiny ? r : (r ≥ 0 ? tiny : -tiny)
        r = (a[i] - λ) - (b[i-1]^2) / denom
        if r < 0
            neg += 1
        end
    end

    return neg
end

# ------------------------------
# Histogram via Sturm sequence
# ------------------------------
sturm_histogram(a, b, bin_edges) =
    diff([count_neg_eigs_ratio(a, b, λ) for λ in bin_edges])

# ------------------------------
# β-Hermite scaled tridiagonal sampler
# ------------------------------
function sample_beta_hermite_scaled(n::Int, β::Float64, rng::AbstractRNG)
    diag = randn(rng, n)
    k = (n-1):-1:1
    dof = β .* k
    off = sqrt.(rand.(Ref(rng), Distributions.Chisq.(dof))) ./ sqrt(2.0)
    s = 1.0 / sqrt(2.0 * n * β)
    return diag .* s, off .* s
end

# ------------------------------
# Main (Plotting)
# ------------------------------
n = 2000
β = 2.0
rng = MersenneTwister(0)

a, b = sample_beta_hermite_scaled(n, β, rng)
bin_edges = range(-1.25, 1.25; length=201)

hist = sturm_histogram(a, b, collect(bin_edges))
centers = (bin_edges[1:end-1] .+ bin_edges[2:end]) ./ 2
width = step(bin_edges)
density = hist ./ (sum(hist) * width)

# Semicircle law
R = 1.0
semicircle = (2/pi) .* sqrt.(max.(0.0, R^2 .- centers.^2)) ./ R^2

# Plot (exact matplotlib look)
figure(figsize=(6,4))
bar(centers, density; width=width, color="gray", edgecolor="black",
    alpha=0.7, label="Sturm histogram")
plot(centers, semicircle; lw=2, linestyle="--", label="Semicircle law", color="red")

title("β-Hermite Sturm Histogram (β=$β, n=$n)")
xlabel("Eigenvalue λ")
ylabel("Density (normalized)")
legend()

savefig("histogram_sturm_julia.png")


# using CSV
# using DataFrames

# # ------------------------------
# #  Timing experiment (sturm histogramming)
# # ------------------------------


# β = 2.0
# filename = "timing_results_sturm_julia.csv"

# df = DataFrame(n=Int[], bins_n=Int[], time_ms=Float64[])

# println("Running timing experiment...")

# for n in [100, 200, 400, 600, 800, 1000, 1500, 2000]
#     for bins_n in [20, 40, 60, 80, 100, 150, 200]
#         total_ms = 0.0

#         for trial in 1:20
#             rng = MersenneTwister(0)

#             t0 = time_ns()

#             a, b = sample_beta_hermite_scaled(n, β, rng)
#             bin_edges = range(-1.25, 1.25; length=bins_n+1)

#             hist = sturm_histogram(a, b, collect(bin_edges))

#             t1 = time_ns()
#             total_ms += (t1 - t0) / 1e6     # convert ns → ms
#         end

#         avg_ms = total_ms / 20
#         push!(df, (n=n, bins_n=bins_n, time_ms=avg_ms))

#         println("n=$n, bins=$bins_n → $(round(avg_ms, digits=3)) ms")
#     end
# end

# CSV.write(filename, df)
# println("Saved timing results to $filename")
