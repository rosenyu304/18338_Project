using LinearAlgebra
using Distributions
using StatsBase
using PyPlot
using CSV
using DataFrames

########################################################
# β-Hermite sampler (Dumitriu–Edelman, scaled to [-1,1])
########################################################

function sample_beta_hermite_tridiagonal(n::Int, β::Float64)
    a = randn(n)
    b = zeros(n-1)
    for i in 1:n-1
        dof = β * (n - i)
        b[i] = sqrt(rand(Chisq(dof))) / sqrt(2.0)
    end
    scale = 1.0 / sqrt(2n*β)
    return a .* scale, b .* scale
end

function sample_eigs(n::Int, β::Float64)
    a, b = sample_beta_hermite_tridiagonal(n, β)
    return sort(eigen(SymTridiagonal(a, b)).values)
end

############################
# Utilities
############################

function trapz(x::AbstractVector, y::AbstractVector)
    s = 0.0
    @inbounds for i in 1:length(x)-1
        s += 0.5*(y[i] + y[i+1])*(x[i+1] - x[i])
    end
    return s
end

###############################################################
# Global repulsion: ∫ (1 - g(r)) dr
###############################################################

global_repulsion_β(β; kwargs...) = begin
    r, g = pair_correlation_β(β; kwargs...)
    trapz(r, 1 .- g)
end

global_repulsion_poisson(; kwargs...) = begin
    r, g = pair_correlation_poisson(; kwargs...)
    trapz(r, 1 .- g)
end



###############################################################
# Local repulsion: g''(0)
###############################################################

function local_repulsion_β(β; r_fit=0.5, kwargs...)
    r, g = pair_correlation_β(β; kwargs...)
    idx = findall(r .< r_fit)
    X = hcat(r[idx].^2, r[idx], ones(length(idx)))
    a = (X \ g[idx])[1]
    return 2a
end

function local_repulsion_poisson(; r_fit=0.5, kwargs...)
    r, g = pair_correlation_poisson(; kwargs...)
    idx = findall(r .< r_fit)
    X = hcat(r[idx].^2, r[idx], ones(length(idx)))
    a = (X \ g[idx])[1]
    return 2a
end




###############################################################
# Pair-correlation estimator (β-Hermite) - FIXED NORMALIZATION
###############################################################

function pair_correlation_β(β; n=500, num_trials=300, r_max=5.0, nbins=200)
    dists = Float64[]
    total_bulk_length = 0.0

    for _ in 1:num_trials
        eigs = sample_eigs(n, β)
        # Focus on bulk
        bulk = eigs[Int(0.2n):Int(0.8n)]

        # Unfold
        Δ = diff(bulk)
        m = mean(Δ)
        unfolded = (bulk .- bulk[1]) ./ m
        
        # Track total length for normalization
        total_bulk_length += length(unfolded)

        L = length(unfolded)
        # Collect pairwise distances
        for i in 1:L-1
            for j in i+1:L
                d = abs(unfolded[j] - unfolded[i])
                if d <= r_max
                    push!(dists, d)
                end
            end
        end
    end

    edges = collect(range(0, r_max; length=nbins+1))
    H = fit(Histogram, dists, edges)
    dr = edges[2] - edges[1]
    
    r_centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    
    # CORRECT NORMALIZATION:
    # Expected counts = (Total Number of Points) * dr
    # We ignore (L-r) edge effect correction because r << L
    g_r = H.weights ./ (total_bulk_length * dr)

    return r_centers, g_r
end

###############################################################
# Pair-correlation for Poisson (rand()) - FIXED NORMALIZATION
###############################################################

function pair_correlation_poisson(; n=500, num_trials=300, r_max=5.0, nbins=200)
    dists = Float64[]
    total_len = 0.0

    for _ in 1:num_trials
        x = sort(rand(n))
        Δ = diff(x)
        m = mean(Δ)
        unfolded = (x .- x[1]) ./ m
        
        total_len += length(unfolded)

        L = length(unfolded)
        for i in 1:L-1
            for j in i+1:L
                d = abs(unfolded[j] - unfolded[i])
                if d <= r_max
                    push!(dists, d)
                end
            end
        end
    end

    edges = collect(range(0, r_max; length=nbins+1))
    H = fit(Histogram, dists, edges)
    dr = edges[2] - edges[1]

    r_centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    
    # CORRECT NORMALIZATION
    g_r = H.weights ./ (total_len * dr)

    return r_centers, g_r
end

###############################################################
# Probability of Difference: p_u = ∫ (1 - g(v)) dv
###############################################################

function probability_of_difference_β(β; kwargs...)
    r, g = pair_correlation_β(β; kwargs...)
    # p_u should converge to 1.0 for rigid systems, 0.0 for Poisson
    integral_val = trapz(r, 1 .- g)
    return clamp(integral_val, 0.0, 1.0)
end

function probability_of_difference_poisson(; kwargs...)
    r, g = pair_correlation_poisson(; kwargs...)
    integral_val = trapz(r, 1 .- g)
    return clamp(integral_val, 0.0, 1.0)
end

###############################################################
# Spacing Distribution Distance (The Metric you NEED for the Chart)
###############################################################

function spacing_distance_β(β::Float64; n=500, num_trials=300, s_max=3.0, nbins=200)
    all_spacings = Float64[]
    i1 = Int(round(0.2n)); i2 = Int(round(0.8n))

    for _ in 1:num_trials
        eigs = sample_eigs(n, β)
        bulk = eigs[i1:i2]
        if length(bulk) > 1
            Δ = diff(bulk)
            append!(all_spacings, Δ ./ mean(Δ))
        end
    end

    edges = collect(range(0, s_max; length=nbins+1))
    H = fit(Histogram, all_spacings, edges)
    Δs = edges[2] - edges[1]
    s_centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    
    # PDF Normalization (This one IS a PDF, so this is correct)
    p_hat = H.weights ./ (sum(H.weights)*Δs)
    p_pois = exp.(-s_centers)

    return trapz(s_centers, abs.(p_pois .- p_hat))
end

function spacing_distance_poisson(; n=500, num_trials=300, s_max=3.0, nbins=200)
    all_spacings = Float64[]
    for _ in 1:num_trials
        x = sort(rand(n)); Δ = diff(x); append!(all_spacings, Δ ./ mean(Δ))
    end
    edges = collect(range(0, s_max; length=nbins+1))
    H = fit(Histogram, all_spacings, edges)
    Δs = edges[2] - edges[1]
    s_centers = (edges[1:end-1] .+ edges[2:end]) ./ 2
    p_hat = H.weights ./ (sum(H.weights)*Δs)
    p_pois = exp.(-s_centers)
    
    return trapz(s_centers, abs.(p_pois .- p_hat))
end

###############################################################
# MAIN LOOP
###############################################################

# Only run for n=500 to save time for this test
n_val = 500
β_list = [0.1, 0.3, 0.5, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

for n in [10, 50, 100, 200, 300, 400, 500]
    rows = Vector{Dict{Symbol,Any}}()

    println("\nComputing metrics with CORRECTED normalization...")

    for β in β_list
        println("β = $β")
        # Using arrays to store trial results
        Rnn_vals = Float64[]
        Glob_vals = Float64[]
        Loc_vals = Float64[]
        Pu_vals = Float64[]
        
        # Small num_trials for speed in this test
        for trial in 1:10
            # 1. Spacing Distance (The gradient metric)
            Rnn = spacing_distance_β(Float64(β); n=n_val, num_trials=10)
            
            # 2. Global Repulsion (Integrals)
            Glob = global_repulsion_β(Float64(β); n=n_val, num_trials=10)
            
            # 3. Local Repulsion (Curvature)
            Loc = local_repulsion_β(Float64(β); n=n_val, num_trials=10)
            
            # 4. Probability of Difference (Should be 1.0 for all beta)
            Pu = probability_of_difference_β(Float64(β); n=n_val, num_trials=10)

            # println("Pu = $Pu")
            
            push!(Rnn_vals, Rnn)
            push!(Glob_vals, Glob)
            push!(Loc_vals, Loc)
            push!(Pu_vals, Pu)
        end

        push!(rows, Dict(
            :beta => β,
            :spacing_dist => mean(Rnn_vals),
            :spacing_dist_std => std(Rnn_vals),
            :global_r => mean(Glob_vals),
            :global_r_std => std(Glob_vals),
            :local_r => mean(Loc_vals),
            :local_r_std => std(Loc_vals),
            :p_u => mean(Pu_vals),
            :p_u_std => std(Pu_vals)
        ))
    end

    # Poisson Baseline
    println("Computing Poisson Baseline...")
    RnnP_vals = Float64[]; GlobP_vals = Float64[]; LocP_vals = Float64[]; PuP_vals = Float64[]
    for trial in 1:10
        push!(RnnP_vals, spacing_distance_poisson(n=n_val, num_trials=10))
        push!(GlobP_vals, global_repulsion_poisson(n=n_val, num_trials=10))
        push!(LocP_vals, local_repulsion_poisson(n=n_val, num_trials=10))
        push!(PuP_vals, probability_of_difference_poisson(n=n_val, num_trials=10))
    end

    # println("Pu poisson = $PuP_vals")

    push!(rows, Dict(
        :beta => "Poisson",
        :spacing_dist => mean(RnnP_vals),
        :spacing_dist_std => std(RnnP_vals),
        :global_r => mean(GlobP_vals),
        :global_r_std => std(GlobP_vals),
        :local_r => mean(LocP_vals),
        :local_r_std => std(LocP_vals),
        :p_u => mean(PuP_vals),
        :p_u_std => std(PuP_vals)
    ))

    # Save
    df = DataFrame(rows)
    CSV.write("repulsion_metrics_v4_oreilly_corrected_n=$n.csv", df)
    # println(df)
    println("\nSaved → repulsion_metrics_v4_oreilly_corrected_n=$n.csv")
end
