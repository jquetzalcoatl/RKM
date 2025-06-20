include("von_mises.jl")
include("tikhonov_rand.jl")

function v_to_h(θ::CuArray{Float64,2},J::Weights,β=1.0)::CuArray{Float64,2} 
    a,b = A_h(θ,J,β),B_h(θ,J,β)
    k = CUDA.sqrt.(a .^ 2 .+ b .^ 2)
    μ = CUDA.atan.(b, a)
    
    # gpu(vonMises_sample.(cpu(μ),cpu(k)))
    s = size(k)
    psi = reshape(TikhonovGenGPUv2(reshape(k,:), CuArray(zeros(prod(s))),
        CuArray(zeros(prod(s))),CuArray(zeros(prod(s))), CuArray(ones(Float64,prod(s),8)),
        CuArray(ones(Float64,8,3,prod(s)))),s) .+ μ
    psi
end

function h_to_v(θ::CuArray{Float64,2},J::Weights, β=1.0)::CuArray{Float64,2} 
    a,b = A_v(θ,J,β),B_v(θ,J,β)
    k = CUDA.sqrt.(a .^ 2 .+ b .^ 2)
    μ = CUDA.atan.(b, a)
    
    # gpu(vonMises_sample.(cpu(μ),cpu(k)))
    s = size(k)
    psi = reshape(TikhonovGenGPUv2(reshape(k,:), CuArray(zeros(prod(s))),
        CuArray(zeros(prod(s))),CuArray(zeros(prod(s))), CuArray(ones(Float64, prod(s),8)),
        CuArray(ones(Float64, 8,3,prod(s)))),s) .+ μ
    psi
end

function v_to_h(θ::Array{Float64,2},J::Weights,β=1.0)::Array{Float64,2} 
    a,b = A_h(θ,J,β),B_h(θ,J,β)
    k = sqrt.(a .^ 2 .+ b .^ 2)
    μ = atan.(b, a)
    
    # vonMises_sample.(μ,k)
    s = size(k)
    psi = reshape(TikhonovGen(reshape(k,:), Array(zeros(prod(s))),
        Array(zeros(prod(s))),Array(zeros(prod(s))), Array(ones(prod(s),8)),
        Array(ones(8,3,prod(s)))),s) .+ μ
    psi
end

function h_to_v(θ::Array{Float64,2},J::Weights, β=1.0)::Array{Float64,2} 
    a,b = A_v(θ,J,β),B_v(θ,J,β)
    k = sqrt.(a .^ 2 .+ b .^ 2)
    μ = atan.(b, a)
    
    # vonMises_sample.(μ,k)
    s = size(k)
    psi = reshape(TikhonovGen(reshape(k,:), Array(zeros(prod(s))),
        Array(zeros(prod(s))),Array(zeros(prod(s))), Array(ones(prod(s),8)),
        Array(ones(8,3,prod(s)))),s) .+ μ
    psi
end

function neg_phase(θ::CuArray{Float64,2},J::Weights,steps::Int, bs::Int)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h(v,J)
        v = h_to_v(h,J)
    end
    (CUDA.cos.(v) * CUDA.cos.(h)' .+ CUDA.sin.(v) * CUDA.sin.(h)') ./ bs
end

function neg_phase(θ::Array{Float64,2},J::Weights,steps::Int, bs::Int)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h(v,J)
        v = h_to_v(h,J)
    end
    (cos.(v) * cos.(h)' .+ sin.(v) * sin.(h)') ./ bs
end

function neg_phase_centred(θ::CuArray{Float64,2},J::Weights,steps::Int, bs::Int)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h(v,J)
        v = h_to_v(h,J)
    end
    vh_m = (CUDA.cos.(v) * CUDA.cos.(h)' .+ CUDA.sin.(v) * CUDA.sin.(h)') ./ bs
    vm_hm = mean(cos.(v), dims=2) * mean(cos.(h), dims=2)' .+ mean(sin.(v), dims=2) * mean(sin.(h), dims=2)'
    vh_m .- vm_hm
end

function neg_phase_centred(θ::Array{Float64,2},J::Weights,steps::Int, bs::Int)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h(v,J)
        v = h_to_v(h,J)
    end
    vh_m = (cos.(v) * cos.(h)' .+ sin.(v) * sin.(h)') ./ bs
    vm_hm = mean(cos.(v), dims=2) * mean(cos.(h), dims=2)' .+ mean(sin.(v), dims=2) * mean(sin.(h), dims=2)'
    vh_m .- vm_hm
end

function compute_gradients(x::CuArray{Float64,2},rkm::RKM,J::Weights, config)
    pph = pos_phase(x, J, config.rkm["batch_size"])
    nph = neg_phase(rkm.v,J,config.rkm["t"],config.rkm["batch_size"])
    pph .- nph
end

function compute_gradients(x::Array{Float64,2},rkm::RKM,J::Weights, config)
    pph = pos_phase(x, J, config.rkm["batch_size"])
    nph = neg_phase(rkm.v,J,config.rkm["t"],config.rkm["batch_size"])
    pph .- nph
end

function compute_gradients_centred(x::CuArray{Float64,2},rkm::RKM,J::Weights, config)
    pph = pos_phase(x, J, config.rkm["batch_size"])
    nph = neg_phase_centred(rkm.v,J,config.rkm["t"],config.rkm["batch_size"])
    # θ = CuArray{Float64}(rand(size(rkm.v,1),size(rkm.v,2)) * 2π)
    # nph = neg_phase_centred(θ,J,config.rkm["t"],config.rkm["batch_size"])
    pph .- nph
end

function compute_gradients_centred_rdm(x::CuArray{Float64,2},rkm::RKM,J::Weights, config)
    pph = pos_phase(x, J, config.rkm["batch_size"])
    θ = CuArray{Float64}(rand(size(rkm.v,1),size(rkm.v,2)) * 2π)
    nph = neg_phase_centred(θ,J,config.rkm["t"],config.rkm["batch_size"])
    pph .- nph
end

function compute_gradients_centred(x::Array{Float64,2},rkm::RKM,J::Weights, config)
    pph = pos_phase(x, J, config.rkm["batch_size"])
    nph = neg_phase_centred(rkm.v,J,config.rkm["t"],config.rkm["batch_size"])
    # θ = Array{Float64}(rand(size(rkm.v,1),size(rkm.v,2)) * 2π)
    # nph = neg_phase_centred(θ,J,config.rkm["t"],config.rkm["batch_size"])
    pph .- nph
end

function bgs(θ::CuArray{Float64,2},J::Weights,steps::Int)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h(v,J)
        v = h_to_v(h,J)
    end
    v
end

function bgs(θ::Array{Float64,2},J::Weights,steps::Int)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h(v,J)
        v = h_to_v(h,J)
    end
    v
end

function bgs_vh(θ::CuArray{Float64,2},J::Weights,steps::Int, β=1.0)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h(v,J,β)
        v = h_to_v(h,J,β)
    end
    v,h
end