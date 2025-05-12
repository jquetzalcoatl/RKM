function v_to_h_exact(θ::CuArray{Float32,2},J::Weights,β=1.0)::CuArray{Float32,2} 
    a,b = A_h(θ,J,β),B_h(θ,J,β)
    k = CUDA.sqrt.(a .^ 2 .+ b .^ 2)
    μ = CUDA.atan.(b, a)
    # vM = VonMises.(cpu(μ),cpu(k))
    # gpu(rand.(vM))
    gpu(rand.(VonMises.(cpu(μ),cpu(k))))
end

function h_to_v_exact(θ::CuArray{Float32,2},J::Weights, β=1.0)::CuArray{Float32,2} 
    a,b = A_v(θ,J,β),B_v(θ,J,β)
    k = CUDA.sqrt.(a .^ 2 .+ b .^ 2)
    μ = CUDA.atan.(b, a)
    # vM = VonMises.(cpu(μ),cpu(k))
    # gpu(rand.(vM))
    gpu(rand.(VonMises.(cpu(μ),cpu(k))))
end

function v_to_h_exact(θ::Array{Float64,2},J::Weights,β=1.0)::Array{Float64,2} 
    a,b = A_h(θ,J,β),B_h(θ,J,β)
    k = CUDA.sqrt.(a .^ 2 .+ b .^ 2)
    μ = CUDA.atan.(b, a)
    vM = VonMises.(μ,k)
    rand.(vM)
end

function h_to_v_exact(θ::Array{Float64,2},J::Weights, β=1.0)::Array{Float64,2} 
    a,b = A_v(θ,J,β),B_v(θ,J,β)
    k = sqrt.(a .^ 2 .+ b .^ 2)
    μ = atan.(b, a)
    vM = VonMises.(μ,k)
    rand.(vM)
end

function neg_phase_exact(θ::CuArray{Float32,2},J::Weights,steps::Int, bs::Int)
    v = θ
    local h
    for _ in 1:steps
        # h = v_to_h_argmax(v,J)
        # v = h_to_v_argmax(h,J)
        h = v_to_h(v,J) .% Float32(2π)
        v = h_to_v(h,J) .% Float32(2π)
    end
    (CUDA.cos.(v) * CUDA.cos.(h)' .+ CUDA.sin.(v) * CUDA.sin.(h)') ./ bs
end

function neg_phase_exact(θ::Array{Float64,2},J::Weights,steps::Int, bs::Int)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h(v,J) .% 2π
        v = h_to_v(h,J) .% 2π
    end
    (cos.(v) * cos.(h)' .+ sin.(v) * sin.(h)') ./ bs
end

function compute_gradients_exact(x::CuArray{Float32,2},rkm::RKM,J::Weights, config)
    pph = pos_phase(x, J, config.rkm["batch_size"])
    nph = neg_phase(rkm.v,J,config.rkm["t"],config.rkm["batch_size"])
    pph .- nph
end

function compute_gradients_exact(x::Array{Float32,2},rkm::RKM,J::Weights, config)
    pph = pos_phase(x, J, config.rkm["batch_size"])
    nph = neg_phase(rkm.v,J,config.rkm["t"],config.rkm["batch_size"])
    pph .- nph
end

function bgs_exact(θ::CuArray{Float32,2},J::Weights,steps::Int)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h(v,J) .% Float32(2π)
        v = h_to_v(h,J) .% Float32(2π)
    end
    v
end

function bgs_exact(θ::Array{Float64,2},J::Weights,steps::Int)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h(v,J) .% 2π
        v = h_to_v(h,J) .% 2π
    end
    v
end