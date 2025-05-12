h_to_v_argmax(θ::CuArray{Float32,2},J::Weights)::CuArray{Float32,2} = CUDA.atan.(J.w * CUDA.sin.(θ), J.w * CUDA.cos.(θ))
h_to_v_argmax(θ::Array{Float64,2},J::Weights) = atan.(J.w * sin.(θ), J.w * cos.(θ))

v_to_h_argmax(θ::CuArray{Float32,2},J::Weights)::CuArray{Float32,2} = CUDA.atan.(J.w' * CUDA.sin.(θ), J.w' * CUDA.cos.(θ))
v_to_h_argmax(θ::Array{Float64,2},J::Weights) = atan.(J.w' * sin.(θ), J.w' * cos.(θ))

function neg_phase_argmax(θ::CuArray{Float32,2},J::Weights,steps::Int, bs::Int)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h_argmax(v,J)
        v = h_to_v_argmax(h,J)
    end
    (CUDA.cos.(v) * CUDA.cos.(h)' .+ CUDA.sin.(v) * CUDA.sin.(h)') ./ bs
end

function neg_phase_argmax(θ::Array{Float64,2},J::Weights,steps::Int, bs::Int)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h_argmax(v,J) .% 2π
        v = h_to_v_argmax(h,J) .% 2π
    end
    (cos.(v) * cos.(h)' .+ sin.(v) * sin.(h)') ./ bs
end

function compute_gradients_argmax(x::CuArray{Float32,2},rkm::RKM,J::Weights, config)
    pph = pos_phase(x, J, config.rkm["batch_size"])
    nph = neg_phase_argmax(rkm.v,J,config.rkm["t"],config.rkm["batch_size"])
    pph .- nph
end

function compute_gradients_argmax(x::Array{Float64,2},rkm::RKM,J::Weights, config)
    pph = pos_phase(x, J, config.rkm["batch_size"])
    nph = neg_phase_argmax(rkm.v,J,config.rkm["t"],config.rkm["batch_size"])
    pph .- nph
end

function bgs_argmax(θ::CuArray{Float32,2},J::Weights,steps::Int)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h_argmax(v,J)
        v = h_to_v_argmax(h,J)
    end
    v
end

function bgs_argmax(θ::Array{Float64,2},J::Weights,steps::Int)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h_argmax(v,J)
        v = h_to_v_argmax(h,J)
    end
    v
end