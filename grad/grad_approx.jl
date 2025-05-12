function v_to_h_approx(θ::CuArray{Float32,2},J::Weights,β=1.0)::CuArray{Float32,2} 
    a,b = A_h(θ,J,β),B_h(θ,J,β)
    k = CUDA.sqrt.(a .^ 2 .+ b .^ 2)
    μ = CUDA.atan.(b, a)
    # bessel_ratio = besseli.(1,cpu(k)) ./ besseli.(0,cpu(k)) |> gpu
    bessel_ratio = besseli.(1,k) ./ besseli.(0,k)
    if minimum(bessel_ratio)<0 || maximum(bessel_ratio)>exp(1)
        @info minimum(bessel_ratio), maximum(bessel_ratio)
    end
    σ = CUDA.sqrt.(-2 .* CUDA.log.(bessel_ratio))
    ((randn!(σ) .* σ ) .% π) .+ μ
end

function h_to_v_approx(θ::CuArray{Float32,2},J::Weights, β=1.0)::CuArray{Float32,2} 
    a,b = A_v(θ,J,β),B_v(θ,J,β)
    k = CUDA.sqrt.(a .^ 2 .+ b .^ 2)
    μ = CUDA.atan.(b, a)
    # bessel_ratio = besseli.(1,cpu(k)) ./ besseli.(0,cpu(k)) |> gpu
    bessel_ratio = besseli.(1,k) ./ besseli.(0,k)
    if minimum(bessel_ratio)<0 || maximum(bessel_ratio)>exp(1)
        @info minimum(bessel_ratio), maximum(bessel_ratio)
    end
    σ = CUDA.sqrt.(-2 .* CUDA.log.(bessel_ratio))
    ((randn!(σ) .* σ ) .% π) .+ μ
end

function neg_phase_approx(θ::CuArray{Float32,2},J::Weights,steps::Int, bs::Int)
    v = θ
    local h
    for _ in 1:steps
        # h = v_to_h_argmax(v,J)
        # v = h_to_v_argmax(h,J)
        h = v_to_h_approx(v,J)
        v = h_to_v_approx(h,J)
    end
    (CUDA.cos.(v) * CUDA.cos.(h)' .+ CUDA.sin.(v) * CUDA.sin.(h)') ./ bs
end

function compute_gradients_approx(x::CuArray{Float32,2},rkm::RKM,J::Weights, config)
    pph = pos_phase(x, J, config.rkm["batch_size"])
    nph = neg_phase_approx(rkm.v,J,config.rkm["t"],config.rkm["batch_size"])
    pph .- nph
end


function bgs_approx(θ::CuArray{Float32,2},J::Weights,steps::Int)
    v = θ
    local h
    for _ in 1:steps
        h = v_to_h_approx(v,J)
        v = h_to_v_approx(h,J)
    end
    v
end