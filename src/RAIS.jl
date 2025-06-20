include("../grad/gradient.jl")

function AIS(rkm::RKM, J::Weights, hparams::HyperParams, nbeta::Float64=30)
    lnZa = sum(log1p.(exp.(J.a))) + sum(log1p.(exp.(J.b)))
    FreeEnergy_ratios = 0.0
    Δbeta = 1.0 / nbeta

    θ = CuArray{Float64}(rand(size(rkm.v,1),size(rkm.v,2)) * 2π);
    for β in 0:Δbeta:1.0-Δbeta
        @info "AIS annealing $β"
        rkm.v,rkm.h = bgs_vh(θ,J,hparams.t, β)

        energy_samples_i = energy(rkm,J,β)
        energy_samples_i_plus = energy(rkm,J,β + Δbeta)
        FreeEnergy_ratios += log(mean(exp.(energy_samples_i .- energy_samples_i_plus)))
    end
    logZb = FreeEnergy_ratios + lnZa
    return logZb
end

function RAIS(rkm::RKM, J::Weights, hparams::HyperParams, nbeta::Float64=30)
    lnZb = sum(log1p.(exp.(J.a))) + sum(log1p.(exp.(J.b)))
    FreeEnergy_ratios = 0.0
    Δbeta = 1.0 / nbeta

    θ = CuArray{Float64}(rand(size(rkm.v,1),size(rkm.v,2)) * 2π);
    for β in 1:-Δbeta:Δbeta
        @info "RAIS annealing $β"
        rkm.v,rkm.h = bgs_vh(θ,J,hparams.t, β)

        energy_samples_i = energy(rkm,J,β)
        energy_samples_i_minus = energy(rkm,J,β - Δbeta)
        FreeEnergy_ratios += log(mean(exp.(energy_samples_i .- energy_samples_i_minus)))
    end
    logZa = - FreeEnergy_ratios + lnZb
    return logZa
end

energy(rkm::RKM,J::Weights, β) = - diag(CUDA.cos.(rkm.v)' * β * J.w * CUDA.cos.(rkm.h) .+ CUDA.sin.(rkm.v)' * β * J.w * CUDA.sin.(rkm.h))

function LL_numerator(v,J::Weights)
    θ = CuArray{Float64}(v)
    a,b = A_h(θ,J,1.0), B_h(θ,J,1.0)
    k_h=CUDA.sqrt.( a .^ 2 .+ b .^ 2)
    bsl = besseli.(0,k_h)
    mean(sum(log.(bsl),dims=2)) + size(bsl,1)*log(2π)
end