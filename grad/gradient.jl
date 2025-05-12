# using SpecialFunctions, 
using Bessels, Distributions, Random

include("grad_argmax.jl")
include("grad_approx.jl")
include("grad_exact.jl")
include("grad.jl")

A_h(θ::CuArray{Float64,2},J::Weights,β=1.0)::CuArray{Float64,2} = β * J.w' * CUDA.cos.(θ)
B_h(θ::CuArray{Float64,2},J::Weights,β=1.0)::CuArray{Float64,2} = β * J.w' * CUDA.sin.(θ)

A_h(θ::Array{Float64,2},J::Weights,β=1.0) = β * J.w' * cos.(θ)
B_h(θ::Array{Float64,2},J::Weights,β=1.0) = β * J.w' * sin.(θ)

A_v(θ::CuArray{Float64,2},J::Weights,β=1.0)::CuArray{Float64,2} = β * J.w * CUDA.cos.(θ)
B_v(θ::CuArray{Float64,2},J::Weights,β=1.0)::CuArray{Float64,2} = β * J.w * CUDA.sin.(θ)

A_v(θ::Array{Float64,2},J::Weights,β=1.0) = β * J.w * cos.(θ)
B_v(θ::Array{Float64,2},J::Weights,β=1.0) = β * J.w * sin.(θ)

function pos_phase(θ::CuArray{Float64,2}, J::Weights, bs::Int, β=1.0)
    a,b = A_h(θ,J,β), B_h(θ,J,β)
    k_h=CUDA.sqrt.( a .^ 2 .+ b .^ 2)
    # bessel_ratio = besseli.(1,cpu(k_h)) ./ besseli.(0,cpu(k_h)) |> gpu
    bessel_ratio = besseli.(1,k_h) ./ besseli.(0,k_h)
    arg2_c, arg2_s = a .* bessel_ratio ./ k_h , b .* bessel_ratio ./ k_h
    vh_m = (CUDA.cos.(θ) * arg2_c' + CUDA.sin.(θ) * arg2_s') ./ bs
    vh_m
end

function pos_phase(θ::Array{Float64,2}, J::Weights, bs::Int, β=1.0)
    a,b = A_h(θ,J,β), B_h(θ,J,β)
    k_h=sqrt.( a .^ 2 .+ b .^ 2)
    bessel_ratio = besseli.(1,k_h) ./ besseli.(0,k_h)
    arg2_c, arg2_s = a .* bessel_ratio ./ k_h , b .* bessel_ratio ./ k_h
    vh_m = (cos.(θ) * arg2_c' + sin.(θ) * arg2_s') ./ bs
    vh_m
end

function updateJAdam!(J, Δw, opt; hparams)
    J.w = step!(opt.w, Δw) #(1 - hparams.γ) .* 
end