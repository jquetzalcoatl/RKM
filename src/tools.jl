using Statistics

magnetization(spin_array::CuArray{Float32,2}) = sqrt.(sum(CUDA.cos.(spin_array), dims=1) .^ 2 .+ sum(CUDA.sin.(spin_array) , dims=1) .^2)/size(spin_array,1)

energy(rkm::RKM,J::Weights) = - diag(CUDA.cos.(rkm.v)' * J.w * CUDA.cos.(rkm.h) .+ CUDA.sin.(rkm.v)' * J.w * CUDA.sin.(rkm.h))