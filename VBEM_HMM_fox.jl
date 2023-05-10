### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# ╔═╡ db588217-5856-47b9-bfa2-637f84d19cd4
begin
	using Distributions
	using LinearAlgebra
	using StatsFuns
	using SpecialFunctions
	using Random
	using HMMBase
	using PlutoUI
	using PDMats
end

# ╔═╡ 06692696-8edb-42dd-be97-b1ea41e3d60f
TableOfContents()

# ╔═╡ ed4603db-b956-4098-85e9-dcd97b132cb4
md"""
# Derive VB-EM for HMM
The following notes are taken from  M. J Beale's 2003 Paper

Marginal likelihood:

$$p(y_{1:T}) = \int d\mathbf{θ} \, p(\mathbf{θ})p(y_{1:T}|\mathbf{θ})$$

where $$\mathbf{θ} = \{ π, A, B\}$$

Log marginal likelihood of HMM, let Y denote $y_{1:T}$, S denote $s_{1:T}$:

$$\ln p(Y) = \ln \int dπ \int dA \int dB \sum_{s \in S} p(π, A, B) p(Y, S|π, A, B)$$

Using Jensen's inequality and introduce $q(π, A, B, S)$:

$$\ln p(Y) \geq \int dπ \int dA \int dB \sum_{s \in S} q(π, A, B, S) \, \ln\frac{p(π, A, B) p(Y, S|π, A, B)}{q(π, A, B, S)}$$

Making a factorised assumption of the posterior into variational posterior:

$$p(π, A, B, S | Y) \approx q(π, A, B|Y) q(S|Y)$$

Hereafter we assume implicit dependence on $Y = y_{1:T}$, for notational ease.

Giving rise to the expression for the lower bound $\mathcal F$:

$$\mathcal F(q(π, A, B), q(S)) = \int dπ \int dA \int dB \, q(π, A, B) \{\ln\frac{p(π, A, B)}{q(π, A, B)} + \sum_{s \in S} q(S) \ln\frac{p(Y,S|π, A, B)}{q(S)} \}$$

$\ln p(Y) \geq \mathcal F(q(π, A, B), q(S))$
"""

# ╔═╡ 009cfa59-55d5-4d11-8e29-7f09f58a7354
md"""
## VB_M Step 
$\begin{align}
\ln q(π, A, B) &= \ln p(π, A, B) + \langle \ln p(Y, S| π, A, B) \rangle_{\hat q(S)} + c \\
&= \ln p(π) + \ln p(A) + \ln p(B) \\ &+ \langle \ln p(s_1| π) \rangle_{\hat q(s_1)} +  \langle \ln p(s_{2:T}| s_1, A) \rangle_{\hat q(S)} +  \langle \ln p(Y| S, B) \rangle_{\hat q(S)} + c
\end{align}$

where $c$ is the normalizing constant; 

2nd line given by: $q(π, A, B) = q(π) q(A) q(B)$

3rd line given by: log complete likelihood $\ln p(S, Y| \mathbf{θ}) = s_1^T \ln π + \sum_{t=2}^T s_{t-1}^T \ln A \,s_t + \sum_{t=1}^T s_t^T \ln B \, y_t$

By conjugacy, variational posterior distributions have the same form as the priors with their hyperparameters `u` augmented by the sufficient statistics of the hidden states and observations `t(S, Y)`, these are computed from the E-step.

$$\mathbf{w} = \mathbf{u} + \langle t(S, Y) \rangle_{\hat q_(S)}$$

### Compare with MLE Baum-Welch
Unlike the Baum-Welch M-step, which uses frequency counting to extract the maxium of the MLE, resulting in point estimates of π, A and MVN parameters μ, Σ. 

The VBEM M-step, we update the parameters of the variational posterior distributions for the HMM parameters. We use the expected sufficient statistics calculated during the E-step to update the parameters of the variational posterior distributions.

The VBEM M-step incorporates prior knowledge about the HMM parameters through the use of conjugate priors. 

	Transition matrix A, we use Dirichlet priors on rows of A, 

	Multivariate Gaussian (MVN) emissions, we use a Normal (inverse) Wishart prior. 

These priors choice can influence the updates of the variational posterior distributions, leading to a Bayesian estimation of the parameters. 

The VBEM M-step provides a form of regularization due to the incorporation of priors. In theory, this can prevent overfitting, especially when there is limited data, and lead to more robust estimates of the HMM parameters. 
"""

# ╔═╡ 4c7b01e4-b201-4605-8bf3-16d404c8be39
md"""
## VB_E Step

$\begin{align}
\ln q(S) &= \langle \ln p(Y, S|π, A, B) \rangle_{\hat q_(π, A, B)} - \ln \mathcal Z(Y) \\
&= \langle s_1^T \ln π + \sum_{t=2}^T s_{t-1}^T \ln A \,s_t + \sum_{t=1}^T s_t^T \ln B \, y_t \rangle_{\hat q(π) \hat q(A) \hat q(B)} - \ln \mathcal Z(Y)
\end{align}$


The 2nd line is given by the vectorised expression of log complete data likelihood, and $\ln \mathcal Z(Y)$ is another normalization constant

Use modified parameter $\tilde{θ}$ for the forward-backward algorithm

$$\tilde{θ} = \{\exp \langle \ln π \rangle_{\hat q(π)},  \exp \langle \ln A \rangle_{\hat q(A)}, \exp \langle \ln B \rangle_{\hat q(B)}\}$$

"""

# ╔═╡ 3da884a5-eacb-4749-a10c-5a6ad6da34b7
md"""
# VBEM MVN Emission
The following (batch) VBEM follows largely from Beale's 2003 paper and Fox et al's paper in 2014

Using the following structured mean-field approximation:

$$p(A, \{ϕ_k\}, S | Y) \approx q(A) q(\{ϕ_k\}) q(S)$$

where $\{ϕ_k\}$ are parameters Normal (inverse) Wishart distributed, $ϕ_k = \{μ_k, Σ_k\}$, mean vector and co-variance matrix.
"""

# ╔═╡ 4a573a35-51b2-4a42-b175-3ba017ef7245
md"""
## VB_M NIW update

The NIW prior is characterized by four parameters μ (mu), κ (kappa), ν (nu), Σ (Sigma):
* μ₀ (prior mean), 
* κ₀ (prior degrees of freedom for the mean), 
* ν₀ (prior degrees of freedom for the covariance), 
* Σ₀ (prior scale matrix).

The natural parameterization of the Normal-Inverse-Wishart distribution has four parameters:

$u^ϕ_1 = κ₀ μ₀\\$ A vector representing the expected value of the multivariate normal distribution. 


$u^ϕ_2 = κ₀\\$ The scale factor for the precision of the multivariate normal distribution.


$u^ϕ_3 = Σ₀ + κ₀ μ₀ μ₀^T\\$ A symmetric positive definite matrix representing the expected value of the inverse Wishart distribution.


$u^ϕ_4 = ν₀ + 2 + D\\$ The degrees of freedom of the inverse Wishart distribution.


By working with these natural parameters, the updates and calculations in the VBEM algorithm can be simplified, and the optimization can be more efficient.
"""

# ╔═╡ ae313199-fe27-478a-b6b2-bee2923b5a54
begin
	struct Exp_MVN{T <: Real}
	    m::Array{T, 1}  # d-dimensional mean vector
		κ::T         # scalar precision parameter for the mean vector [β -> Bishop 10.2.1]
	 	ν::T            # degrees of freedom (ν > d - 1)
		Σ::Matrix{T}   # d×d positive definite scale matrix
	end

	# according to Fox Supplement S7, VB-M Update unsure of cov/precision params
	function Exp_MVN(prior::Exp_MVN, k::Int64, rs, ys)
		T, D = size(ys)
		
		N_k = sum(rs, dims=2)[k]
		
		κ_k = prior.κ + N_k
		
		ν_k = (prior.ν + 2.0 + D) + N_k
		
		m_k = prior.m * prior.κ + sum([ys[t, :] * rs[k, t] for t in 1:T])
		
		Σ_k = prior.Σ + prior.κ*prior.m*prior.m' + sum([ys[t, :]*ys[t, :]'*rs[k, t] for t in 1:T])
		
		return Exp_MVN(m_k, κ_k, ν_k, Σ_k) # should parse a list of len K to forward/backward
	end
end

# ╔═╡ a0fa9433-3ae7-4697-8eea-5d6ddefbf62b
begin
	# test M-step update to recover true mean and precision
	Random.seed!(123)

	# ground-truth
	μ = [-2, 1]
	Σ = Matrix{Float64}([2 0; 0 3])
	
	data = rand(MvNormal(μ, Σ), 500) # D X T

	# 3 x 100 sufficient stats q(s)
	logγ = zeros(3, 500)
	
	logγ[1,:] .= 1
	logγ[2,:] .= -Inf
	logγ[3,:] .= -Inf

	logrs = (logγ .- logsumexp(logγ, dims=1))
	
	rs = exp.(logrs)

	prior = Exp_MVN(zeros(2), 0.1, 1.0, Matrix{Float64}(1.0 * I, 2, 2))

	post = Exp_MVN(prior, 1, rs, data')
end

# ╔═╡ e9b7bbd0-2de8-4a5b-af73-3b576e8c54a2
md"""
Test with m-step update -> mean
"""

# ╔═╡ 6905f394-3da2-4207-a511-888f7521d82a
μ_m = post.m/post.κ # recover mean from natural paramerization

# ╔═╡ ac620936-66fb-4457-b9fd-9180fc9ba994
Σ_m = post.Σ - post.κ*μ_m*μ_m'

# ╔═╡ 5d746477-0282-4656-8db4-5a61561d3ccb
md"""
Test with m-step update -> Co-variance
"""

# ╔═╡ a4b38c35-e320-439e-b169-b8af974e2635
Σ_m / (post.ν - 2 - 2)

# ╔═╡ 61a5e1fd-5480-4ed1-83ff-d3ad140cbcbc
md"""
## VB_M, Dirichlet update

During the E-step, you calculate the expected sufficient statistics, which can be the expected number of transitions between states `log_ξ`. After exponentiating and summing over time, you get the following sufficient statistics:

    A matrix sufficient_A of shape (K, K) representing the expected number of transitions between states.

The natural parameterization refers to the representation of the distributions in terms of their natural parameters. Using the natural parameterization often simplifies the calculations and updates in variational inference.

    For the Dirichlet distribution, the natural parameterization is given by the vector of concentration parameters α = (α₁, α₂, ..., αₖ), where k is the number of states. The Dirichlet distribution is typically represented as:

$Dir(π | α) \propto Πᵢ πᵢ^{αᵢ - 1}$

The natural parameters are the exponents minus one: αᵢ - 1.

When updating the Dirichlet parameters in the M-step, you add the prior parameters and the sufficient statistics:

    dirichlet_params_A = A_prior + sufficient_A

	η_new = α - 1 .+ sum(exp.(log_ξ), dims=3)
"""

# ╔═╡ 5a1de95b-deef-4199-8992-28fcfe2157b8
md"""
## VB_M Step
`vbem_m_step` function for the Variational Bayesian EM algorithm, which updates the Dirichlet and Normal-Inverse-Wishart(NIW) parameters for the transition probabilities (A), and the Multivariate Gaussian (MVN) emission distributions."""

# ╔═╡ b4d1bdf3-efd6-4851-9d33-e1564b8ed769
# log_γ, log_ξ are sufficient stats from VBEM E-step
# alpha, prior are initial prior hyperparameters
function vbem_m_step(ys, log_γ, log_ξ, alpha, prior::Exp_MVN)
    K, T = size(log_γ)
    
    # Update Dirichlet parameters [prior natural param + sufficient stats]
    dirichlet_params_new = (alpha - 1) .+ sum(exp.(log_ξ), dims=3)[:, :, 1]
	
	# Update NIW parameters [prior natural param + sufficient stats]
    Exp_MVNs = []
	
    rs = exp.(log_γ)
	
    for k in 1:K
		Exp_mvn_k = Exp_MVN(prior, k, rs, ys)
		push!(Exp_MVNs, Exp_mvn_k)
    end
    
    return dirichlet_params_new, Exp_MVNs
end

# ╔═╡ 54f38d75-aa88-4eb8-983e-e2a3038f910f
md"""
## VB_E Step

From the modified parameter $\tilde{θ}$ for the forward-backward algorithm:

$$\tilde{θ} = \{\exp \langle \ln π \rangle_{\hat q(π)},  \exp \langle \ln A \rangle_{\hat q(A)}, \exp \langle \ln B \rangle_{\hat q(B)}\}$$

Since we are working with conjugate exponential model, the expectation of the log of Dirichlet distributed probabilities have a ready-to-use propety to compute using the Di-gamma $Ψ$ function. 

This works for $\tilde{A}$ and $\tilde{π}$. Here we present the more numerically stable version working in log-space. Details refer back to Beal 2003 paper equations `3.69, 3.70`.
"""

# ╔═╡ ebed75fc-0c10-48c7-9352-fc61bc3dcfd8
md"""
### Expected log Ã
"""

# ╔═╡ 072762bc-1f27-4a95-ad46-ddbdf45292cc
# E_q[ln(A)]
function log_Ã(dirichlet_params)
	row_sums = sum(dirichlet_params, dims=2)
    log_A_exp = digamma.(dirichlet_params) .- digamma.(row_sums) #broadcasting
    return log_A_exp
end

# ╔═╡ 820a3a70-6df9-4300-8037-21b2d51e8247
md"""
### Expected log p̃

For the case with MVN emission under NIW prior, the 3rd term of $\tilde θ$ becomes:

$\exp \langle \ln p(y_t|s_t = k) \rangle_{\hat q(ϕ)}$


Another function needed for the forward pass is the expected log probability density of the Gaussian emission distributions under their NIW variational distributions. Expressed as $\log p̃(y_t|s_t = k)$ in Fox et al.

We would like to formulate a funcion for calculating this log expected likelihood:

$\ln p̃(y_t|s_t = k) = \langle \ln p(y_t|s_t = k, ϕ) \rangle_{\hat q(ϕ)}$
"""

# ╔═╡ 49d2d576-e58f-4988-a7df-cdc44c1448db
md"""
#### Log Likelihood of MVN
$p(D|μ, Λ) = (2π)^{-nd/2} |Λ|^{n/2} \exp(- 0.5 ∑ (x_i - μ)^T Λ (x_i - μ))$

$\ln p(D| Λ, μ) = \frac{n}{2} \ln |Λ| - \frac{1}{2} ∑ (x_i - μ)^T Λ (x_i - μ) + c$

$c = \frac{nd}{2} \ln(\frac{1}{2π})$
"""

# ╔═╡ 9390462c-ee9e-415c-a0cc-3942989ad494
function log_p̃(mvn::Exp_MVN, y::Vector{Float64})
	# cf. Bishop 10.64, 10.65 and Normal-Wishart notebook
	d = size(mvn.m)[1]

	if mvn.Σ == Matrix{Float64}(I, d, d) # exclude first iteration
		κm = mvn.κ
		mm = mvn.m
		νm = mvn.ν
		Σm = Matrix{Float64}(I, d, d)

	else # recover from natural parameterization for expected log lik computation
		κm = mvn.κ
		mm = mvn.m/κm
		νm = mvn.ν - 2.0 - d
		Σm = mvn.Σ - κm*mm*mm'
	end
	
	# quadratic term
	term1 = -0.5(d/κm + νm * (y-mm)'* inv(Σm) * (y-mm))
	
	# log determinant term	
	term2 = 0.5 * ((digamma.(0.5 .* (νm .+ (1 .- collect(1:d)))) |> sum) + d * log(2) + logdet(inv(Σm)))
	
	return term1 + term2
end

# ╔═╡ bc0d9def-56a6-4c66-9ba4-d3237cdf2343
md"""
### Forward, Backward
"""

# ╔═╡ 20753ea0-a5b0-403d-ab33-98c90263e314
md"""
Fox et al (2014) presented an alternative method of working out $π$ as the left leading eigenvector of $A$. This assumption that $π$ is the leading left eigenvector of $A$ is only valid under certain conditions, such as ergodicity of the Markov chain.
"""

# ╔═╡ 7fda653c-2f37-4901-9676-8340978bca4d
function forward_log(ys, log_Ã, Exp_MVNs)
	T, D = size(ys)
	K = length(Exp_MVNs) # K Hidden states
	log_alpha = zeros(K, T)
	l_ζs = zeros(T) # used for ELBO 
	
	# Leading left eigen vector of Ã
	F = eigen((exp.(log_Ã))')
	idx = argmax(F.values)
	pi_sub = F.vectors[:, idx]
	pi_sub = pi_sub / sum(pi_sub)
	log_π = log.(pi_sub)

	log_alpha[:, 1] = log_π .+ [log_p̃(Exp_MVNs[k], ys[1, :]) for k in 1:K]
    
	# Normalize t=1
	l_ζs[1] = logsumexp(log_alpha[:, 1])
    log_alpha[:, 1] .-= l_ζs[1] 

    # Iterate through the remaining time steps
    for t in 2:T
        for k in 1:K
			log_emiss = log_p̃(Exp_MVNs[k], ys[t, :])
            log_alpha[k, t] = log_emiss + logsumexp(log_alpha[:, t-1] .+ log_Ã[:, k])
        end
		
        # Normalize log_alpha for t > 1
		l_ζs[t] = logsumexp(log_alpha[:, t])
        log_alpha[:, t] .-= l_ζs[t]
    end

    return log_alpha, l_ζs
end

# ╔═╡ 43660616-f1a8-434c-afc3-ac200d82425a
md"""
The `backward_log` function returns the backward variables log_beta of size (K, T), which can be used in the E-step of the Variational Bayesian EM algorithm.
"""

# ╔═╡ 3c4f3444-2547-4ecd-816c-05b35c73f050
# updated backward_log
function backward_log(ys, log_Ã, Exp_MVNs)
	T, D = size(ys)
	K = length(Exp_MVNs)    
	log_beta = zeros(K, T)

    # Iterate through the remaining time steps [backward pass]
    for t in T-1:-1:1
        for k in 1:K
			log_emiss = [log_p̃(Exp_MVNs[j], ys[t+1, :]) for j in 1:K]
			log_beta[k, t] = logsumexp(log_Ã[k, :] .+ log_emiss .+ log_beta[:, t+1])
        end
		
        # Normalize log_beta
        log_beta[:, t] .-= logsumexp(log_beta[:, t])
    end

    return log_beta
end

# ╔═╡ 48f0208e-8b99-4b68-9982-02fa18bc0ab1
md"""
Putting together forward, backward and helper functions together constitute the `vbem_e_step` function below
"""

# ╔═╡ 268464af-2842-4240-8551-ffc0cc130b70
function vbem_e_step(ys, dirichlet_params, Exp_MVNs)
    # Compute expected log transition probabilities 
    log_A = log_Ã(dirichlet_params)

	log_α, log_ζs = forward_log(ys, log_A, Exp_MVNs)
	log_β = backward_log(ys, log_A, Exp_MVNs)
	
    # Compute log_ξ and log_γ [identical to Baum-Welch E-step]
    log_γ = log_α .+ log_β
	log_γ .-= logsumexp(log_γ, dims=1) # normalize

    K, T = size(log_α)
    log_ξ = zeros(K, K, T-1)
	
	log_emiss = zeros(K, T)
	
	for i in 1:K
		log_emiss[i, :] = [log_p̃(Exp_MVNs[i], ys[t, : ]) for t in 1:T]
    end

	for t in 1:T-1
        log_ξ[:, :, t] = log_α[:, t] .+ log_A .+ log_emiss[:, t+1]' .+ log_β[:, t+1]'
        log_ξ[:, :, t] .-= logsumexp(log_ξ[:, :, t]) # normalize
    end
	
    return log_γ, log_ξ, log_ζs
end

# ╔═╡ 84a4bb31-26f8-4a9e-a0b2-f5f8952ef08b
# α, (μ0, κ0, ν0, Σ0) hyperparameters for Dirichlet and NIW priors
function vbem(ys, K, mvn_prior::Exp_MVN, max_iter=100; α=1.0)
	T, D = size(ys)
	
    dirichlet_params = ones(K, K) * α # K X K
	
    # Exp_MVNs = [mvn_prior for _ in 1:K] NEED more randomness
	Random.seed!(111)
	μs = [rand(D) for _ in 1:K]
	κ_0 = 0.1
	ν_0 = D + 1.0
	Σ_0 = Matrix{Float64}(I, D, D)

	Exp_MVNs = [Exp_MVN(μs[i], κ_0, ν_0, Σ_0) for i in 1:K]
	
    for iter in 1:max_iter
        # E-step
        log_γ, log_ξ, _ = vbem_e_step(ys, dirichlet_params, Exp_MVNs)

        # M-step
        dirichlet_params, Exp_MVNs = vbem_m_step(ys, log_γ, log_ξ, α, mvn_prior)

		# Convergence check [TO-DO]
    end

    return dirichlet_params, Exp_MVNs
end

# ╔═╡ 453d68e3-9a04-47c3-9af3-37d2347bfd64
begin
	Random.seed!(111)
	m1 = [-2.0, -3.0]
	m2 = [1.0, 1.0]
	Σ_true = [Matrix{Float64}(0.8 * I, 2, 2) for _ in 1:2]
	
	mvn1 = MvNormal(m1, Σ_true[1])
	mvn2 = MvNormal(m2, Σ_true[2])
	
	A_mvn = [0.8 0.2; 0.2 0.8] 

	π_0 = [1.0, 0.0]
	
	mvnHMM = HMM(π_0, A_mvn, [mvn1, mvn2])

	# test with 2000 data points 
	mvn_data = rand(mvnHMM, 2000)
end;

# ╔═╡ 20f39921-7187-4651-9b42-e1c4fc8f1056
md"""
Ground truth HMM
"""

# ╔═╡ 70c8ef1b-b2a9-4ecb-a8c9-70dd57411a8a
mvnHMM

# ╔═╡ e5cfd369-eb38-4bb9-a38d-e9923b5ec199
md"""
Get Dirichlet and NIW parameters from VBEM
"""

# ╔═╡ c7f6c9b4-b0dc-4da4-9ca4-dd96b4afb640
begin
	# Initialize the NormalWishart prior
	d = 2
	μ_0 = zeros(d)
	κ_0 = 0.1
	ν_0 = d + 1.0
	Σ_0 = Matrix{Float64}(I, d, d)
	
	mvn_prior = Exp_MVN(μ_0, κ_0, ν_0, Σ_0)
end;

# ╔═╡ 29c0c7f7-4199-4a32-8d55-8ccd7759450d
dirichlet_f, niw_f = vbem(mvn_data, 2, mvn_prior, 5)

# ╔═╡ f6d66591-f9a6-48f4-9d40-2fd08a220a38
md"""
### Testing VBEM [Batch] Results
"""

# ╔═╡ 1fdc1902-e892-4b4c-ac2e-5ea2222a6228
A_est = exp.(log_Ã(dirichlet_f))

# ╔═╡ 7333ac70-4f79-45f5-875a-3df38b55052a
μs_est = [niw_f[i].m/niw_f[i].κ for i in 1:2]

# ╔═╡ c6481a69-b6cb-4f90-a557-303eb7e42f09
[(niw_f[i].Σ - niw_f[i].m*(niw_f[i].m)'/niw_f[i].κ)/niw_f[i].ν for i in 1:2]

# ╔═╡ 5b9cd9ce-df0a-49c8-97b2-b0b9144ff323
md"""
#### Test with $K=3$
"""

# ╔═╡ 4ab65b62-2f57-4c7b-a58a-b50b773863d0
begin
	# debug required
	Random.seed!(111)
	m3 = [2.0, 2.0]
	A_mvn3 = [0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.8]
	mvn3 = MvNormal(m3,  Σ_true[1])
	π_3 = [1.0, 0.0, 0.0]
	mvnHMM_k3 = HMM(π_3, A_mvn3, [mvn1, mvn2, mvn3])
	mvn_data_k3 = rand(mvnHMM_k3, 10000)
end;

# ╔═╡ 66a1a7d4-db54-47f7-bdfe-953ba155e35a
mvnHMM_k3

# ╔═╡ 25fa9c45-82d1-44a0-814c-ecf9db6234c4
dirichlet_3, niw_3 = vbem(mvn_data_k3, 3, mvn_prior)

# ╔═╡ da615595-cffd-4bf8-abd3-5498b7a4d202
md"""
Recover transition matrix estimation
"""

# ╔═╡ 4a6712c6-955f-445e-9c68-b09ae5b00d3d
A_est3 = exp.(log_Ã(dirichlet_3))

# ╔═╡ f0a47b5e-40e6-4efe-b5c7-e3cc01270a0a
md"""
Recover mean estimates from natural params:
"""

# ╔═╡ 50b5bdaf-2467-450f-a0c6-80d55a68586c
[niw_3[i].m/niw_3[i].κ for i in 1:3]

# ╔═╡ 1877d3d1-cc43-4271-9542-11d9d1bb7208
md"""
Recover co-variances estimates from natural params:
"""

# ╔═╡ 0c1e1848-67d1-4ebb-9b8c-534f7e0cb3c1
[(niw_3[i].Σ - niw_3[i].m*(niw_3[i].m)'/niw_3[i].κ)/niw_3[i].ν for i in 1:3]

# ╔═╡ 18e35e5b-dd37-4f0a-aa8e-f0aedcb658e2
md"""
### Convergence Check

**ELBO of Gaussian HMM** can be expressed as:

$\begin{align}
\mathcal F &= \langle \ln p(D, S, \mathbf{θ}) \rangle_{q(S, θ)} - \langle \ln q(S, \mathbf{θ}) \rangle_{q(S, θ)} \\
&= \langle \ln p(\mathbf{θ}) \rangle_{q(θ)} - \langle \ln q(\mathbf{θ}) \rangle_{q(θ)} + \langle \ln p(D, S|\mathbf{\theta}) \rangle_{q(S, θ)} - \langle \ln q(S) \rangle_{q(S)}
\end{align}$

Maximising ELBO is equivalent to minimising the KL divergence between $q(S, \mathbf{θ})$ and $p(S, \mathbf{θ}|D)$

According to Beal Chap 3, 3.79, the ELBO can be further simplified after VBE step:

$\begin{align}
\mathcal F &= \langle \ln p(\mathbf{θ}) \rangle_{q(θ)} - \langle \ln q(\mathbf{θ}) \rangle_{q(θ)} + \ln \tilde{Z} \\
&= KL(p(A)|q(A)) + KL(p(ϕ)|q(ϕ)) + \ln \tilde{Z} \\
\end{align}$

where $\tilde{Z} = \prod_{t=1}^T \tilde{ζ_t}$, $\tilde{ζ_t}$ is the normalisation term from the modified forward pass.

Hence, given the forward_log implementation:

$\ln \tilde{Z} = \sum_{t=1}^T \ln \tilde{ζ_t}$
"""

# ╔═╡ 3ac18f00-7e85-4abb-90fc-56b47d9fdd27
md"""
Compute KL divergence of Dirichlet distributed rows of A
"""

# ╔═╡ d0cba4d1-2d02-4524-afd8-3150c5019f83
# to be used for each row of A
function kl_dirichlet(α::Array{Float64, 1}, β::Array{Float64, 1})
    # cf. Beal Appendix A, p261
	α_sum, β_sum = sum(α), sum(β)
    kl = loggamma(β_sum) - loggamma(α_sum) - sum(loggamma.(β) - loggamma.(α))
    kl += sum((α - β) .* (digamma.(α) .- digamma(α_sum)))
    return kl
end

# ╔═╡ 429c5dfd-4f78-4697-b710-777bd5a38240
md"""
Compute KL divergence of MVN emission (using natural parameters of NIW prior)
"""

# ╔═╡ 51a5c3e6-7b0a-4d54-b9d3-5eb6054ec72d
function kl_niw(p::Exp_MVN, q::Exp_MVN)
	D = size(p.m, 1)

	if p.Σ == Matrix{Float64}(I, d, d) # exclude first iteration
		κ_p = p.κ
		m_p = p.m
		ν_p = p.ν
		Σ_p = Matrix{Float64}(I, d, d)
	else
		κ_p = p.κ
		m_p = p.m/κ_p
		ν_p = p.ν - 2.0 - D
		Σ_p = p.Σ - κ_p*m_p*m_p'
	end

	κ_q = q.κ
	m_q = q.m/κ_q
	ν_q = q.ν - 2.0 - D
	Σ_q = q.Σ - κ_q*m_q*m_q'

	Σ_inv = inv(Σ_q)

	kl = -0.5*logdet(Σ_p*Σ_inv)

	kl -= 0.5*tr(I - (Σ_p + (m_p - m_q)*(m_p - m_q)')*Σ_inv)
	
	"""
    kl = 0.5 * (trace(Σ_inv * Σ_p) + (m_p - m_q)' * Σ_inv * (m_p - m_q) - D)
    kl += 0.5 * (logdet(Σ_q) - logdet(Σ_p)) * (ν_p - ν_q)
    
    for j in 1:D
        kl += 0.5 * (ν_p - j) * (digamma((ν_q - j) / 2) - digamma((ν_p - j) / 2)) * (ν_q - ν_p)
    end
	"""
	return kl
end

# ╔═╡ a2e69fa3-17a3-4588-8f65-be6d77b09c4f
function vbem_c(ys, K, mvn_prior::Exp_MVN, max_iter=200; α=1.0, tol=1e-3)
	T, D = size(ys)
	
    dirichlet_params_p = ones(K, K) * α # K X K
	
    # Exp_MVNs = [mvn_prior for _ in 1:K] NEED more randomness
	Random.seed!(111)
	μs = [rand(D) for _ in 1:K]
	κ_0 = 0.1
	ν_0 = D + 1.0
	Σ_0 = Matrix{Float64}(I, D, D)

	Exp_MVNs_p = [Exp_MVN(μs[i], κ_0, ν_0, Σ_0) for i in 1:K]
	elbo_prev = -Inf

    for iter in 1:max_iter
        # E-step
        log_γ, log_ξ, log_ζs = vbem_e_step(ys, dirichlet_params_p, Exp_MVNs_p)

        # M-step
        dirichlet_params, Exp_MVNs = vbem_m_step(ys, log_γ, log_ξ, α, mvn_prior)

		# Convergence check
		kl_A = sum(kl_dirichlet(dirichlet_params_p[i, :], dirichlet_params[i, :]) for i in 1:K)

		kl_niw_ = sum(kl_niw(Exp_MVNs_p[i], Exp_MVNs[i]) for i in 1:K)

		log_Z̃ = sum(log_ζs)

		elbo = kl_A + kl_niw_ + log_Z̃
		
		if abs(elbo - elbo_prev) < tol
			dirichlet_params_p = dirichlet_params
			Exp_MVNs_p = Exp_MVNs
			println("Stopped at iteration: $iter")
            break
		end
		
        elbo_prev = elbo
		dirichlet_params_p = dirichlet_params
		Exp_MVNs_p = Exp_MVNs
    end

    return dirichlet_params_p, Exp_MVNs_p
end

# ╔═╡ 875aa3bc-c4b9-4dce-987d-62dbe1d40d37
dirichlet_c, niw_c = vbem_c(mvn_data, 2, mvn_prior)

# ╔═╡ 289d16d6-528e-4e07-9440-9573d2f71724
A_est_c = exp.(log_Ã(dirichlet_c))

# ╔═╡ 30cbc8bf-f0ae-4514-b4f3-1e8734f38377
[niw_c[i].m/niw_c[i].κ for i in 1:2]

# ╔═╡ 58a00d74-3383-4e98-a4ec-97655801516a
[(niw_c[i].Σ - niw_c[i].m*(niw_c[i].m)'/niw_c[i].κ)/niw_c[i].ν for i in 1:2]

# ╔═╡ fbdd786f-24f2-497d-a804-7af8cf5cb72e
dirichlet_c3, niw_c3 = vbem_c(mvn_data_k3, 3, mvn_prior)

# ╔═╡ 1e8b3bed-943e-4e93-a096-958e1dbdfa6d
A_est_c3 = exp.(log_Ã(dirichlet_c3))

# ╔═╡ 54c2fe20-7b86-4d72-9449-5e942496fdb6
[niw_c3[i].m/niw_c3[i].κ for i in 1:3]

# ╔═╡ 4fad4cf7-20ce-4a7d-bbc0-c91a0d1611d4
[(niw_c3[i].Σ - niw_c3[i].m*(niw_c3[i].m)'/niw_c3[i].κ)/niw_c3[i].ν for i in 1:3]

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HMMBase = "b2b3ca75-8444-5ffa-85e6-af70e2b64fe7"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"

[compat]
Distributions = "~0.25.86"
HMMBase = "~1.0.7"
PDMats = "~0.11.17"
PlutoUI = "~0.7.50"
SpecialFunctions = "~2.2.0"
StatsFuns = "~1.3.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.5"
manifest_format = "2.0"
project_hash = "f5a8953b04eed71931715b96624fcd92f64a25bc"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c6d890a52d2c4d55d326439580c3b8d0875a77d9"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.7"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "485193efd2176b88e6622a39a246f8c5b600e74e"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.6"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "7ebbd653f74504447f1c33b91cd706a69a1b189f"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.14.4"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.1+0"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "49eba9ad9f7ead780bfb7ee319f962c811c6d3b2"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.8"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "da9e1a9058f8d3eec3a8c9fe4faacfb89180066b"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.86"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "0ba171480d51567ba337e5eea4e68a8231b7a2c3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.10"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.HMMBase]]
deps = ["ArgCheck", "Clustering", "Distributions", "Hungarian", "LinearAlgebra", "Random"]
git-tree-sha1 = "47d95dcc06cafd4a1c100bfad64da3ab06ad38c7"
uuid = "b2b3ca75-8444-5ffa-85e6-af70e2b64fe7"
version = "1.0.7"

[[deps.Hungarian]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "371a7df7a6cce5909d6c576f234a2da2e3fa0c98"
uuid = "e91730f6-4275-51fb-a7a0-7064cfbd3b39"
version = "0.6.0"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "67eae2738d63117a196f497d7db789821bce61d1"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.17"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "478ac6c952fddd4399e71d4779797c538d0ff2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.8"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "5bb5129fdd62a2bbbe17c2756932259acf467386"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.50"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "6ec7ac8412e83d57e313393220879ede1740f9ee"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.8.2"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "b8d897fe7fa688e93aef573711cb207c08c9e11e"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.19"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "aadb748be58b492045b4f56166b5188aa63ce549"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.7"

[[deps.URIs]]
git-tree-sha1 = "074f993b0ca030848b897beff716d93aca60f06a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═db588217-5856-47b9-bfa2-637f84d19cd4
# ╟─06692696-8edb-42dd-be97-b1ea41e3d60f
# ╟─ed4603db-b956-4098-85e9-dcd97b132cb4
# ╟─009cfa59-55d5-4d11-8e29-7f09f58a7354
# ╟─4c7b01e4-b201-4605-8bf3-16d404c8be39
# ╟─3da884a5-eacb-4749-a10c-5a6ad6da34b7
# ╟─4a573a35-51b2-4a42-b175-3ba017ef7245
# ╠═ae313199-fe27-478a-b6b2-bee2923b5a54
# ╠═a0fa9433-3ae7-4697-8eea-5d6ddefbf62b
# ╟─e9b7bbd0-2de8-4a5b-af73-3b576e8c54a2
# ╠═6905f394-3da2-4207-a511-888f7521d82a
# ╠═ac620936-66fb-4457-b9fd-9180fc9ba994
# ╟─5d746477-0282-4656-8db4-5a61561d3ccb
# ╠═a4b38c35-e320-439e-b169-b8af974e2635
# ╟─61a5e1fd-5480-4ed1-83ff-d3ad140cbcbc
# ╟─5a1de95b-deef-4199-8992-28fcfe2157b8
# ╠═b4d1bdf3-efd6-4851-9d33-e1564b8ed769
# ╟─54f38d75-aa88-4eb8-983e-e2a3038f910f
# ╟─ebed75fc-0c10-48c7-9352-fc61bc3dcfd8
# ╠═072762bc-1f27-4a95-ad46-ddbdf45292cc
# ╟─c83c2754-5cce-46c7-b8df-f006b70f41e0
# ╟─820a3a70-6df9-4300-8037-21b2d51e8247
# ╟─49d2d576-e58f-4988-a7df-cdc44c1448db
# ╠═9390462c-ee9e-415c-a0cc-3942989ad494
# ╟─bc0d9def-56a6-4c66-9ba4-d3237cdf2343
# ╟─20753ea0-a5b0-403d-ab33-98c90263e314
# ╠═7fda653c-2f37-4901-9676-8340978bca4d
# ╟─43660616-f1a8-434c-afc3-ac200d82425a
# ╠═3c4f3444-2547-4ecd-816c-05b35c73f050
# ╟─48f0208e-8b99-4b68-9982-02fa18bc0ab1
# ╠═268464af-2842-4240-8551-ffc0cc130b70
# ╠═84a4bb31-26f8-4a9e-a0b2-f5f8952ef08b
# ╠═453d68e3-9a04-47c3-9af3-37d2347bfd64
# ╟─20f39921-7187-4651-9b42-e1c4fc8f1056
# ╠═70c8ef1b-b2a9-4ecb-a8c9-70dd57411a8a
# ╟─e5cfd369-eb38-4bb9-a38d-e9923b5ec199
# ╠═c7f6c9b4-b0dc-4da4-9ca4-dd96b4afb640
# ╠═29c0c7f7-4199-4a32-8d55-8ccd7759450d
# ╟─f6d66591-f9a6-48f4-9d40-2fd08a220a38
# ╠═1fdc1902-e892-4b4c-ac2e-5ea2222a6228
# ╠═7333ac70-4f79-45f5-875a-3df38b55052a
# ╠═c6481a69-b6cb-4f90-a557-303eb7e42f09
# ╟─5b9cd9ce-df0a-49c8-97b2-b0b9144ff323
# ╠═4ab65b62-2f57-4c7b-a58a-b50b773863d0
# ╠═66a1a7d4-db54-47f7-bdfe-953ba155e35a
# ╠═25fa9c45-82d1-44a0-814c-ecf9db6234c4
# ╟─da615595-cffd-4bf8-abd3-5498b7a4d202
# ╠═4a6712c6-955f-445e-9c68-b09ae5b00d3d
# ╟─f0a47b5e-40e6-4efe-b5c7-e3cc01270a0a
# ╠═50b5bdaf-2467-450f-a0c6-80d55a68586c
# ╟─1877d3d1-cc43-4271-9542-11d9d1bb7208
# ╠═0c1e1848-67d1-4ebb-9b8c-534f7e0cb3c1
# ╟─18e35e5b-dd37-4f0a-aa8e-f0aedcb658e2
# ╟─3ac18f00-7e85-4abb-90fc-56b47d9fdd27
# ╠═d0cba4d1-2d02-4524-afd8-3150c5019f83
# ╟─429c5dfd-4f78-4697-b710-777bd5a38240
# ╠═51a5c3e6-7b0a-4d54-b9d3-5eb6054ec72d
# ╠═a2e69fa3-17a3-4588-8f65-be6d77b09c4f
# ╠═875aa3bc-c4b9-4dce-987d-62dbe1d40d37
# ╠═289d16d6-528e-4e07-9440-9573d2f71724
# ╠═30cbc8bf-f0ae-4514-b4f3-1e8734f38377
# ╠═58a00d74-3383-4e98-a4ec-97655801516a
# ╠═fbdd786f-24f2-497d-a804-7af8cf5cb72e
# ╠═1e8b3bed-943e-4e93-a096-958e1dbdfa6d
# ╠═54c2fe20-7b86-4d72-9449-5e942496fdb6
# ╠═4fad4cf7-20ce-4a7d-bbc0-c91a0d1611d4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
