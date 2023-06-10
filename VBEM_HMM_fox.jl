### A Pluto.jl notebook ###
# v0.19.25

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
end

# ╔═╡ 15eaf87f-865c-49e4-82a7-bcd8d648d831
begin
	using Flux: onehot
	function one_hot_yt(y_t, labels)
	    y_t = y_t[1] # Extract value if y_t is an array
	    one_hot_yt = onehot(y_t, labels)
	    return Int.(one_hot_yt)
	end
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

## Compare with MLE Baum-Welch
Unlike the Baum-Welch M-step, which uses frequency counting to extract the maxium of the MLE, resulting in point estimates of π, A and MVN parameters μ, Σ. 

The VBEM M-step, we update the parameters of the variational posterior distributions for the HMM parameters. We use the expected sufficient statistics calculated during the E-step to update the parameters of the variational posterior distributions.

The VBEM M-step incorporates prior knowledge about the HMM parameters through the use of conjugate priors. 

	Transition matrix A, we use Dirichlet priors on rows of A, 

	Multivariate Gaussian (MVN) emissions, we use a Normal (inverse) Wishart prior. 

These priors choice can influence the updates of the variational posterior distributions, leading to a Bayesian estimation of the parameters. 

The VBEM M-step provides a form of regularization due to the incorporation of priors. In theory, this can prevent overfitting, especially when there is limited data, and lead to more robust estimates of the HMM parameters. 
"""

# ╔═╡ a98e5e92-d4e8-4ac2-a4ea-b21006fe1558
md"""
# VBEM HMM
"""

# ╔═╡ e74c5c83-9cf5-4d4a-ad9f-2785e18bc3ef
md"""
For conventional HMM setup, priors over $π$, the rows of $A$, and the rows of $C$ are Dirichlet distributions:

$p(π) = \mathcal Dir(π_1 , . . . , π_K | u^{(π)} )$

$p(A) = \prod_{j=1}^K \mathcal Dir( a_{j, 1}, ...,  a_{j, K} | u^{(A)})$


$p(B) = \prod_{j=1}^K \mathcal Dir( b_{j, 1}, ...,  b_{j, D} | u^{(B)})$


Whilst there are many possible choices, Dirichlet distributions have the advantage that they are conjugate to the complete-data likelihood.

## VB_M Variational update
	dirichlet_params_() = ()_prior + suffstats_()

$q(π) = \mathcal Dir(π_1 , . . . , π_K | w^{(π)})$

$w_j^{(π)} = w_j^{(π)} + γ(s_1, j)$

$q(A) = \prod_{j=1}^K \mathcal Dir(a_{j, 1}, ...,  a_{j, K} | w^{(A)})$

$w_{j, j'}^{(A)} = u_{j'}^{(A)} + \sum_{t=2}^T ξ(s_t, s_{t+1})$

$q(B) = \prod_{j=1}^K \mathcal Dir(b_{j, 1}, ...,  b_{j, D} | w^{(B)})$

$w_{j, q}^{(B)} = u_{q}^{(B)} + \sum_{t=1}^T γ(s_t, j)p(y_t , q)$
"""

# ╔═╡ 93d378e6-b6bf-4065-b483-5010a54b04b4
# prior struct for HMM (discrete emission)
struct U_Prior
	u_π
	u_A
	u_B
end

# ╔═╡ 99234c60-3250-4c63-82ee-f15b6299d856
# log_γ, log_ξ are sufficient stats from VBEM E-step
function vbem_m(ys, labels, log_γ, log_ξ, prior::U_Prior)
	K, T = size(log_γ)
	V = length(labels)

    # Update Dirichlet parameters [prior + sufficient stats]
	γ_counts = exp.(log_γ)
	
	w_π = prior.u_π .+ γ_counts[:, 1]
	w_A = prior.u_A .+ sum(exp.(log_ξ), dims=3)[:, :, 1]
	w_B = prior.u_B
	
    for t in 1:T
        # Apply one-hot encoding to the t-th observation
        yt_o = one_hot_yt(ys[:, t], labels)
		w_B += γ_counts[:, t] * yt_o' #broadcast
		
		"""
        for k in 1:K
            for v in 1:V
                # Add the weighted count for the t-th time step
                w_B[k, v] += γ_counts[k, t] * yt_o[v]
            end
        end
		"""
    end

    return w_π, w_A, w_B
end

# ╔═╡ a4122178-c8c6-4e67-9a9a-ccf95d3bc96f
md"""
Test M-step (uni-variate y)
"""

# ╔═╡ 5baf0c31-717c-4e0c-84a5-ebe9da192dbd
function log_π̃(w_π)
	log_π_exp = digamma.(w_π) .- digamma(sum(w_π))
    return log_π_exp
end

# ╔═╡ 072762bc-1f27-4a95-ad46-ddbdf45292cc
# E_q[ln(A)] # RE-USE
function log_Ã(w_A)
	row_sums = sum(w_A, dims=2)
    log_A_exp = digamma.(w_A) .- digamma.(row_sums) # broadcasting
    return log_A_exp
end

# ╔═╡ 495d7a29-5b02-49d8-b376-bf1d088026c1
function log_B̃(w_B)
	row_sums = sum(w_B, dims=2)
    log_B_exp = digamma.(w_B) .- digamma.(row_sums) # broadcasting
    return log_B_exp
end

# ╔═╡ 261c15ff-a80a-43df-bf51-33591d914aea
md"""
## VB_E
"""

# ╔═╡ e7e199fd-cc7f-483f-b588-00013b9069d1
function forward_l(ys, labels, log_π̃, log_Ã, log_B̃)
	D, T = size(ys)
	K = length(log_π̃) # K Hidden states
	log_alpha = zeros(K, T)
	
	l_ζs = zeros(T) # used for ELBO 

	# t = 1
	y_1_o = one_hot_yt(ys[:, 1], labels)
	log_alpha[:, 1] = log_π̃ + log_B̃ * y_1_o
	
	l_ζs[1] = logsumexp(log_alpha[:, 1])
    log_alpha[:, 1] .-= l_ζs[1] 

    # Iterate through the remaining time steps
    for t in 2:T
		yt_o = one_hot_yt(ys[:, t], labels)

		#log_alpha[:, t] = log_B̃ * yt_o .+ logsumexp(log_Ã .+ log_alpha[:, t-1], dims=1)'

        for k in 1:K
            log_alpha[k, t] = logsumexp(log_alpha[:, t-1] .+ log_Ã[:, k]) + log_B̃[k, findfirst(Bool.(yt_o))]
        end
	
        # Normalize log_alpha for t > 1
		l_ζs[t] = logsumexp(log_alpha[:, t])
        log_alpha[:, t] .-= l_ζs[t]
    end

    return log_alpha, l_ζs
end

# ╔═╡ a2019816-e7b9-4eda-be76-9b3ebbb990e3
function backward_l(ys, labels, log_Ã, log_B̃)
	D, T = size(ys)
    K = size(log_Ã, 1)
    log_beta = zeros(K, T)
	
	# casino data
	for t in T-1:-1:1
		yt_o = one_hot_yt(ys[:, t+1], labels)
		
    	#log_beta[:, t] = logsumexp(log_Ã .+ log_B̃ * yt_o .+ log_beta[:, t+1], dims=1)

		for k in 1:K
			log_beta[k, t] = logsumexp(log_Ã[k, :] .+ log_B̃[:, findfirst(Bool.(yt_o))] .+ log_beta[:, t+1])
		end
		log_beta[:, t] .-= logsumexp(log_beta[:, t]) 
	end

	return log_beta
end

# ╔═╡ 2b01dd4d-fba9-49dd-9802-67cea335d49c
function vbem_e(ys, P, w_π, w_A, w_B)
	D, T = size(ys)
    K = length(w_π)
	
	log_π = log_π̃(w_π)
	log_A = log_Ã(w_A)
	log_B = log_B̃(w_B)
		
	log_α, log_ζs = forward_l(ys, P, log_π, log_A, log_B)
	log_β = backward_l(ys, P, log_A, log_B)
	
    log_γ = log_α .+ log_β
	log_γ .-= logsumexp(log_γ, dims=1)
	
	log_ξ = zeros(K, K, T-1)
	
    for t in 1:T-1
		y_to = one_hot_yt(ys[:, t], P)
		log_ξ[:, :, t] = log_α[:, t] .+ log_A .+ log_B[:, findfirst(Bool.(y_to))]' .+ log_β[:, t+1]'
        log_ξ[:, :, t] .-= logsumexp(log_ξ[:, :, t]) 
    end
	
    return log_γ, log_ξ, log_ζs
end

# ╔═╡ c38bfc75-4281-4b52-aeac-0a9dbf7f8776
md"""
## VBEM
"""

# ╔═╡ 3da884a5-eacb-4749-a10c-5a6ad6da34b7
md"""
# VBEM for HMM with MVN Emission
The following (batch) VBEM follows largely from Beale's 2003 paper and Fox et al's paper in 2014

Using the following structured mean-field approximation:

$$p(A, \{ϕ_k\}, S | Y) \approx q(A) q(\{ϕ_k\}) q(S)$$

where $\{ϕ_k\}$ are parameters Normal (inverse) Wishart distributed, $ϕ_k = \{μ_k, Σ_k\}$, mean vector and co-variance matrix.
"""

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

# ╔═╡ d7aaea92-8cf6-4ac3-9929-559ab88d65e3
md"""
### Test M-Step
"""

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
Test m-step update on mean
"""

# ╔═╡ 6905f394-3da2-4207-a511-888f7521d82a
μ_m = post.m/post.κ # recover mean from natural param

# ╔═╡ 5d746477-0282-4656-8db4-5a61561d3ccb
md"""
Test m-step update on Co-variance
"""

# ╔═╡ ac620936-66fb-4457-b9fd-9180fc9ba994
Σ_m = post.Σ - post.κ*μ_m*μ_m'

# ╔═╡ a4b38c35-e320-439e-b169-b8af974e2635
Σ_m / (post.ν - 2 - 2) # recover from natural param

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

# ╔═╡ 137cafd1-fe56-4db8-9e28-f1e133887be4
md"""
Re-use from discrete HMM
"""

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

# ╔═╡ fcf0b465-7d59-49af-97df-dcf379305671
md"""
## VBEM
"""

# ╔═╡ 84a4bb31-26f8-4a9e-a0b2-f5f8952ef08b
# α, (μ0, κ0, ν0, Σ0) hyperparameters for Dirichlet and NIW priors
function vbem(ys, K::Int64, mvn_prior::Exp_MVN, max_iter=100; α=1.0)
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

# ╔═╡ f6d66591-f9a6-48f4-9d40-2fd08a220a38
md"""
# Testing VBEM [Batch] Results
"""

# ╔═╡ 171ea1db-a425-4cf2-93a8-01e94ff329cb
md"""
## Test $K=2$
"""

# ╔═╡ 453d68e3-9a04-47c3-9af3-37d2347bfd64
begin
	Random.seed!(111)
	m1 = [-2.0, -3.0]
	m2 = [1.0, 1.0]
	Σ_true = [Matrix{Float64}(0.8 * I, 2, 2) for _ in 1:2]
	mvn1 = MvNormal(m1, Σ_true[1])
	mvn2 = MvNormal(m2, Σ_true[2])
	A_mvn = [0.8 0.2; 0.1 0.9] 
	π_0 = [1.0, 0.0]

	# use HMMBase to construct test HMM
	mvnHMM = HMM(π_0, A_mvn, [mvn1, mvn2])

	# generate test data
	s_true, mvn_data = rand(mvnHMM, 2000, seq=true)
end;

# ╔═╡ c6dbe013-c810-4be3-b7d8-a072bd6432c1
mvn_data # T X D matrix

# ╔═╡ 20f39921-7187-4651-9b42-e1c4fc8f1056
md"""
**Ground truth** HMM
"""

# ╔═╡ 70c8ef1b-b2a9-4ecb-a8c9-70dd57411a8a
mvnHMM

# ╔═╡ c7f6c9b4-b0dc-4da4-9ca4-dd96b4afb640
begin
	d = 2
	μ_0 = zeros(d)
	κ_0 = 0.1
	ν_0 = d + 1.0
	Σ_0 = Matrix{Float64}(I, d, d)
	mvn_prior = Exp_MVN(μ_0, κ_0, ν_0, Σ_0)
	dirichlet_f, niw_f = vbem(mvn_data, 2, mvn_prior, 5) # 5 iter
end;

# ╔═╡ cfa4a7d2-3a6a-40bd-80fb-449f91584ed9
md"""
### Infer model parameters
"""

# ╔═╡ 1fdc1902-e892-4b4c-ac2e-5ea2222a6228
A_est = exp.(log_Ã(dirichlet_f))

# ╔═╡ 7333ac70-4f79-45f5-875a-3df38b55052a
μs_est = [niw_f[i].m/niw_f[i].κ for i in 1:2]

# ╔═╡ c6481a69-b6cb-4f90-a557-303eb7e42f09
[(niw_f[i].Σ - niw_f[i].m*(niw_f[i].m)'/niw_f[i].κ)/niw_f[i].ν for i in 1:2]

# ╔═╡ da6d3c66-4f8f-4d18-84ec-c84be9153474
md"""
### Hidden state inference
"""

# ╔═╡ 0ff7f6fc-81a7-4eac-b933-5c8e71b6843b
let
	log_A = log_Ã(dirichlet_f)
	log_α, _ = forward_log(mvn_data, log_A, niw_f)
	log_β = backward_log(mvn_data, log_A, niw_f)
	
    # Compute log_ξ and log_γ [identical to Baum-Welch E-step]
    log_γ = log_α + log_β
	log_γ .-= logsumexp(log_γ, dims=1)
	γ = exp.(log_γ)

	[argmax(γ[:, t])[1] for t in 1:size(γ, 2)] #
end

# ╔═╡ a33e7ee9-f8fb-4483-8b15-21d098e9b65d
s_true # true hidden states

# ╔═╡ 36b7ede6-6ed5-4daa-8a74-ccf6d049c3fa
s_f = exp.(vbem_e_step(mvn_data, dirichlet_f, niw_f)[1])

# ╔═╡ 10b8f741-ed45-4cd3-9817-a63f2e4a3aaf
let
	ss = [x[1]*2 + x[2]*1 for x in eachcol(s_f)]
	mad = mean(abs.(s_true - ss))
	rmse = sqrt(mean((s_true - ss).^2))
	println("MAD: ", mad)
	println("RMSE: ", rmse)
end

# ╔═╡ 5b9cd9ce-df0a-49c8-97b2-b0b9144ff323
md"""
## Test $K=3$
"""

# ╔═╡ 4ab65b62-2f57-4c7b-a58a-b50b773863d0
begin
	Random.seed!(111)
	m3 = [2.0, 2.0]
	A_mvn3 = [0.8 0.1 0.1; 0.1 0.8 0.1; 0.1 0.1 0.8]
	mvn3 = MvNormal(m3,  Σ_true[1])
	π_3 = [1.0, 0.0, 0.0]
	mvnHMM_k3 = HMM(π_3, A_mvn3, [mvn1, mvn2, mvn3])
	s3_true, mvn_data_k3 = rand(mvnHMM_k3, 8000, seq = true)
	dirichlet_3, niw_3 = vbem(mvn_data_k3, 3, mvn_prior)
end;

# ╔═╡ a6143756-dafc-41d3-9a46-8a5c23dfd87e
md"""
**Ground truth** HMM $K=3$
"""

# ╔═╡ 66a1a7d4-db54-47f7-bdfe-953ba155e35a
mvnHMM_k3

# ╔═╡ 1abdb6ea-8de2-4df2-b278-03335639a202
md"""
### Infer model parameters
"""

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
[niw_3[i].m/niw_3[i].κ for i in 1:3] # index 1 and 2 also mixed up, like K=2 case

# ╔═╡ 1877d3d1-cc43-4271-9542-11d9d1bb7208
md"""
Recover co-variances estimates from natural params:
"""

# ╔═╡ 0c1e1848-67d1-4ebb-9b8c-534f7e0cb3c1
[(niw_3[i].Σ - niw_3[i].m*(niw_3[i].m)'/niw_3[i].κ)/niw_3[i].ν for i in 1:3]

# ╔═╡ d5bbbd58-a4c0-466f-987a-a6f32a80a775
md"""
### Hidden state inference
"""

# ╔═╡ a3a23fb9-baad-4d04-b8da-82ce2267f0b7
s3_true

# ╔═╡ ae6352cf-0343-4b4c-9fa7-8ba3792eafc7
s_f3 = exp.(vbem_e_step(mvn_data_k3, dirichlet_3, niw_3)[1])

# ╔═╡ 8f21e2cd-5f8b-4242-885b-9d42b22fad74
s_f3[:, 1000] # index of 1 and 2 seemingly reversed? 

# ╔═╡ 9400ad3d-f2af-418e-971e-6031c7164c78
let
	ss = [x[1]*2 + x[2]*1 + x[3]*3 for x in eachcol(s_f3)]
	mad = mean(abs.(s3_true - ss))
	rmse = sqrt(mean((s3_true - ss).^2))
	println("MAD: ", mad)
	println("RMSE: ", rmse)
	ss
end

# ╔═╡ 18e35e5b-dd37-4f0a-aa8e-f0aedcb658e2
md"""
# Convergence Check

## ELBO Computation
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
# to be used for each row of A, α -> p(A), β -> q(A)
function kl_dirichlet(α::Array{Float64, 1}, β::Array{Float64, 1})
    # cf. Beal Appendix A, p261
	
	α_sum, β_sum = sum(α), sum(β)
	
    #kl = -loggamma(β_sum) + loggamma(α_sum) - sum(loggamma.(α) - loggamma.(β))
    #kl += sum((α - β) .* (digamma.(α) .- digamma(α_sum)))

	kl = loggamma(β_sum) - loggamma(α_sum) + sum(loggamma.(α) - loggamma.(β))
	kl += sum((β - α) .* (digamma.(β) .- digamma(β_sum)))
    return kl
end

# ╔═╡ e4991173-290e-4a28-838b-f2afbec67c93
function vbem_hmm(ys, K, prior::U_Prior, max_iter=500, r_seed=69, tol=5e-3)
	D, T = size(ys)
	P = sort(unique(ys[:])) # labels
	V = length(P)
	
	# random initialisation (sensitive start
	Random.seed!(r_seed)
	w_π = rand(Dirichlet(ones(K)))
    w_A = [rand(Dirichlet(ones(K))) for _ in 1:K]
    w_A = hcat(w_A...)'
    w_B = [rand(Dirichlet(ones(V))) for _ in 1:K]
    w_B = hcat(w_B...)'
	
	elbo_prev = -Inf
	
	for iter in 1:max_iter
        # E-step
        log_γ, log_ξ, log_ζs = vbem_e(ys, P, w_π, w_A, w_B)

        # M-step
        w_π, w_A, w_B = vbem_m(ys, P, log_γ, log_ξ, prior)

		# Convergence check
		kl_π = kl_dirichlet(prior.u_π, w_π)
		kl_A = sum(kl_dirichlet(prior.u_A[i, :], w_A[i, :]) for i in 1:K)
		kl_B = sum(kl_dirichlet(prior.u_B[i, :], w_B[i, :]) for i in 1:K)
		log_Z̃ = sum(log_ζs)

		elbo = kl_π + kl_A + kl_B + log_Z̃
		
		if abs(elbo - elbo_prev) < tol
			println("Stopped at iteration: $iter")
            break
		end
		
        elbo_prev = elbo
		
		if (iter == max_iter)
			println("Warning: VB has not necessarily converged at $max_iter iterations")
		end

    end

    return w_π, w_A, w_B
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

	Σ_inv = inv(Σ_p)

	kl = -0.5*logdet(Σ_q*Σ_inv)
	kl -= 0.5*tr(I - (Σ_q + (m_p - m_q)*(m_p - m_q)')*Σ_inv)
	
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

# ╔═╡ 482e735a-610c-414c-83ac-0e1e1e2d4d86
md"""
## VBEM with convergence check
"""

# ╔═╡ 99c2a566-38a2-40f0-9184-d592ec79694e
md"""
### K = 2
"""

# ╔═╡ 875aa3bc-c4b9-4dce-987d-62dbe1d40d37
dirichlet_c, niw_c = vbem_c(mvn_data, 2, mvn_prior)

# ╔═╡ 289d16d6-528e-4e07-9440-9573d2f71724
A_est_c = exp.(log_Ã(dirichlet_c))

# ╔═╡ 30cbc8bf-f0ae-4514-b4f3-1e8734f38377
[niw_c[i].m/niw_c[i].κ for i in 1:2]

# ╔═╡ 58a00d74-3383-4e98-a4ec-97655801516a
[(niw_c[i].Σ - niw_c[i].m*(niw_c[i].m)'/niw_c[i].κ)/niw_c[i].ν for i in 1:2]

# ╔═╡ 9365ad98-d6ff-424f-94ff-6c754315417c
md"""
### K = 3
"""

# ╔═╡ fbdd786f-24f2-497d-a804-7af8cf5cb72e
dirichlet_c3, niw_c3 = vbem_c(mvn_data_k3, 3, mvn_prior)

# ╔═╡ 1e8b3bed-943e-4e93-a096-958e1dbdfa6d
A_est_c3 = exp.(log_Ã(dirichlet_c3))

# ╔═╡ 54c2fe20-7b86-4d72-9449-5e942496fdb6
[niw_c3[i].m/niw_c3[i].κ for i in 1:3]

# ╔═╡ 4fad4cf7-20ce-4a7d-bbc0-c91a0d1611d4
[(niw_c3[i].Σ - niw_c3[i].m*(niw_c3[i].m)'/niw_c3[i].κ)/niw_c3[i].ν for i in 1:3]

# ╔═╡ 4569a084-ba82-4e12-856e-719ebc3cecb3
md"""
# Appendix
"""

# ╔═╡ 5597ece4-8184-4495-a2f9-a8ec1b7c5d2c
md"""
## MLE/ Baum-Welch (Casino)
"""

# ╔═╡ df01dca6-6c72-40bc-92d9-d38b5938f09e
md"""
Generate data using HMMBase
"""

# ╔═╡ 45ffda31-984f-4144-8b3f-7c452ed69d06
begin
	die1 = Categorical(1/6 * ones(6))
	die2 = Categorical([1/10, 1/10, 1/10, 1/10, 1/10, 1/2])
	A_casino = [0.9 0.1; 0.1 0.9]
	π_i = [1.0, 0.0]
	casinoHMM = HMM(π_i, A_casino, [die1, die2])
	B_casino = hcat(die1.p, die2.p)'
	Random.seed!(99)
	true_coins, c_data = rand(casinoHMM, 2000, seq=true)
	c_data = Int.(c_data) 
end;

# ╔═╡ 6ddf3db6-11f2-4cbc-a71e-e818ca1f27f4
let
	u = U_Prior(ones(2) .* 0.01 ./2 , ones(2, 2) .* 0.01 ./2 , ones(2, 6) .* 0.01 ./2)
	
	w_π, w_A, w_B = vbem_hmm(c_data', 2, u)
	
	P = sort(unique(c_data'[:]))
	s_f = exp.(vbem_e(c_data', P, w_π, w_A, w_B)[1])
	
	ss = [x[1]*1 + x[2]*2 for x in eachcol(s_f)]
	mad = mean(abs.(true_coins - ss))
	rmse = sqrt(mean((true_coins - ss).^2))
	
	println("MAD: ", mad)
	println("RMSE: ", rmse)
	
	π_vb, A_vb, B_vb = exp.(log_π̃(w_π)), exp.(log_Ã(w_A)), exp.(log_B̃(w_B))
end

# ╔═╡ e7837cab-749c-43e4-bb82-c9f87afde848
π_i, A_casino, B_casino

# ╔═╡ 276c73b9-0699-45da-b35c-317f42d18227
c_data'

# ╔═╡ f674e936-f124-48b6-bd3b-cce7f9b7e6b4
function forward_mle(ys, π, A, B)
    D, T = size(ys)
    K = length(π)
    
	log_alpha = zeros(K, T)

	# casino data - not one-hot encoded
	log_alpha[:, 1] = log.(π) .+ log.(B[:, ys[1]])
	log_alpha[:, 1] .-= logsumexp(log_alpha[:, 1])
	
	for t in 2:T
		for j in 1:K
			log_alpha[j, t] = logsumexp(log_alpha[:, t-1] .+ log.(A[:, j])) + log(B[j, ys[t]])
		end

		log_alpha[:, t] .-= logsumexp(log_alpha[:, t]) # normalize
	end
	
    return log_alpha
end

# ╔═╡ 5b0ccc1c-ea6e-439d-9c04-c24dde87a22f
function backward_mle(ys, A, B)
    D, T = size(ys)
    K = size(A, 1)
    log_beta = zeros(K, T)
	
	# casino data - not one-hot encoded
	for t in T-1:-1:1
		for i in 1:K
			log_beta[i, t] = logsumexp(log.(A[i, :]) .+ log.(B[:, ys[:, t+1]]) .+ 							   log_beta[:, t+1])
		end
		
		log_beta[:, t] .-= logsumexp(log_beta[:, t]) 
	end

	return log_beta
end

# ╔═╡ 09d078fa-2346-44fd-9e33-32203fc88b23
forward(casinoHMM, c_data)[1]', backward(casinoHMM, c_data)[1]'

# ╔═╡ 69e878f5-f893-424e-b059-ca848a1dd23f
exp.(forward_mle(c_data', π_i, A_casino, B_casino)), exp.(backward_mle(c_data', A_casino, B_casino))

# ╔═╡ 6a78f8a9-0d19-4eaa-8e2a-ce9290dbea15
md"""
Testing forward, backward
"""

# ╔═╡ 5fd43cac-f673-4218-b593-7453ffbd0c64
function e_mle(ys, π, A, B)
    D, T = size(ys)
    K = size(A, 1)
	
	log_α = forward_mle(ys, π, A, B)
    log_β = backward_mle(ys, A, B)
	
    log_γ = log_α .+ log_β
    log_γ .-= logsumexp(log_γ, dims=1) 

    log_ξ = zeros(K, K, T - 1)
    for t in 1:(T - 1)
		log_ξ[:, :, t] = log_α[:, t] .+ log.(A) .+ log.(B)[:, ys[t+1]]' .+ log_β[:, t+1]'
        log_ξ[:, :, t] .-= logsumexp(log_ξ[:, :, t]) 
    end

    return log_γ, log_ξ
end

# ╔═╡ f8fc9f13-e00f-4da7-b244-7bb497a1c669
let
	log_γ, log_ξ = e_mle(c_data', π_i, A_casino, B_casino)

	# symmetric prior, fixed strength f = 0.1 scaled by K
	u = U_Prior(ones(2) .* 0.1 ./2 , ones(2, 2) * 0.1 ./ 2, ones(2, 6) * 0.1 ./2)
	
	P = sort(unique(c_data'[:]))

	w_π, w_A, w_B = vbem_m(c_data', P, log_γ, log_ξ, u::U_Prior)

	# π̃, Ã, B̃
	exp.(log_π̃(w_π)), exp.(log_Ã(w_A)), exp.(log_B̃(w_B))
end

# ╔═╡ 15ebed48-40f0-4c53-aaf3-e56cf42f0a74
let
	log_γ, log_ξ = e_mle(c_data', π_i, A_casino, B_casino)

	# symmetric prior, fixed strength f = 0.1 scaled by K
	u = U_Prior(ones(2) .* 0.1 ./2 , ones(2, 2) * 0.1 ./ 2, ones(2, 6) * 0.1 ./2)
	
	P = sort(unique(c_data'[:]))

	w_π, w_A, w_B = vbem_m(c_data', P, log_γ, log_ξ, u::U_Prior)

	log_π = log_π̃(w_π)
	log_B = log_B̃(w_B)
	y_1_o = one_hot_yt(c_data'[:, 1], P)
	
	log_f1 = log_π + log_B * y_1_o
	
	log_f1 .-= logsumexp(log_f1)

	log_ff1 = log_π .+ log_B[:, findfirst(Bool.(y_1_o))]
	log_ff1 .-= logsumexp(log_ff1)

	log_f1, log_ff1
end

# ╔═╡ 1e44a7f5-75bc-4476-81b6-54acf093fa51
let
	# use MLE e-step results
	log_γ, log_ξ = e_mle(c_data', π_i, A_casino, B_casino)
	
	# symmetric prior, fixed strength f = 0.1 scaled by K
	u = U_Prior(ones(2) .* 0.1 ./2 , ones(2, 2) * 0.1 ./ 2, ones(2, 6) * 0.1 ./2)
	
	P = sort(unique(c_data'[:]))
	w_π, w_A, w_B = vbem_m(c_data', P, log_γ, log_ξ, u::U_Prior)

	log_γ, log_ξ, _ = vbem_e(c_data', P, w_π, w_A, w_B)
	w_π, w_A, w_B = vbem_m(c_data', P, log_γ, log_ξ, u::U_Prior)
end

# ╔═╡ d5bf0a63-4d30-463f-ac3e-00cdab3c7331
function m_mle(log_γ, log_ξ, ys, K)
   	D, T = size(ys)
    P = length(unique(vec(ys)))

    # Update initial state probabilities - π
    log_π_new = log_γ[:, 1]

    # Update state transition probabilities - A
    log_A_numerator = logsumexp(log_ξ, dims=3)[:, :, 1]
	log_A_denominator = logsumexp(log_γ[:, 1:end-1], dims=2)
	log_A_new = log_A_numerator .- log_A_denominator

    # Update emission probabilities - B
    log_B_new = zeros(K, P)
	
	for j in 1:P
	    obs_filter = (ys' .== j) # uni-var
	    log_gamma_obs = logsumexp(log_γ[:, obs_filter], dims=2)
	    log_gamma_total = logsumexp(log_γ, dims=2)
		log_B_new[:, j] = log_gamma_obs .- log_gamma_total
	end
	
	log_B_new .-= logsumexp(log_B_new, dims=2)

    return exp.(log_π_new), exp.(log_A_new), exp.(log_B_new)
end

# ╔═╡ a3a1545f-9af4-409b-9b55-d7a62bdf5c5e
function em_mle(ys, K, max_iter=100)
    D, T = size(ys)
    P = length(unique(vec(ys)))
    # not one-hot encoded
	
	Random.seed!(111)
    # Initialize model parameters randomly
    π = rand(Dirichlet(ones(K)))
	
    A = [rand(Dirichlet(ones(K))) for _ in 1:K]
    A = hcat(A...)'
	
    B = [rand(Dirichlet(ones(P))) for _ in 1:K]
    B = hcat(B...)'

    for _ in 1:max_iter
        # E-step
        γ, ξ = e_mle(ys, π, A, B)
        
        # M-step
        π, A, B = m_mle(γ, ξ, ys, K)
    end
    
    return π, A, B
end

# ╔═╡ c0af3e5c-ad73-4fe2-92d3-fdecf20c5512
oh_y3 = one_hot_yt(c_data'[3], sort(unique(c_data')))

# ╔═╡ 1d41f8ef-565d-41d6-955e-0e7fdbabbfc6
oh_y7 = one_hot_yt(c_data'[7], unique(c_data'))

# ╔═╡ 22cd1223-dadd-47a4-8d8b-686b554b4869
c_data'[:, 1]

# ╔═╡ 18cc49b7-86d5-4f8d-a235-16b02d0036e3
c_data'[1]

# ╔═╡ 35fc03f2-21b2-452c-abc2-216579817fc8
function log_dot(vec1, vec2)
    return logsumexp(log(vec1[i]) + log(vec2[i]) for i in 1:length(vec1))
end

# ╔═╡ 096ccd02-4844-466d-8532-cbb7cda80910
function forward_oh(ys, P, π, A, B)
    D, T = size(ys)
    K = length(π)
	log_alpha = zeros(K, T)
	y_1_oh = one_hot_yt(ys[:, 1], P)

	for i in 1:K
		log_alpha[i, 1] = log(π[i]) + log_dot(B[i, :], y_1_oh)
	end
	
	log_alpha[:, 1] .-= logsumexp(log_alpha[:, 1])

	# optim?
	for t in 2:T
		y_t = one_hot_yt(ys[:, t], P)
		for i in 1:K
			log_alpha[i, t] = log_dot(B[i, :], y_t) + logsumexp(log(A[j, i]) + log_alpha[j, t-1] for j in 1:K)
		end
		
		log_alpha[:, t] .-= logsumexp(log_alpha[:, t])
	end
	
    return log_alpha
end

# ╔═╡ f9b02790-d6c5-4d69-896f-cba1b6c622b7
exp.(forward_oh(c_data', unique(c_data'), π_i, A_casino, B_casino))

# ╔═╡ a9393ab8-1334-48a6-a47c-4bcfaab9929e
function backward_oh(ys, P, A, B)
	D, T = size(ys)
    K = size(A, 1)
    log_beta = zeros(K, T)

	# optim?
	for t in T-1:-1:1
		y_t = one_hot_yt(ys[:, t+1], P)
		for i in 1:K
			log_beta[i, t] = logsumexp([log(A[i, j]) + log_dot(B[j, :], y_t) + log_beta[j, t+1] for j in 1:K])
		end
		
		log_beta[:, t] .-= logsumexp(log_beta[:, t]) 
	end

	return log_beta
end

# ╔═╡ cda23c77-a219-4bbd-a4f0-22e43f8037d9
exp.(backward_oh(c_data', unique(c_data'), A_casino, B_casino))

# ╔═╡ 51a60031-6636-47d3-a447-058ead85229d
function e_oh(ys, π, A, B)
    D, T = size(ys)
    K = size(A, 1)
	P = sort(unique(ys[:]))
	
	log_α = forward_oh(ys, P, π, A, B)
    log_β = backward_oh(ys, P, A, B)
	
    log_γ = log_α .+ log_β
    log_γ .-= logsumexp(log_γ, dims=1) 

    log_ξ = zeros(K, K, T - 1)

	# optim?
    for t in 1:(T - 1)
		y_t = one_hot_yt(ys[:, t+1], P)
		for i in 1:K
            for j in 1:K
				log_ξ[i, j, t] = log_α[i, t] + log(A[i, j]) + log_dot(B[j, :], y_t) + log_β[j, t+1]
			end
		end
        log_ξ[:, :, t] .-= logsumexp(log_ξ[:, :, t]) 
    end

    return log_γ, log_ξ
end

# ╔═╡ 5f4a79c6-b319-41a9-a35f-094fbdfe68b6
md"""
Test E-step (with one-hot encoding)
"""

# ╔═╡ 5a264cf0-1d7b-468e-91fa-467e9ecb21bf
e_mle(c_data', π_i, A_casino, B_casino), e_oh(c_data', π_i, A_casino, B_casino)

# ╔═╡ 82fbde19-1a8c-4b6a-94db-ac49b59e2ac4
function m_oh(log_γ, log_ξ, ys)
   	D, T = size(ys)
	labels = sort(unique(ys[:]))
	P = length(labels) 
	K, _ = size(log_γ)
    # Update initial state probabilities - π
    log_π_new = log_γ[:, 1]

    # Update state transition probabilities - A
    log_A_numerator = logsumexp(log_ξ, dims=3)[:, :, 1]
	log_A_denominator = logsumexp(log_γ[:, 1:end-1], dims=2)
	log_A_new = log_A_numerator .- log_A_denominator

    # Update emission probabilities - B
	B_new = zeros(K, P)
    
	for t in 1:T
        y_onehot = one_hot_yt(ys[:, t], labels)
		B_new .+= exp.(log_γ[:, t]) .* y_onehot'
    end
	
	log_B_ = logsumexp(log_γ, dims=2) 

	#B_new = B_new ./ sum(B_new, dims=2)
	B_new = B_new ./ exp.(log_B_)
    return exp.(log_π_new), exp.(log_A_new), B_new
end

# ╔═╡ 95dbae39-8221-4475-8d81-8d8cdfccefe7
md"""
Test M-step
"""

# ╔═╡ f1052288-7249-4de0-afea-dadd75da68ed
let
	log_g, log_x = e_oh(c_data', π_i, A_casino, B_casino)
	pi, A, B = m_oh(log_g, log_x, c_data')
end

# ╔═╡ 1280da63-0964-4db0-91e8-6cd0c9d408de
function em_oh(ys, K, max_iter=100)
    D, T = size(ys)
    P = length(unique(ys[:]))

	Random.seed!(111)
    # Initialize model parameters randomly
    π = rand(Dirichlet(ones(K)))
	
    A = [rand(Dirichlet(ones(K))) for _ in 1:K]
    A = hcat(A...)'
	
    B = [rand(Dirichlet(ones(P))) for _ in 1:K]
    B = hcat(B...)'

    for _ in 1:max_iter
        # E-step
        γ, ξ = e_oh(ys, π, A, B)
        
        # M-step
        π, A, B = m_oh(γ, ξ, ys)
    end
    
    return π, A, B
end

# ╔═╡ 0bfa1922-bb11-40e2-a972-1d656d4390fd
md"""
Test EM (Baum-Welch)
"""

# ╔═╡ ee5a5e53-7811-4d70-b856-c89eab5f89f7
em_oh(c_data', 2) 

# ╔═╡ 9eda917a-ac0b-48b9-a12f-960708deadf4
md"""
Test hidden var (X) inference
"""

# ╔═╡ f35f68c3-df08-449a-9209-42f0c793f3de
π_mle, A_mle, B_mle = em_mle(c_data', 2)

# ╔═╡ be63f9e1-8886-4cda-9678-16ad78748d50
let
	s_f = exp.(e_mle(c_data', π_mle, A_mle, B_mle)[1])
	ss = [x[1]*1 + x[2]*2 for x in eachcol(s_f)]
	mad = mean(abs.(true_coins - ss))
	rmse = sqrt(mean((true_coins - ss).^2))
	println("MAD: ", mad)
	println("RMSE: ", rmse)
end

# ╔═╡ 1fe0e1c1-44e1-4a15-82c5-8da8f12d843e
md"""
Aside, alter labels, should still work with one-hot encoding and identical to above
"""

# ╔═╡ fd501652-6a71-411d-b805-3be29390be03
c_data_incre = c_data .+ 1

# ╔═╡ aa44d78a-0625-480a-96c3-012914b1c718
em_oh(c_data_incre', 2) 

# ╔═╡ 2e015146-94f4-47d8-b52e-2a403bfe72db
let
	π_mle, A_mle, B_mle = em_oh(c_data_incre', 2)
	s_f = exp.(e_oh(c_data_incre', π_mle, A_mle, B_mle)[1])
	ss = [x[1]*1 + x[2]*2 for x in eachcol(s_f)]
	mad = mean(abs.(true_coins - ss))
	rmse = sqrt(mean((true_coins - ss).^2))
	println("MAD: ", mad)
	println("RMSE: ", rmse)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
HMMBase = "b2b3ca75-8444-5ffa-85e6-af70e2b64fe7"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
StatsFuns = "4c63d2b9-4356-54db-8cca-17b64c39e42c"

[compat]
Distributions = "~0.25.86"
Flux = "~0.13.16"
HMMBase = "~1.0.7"
PlutoUI = "~0.7.50"
SpecialFunctions = "~2.2.0"
StatsFuns = "~1.3.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.0"
manifest_format = "2.0"
project_hash = "d27ccddcce64d93aaffaa38eb9e9dbc72d3aa1cb"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "16b6dbc4cf7caee4e1e75c49485ec67b667098a0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "a4f8669e46c8cdf68661fe6bb0f7b89f51dd23cf"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.30"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cc37d689f599e8df4f464b2fa3870ff7db7492ef"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.6.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "a3a402a35a2f7e0b87828ccabbd5ebfbebe356b4"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.3.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c06a868224ecba914baa6942988e2f2aade419be"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "0.1.0"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random", "Test"]
git-tree-sha1 = "dbf84058d0a8cbbadee18d25cf606934b22d7c66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.4.2"

[[deps.BangBang]]
deps = ["Compat", "ConstructionBase", "Future", "InitialValues", "LinearAlgebra", "Requires", "Setfield", "Tables", "ZygoteRules"]
git-tree-sha1 = "7fe6d92c4f281cf4ca6f2fba0ce7b299742da7ca"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.3.37"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.CEnum]]
git-tree-sha1 = "eb4cb44a499229b3b8426dcfb5dd85333951ff90"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.2"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "CompilerSupportLibraries_jll", "ExprTools", "GPUArrays", "GPUCompiler", "KernelAbstractions", "LLVM", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "Preferences", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "SpecialFunctions", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "280893f920654ebfaaaa1999fbd975689051f890"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "4.2.0"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "498f45593f6ddc0adff64a9310bb6710e851781b"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.5.0+1"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "bcc4a23cbbd99c8535a5318455dcf0f2546ec536"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.2.2"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "5248d9c45712e51e27ba9b30eebec65658c6ce29"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.6.0+0"

[[deps.CUDNN_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "2918fbffb50e3b7a0b9127617587afa76d4276e8"
uuid = "62b44479-cb7b-5706-934f-f13b2eb2e645"
version = "8.8.1+0"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "Statistics", "StructArrays"]
git-tree-sha1 = "8bae903893aeeb429cf732cf1888490b93ecf265"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.49.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e30f2f4e20f7f186dc36529910beaedc60cfa644"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.16.0"

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

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "738fec4d684a9a6ee9598a8bfee305b26831f28c"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.2"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "a4ad7ef19d2cdc2eff57abbbe68032b1cd0bd8f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.13.0"

[[deps.Distances]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "49eba9ad9f7ead780bfb7ee319f962c811c6d3b2"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.8"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "da9e1a9058f8d3eec3a8c9fe4faacfb89180066b"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.86"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"

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

[[deps.ExprTools]]
git-tree-sha1 = "c1d06d129da9f55715c6c212866f5b1bddc5fa00"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.9"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "ffb97765602e3cbe59a0589d237bf07f245a8576"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.1"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

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

[[deps.Flux]]
deps = ["Adapt", "CUDA", "ChainRulesCore", "Functors", "LinearAlgebra", "MLUtils", "MacroTools", "NNlib", "NNlibCUDA", "OneHotArrays", "Optimisers", "Preferences", "ProgressLogging", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "Zygote", "cuDNN"]
git-tree-sha1 = "64005071944bae14fc145661f617eb68b339189c"
uuid = "587475ba-b771-5e3f-ad9e-33799f191a9c"
version = "0.13.16"

    [deps.Flux.extensions]
    AMDGPUExt = "AMDGPU"

    [deps.Flux.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"

[[deps.FoldsThreads]]
deps = ["Accessors", "FunctionWrappers", "InitialValues", "SplittablesBase", "Transducers"]
git-tree-sha1 = "eb8e1989b9028f7e0985b4268dabe94682249025"
uuid = "9c68100b-dfe1-47cf-94c8-95104e173443"
version = "0.1.1"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "478f8c3145bb91d82c2cf20433e8c1b30df454cc"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.4"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "Serialization", "Statistics"]
git-tree-sha1 = "9ade6983c3dbbd492cf5729f865fe030d1541463"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "8.6.6"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "1cd7f0af1aa58abc02ea1d872953a97359cb87fa"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.4"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "Scratch", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "e9a9173cd77e16509cdf9c1663fda19b22a518b7"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "0.19.3"

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

[[deps.IRTools]]
deps = ["InteractiveUtils", "MacroTools", "Test"]
git-tree-sha1 = "eac00994ce3229a464c2847e956d77a2c64ad3a5"
uuid = "7869d1d1-7146-5819-86e3-90919afe41df"
version = "0.4.10"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "6667aadd1cdee2c6cd068128b3d226ebc4fb0c67"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.9"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "LinearAlgebra", "MacroTools", "PrecompileTools", "SparseArrays", "StaticArrays", "UUIDs", "UnsafeAtomics", "UnsafeAtomicsLLVM"]
git-tree-sha1 = "47be64f040a7ece575c2b5f53ca6da7b548d69f4"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.4"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Printf", "Unicode"]
git-tree-sha1 = "a8960cae30b42b66dd41808beb76490519f6f9e2"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "5.0.0"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "09b7505cc0b1cee87e5d4a26eea61d2e1b0dcd35"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.21+0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

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
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "FoldsThreads", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "ca31739905ddb08c59758726e22b9e25d0d1521b"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.2"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.MicroCollections]]
deps = ["BangBang", "InitialValues", "Setfield"]
git-tree-sha1 = "629afd7d10dbc6935ec59b32daeb33bc4460a42e"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.1.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Pkg", "Random", "Requires", "Statistics"]
git-tree-sha1 = "99e6dbb50d8a96702dc60954569e9fe7291cc55d"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.8.20"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"

[[deps.NNlibCUDA]]
deps = ["Adapt", "CUDA", "LinearAlgebra", "NNlib", "Random", "Statistics", "cuDNN"]
git-tree-sha1 = "f94a9684394ff0d325cc12b06da7032d8be01aaf"
uuid = "a00861dc-f156-4864-bf3c-e6376f28a68d"
version = "0.2.7"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "2c3726ceb3388917602169bed973dbc97f1b51a8"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.13"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OneHotArrays]]
deps = ["Adapt", "ChainRulesCore", "Compat", "GPUArraysCore", "LinearAlgebra", "NNlib"]
git-tree-sha1 = "f511fca956ed9e70b80cd3417bb8c2dde4b68644"
uuid = "0b1bfda6-eb8a-41d2-88d8-f5af5cad476f"
version = "0.2.3"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optimisers]]
deps = ["ChainRulesCore", "Functors", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "6a01f65dd8583dee82eecc2a19b0ff21521aa749"
uuid = "3bd65402-5787-11e9-1adc-39752487f4e2"
version = "0.2.18"

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
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "5bb5129fdd62a2bbbe17c2756932259acf467386"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.50"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "259e206946c293698122f63e2b513a7c99a244e8"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.1.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

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

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "552f30e847641591ba3f39fd1bed559b9deb0ef3"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.6.1"

[[deps.RandomNumbers]]
deps = ["Random", "Requires"]
git-tree-sha1 = "043da614cc7e95c703498a491e2c21f58a2b8111"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.5.3"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

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

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "30449ee12237627992a99d5e30ae63e4d78cd24a"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

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
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

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
version = "1.9.0"

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
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StructArrays]]
deps = ["Adapt", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "521a0e828e98bb69042fec1809c1b5a680eb7389"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.15"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.Transducers]]
deps = ["Adapt", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "Setfield", "SplittablesBase", "Tables"]
git-tree-sha1 = "25358a5f2384c490e98abd565ed321ffae2cbb37"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.76"

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

[[deps.UnsafeAtomics]]
git-tree-sha1 = "6331ac3440856ea1988316b46045303bef658278"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.2.1"

[[deps.UnsafeAtomicsLLVM]]
deps = ["LLVM", "UnsafeAtomics"]
git-tree-sha1 = "ea37e6066bf194ab78f4e747f5245261f17a7175"
uuid = "d80eeb9a-aca5-4d75-85e5-170c8b632249"
version = "0.1.2"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zygote]]
deps = ["AbstractFFTs", "ChainRules", "ChainRulesCore", "DiffRules", "Distributed", "FillArrays", "ForwardDiff", "GPUArrays", "GPUArraysCore", "IRTools", "InteractiveUtils", "LinearAlgebra", "LogExpFunctions", "MacroTools", "NaNMath", "Random", "Requires", "SnoopPrecompile", "SparseArrays", "SpecialFunctions", "Statistics", "ZygoteRules"]
git-tree-sha1 = "987ae5554ca90e837594a0f30325eeb5e7303d1e"
uuid = "e88e6eb3-aa80-5325-afca-941959d7151f"
version = "0.6.60"

    [deps.Zygote.extensions]
    ZygoteColorsExt = "Colors"
    ZygoteDistancesExt = "Distances"
    ZygoteTrackerExt = "Tracker"

    [deps.Zygote.weakdeps]
    Colors = "5ae59095-9a9b-59fe-a467-6f913c188581"
    Distances = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ZygoteRules]]
deps = ["ChainRulesCore", "MacroTools"]
git-tree-sha1 = "977aed5d006b840e2e40c0b48984f7463109046d"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.3"

[[deps.cuDNN]]
deps = ["CEnum", "CUDA", "CUDNN_jll"]
git-tree-sha1 = "ec954b59f6b0324543f2e3ed8118309ac60cb75b"
uuid = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"
version = "1.0.3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.7.0+0"

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
# ╟─4c7b01e4-b201-4605-8bf3-16d404c8be39
# ╟─009cfa59-55d5-4d11-8e29-7f09f58a7354
# ╟─a98e5e92-d4e8-4ac2-a4ea-b21006fe1558
# ╟─e74c5c83-9cf5-4d4a-ad9f-2785e18bc3ef
# ╠═93d378e6-b6bf-4065-b483-5010a54b04b4
# ╠═99234c60-3250-4c63-82ee-f15b6299d856
# ╟─a4122178-c8c6-4e67-9a9a-ccf95d3bc96f
# ╠═f8fc9f13-e00f-4da7-b244-7bb497a1c669
# ╠═5baf0c31-717c-4e0c-84a5-ebe9da192dbd
# ╠═072762bc-1f27-4a95-ad46-ddbdf45292cc
# ╠═495d7a29-5b02-49d8-b376-bf1d088026c1
# ╠═15ebed48-40f0-4c53-aaf3-e56cf42f0a74
# ╟─261c15ff-a80a-43df-bf51-33591d914aea
# ╠═e7e199fd-cc7f-483f-b588-00013b9069d1
# ╠═a2019816-e7b9-4eda-be76-9b3ebbb990e3
# ╠═2b01dd4d-fba9-49dd-9802-67cea335d49c
# ╠═1e44a7f5-75bc-4476-81b6-54acf093fa51
# ╟─c38bfc75-4281-4b52-aeac-0a9dbf7f8776
# ╠═e4991173-290e-4a28-838b-f2afbec67c93
# ╠═6ddf3db6-11f2-4cbc-a71e-e818ca1f27f4
# ╠═e7837cab-749c-43e4-bb82-c9f87afde848
# ╟─3da884a5-eacb-4749-a10c-5a6ad6da34b7
# ╟─61a5e1fd-5480-4ed1-83ff-d3ad140cbcbc
# ╟─4a573a35-51b2-4a42-b175-3ba017ef7245
# ╠═ae313199-fe27-478a-b6b2-bee2923b5a54
# ╟─5a1de95b-deef-4199-8992-28fcfe2157b8
# ╠═b4d1bdf3-efd6-4851-9d33-e1564b8ed769
# ╟─d7aaea92-8cf6-4ac3-9929-559ab88d65e3
# ╠═a0fa9433-3ae7-4697-8eea-5d6ddefbf62b
# ╟─e9b7bbd0-2de8-4a5b-af73-3b576e8c54a2
# ╠═6905f394-3da2-4207-a511-888f7521d82a
# ╟─5d746477-0282-4656-8db4-5a61561d3ccb
# ╠═ac620936-66fb-4457-b9fd-9180fc9ba994
# ╠═a4b38c35-e320-439e-b169-b8af974e2635
# ╟─54f38d75-aa88-4eb8-983e-e2a3038f910f
# ╟─ebed75fc-0c10-48c7-9352-fc61bc3dcfd8
# ╟─137cafd1-fe56-4db8-9e28-f1e133887be4
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
# ╟─fcf0b465-7d59-49af-97df-dcf379305671
# ╠═84a4bb31-26f8-4a9e-a0b2-f5f8952ef08b
# ╟─f6d66591-f9a6-48f4-9d40-2fd08a220a38
# ╟─171ea1db-a425-4cf2-93a8-01e94ff329cb
# ╠═453d68e3-9a04-47c3-9af3-37d2347bfd64
# ╠═c6dbe013-c810-4be3-b7d8-a072bd6432c1
# ╟─20f39921-7187-4651-9b42-e1c4fc8f1056
# ╟─70c8ef1b-b2a9-4ecb-a8c9-70dd57411a8a
# ╠═c7f6c9b4-b0dc-4da4-9ca4-dd96b4afb640
# ╟─cfa4a7d2-3a6a-40bd-80fb-449f91584ed9
# ╠═1fdc1902-e892-4b4c-ac2e-5ea2222a6228
# ╠═7333ac70-4f79-45f5-875a-3df38b55052a
# ╠═c6481a69-b6cb-4f90-a557-303eb7e42f09
# ╟─da6d3c66-4f8f-4d18-84ec-c84be9153474
# ╠═0ff7f6fc-81a7-4eac-b933-5c8e71b6843b
# ╠═a33e7ee9-f8fb-4483-8b15-21d098e9b65d
# ╠═36b7ede6-6ed5-4daa-8a74-ccf6d049c3fa
# ╠═10b8f741-ed45-4cd3-9817-a63f2e4a3aaf
# ╟─5b9cd9ce-df0a-49c8-97b2-b0b9144ff323
# ╠═4ab65b62-2f57-4c7b-a58a-b50b773863d0
# ╟─a6143756-dafc-41d3-9a46-8a5c23dfd87e
# ╠═66a1a7d4-db54-47f7-bdfe-953ba155e35a
# ╟─1abdb6ea-8de2-4df2-b278-03335639a202
# ╟─da615595-cffd-4bf8-abd3-5498b7a4d202
# ╠═4a6712c6-955f-445e-9c68-b09ae5b00d3d
# ╟─f0a47b5e-40e6-4efe-b5c7-e3cc01270a0a
# ╠═50b5bdaf-2467-450f-a0c6-80d55a68586c
# ╟─1877d3d1-cc43-4271-9542-11d9d1bb7208
# ╠═0c1e1848-67d1-4ebb-9b8c-534f7e0cb3c1
# ╟─d5bbbd58-a4c0-466f-987a-a6f32a80a775
# ╠═a3a23fb9-baad-4d04-b8da-82ce2267f0b7
# ╠═8f21e2cd-5f8b-4242-885b-9d42b22fad74
# ╠═ae6352cf-0343-4b4c-9fa7-8ba3792eafc7
# ╠═9400ad3d-f2af-418e-971e-6031c7164c78
# ╟─18e35e5b-dd37-4f0a-aa8e-f0aedcb658e2
# ╟─3ac18f00-7e85-4abb-90fc-56b47d9fdd27
# ╠═d0cba4d1-2d02-4524-afd8-3150c5019f83
# ╟─429c5dfd-4f78-4697-b710-777bd5a38240
# ╠═51a5c3e6-7b0a-4d54-b9d3-5eb6054ec72d
# ╠═a2e69fa3-17a3-4588-8f65-be6d77b09c4f
# ╟─482e735a-610c-414c-83ac-0e1e1e2d4d86
# ╟─99c2a566-38a2-40f0-9184-d592ec79694e
# ╠═875aa3bc-c4b9-4dce-987d-62dbe1d40d37
# ╠═289d16d6-528e-4e07-9440-9573d2f71724
# ╠═30cbc8bf-f0ae-4514-b4f3-1e8734f38377
# ╠═58a00d74-3383-4e98-a4ec-97655801516a
# ╟─9365ad98-d6ff-424f-94ff-6c754315417c
# ╠═fbdd786f-24f2-497d-a804-7af8cf5cb72e
# ╠═1e8b3bed-943e-4e93-a096-958e1dbdfa6d
# ╠═54c2fe20-7b86-4d72-9449-5e942496fdb6
# ╠═4fad4cf7-20ce-4a7d-bbc0-c91a0d1611d4
# ╟─4569a084-ba82-4e12-856e-719ebc3cecb3
# ╟─5597ece4-8184-4495-a2f9-a8ec1b7c5d2c
# ╟─df01dca6-6c72-40bc-92d9-d38b5938f09e
# ╠═45ffda31-984f-4144-8b3f-7c452ed69d06
# ╠═276c73b9-0699-45da-b35c-317f42d18227
# ╠═f674e936-f124-48b6-bd3b-cce7f9b7e6b4
# ╠═5b0ccc1c-ea6e-439d-9c04-c24dde87a22f
# ╠═09d078fa-2346-44fd-9e33-32203fc88b23
# ╠═69e878f5-f893-424e-b059-ca848a1dd23f
# ╟─6a78f8a9-0d19-4eaa-8e2a-ce9290dbea15
# ╠═f9b02790-d6c5-4d69-896f-cba1b6c622b7
# ╠═cda23c77-a219-4bbd-a4f0-22e43f8037d9
# ╟─5fd43cac-f673-4218-b593-7453ffbd0c64
# ╟─d5bf0a63-4d30-463f-ac3e-00cdab3c7331
# ╠═a3a1545f-9af4-409b-9b55-d7a62bdf5c5e
# ╠═15eaf87f-865c-49e4-82a7-bcd8d648d831
# ╠═c0af3e5c-ad73-4fe2-92d3-fdecf20c5512
# ╠═1d41f8ef-565d-41d6-955e-0e7fdbabbfc6
# ╠═22cd1223-dadd-47a4-8d8b-686b554b4869
# ╠═18cc49b7-86d5-4f8d-a235-16b02d0036e3
# ╠═35fc03f2-21b2-452c-abc2-216579817fc8
# ╠═096ccd02-4844-466d-8532-cbb7cda80910
# ╠═a9393ab8-1334-48a6-a47c-4bcfaab9929e
# ╠═51a60031-6636-47d3-a447-058ead85229d
# ╟─5f4a79c6-b319-41a9-a35f-094fbdfe68b6
# ╠═5a264cf0-1d7b-468e-91fa-467e9ecb21bf
# ╠═82fbde19-1a8c-4b6a-94db-ac49b59e2ac4
# ╟─95dbae39-8221-4475-8d81-8d8cdfccefe7
# ╠═f1052288-7249-4de0-afea-dadd75da68ed
# ╠═1280da63-0964-4db0-91e8-6cd0c9d408de
# ╟─0bfa1922-bb11-40e2-a972-1d656d4390fd
# ╠═ee5a5e53-7811-4d70-b856-c89eab5f89f7
# ╟─9eda917a-ac0b-48b9-a12f-960708deadf4
# ╠═f35f68c3-df08-449a-9209-42f0c793f3de
# ╠═be63f9e1-8886-4cda-9678-16ad78748d50
# ╟─1fe0e1c1-44e1-4a15-82c5-8da8f12d843e
# ╠═fd501652-6a71-411d-b805-3be29390be03
# ╠═aa44d78a-0625-480a-96c3-012914b1c718
# ╠═2e015146-94f4-47d8-b52e-2a403bfe72db
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
