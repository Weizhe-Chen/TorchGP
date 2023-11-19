---
title: "Online Sparse Gaussian Process Regression"
---

``` {=html}
<style>
body { min-width: 80% !important; }
</style>
```

## Notation

- $\mathbf{X}$: $N \times D$ matrix of inputs
- $\mathbf{y}$: $N \times 1$ vector of outputs
- $\mathbf{Z}^{\text{old}}$: $M \times D$ matrix of old inducing inputs
- $\mathbf{u}$: $M \times 1$ vector of old inducing outputs
- $\mathbf{Z}$: $M \times D$ matrix of new inducing inputs
- $\mathbf{v}$: $M \times 1$ vector of new inducing outputs
- $\mathbf{\tilde{f}} = [\mathbf{f}^{\top}, \mathbf{u}_{\text{old}}^{\top}]^{\top}$: $(N + M) \times 1$ vector of concatenated function values and old inducing outputs
- $q(\mathbf{u})=\mathcal{N}(\mathbf{u} | \mathbf{m}_{u}, \mathbf{S}_{u u})$: posterior distribution of old inducing outputs
- $\mathbf{C}_{u u} = (\mathbf{S}_{u u}^{-1} - {\mathbf{K}^{\text{old}}_{u u}}^{-1})^{-1}$

## Prediction

$$
\begin{aligned}
\boldsymbol{\mu}= & \mathbf{K}_{s v} \sqrt{\mathbf{K}_{v v}}^{-\top} {\color{red}\mathbf{D}^{-1}} \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v \tilde{f}} \mathbf{\Sigma}_{\tilde{y}}^{-1} \mathbf{\tilde{y}}\\
=& \underbrace{\mathbf{K}_{s v} \sqrt{\mathbf{K}_{v v}}^{-\top} \sqrt{\mathbf{D}}^{-\top}} \underbrace{\sqrt{\mathbf{D}}^{-1} \sqrt{\mathbf{K}_{v v}}^{-1} {\color{blue}\mathbf{K}_{v \tilde{f}} \mathbf{\Sigma}_{\tilde{y}}^{-1} \mathbf{\tilde{y}}}}\\
=& (\sqrt{\mathbf{D}}^{-1} \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v s})^{\top} \sqrt{\mathbf{D}}^{-1} \sqrt{\mathbf{K}_{v v}}^{-1} {\color{blue} \mathbf{c}}
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{\Sigma} =& \mathbf{K}_{s s} - \mathbf{K}_{s v} \mathbf{K}_{v v}^{-1} \mathbf{K}_{v s} + \mathbf{K}_{s v} \sqrt{\mathbf{K}_{v v}}^{-\top} \mathbf{D}^{-1} \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v s}\\
=& \mathbf{K}_{s s} - \mathbf{K}_{s v} \sqrt{\mathbf{K}_{v v}}^{-\top} \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v s}\\ 
&~~~~~~~+ \mathbf{K}_{s v} \sqrt{\mathbf{K}_{v v}}^{-\top} \mathbf{D}^{-1} \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v s}\\
=& \mathbf{K}_{s s} - (\sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v s})^{\top} \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v s}\\ 
&~~~~~~~+ \mathbf{K}_{s v} \sqrt{\mathbf{K}_{v v}}^{-\top} \sqrt{\mathbf{D}}^{-\top} \sqrt{\mathbf{D}}^{-1} \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v s}\\
=& \mathbf{K}_{s s} - (\sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v s})^{\top} \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v s}\\ 
&~~~~~~~+ \left(\sqrt{\mathbf{D}}^{-1} \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v s}\right)^{\top} \sqrt{\mathbf{D}}^{-1} \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v s}
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{D}
=& \mathbf{I} + \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v \tilde{f}} \mathbf{\Sigma}_{\tilde{y}}^{-1} \mathbf{K}_{\tilde{f} v} \sqrt{\mathbf{K}_{v v}}^{-\top}\\
=& \mathbf{I} + \sqrt{\mathbf{K}_{v v}}^{-1} \begin{bmatrix} \mathbf{K}_{v f} & \mathbf{K}_{v u} \end{bmatrix} \begin{bmatrix} \sigma^{2} \mathbf{I}  & \mathbf{0} \\ \mathbf{0} & \mathbf{C}_{u u}  \end{bmatrix}^{-1} \begin{bmatrix} \mathbf{K}_{f v} \\ \mathbf{K}_{u v} \end{bmatrix} \sqrt{\mathbf{K}_{v v}}^{-\top}\\
=& \mathbf{I} + \sqrt{\mathbf{K}_{v v}}^{-1} \begin{bmatrix} \mathbf{K}_{v f} & \mathbf{K}_{v u} \end{bmatrix} \begin{bmatrix} \sigma^{-2} \mathbf{I}  & \mathbf{0} \\ \mathbf{0} & \mathbf{C}_{u u}^{-1}  \end{bmatrix} \begin{bmatrix} \mathbf{K}_{f v} \\ \mathbf{K}_{u v} \end{bmatrix} \sqrt{\mathbf{K}_{v v}}^{-\top}\\
=& \mathbf{I} + \sqrt{\mathbf{K}_{v v}}^{-1} \begin{bmatrix} \mathbf{K}_{v f} & \mathbf{K}_{v u} \end{bmatrix}  \begin{bmatrix} \sigma^{-2} \mathbf{K}_{f v} \\ \mathbf{C}_{u u}^{-1} \mathbf{K}_{u v} \end{bmatrix} \sqrt{\mathbf{K}_{v v}}^{-\top}\\
=& \mathbf{I} + \sqrt{\mathbf{K}_{v v}}^{-1} (\sigma^{-2} \mathbf{K}_{v f} \mathbf{K}_{f v} + \mathbf{K}_{v u} \mathbf{C}_{u u}^{-1} \mathbf{K}_{u v}) \sqrt{\mathbf{K}_{v v}}^{-\top}\\
=& \mathbf{I} + \sqrt{\mathbf{K}_{v v}}^{-1} (\sigma^{-2} \mathbf{K}_{v f} \mathbf{K}_{f v} + \mathbf{K}_{v u} (\mathbf{S}_{u u}^{-1} - {\mathbf{K}^{\text{old}}_{u u}}^{-1}) \mathbf{K}_{u v}) \sqrt{\mathbf{K}_{v v}}^{-\top}\\
=& \mathbf{I} + \sqrt{\mathbf{K}_{v v}}^{-1} (\sigma^{-2} \mathbf{K}_{v f} \mathbf{K}_{f v} + \mathbf{K}_{v u} \mathbf{S}_{u u}^{-1} \mathbf{K}_{u v} - \mathbf{K}_{v u} {\mathbf{K}^{\text{old}}_{u u}}^{-1} \mathbf{K}_{u v}) \sqrt{\mathbf{K}_{v v}}^{-\top}\\
=& \mathbf{I} + \underbrace{\sqrt{\mathbf{K}_{v v}}^{-1} (\sigma^{-2} \mathbf{K}_{v f} \mathbf{K}_{f v}) \sqrt{\mathbf{K}_{v v}}^{-\top}}_{\mathbf{D}_{1}}\\
&~+ \underbrace{\sqrt{\mathbf{K}_{v v}}^{-1} (\mathbf{K}_{v u} \mathbf{S}_{u u}^{-1} \mathbf{K}_{u v}) \sqrt{\mathbf{K}_{v v}}^{-\top}}_{\mathbf{D}_{2}}\\
&~+ \underbrace{\sqrt{\mathbf{K}_{v v}}^{-1} (\mathbf{K}_{v u} {\mathbf{K}^{\text{old}}_{u u}}^{-1} \mathbf{K}_{u v}) \sqrt{\mathbf{K}_{v v}}^{-\top}}_{\mathbf{D}_{3}}\\
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{D}_{1}
&= \sigma^{-2} \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v f} \mathbf{K}_{f v} \sqrt{\mathbf{K}_{v v}}^{-\top}\\
&= \sigma^{-2} (\sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v f}) (\sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v f})^{\top}\\
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{D}_{2}
&= \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v u} \mathbf{S}_{u u}^{-1} \mathbf{K}_{u v} \sqrt{\mathbf{K}_{v v}}^{-\top}\\
&= \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v u} \sqrt{\mathbf{S}_{u u}}^{-\top} \sqrt{\mathbf{S}_{u u}}^{-1} \mathbf{K}_{u v} \sqrt{\mathbf{K}_{v v}}^{-\top}\\
&= \left(\sqrt{\mathbf{S}_{u u}}^{-1} \mathbf{K}_{u v} \sqrt{\mathbf{K}_{v v}}^{-\top}\right)^{\top} \sqrt{\mathbf{S}_{u u}}^{-1} \mathbf{K}_{u v} \sqrt{\mathbf{K}_{v v}}^{-\top}\\
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{D}_{3}
&= \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v u} {\mathbf{K}^{\text{old}}_{u u}}^{-1} \mathbf{K}_{u v} \sqrt{\mathbf{K}_{v v}}^{-\top}\\
&= \sqrt{\mathbf{K}_{v v}}^{-1} \mathbf{K}_{v u} \sqrt{\mathbf{K}^{\text{old}}_{u u}}^{-\top} \sqrt{\mathbf{K}^{\text{old}}_{u u}}^{-1} \mathbf{K}_{u v} \sqrt{\mathbf{K}_{v v}}^{-\top}\\
&= \left(\sqrt{\mathbf{K}^{\text{old}}_{u u}}^{-1} \mathbf{K}_{u v} \sqrt{\mathbf{K}_{v v}}^{-\top}\right)^{\top} \sqrt{\mathbf{K}^{\text{old}}_{u u}}^{-1} \mathbf{K}_{u v} \sqrt{\mathbf{K}_{v v}}^{-\top}\\
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{c} 
=& \mathbf{K}_{v \tilde{f}} \mathbf{\Sigma}_{\tilde{y}}^{-1} \mathbf{\tilde{y}}\\
=& \begin{bmatrix} \mathbf{K}_{v f} & \mathbf{K}_{v u} \end{bmatrix} \begin{bmatrix} \sigma^{2} \mathbf{I}  & \mathbf{0} \\ \mathbf{0} & \mathbf{C}_{u u}  \end{bmatrix}^{-1} \begin{bmatrix} \mathbf{y} \\ \mathbf{C}_{u u} \mathbf{S}_{uu}^{-1} \mathbf{m}_{u} \end{bmatrix}\\
=& \begin{bmatrix} \mathbf{K}_{v f} & \mathbf{K}_{v u} \end{bmatrix} \begin{bmatrix} \sigma^{-2} \mathbf{y} \\ \mathbf{S}_{u u}^{-1} \mathbf{m}_{u} \end{bmatrix}\\
=& \underbrace{\sigma^{-2} \mathbf{K}_{v f} \mathbf{y}}_{\mathbf{c}_{1}} + \underbrace{\mathbf{K}_{v u} \mathbf{S}_{u u}^{-1} \mathbf{m}_{u}}_{\mathbf{c}_{2}}\\
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{c}_{2} &= \mathbf{K}_{v u} \mathbf{S}_{u u}^{-1} \mathbf{m}_{u}\\
&= \mathbf{K}_{v u} \sqrt{\mathbf{S}_{u u}}^{-\top} \sqrt{\mathbf{S}_{u u}}^{-1}\mathbf{m}_{u}\\
&= (\sqrt{\mathbf{S}_{u u}}^{-1} \mathbf{K}_{u v})^{\top} \sqrt{\mathbf{S}_{u u}}^{-1}\mathbf{m}_{u}
\end{aligned}
$$

## Evidence

$$
\begin{aligned}
\log p(\mathbf{y} | \boldsymbol{\theta}, \mathbf{Z}) =& -\frac{N}{2}\log(2\pi\sigma^{2})\\
&- \frac{1}{2\sigma^{2}}\mathbf{y}^{\top}\mathbf{y} + \frac{1}{2}\mathbf{c}^{\top}\sqrt{\mathbf{K}_{v v}}^{-\top}\mathbf{D}^{-1}\sqrt{\mathbf{K}_{v v}}^{-1}\mathbf{c} - \frac{1}{2}\mathbf{m}_{u}\mathbf{S}_{u u}^{-1}\mathbf{m}_{u}\\
&- \frac{1}{2}\log|\mathbf{D}| - \frac{1}{2}\log|\mathbf{S}_{u u}| + \frac{1}{2}\log|\mathbf{K}_{u u}^{\text{old}}|\\
&- \frac{1}{2}\mathtt{Trace}(\mathbf{C}_{u u}^{-1} \mathbf{E}_{u u}) - \frac{1}{2\sigma^{2}}\texttt{Trace}(\mathbf{E}_{f f})
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{E}_{u u} &= \mathbf{K}_{u u} - \mathbf{K}_{u v} \mathbf{K}_{v v}^{-1} \mathbf{K}_{v u}\\
&= \mathbf{K}_{u u} - \mathbf{Q}_{u u}
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{E}_{f f} &= \mathbf{K}_{f f} - \mathbf{K}_{f v} \mathbf{K}_{v v}^{-1} \mathbf{K}_{v f}\\
&= \mathbf{K}_{f f} - \mathbf{Q}_{f f}
\end{aligned}
$$

$$
\begin{aligned}
\mathtt{Trace}(\mathbf{C}_{u u}^{-1} \mathbf{E}_{u u})
&= \mathtt{Trace}\left((\mathbf{S}_{u u}^{-1} - {\mathbf{K}^{\text{old}}_{u u}}^{-1}) \mathbf{E}_{u u}\right)\\
&= \mathtt{Trace}\left(\mathbf{S}_{u u}^{-1} \mathbf{E}_{u u} - {\mathbf{K}^{\text{old}}_{u u}}^{-1} \mathbf{E}_{u u}\right)\\
&= \mathtt{Trace}\left(\mathbf{S}_{u u}^{-1} \mathbf{E}_{u u}\right) - \mathtt{Trace}\left({\mathbf{K}^{\text{old}}_{u u}}^{-1} \mathbf{E}_{u u}\right)\\
\end{aligned}
$$

$$
\begin{aligned}
\mathtt{Trace}(\mathbf{E}_{f f}) &= \mathtt{Trace}(\mathbf{K}_{f f} - \mathbf{Q}_{f f})\\
&= \mathtt{Trace}(\mathbf{K}_{f f}) - \mathtt{Trace}(\mathbf{Q}_{f f})\\
\end{aligned}
$$

## Implementation

- Compute `mean = InvTrilD_InvTrilKvv_Kvs.T @ InvTrilD_InvTrilKvv_c`
    - `InvTrilD_InvTrilKvv_Kvs`
        - `InvTrilKvv_Kvs = tril_solve(TrilKvv, Kvs)`
        - `InvTrilD_InvTrilKvv_Kvs = tril_solve(TrilD, InvTrilKvv_Kvs)`
    - `InvTrilD_InvTrilKvv_c`
        - `c = c1 + c2`
            - `c1 = (Kvf @ y).div(noise_variance)`
            - `c2 = InvTrilSuu_Kuv.T @ InvTrilSuu_mu`
                - `InvTrilSuu_Kuv = tril_solve(TrilSuu, Kuv)`
                - `InvTrilSuu_mu = tril_solve(TrilSuu, mu)
        - `InvTrilKvv_c = tril_solve(TrilKvv, c)`
        - `InvTrilD_InvTrilKvv_c = tril_solve(TrilD, InvTrilKvv_c)`
- Compute `cov = Kss - InvTrilKvv_Kvs.T @ InvTrilKvv_Kvs + InvTrilD_InvTrilKvv_Kvs.T @ InvTrilD_InvTrilKvv_Kvs`
    - `InvTrilKvv_Kvs`
        - `InvTrilKvv_Kvs = tril_solve(TrilKvv, Kvs)`
    - `InvTrilD_InvTrilKvv_Kvs`
        - `InvTrilD_InvTrilKvv_Kvs = tril_solve(TrilD, InvTrilKvv_Kvs)`
- Compute `evidence = constant_term + quadratic_terms + logdet_terms + trace_terms`
    - `constant_term = -0.5 * N * np.log(2 * np.pi * noise_variance)`
    - `quadratic_terms = -0.5 * quadratic_form(y).div(noise_variance) + 0.5 * quadratic_form(InvTrilD_InvTrilKvv_c) - 0.5 * quadratic_form(InvTrilSuu_mu)`
        - `quadratic_form(x) = x.T @ x = x.square().sum()`
    - `logdet_terms = -0.5 * tril_logdet(TrilD) - 0.5 * tril_logdet(TrilSuu) + 0.5 * tril_logdet(TrilOldKuu)`
        - `tril_logdet(tril) = 2.0 * tril.diagonal().log().sum()`
    - `trace_terms = -0.5 * InvSuu_Euu.trace() + 0.5 * InvOldKuu_Euu.trace() - (Kff.trace() - trace_quadratic_form(InvTrilKvv_Kvf)).div(2.0 * noise_variance)`
