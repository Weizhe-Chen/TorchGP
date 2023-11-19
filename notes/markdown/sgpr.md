---
title: "Sparse Gaussian Process Regression"
---

``` {=html}
<style>
body { min-width: 80% !important; }
</style>
```

## Optimal Variational Distribution

$$
\begin{align}
q(\mathbf u) &= \mathcal N(\mathbf u\,|\,  \mathbf m, \mathbf \Lambda^{-1})\\
\mathbf \Lambda &= \mathbf K_{uu}^{-1} + \mathbf K_{uu}^{-1}\mathbf K_{uf}\mathbf K_{fu}\mathbf K_{uu}^{-1} \sigma^{-2}\\
\mathbf m &= \mathbf \Lambda^{-1} \mathbf K_{uu}^{-1}\mathbf K_{uf}\mathbf y\sigma^{-2}
\end{align}
$$

## Prediction

$$
\begin{equation}
p(\mathbf f^\star) = \mathcal N(\mathbf f^\star\,|\, \mathbf K_{\star u}\mathbf L^{-\top}\mathbf L_{\mathbf B}^{-\top}\mathbf c, \,\mathbf K_{\star\star} - \mathbf K_{\star u}\mathbf L^{-\top}(\mathbf I - \mathbf B^{-1})\mathbf L^{-1}\mathbf K_{u\star}),
\end{equation}
$$
where
$$
\mathbf c \triangleq \mathbf L_{\mathbf B}^{-1}\mathbf A\mathbf y \sigma^{-1}
$$

## Model Evidence


$$
\begin{align}
\mathcal{L} = &-\tfrac{N}{2}\log{2\pi} -\tfrac{1}{2}\log|\mathbf B| -\tfrac{N}{2}\log\sigma^{2} -\tfrac{1}{2}\sigma^{-2}\mathbf y^\top\mathbf y +\tfrac{1}{2}\sigma^{-2}\mathbf y^\top\mathbf A^{\top} \mathbf B^{-1}\mathbf A\mathbf y\\
&-\tfrac{1}{2}\sigma^{-2}\textrm{tr}(\mathbf K_{ff}) + \tfrac{1}{2}\textrm{tr}(\mathbf {AA}^\top),
\end{align}
$$
where
$$
\begin{align}
\mathbf A \triangleq & \mathbf L^{-1}\mathbf K_{uf}\sigma^{-1}\\
\mathbf B \triangleq & \mathbf I + \mathbf A\mathbf A^\top
\end{align}
$$
