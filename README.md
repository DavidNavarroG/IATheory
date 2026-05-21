# IATheory

## Description

IATheory is a Python package built on top of PyCCL that provides a unified framework to compute galaxy clustering and intrinsic alignment (IA) estimators in both box and lightcone configurations. In the case of the lightcone, one can use spectroscopic or photometric redshifts. Magnification and lensing contributions can be added to photometric analyses.

Examples to evaluate the models and to run likelihood analyses are provided in the folder "examples".

## Theory summary

1. Box:

  $$ w_{\rm{gg}}(r_{\rm{p}}) = \int \frac{{\rm d} k_{\perp} k_{\perp}}{2\pi}J_{0}(k_{\perp} r_{\rm {p}})P_{\rm{gg}}(k_{\perp}, z) $$
  $$ w_{\rm{gp}}(r_{\rm{p}}) = - \int \frac{{\rm d} k_{\perp} k_{\perp}}{2\pi}J_{2}(k_{\perp} r_{\rm {p}})P_{\rm{gI}}(k_{\perp}, z) $$

where $r_{\rm{p}}$ is the projected distance, $k_{\perp}$ the perpendicular wavelength, and $J_{0}$ and $J_{2}$ are the 0th and 2nd-order Bessel functions of the first kind.

2. Lightcone spectroscopic redsfhits:

  $$ w_{\rm{gg}}(r_{\rm {p}}) = \int {\rm d}z \mathcal{W}(z) \int \frac{ {\rm d} k_{\perp} k_{\perp}}{2\pi}J_{0}(k_{\perp} r_{\rm {p}})P_{\rm{gg}}(k_{\perp}, z) $$
  $$ w_{\rm{gp}}(r_{\rm {p}}) = -\int {\rm d}z \mathcal{W}(z) \int \frac{ {\rm d} k_{\perp} k_{\perp}}{2\pi}J_{2}(k_{\perp} r_{\rm {p}})P_{\rm{gI}}(k_{\perp}, z) $$

where $\mathcal{W}$ is the projection kernel:

  $$ \mathcal{W}(z) = \frac{n^{i}(z)n^{j}(z)}{\chi^{2}(z)d\chi/dz} \left [ \int {\rm d} z \frac{n^{i}(z)n^{j}(z)}{\chi^{2}(z)d\chi/dz} \right ]^{-1} $$

with $n^{i}$ the redshift distribution of the $i$ sample and $\chi(z)$ the comoving distance along the \gls{los} at redshift $z$.

3. Lightcone photometric redsfhits:

  $$ w_{\rm{gg}} (r_{\rm {p}})=\int_{-\Pi_{{\rm max}}}^{\Pi_{{\rm max}}} {\rm d}\Pi \int {\rm d}z_{\mathrm{m}} \mathcal{W}(z_{\mathrm{m}})\xi_{\rm{gg}}(r_{\rm {p}}, \Pi, z_{\mathrm{m}}) $$
  $$ w_{\rm{gp}} (r_{\rm {p}})=\int_{-\Pi_{{\rm max}}}^{\Pi_{{\rm max}}}  {\rm d}\Pi \int {\rm d}z_{\mathrm{m}} \mathcal{W}(z_{\mathrm{m}})\xi_{\rm{gp}}(r_{\rm {p}}, \Pi, z_{\mathrm{m}}) $$

where

 $$ \xi_{\rm{gg}}(\theta \mid z_{1}, z_{2}) = \frac{1}{2\pi} \int_{0}^{\infty} {\rm d}l l J_{0}(l\theta)C_{\rm{gg}}(l\mid z_{1}, z_{2}) $$
 $$ \xi_{\rm{gp}}(\theta \mid z_{1}, z_{2}) = \frac{1}{2\pi} \int_{0}^{\infty} {\rm d}l l J_{2}(l\theta)C_{\rm{gI}}(l\mid z_{1}, z_{2}) $$

and

 $$ C_{\rm{gg}}(l\mid z_{1}, z_{2}) = \int_{0}^{\chi_{\rm hor}} {\rm d}\chi' \frac{p_{n}(\chi'\mid \chi (z_{1})) p_{n}(\chi'\mid \chi (z_{2}))}{\chi'^{2}}  P_{\rm{gg}}\left (k = \frac{l+0.5}{\chi'}, z(\chi')\right ) $$
 $$ C_{\rm{gI}}(l\mid z_{1}, z_{2}) = \int_{0}^{\chi _{\rm hor}} {\rm d}\chi' \frac{p_{n}(\chi'\mid \chi (z_{1})) p_{e}(\chi'\mid \chi (z_{2}))}{\chi'^{2}}  P_{\rm{gI}}\left (k = \frac{l+0.5}{\chi'}, z(\chi')\right ) $$

with $\chi_{\rm{hor}}$ is the comoving horizon distance and $p_{n/e}(\chi'\mid \chi(z_{1}))$ quantifies the error distribution of the dense and shape samples, respectively. 


Besides the purely galaxy-galaxy and galaxy-intrinsic contributions, the code allows to add the magnification and the lensing contributions to the photometric redshift case, such that the source-source and source-shape correlations, respectively:

 $$ C_{nn}(l) =  C_{\rm{gg}}(l) + C_{\rm{gm}}(l) + C_{\rm{mg}}(l) + C_{\rm{mm}}(l) $$
 $$ C_{ne}(l) =  C_{\mathrm{gI}}(l) + C_{\mathrm{gG}}(l) + C_{\mathrm{mI}}(l) + C_{\mathrm{mG}}(l) $$

with $m$ and $G$ denoting magnification and shear, respectively.

For a more complete explanation, see equations 28-45 of https://doi.org/10.1093/mnras/staf1630.

## Installation

This repository has some prerequisites, which you can find on the requirements.txt file (the python version for these libraries is 3.10.6). We recommend installing it inside a virtual environment:
```bash
python -m venv IA_Theory
source IA_Theory/bin/activate
pip install -r requirements.txt
```

Then, you can directly install **IATheory** from GitHub:
```bash
git clone git@github.com:DavidNavarroG/IATheory.git
cd IATheory
pip install -e .
```

## License

This project is licensed under the MIT License. See the LICENSE file for details

## Citations

If you use this repository, we kindly ask you to cite https://doi.org/10.1093/mnras/staf1630 and https://arxiv.org/abs/2601.15851.