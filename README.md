# IATheory

## Description

IATheory is a Python package built on top of PyCCL that provides a unified framework to compute galaxy clustering and intrinsic alignment (IA) estimators in both box and lightcone configurations. In the case of the lightcone, one can use spectroscopic or photometric redshifts. Magnification and lensing contributions can be added to photometric analyses.

Examples to evaluate the models and to run likelihood analyses are provided in the folder "examples".

## Theory summary

1. Box:

  $$ w_{\rm{gg}}(r_{\rm{p}}) = \int \frac{{\rm d} k_{\perp} k_{\perp}}{2\pi}J_{0}(k_{\perp} r_{\rm {p}})P_{\rm{gg}}(k_{\perp}, z) $$
  $$ w_{\rm{gp}}(r_{\rm{p}}) = - \int \frac{{\rm d} k_{\perp} k_{\perp}}{2\pi}J_{2}(k_{\perp} r_{\rm {p}})P_{\rm{gI}}(k_{\perp}, z) $$

where $r_{\rm{p}}$ is the projected distance, $k_{\perp}$ the perpendicular wavelength, $J_{0}$ and $J_{2}$ are the 0th and 2nd-order Bessel functions of the first kind, $P_{\rm{gg}}$ is the galaxy clustering power spectrum and $P_{\rm{gI}}$ is the galaxy-intrinsic power spectrum.

2. Lightcone spectroscopic redshifts:

  $$ w_{\rm{gg}}(r_{\rm {p}}) = \int {\rm d}z \mathcal{W}(z) \int \frac{ {\rm d} k_{\perp} k_{\perp}}{2\pi}J_{0}(k_{\perp} r_{\rm {p}})P_{\rm{gg}}(k_{\perp}, z) $$
  $$ w_{\rm{gp}}(r_{\rm {p}}) = -\int {\rm d}z \mathcal{W}(z) \int \frac{ {\rm d} k_{\perp} k_{\perp}}{2\pi}J_{2}(k_{\perp} r_{\rm {p}})P_{\rm{gI}}(k_{\perp}, z) $$

where $\mathcal{W}$ is the projection kernel:

  $$ \mathcal{W}(z) = \frac{n^{i}(z)n^{j}(z)}{\chi^{2}(z)d\chi/dz} \left [ \int {\rm d} z \frac{n^{i}(z)n^{j}(z)}{\chi^{2}(z)d\chi/dz} \right ]^{-1} $$

with $n^{i}$ the redshift distribution of the $i$ sample and $\chi(z)$ the comoving distance along the line of sight at redshift $z$.

3. Lightcone photometric redshifts:

  $$ w_{\rm{gg}} (r_{\rm {p}})=\int_{-\Pi_{{\rm max}}}^{\Pi_{{\rm max}}} {\rm d}\Pi \int {\rm d}z_{\mathrm{m}} \mathcal{W}(z_{\mathrm{m}})\xi_{\rm{gg}}(r_{\rm {p}}, \Pi, z_{\mathrm{m}}) $$
  $$ w_{\rm{gp}} (r_{\rm {p}})=\int_{-\Pi_{{\rm max}}}^{\Pi_{{\rm max}}}  {\rm d}\Pi \int {\rm d}z_{\mathrm{m}} \mathcal{W}(z_{\mathrm{m}})\xi_{\rm{gp}}(r_{\rm {p}}, \Pi, z_{\mathrm{m}}) $$

where $\Pi$ is the radial binning and we use the change of coordinates:

- $z_{\mathrm{m}} = (z_{1}+z_{2})/2$
- $r_{\rm{p}}= \theta \chi(z_{\mathrm{m}})$ 
- $\Pi = c(z_{2}-z_{1})/H(z_{\mathrm{m}})$

to obtain

 $$ \xi_{\rm{gg}}(\theta \mid z_{1}, z_{2}) = \frac{1}{2\pi} \int_{0}^{\infty} {\rm d}l l J_{0}(l\theta)C_{\rm{gg}}(l\mid z_{1}, z_{2}) $$
 $$ \xi_{\rm{gp}}(\theta \mid z_{1}, z_{2}) = \frac{1}{2\pi} \int_{0}^{\infty} {\rm d}l l J_{2}(l\theta)C_{\rm{gI}}(l\mid z_{1}, z_{2}) $$

where

 $$ C_{\rm{gg}}(l\mid z_{1}, z_{2}) = \int_{0}^{\chi_{\rm hor}} {\rm d}\chi' \frac{p_{n}(\chi'\mid \chi (z_{1})) p_{n}(\chi'\mid \chi (z_{2}))}{\chi'^{2}}  P_{\rm{gg}}\left (k = \frac{l+0.5}{\chi'}, z(\chi')\right ) $$
 $$ C_{\rm{gI}}(l\mid z_{1}, z_{2}) = \int_{0}^{\chi _{\rm hor}} {\rm d}\chi' \frac{p_{n}(\chi'\mid \chi (z_{1})) p_{e}(\chi'\mid \chi (z_{2}))}{\chi'^{2}}  P_{\rm{gI}}\left (k = \frac{l+0.5}{\chi'}, z(\chi')\right ) $$

with $\chi_{\rm{hor}}$ the comoving horizon distance and $p_{n/e}(\chi'\mid \chi(z_{1}))$ quantifying the error distribution of the dense and shape samples, respectively. 


Besides the purely galaxy-galaxy and galaxy-intrinsic contributions, the code allows to add the magnification and the lensing contributions to the photometric redshift case, such as the source-source and source-shape correlations, respectively:

 $$ C_{nn}(l) =  C_{\rm{gg}}(l) + C_{\rm{gm}}(l) + C_{\rm{mg}}(l) + C_{\rm{mm}}(l) $$
 $$ C_{ne}(l) =  C_{\mathrm{gI}}(l) + C_{\mathrm{gG}}(l) + C_{\mathrm{mI}}(l) + C_{\mathrm{mG}}(l) $$

with $m$ and $G$ denoting magnification and shear, respectively.

For a more complete explanation, see equations 28-45 of https://doi.org/10.1093/mnras/staf1630.

## Installation

This repository has some prerequisites, which you can find on the requirements.txt file (the python version for these libraries is 3.10.6). We recommend installing it inside a virtual environment:

### Option 1 - Conda (recommended for the examples)

The example notebooks need additional packages (`matplotlib`, `jupyter`, ...) that are easiest to install with conda-forge:

```bash
conda create -n IAtheory -y python=3.10
conda activate IAtheory

pip install -e .            # core package (editable install)
```

To also run the sampling/likelihood example:
```bash
pip install emcee nautilus-sampler
```

### Option 2 - pip in a virtual environment

```bash
python -m venv IA_Theory
source IA_Theory/bin/activate
pip install -e .            # core package + runtime deps
```

Install the optional extras only if you want to run the examples:

```bash
pip install -e ".[examples]"     # model-evaluation notebook deps
pip install -e ".[likelihood]"   # emcee + nautilus sampler
```

## Running the examples

The notebook kernel is named `iatheory`:

```bash
conda activate IAtheory
python -m ipykernel install --user --name iatheory
jupyter notebook examples/run_example_model_evaluation.ipynb
```

The box-case portion of `run_example_model_evaluation.ipynb` runs without any external data. The lightcone-spec, lightcone-phot (and likelihood) sections read catalogues and redshift distributions from hardcoded paths (e.g. `/nfs/...` and `/disks/...`) that are not included in this repository. To run those sections, point `path_nz_positions`, `path_nz_shapes`, and the catalogue paths in the notebooks/tests at your own data files.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

`tests/test_evaluate_box.py` runs standalone. The lightcone tests (`test_evaluate_lightcone_spec.py`, `test_evaluate_lightcone_phot.py`) additionally require the redshift-distribution CSV files referenced by the hardcoded NFS paths.

## License

This project is licensed under the MIT License. See the LICENSE file for details

## Citations

If you use this repository, we kindly ask you to cite https://doi.org/10.1093/mnras/staf1630 and https://arxiv.org/abs/2601.15851.
