# alphafold2mmopt

Based on `alphafold2`, `alphafoldmmopt` is developed to optimize molecules from any protein-generation model.

## dependence

```shell
conda install openmm
pip install dm-tree
pip install biopython
pip install dm-haiku
conda install -c omnia pdbfixer
pip install pytorch
```

## installation and run

* install

```shell
pip install alphafold2mmopt
```

* running: for a given any `*.pdb` file

```shell
# shell
alphafold2mmopt *.pdb
```
```python
# python
from proteinopt.common.protein import Protein
from proteinopt.relax.localminimizer import relax
prot = Protein(pdbname)
opt_pdb=relax(prot)
```
# Updation
## v0.2
> 1. Use pytorch instead of Jax to accelerate lDDT matrix calculation.
> > The lDDT calculation time can be reduced from 3s to 0.1s.
> 2. Reconstruct api for Protein object and reduce repeated clean operation.
