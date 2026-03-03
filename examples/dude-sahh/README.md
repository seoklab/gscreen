# DUD-E SAHH target

URL: <https://dude.docking.org/targets/sahh>

## Description

`inputs` is a directory containing the input files for the DUD-E SAHH target.
There are 4 files in this directory:

1. `receptor_clean.pdb` - receptor structure in PDB format, with hydrogens added
   and extraneous molecules like water molecules removed.
2. `crystal_ligand.mol2` - reference (crystal) ligand structure in MOL2 format.
   Hydrogens and partial charges are added.
3. `actives_final_corina_addchg.mol2` - active ligands in MOL2 format. 3D
   structures were generated using CORINA, and hydrogens and partial charges
   are added.
4. `decoys_final_corina_addchg.mol2` - decoy ligands in MOL2 format. 3D
   structures were generated using CORINA, and hydrogens and partial charges
   are added.

`outputs` is a directory containing the expected output files. G-screen is run
in a two-step process. The first step is to generate the "interaction profile"
report from the reference receptor-ligand complex. `ganal` directory contains
the output file from this step (`ganal.json`).

The second step is to screen the active and decoy ligands against the receptor
structure. The output files from this step are in the `gscreen` directory.

Commands run to generate the output files and the cpu time taken are saved in
the `timing.log` file in each output directory.

## Interpreting the results

When run with more than 1 CPU cores, the output files will be split into each
CPU core's output directory (`split/partNN`). The pharmacophore analysis report
is saved in each split directory at `2_pcfilter/scores.csv`. We recommend the
following formula to calculate the final score $\textrm{GS-SP}$ for each ligand:

$$\textrm{GS-SP} = w \textrm{GS-S} + (1-w) \textrm{GS-P} $$

Where $\textrm{GS-S}$ is annotated as `shape_score` and $\textrm{GS-P}$ is
annotated as `pharma_score` in the `scores.csv` file. The weight `w` is
calculated as follows:

$$w =
\begin{cases}
0.9 & \textrm{if } \textrm{Tanimoto} \geq 0.25 \\
6 \cdot (\textrm{Tanimoto} - 0.1) & \textrm{if } 0.25 > \textrm{Tanimoto} \geq 0.1 \\
0 & \textrm{otherwise}
\end{cases}$$

$\textrm{Tanimoto}$ is the ECFP4 Tanimoto similarity between the query ligand
and the reference ligand (annotated as `tani_sim` in the `scores.csv` file).

## Notes

Result is based on commit [c037800](https://github.com/seoklab/gscreen/tree/c037800).
