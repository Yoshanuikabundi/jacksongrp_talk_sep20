#! /bin/bash

source /store/opt/gromacs-2018.8-plumed-2.4.6/bin/GMXRC

set -euf -o pipefail

xtc=/store/joshmitchell/linkers/REST2/cln_025_ext/3_prod/300.00/prod.xtc
tpr=/store/joshmitchell/linkers/REST2/cln_025_ext/3_prod/300.00/prod.tpr

center=Protein
output=Protein
fit=CA

echo $center $output | gmx trjconv -f $xtc -s $tpr -o tmp.xtc -ur compact -center -pbc mol
echo $fit $output | gmx trjconv -f tmp.xtc -s $tpr -o cln025_c22s.xtc -fit rot+trans
rm tmp.xtc

echo $center $output | gmx trjconv -f cln025_c22s.xtc -s $tpr -o cln025_c22s.pdb -dump 0

python3 make_fes.py
