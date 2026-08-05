ln -s ../../BaTiO3-clean/iter-3/models/BaTiO3.ch=32.rmax=5.0.max_ell=2.max_L=3.corr=2.seed=12345_stagetwo_compiled.model BaTiO3.model
pyenv activate fd2bec
for folder in cubic orthorombic tetragonal rhombohedral ; do
	cd ${folder}
	add_oxidation_numbers -i start.extxyz -o start.extxyz -n oxn -c ../oxn.json
	generate_displacements -i start.extxyz -w piezo -o displacements.extxyz
	cd -
done

mace
export PYTHONPATH="${PYTHONPATH}:/home/stoccoel/codes/mace-sabia"
source ~/codes/i-pi/env.sh 
for folder in cubic orthorombic tetragonal rhombohedral ; do
	cd ${folder}
	python ~/codes/i-pi/ipi/pes/_mace.py -m ../BaTiO3.model -i displacements.extxyz -o dataset.extxyz -mk ../mace_kwargs.json
	cd -
done

pyenv activate fd2bec
for folder in cubic orthorombic tetragonal rhombohedral ; do
	cd ${folder}
	dPdS2piezo -r start.extxyz -i dataset.extxyz --quantity dipole --dipole-keyword MACE_dipole --conventional-axes > log.txt
	cd -
done

