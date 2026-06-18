rm -f log.txt
build_dataset4dPdR -i raw/results -r raw/start.extxyz -f aims_polarization -o dataset.extxyz >> log.txt
dPdR2bec -i dataset.extxyz >> log.txt