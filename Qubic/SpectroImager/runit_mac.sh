
export CodeDir=/Users/hamilton/Python/MySoft/Qubic/SpectroImager/

export rep=/Users/hamilton/Qubic/SpectroImager/LocalSims/

export dictfile=/Users/hamilton/Qubic/SpectroImager/testFI.dict
export jobname=TestFI
export tol=1e-2
export noutmin=1
export noutmax=4
export nptg=100
export nfsubin=15
export noI=False


ipython ${CodeDir}/MCmpi_new.py ${dictfile} ${rep}/${jobname} ${tol} ${noutmin} ${noutmax} ${noI} npointings ${nptg} seed 1 nf_sub ${nfsubin}  #> ${rep}/Logs/${jobname}.txt



