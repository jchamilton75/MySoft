
export CodeDir=/Users/hamilton/Python/MySoft/Qubic/SpectroImager/

export rep=/Users/hamilton/Qubic/SpectroImager/LocalSims

export dictfile=/Users/hamilton/Qubic/SpectroImager/testFI.dict
export jobname=/Users/hamilton/Qubic/SpectroImager/LocalSims/TestFI
export tol=1e-4
export noutmin=1
export noutmax=4
export nptg=4000
export nf_sub=15


ipython ${CodeDir}/MCmpi_new.py ${dictfile} ${rep}/${jobname} ${tol} ${noutmin} ${noutmax} npointings ${nptg} seed 1 nf_sub ${nfsubin} #> ${rep}/Logs/${jobname}.txt

