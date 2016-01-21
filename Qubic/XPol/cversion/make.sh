gfortran -c -g wig3j_f.f
gcc -o make_mll_pol -g -std=gnu99 make_mll_pol.c wig3j_f.o -lgfortran -lm
