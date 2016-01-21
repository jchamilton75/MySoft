//=========================================================================
// Routines for Mll computation
//
// input:
//    spectra from the masks applied in temperature and polarization from l=0 to lmax
//    well = [well_TT,well_TP,well_PP]
//
// output:
//    mll matrices in RowMajorOrder, 5 matrices of size (lmax+1,lmax+1)
//    mll = [mll_TT_TT, mll_EE_EE, mll_EE_BB, mll_TE_TE, mll_EB_EB 
//
// dependancy:
//    Need FORTRAN routine wig3j_f.f to compute 3j-wigner
//
// comment:
//    loop on l3 should be 2times the lmax of the Mll matrices
//    in practice if you want to compute Mlls up to 3*nside then you can NOT
//    get Well up to 2*(3*nside) due to HEALPix pixel sizes
//    At first order, you can neglect the points above 3*nside in the l3 loop as
//    the wigner for large l3 get smaller and smaller...
//
// M. Tristram    - sep 2004 -  Polarized version of make_mll
//                - dec 2004 -  Add different mask for temperature and polarization
//                              clean memory in wig3j
//=========================================================================

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX(a,b) (((a)>(b))?(a):(b))

void wig3j_( double *L2, double *L3, double *M2, double *M3,
	     double *L1MIN, double *L1MAX, double *THRCOF, int *NDIM, int *IER);
void wig3j_c( int l2, int l3, int m2, int m3, double *wigner);
void make_mll_pol( long lmax, double *well, double *mll);





void make_mll_pol( long lmax, double *well, double *mll)
{
  double *mll_TT_TT, *mll_EE_EE, *mll_EE_BB, *mll_TE_TE, *mll_EB_EB;
  double *well_TT, *well_TP, *well_PP;
  double sum_TT, sum_TE, sum_EE_EE, sum_EE_BB, sum_EB;
  double *wigner0, *wigner2;
  long ndim, l3i, maxl3;
  long n=lmax+1;
  long well_lmax = lmax;   /* well supposed to be lmax+1 size long also */

  /* split Mll into components */
  mll_TT_TT = &mll[ 0*n*n];
  mll_EE_EE = &mll[ 1*n*n];
  mll_EE_BB = &mll[ 2*n*n];
  mll_TE_TE = &mll[ 3*n*n];
  mll_EB_EB = &mll[ 4*n*n];

  /* split well into components */
  well_TT = &well[ 0*(well_lmax+1)];
  well_TP = &well[ 1*(well_lmax+1)];
  well_PP = &well[ 2*(well_lmax+1)];  

  /* loop over the matrice elements */
  for( long l1=0; l1<=lmax; l1++) {
    for( long l2=0; l2<=lmax; l2++) {

      /* initialization */
      sum_TT    = 0.;
      sum_TE    = 0.;
      sum_EE_EE = 0.;
      sum_EE_BB = 0.;
      sum_EB    = 0.;

      /* alloc wigners */
      ndim = l2+l1-abs(l1-l2)+1;
      wigner0 = (double *) malloc( ndim * sizeof(double));
      printf("ndim_out: %lud\n", ndim);
      if (wigner0 == NULL) {
        printf("Could not allocate wigner0: %ld\n", ndim * sizeof(double));
        return;
      }
      wigner2 = (double *) malloc( ndim * sizeof(double));
      if (wigner2 == NULL) {
        printf("Could not allocate wigner0: %ld\n", ndim * sizeof(double));
        return;
      }

      /* compute wigners */
      wig3j_c( l1, l2,  0., 0., wigner0);
      wig3j_c( l1, l2, -2., 2., wigner2);

      printf("\n l1=%d l2=%d\n",l1,l2);
      for(long nn=0; nn<ndim; nn++)
        printf("      l3=%d w3=%d\n",nn,wigner0[nn]);


      /* loop on l3 */
      /*      maxl3 = (l1+l2 < well_lmax) : l1+l2 ? well_lmax; */
      maxl3 = (l1+l2 < well_lmax) ? l1+l2 : well_lmax;
      for( long l3=abs(l1-l2); l3<=maxl3; l3++) {

	if( (l1+l2+l3)%2 == 0) sum_TT    += well_TT[l3] * (double)(2.*l3+1.) * wigner0[l3] * wigner0[l3];
	if( (l1+l2+l3)%2 == 0) sum_TE    += well_TP[l3] * (double)(2.*l3+1.) * wigner0[l3] * wigner2[l3];
	if( (l1+l2+l3)%2 == 0) sum_EE_EE += well_PP[l3] * (double)(2.*l3+1.) * wigner2[l3] * wigner2[l3];
	if( (l1+l2+l3)%2 != 0) sum_EE_BB += well_PP[l3] * (double)(2.*l3+1.) * wigner2[l3] * wigner2[l3];
	sum_EB += well_PP[l3] * (double)(2.*l3+1.) * wigner2[l3] * wigner2[l3];
      }
      
      mll_TT_TT[ l1*n+l2] = (2.*(double)l2+1.)/(4.*M_PI) * sum_TT;
      mll_EE_EE[ l1*n+l2] = (2.*(double)l2+1.)/(4.*M_PI) * sum_EE_EE;
      mll_EE_BB[ l1*n+l2] = (2.*(double)l2+1.)/(4.*M_PI) * sum_EE_BB;
      mll_TE_TE[ l1*n+l2] = (2.*(double)l2+1.)/(4.*M_PI) * sum_TE;
      mll_EB_EB[ l1*n+l2] = (2.*(double)l2+1.)/(4.*M_PI) * sum_EB;

      free( wigner0);
      free( wigner2);

    } //end loop l2
  } //end loop l1
}







//Wrapper to FORTRAN routine wig3j_f.f
//Be careful, for vanishing wigner coeff, FORTRAN routine does not return 0.
void wig3j_c( int l2, int l3, int m2, int m3, double *wigner)
{
  double *THRCOF;
  double l1min, l1max;
  double dl2, dl3, dm2, dm3;
  int ndim, ier=0, l;

  l1min = MAX( abs(l2-l3), abs(m2+m3));
  l1max = l2 + l3;
  ndim = (int)(l1max-l1min+1);
  printf("ndim_c: %d,  l1min: %lf\n", ndim, l1min);

  dl2 = (double)l2;
  dl3 = (double)l3;
  dm2 = (double)m2;
  dm3 = (double)m3;

  /* wigner=0 for l<l1min */
  for( l=0; l<l1min; l++) wigner[l] = 0.;

  THRCOF = wigner + (int)l1min*0;    /* wig3j start at max( abs(l2-l3), abs(m2+m3)) */
  if( l2<abs(m2) || l3<abs(m3) || (l1max-l1min)<0 ) for( l=0; l<ndim; l++) THRCOF[l] = 0.;
  else wig3j_( &dl2, &dl3, &dm2, &dm3, &l1min, &l1max, THRCOF, &ndim, &ier);

  printf("\nIn wig3j_c\n");
  for(long i=0;i<ndim;i++) printf("i=%d w3=%d\n",i,wigner[i]);

  if( ier) {
    for( l=0; l<ndim; l++) THRCOF[l] = 0.;
    printf( "err=%d  l2=%d, l3=%d, m2=%d, m3=%d : ", ier, l2, l3, m2, m3);
    switch( ier)
    {
    case 1 : printf( "Either L2.LT.ABS(M2) or L3.LT.ABS(M3)\n"); break;
    case 2 : printf( "Either L2+ABS(M2) or L3+ABS(M3) non-integer\n"); break;
    case 3 : printf( "L1MAX-L1MIN not an integer (l1min=%d, l1max=%d)\n", (int)l1min, (int)l1max); break;
    case 4 : printf( "L1MAX less than L1MIN (l1min=%d, l1max=%d)\n", (int)l1min, (int)l1max); break;
    case 5 : printf( "NDIM less than L1MAX-L1MIN+1 (ndim=%d)\n", ndim); break;
    }
    fflush(stdout);
  }

}

int main (int argc, char *argv[]) {
  long lmax = 20;
  double *well;
  double *mll;
  long i, j, k;
  long n = lmax + 1;

  const char *kind[5];
  kind[0] = "TT_TT";
  kind[1] = "EE_EE";
  kind[2] = "EE_BB";
  kind[3] = "TE_TE";
  kind[4] = "EB_EB";

  well = (double *) malloc(3 * n * sizeof(double));
  mll = (double *) malloc(5 * n * n * sizeof(double));

  for (i = 0; i < 3 * n; i++) {
    well[i] = 1.;
  }
  make_mll_pol(lmax, well, mll);
  for (i = 0; i < 5; i++) {
    printf("%s\n=====\n", kind[i]);
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
        printf("%5.2lf ", mll[i*n*n + j*n + k]);
      }
      printf("\n");
    }
    printf("\n");
  }

  free(well);
  free(mll);
}
