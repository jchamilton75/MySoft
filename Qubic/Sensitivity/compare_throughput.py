




#  BICEP2 from http://arxiv.org/abs/1403.4302v1
nchan_b2 = 500 #section 7  
fwhm_deg_b2 = 2.35 * 12./60 #section 4
omega_b2 = np.pi * (np.radians(fwhm_deg_b2)/2.35)**2
s_b2 = (2.8e-3)**2 *1e6  #fig 8
etendue_b2 = nchan_b2 * s_b2 * omega_b2 #unit: mm2.str

# QUBIC
nchan_Q = 400 
fwhm_deg_Q = 12.9 #section 4
omega_Q = np.pi * (np.radians(fwhm_deg_Q/2.35))**2
s_Q = np.pi*(12.33e-3/2)**2 *1e6   #fig 8
frac_integrated = 1
etendue_Q = nchan_Q * s_Q * omega_Q * frac_integrated #unit: mm2.str

print('{0:12s} | {1:12s} | {2:12s} | {3:12s} | {4:12s} | {5:12s} | {6:12s} |'.format('Name', 'Nchan',
    'FWHM[deg]','Omega[sr]','S[mm2]','E[mm2.sr]','sqrt(E)'))
print('{0:12s} | {1:12.2f} | {2:12.2f} | {3:12.2f} | {4:12.2f} | {5:12.2f} | {6:12.2f} |'.format('BICEP2', nchan_b2, 
    fwhm_deg_b2, omega_b2, s_b2, etendue_b2, np.sqrt(etendue_b2)))
print('{0:12s} | {1:12.2f} | {2:12.2f} | {3:12.2f} | {4:12.2f} | {5:12.2f} | {6:12.2f} |'.format('QUBIC', nchan_Q, 
    fwhm_deg_Q, omega_Q, s_Q, etendue_Q,np.sqrt(etendue_Q)))


print(np.sqrt(etendue_Q / etendue_b2))

headers = ['Nchan', 'FWHM[deg]','Omega[sr]','S[mm2]','E[mm2.sr]','sqrt(E)']
tab = np.array([[nchan_b2, 
    fwhm_deg_b2, omega_b2, s_b2, etendue_b2, np.sqrt(etendue_b2)], [nchan_Q, 
    fwhm_deg_Q, omega_Q, s_Q, etendue_Q,np.sqrt(etendue_Q)]])





