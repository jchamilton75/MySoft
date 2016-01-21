### debug IDL/Python sur Da
#### IDL VERSION
zlya=2.46
H0=100.
cc=2.99792458d5
;cosm=[0.3,0.7,-1,0,0.7]
cosm=[0.29085374313206058, 0.70914625686793942, -2.1074896434130177, 0, 0.84454039449454288]
print,da(zlya,cosm)*cc/H0
print,hzval(zlya,cosm),rs(zlya,cosm,ob=0.0456),1./(hzval(zlya,cosm)*rs(zlya,cosm,ob=0.0456)),da(zlya,cosm)*cc/H0/rs(zlya,cosm,ob=0.0456)

wvals=-3+3*findgen(100)/99
da_idl=fltarr(100)
.r
for i=0,99 do begin
    ;cosm=[0.3,0.7,wvals[i],0,0.7]
    cosm=[0.29085374313206058, 0.70914625686793942, wvals[i], 0, 0.84454039449454288]
    da_idl[i]=da(zlya,cosm)*cc/H0
    print,wvals[i],da_idl[i]
endfor
end
save,file='test_w.save',wvals,da_idl



#### PYTHON COSMOLOPY
cosm=[0.1,0.1,-0.001,0,0.7]
zlya=2.46
import cosmolopy
cosmol=cosmolopy.fidcosmo.copy()
cosmol['h']=cosm[4]
cosmol['omega_M_0']=cosm[0]
cosmol['omega_lambda_0']=cosm[1]
cosmol['omega_k_0']=1.-(cosm[0]+cosm[1])
cosmol['w']=3*(cosm[2]+1)-1

#### PYTHON ASTROPY
import astropy
cosast=astropy.cosmology.wCDM(H0=cosm[4]*100,Om0=cosm[0],Ode0=cosm[1],w0=cosm[2])

#### my own code
from McMc import cosmo_utils
reload(cosmo_utils)
cosmojc=cosmolopy.fidcosmo.copy()
cosmojc['h']=cosm[4]
cosmojc['omega_M_0']=cosm[0]
cosmojc['omega_lambda_0']=cosm[1]
cosmojc['omega_k_0']=1.-(cosm[0]+cosm[1])
cosmojc['w']=cosm[2]

cosmolopy.distance.angular_diameter_distance(zlya,**cosmol)
cosast.angular_diameter_distance(zlya)
cosmo_utils.angdist(zlya,accurate=True,**cosmojc)

### Systematic comparison of astropy and cosmolopy and IDL
wvals=linspace(-3,0,100)
da_astropy=np.zeros(100)
da_cosmolopy=np.zeros(100)
for i in np.arange(100):
    cosmol=cosmolopy.fidcosmo.copy()
    cosmol['h']=cosm[4]
    cosmol['omega_M_0']=cosm[0]
    cosmol['omega_lambda_0']=cosm[1]
    cosmol['omega_k_0']=1.-(cosm[0]+cosm[1])
    ##### there is a bug in the cosmolopy.e_z code, a factor 3 missing so one needs to correct w the following way
    cosmol['w']=(wvals[i]+1)*3-1
    da_cosmolopy[i]=cosmolopy.distance.angular_diameter_distance(zlya,**cosmol)
    cosast=astropy.cosmology.wCDM(H0=cosm[4]*100,Om0=cosm[0],Ode0=cosm[1],w0=wvals[i])
    da_astropy[i]=cosast.angular_diameter_distance(zlya)

import scipy.io
bla=scipy.io.readsav('test_w.save')

clf()
plot(wvals,da_astropy,label='Astropy',lw=3)
plot(wvals,da_cosmolopy,'--',label='Cosmolopy',lw=3)
plot(bla['wvals'],bla['da_idl'],':',label='IDL',lw=3)
legend()


clf()
plot(wvals,(da_astropy-da_astropy)/da_astropy*100,label='Astropy',lw=3)
plot(wvals,(da_cosmolopy-da_astropy)/da_astropy*100,'--',label='Cosmolopy',lw=3)
plot(bla['wvals'],(bla['da_idl']-da_astropy)/da_astropy*100,':',label='IDL',lw=3)
legend()

#####
import astropy.cosmology
import cosmolopy
from McMc import cosmo_utils

om=0.27
ol=0.73
w=-1
h=1.
z=2.18728
cosmonico=cosmolopy.fidcosmo.copy()
cosmonico['h']=h
cosmonico['omega_M_0']=om
cosmonico['omega_lambda_0']=ol
cosmonico['omega_k_0']=0.
cosmonico['w']=w

cosmolopy.distance.comoving_distance(z,**cosmonico)
cosmo_utils.propdist(z,**cosmonico)
cos=astropy.cosmology.wCDM(H0=h*100,Om0=om,Ode0=ol,w0=w)
cos.comoving_distance(z)







