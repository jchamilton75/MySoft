# Dear JC,
#
# I hope you're fine
#
# The weather data about Chajnantor (APEX) could be downloaded from the following link:
#
# http://www.apex-telescope.org/weather/Historical_weather
#
# Also, attached to this e-mail are two files:
#
# The named "get_weather_data.sh" is a script for Chajnantor data download. (if you need help how to use it please let me know).
# The named "tipper_macon.dat" you have oppacity information about Macon.
#
# I hope this information be useful for you.
#
#  Please do not hesitate to let me know if you need anything else.
#
# Regards
#
# Emiliano
#

####### Macon tipper data
ts,tau, tau_s = np.genfromtxt('/Users/hamilton/CMB/Interfero/Sites/FromEmiliano/tipper_macon.dat',skiprows=1,invalid_raise=False).T

import jdcal as jdcal
init = np.sum(jdcal.gcal2jd(1992,1,1))
newts = init+ts

yy=[]
mm=[]
dd=[]
ut=[]
for thets in ts:
    theyy, themm, thedd, theut = jdcal.jd2gcal(init,thets)
    yy.append(theyy)
    mm.append(themm)
    dd.append(thedd)
    ut.append(theut)


clf()
plot(time,tau,',')