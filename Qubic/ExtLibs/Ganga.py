"""
Ganga.py

Holds a number of Ken's routines.

Dependencies:
- Julian.py
- numpy

  Date     Author  Remarks
---------- -------- -------
2009-07-24 K. Ganga Added MatplotlibTime2DateTime.
                   Updated DateTime2MatplotlibTime. 
2007-10-28 K. Ganga Added c, Tcmb. 
2007-10-25 K. Ganga First version. Added ll2uv, v2ll, ymd2jd. 
"""

# Conversions
AU2km = 149598000.0

# Parameters
c    = 299792458.0  # http://en.wikipedia.org/wiki/Speed_of_light
h    = 6.626068e-34
k    = 1.38062e-23
Tcmb = 2.725        # K. lambda.gsfc.nasa.gov/product/cobe/firas_overview.cfm.

month_numbers = {
   'January'  :  1, # Full month names in English
   'February' :  2,
   'March'    :  3,
   'April'    :  4,
   'May'      :  5,
   'June'     :  6,
   'July'     :  7,
   'August'   :  8,
   'September':  9,
   'October'  : 10,
   'November' : 11,
   'December' : 12,
   'Jan'      :  1, # Three-letter month name beginnings in English
   'Feb'      :  2,
   'Mar'      :  3,
   'Apr'      :  4,
   'May'      :  5,
   'Jun'      :  6,
   'Jul'      :  7,
   'Aug'      :  8,
   'Sep'      :  9,
   'Oct'      : 10,
   'Nov'      : 11,
   'Dec'      : 12,
   'JAN'      :  1, # Capitalized 3-letter month name starts in English
   'FEB'      :  2,
   'MAR'      :  3,
   'APR'      :  4,
   'MAY'      :  5,
   'JUN'      :  6,
   'JUL'      :  7,
   'JLY'      :  7,
   'AUG'      :  8,
   'SEP'      :  9,
   'OCT'      : 10,
   'NOV'      : 11,
   'DEC'      : 12
   }

# Functions
def BlackBody(nu, T, derivative = False):

   """
   Returns a Blackbody and optionally the derivative of a blackbody

   Inputs:
    nu: Frequency in Hz
    T : Temperature in K
    derivative: if set, calculate the derivative as well. 

   Outputs:
    derivative=False: Blackbody
    derivative=True : Blackbody, dBlackbody     
   """

   # Imports
   from numpy import exp

   x = (h*nu)/(k*T)
   expx = exp(x)
   B = (2*h*nu**3/c**2)/(expx-1.0)

   if derivative: return B, B*x*expx/(T*(expx-1.0))
   else:          return B

def cg(x, b, MultFunc, args = None, tol=1.0e-7, update=False, verbose=False):

   """
   cg

   Solves a linear system using a conjugate gradient method with no
   preconditioning.

   Inputs:
   x: Initial guess at solution vector. This *is* changed. 
   b: The right-hand-side of the equation Ax=b being solved.
   MultFunc: Name of the function that will perform the multiplicaiton by A
   args: Additional args to give to MultFunc
   tol: The tolerance. |d|/|x|<tol.
   update: If true, report after each iteration. 

      Date     Author  Remarks
   ---------- -------- -------
   2009-08-17 K. Ganga Slightly changed 'update' output format.
   2009-08-08 K. Ganga Ported the C from libkmg to Python. 
   """

   from numpy import dot, inf, sqrt, zeros # Imports

   iter = 0   # Initializations
   conv = inf
   rho1 = inf
   r = b - MultFunc(x, args)
   p = zeros(b.size, dtype=b.dtype)

   while conv > tol: # Loop until convergence
       rho2 = rho1
       rho1 = (r*r).sum()
       p = r - (rho1/rho2)*p
       q = MultFunc(p, args)
       alpha = rho1/dot(p, q)
       delta = alpha*p
       x += delta
       r -= alpha*q
       conv = sqrt((delta**2).sum()/(x**2).sum())

       iter += 1

       # Update after each loop, if desired
       if update: print "%d %8.2e %g" % (iter, conv, (x**2).sum()), 
   if update: print

   if verbose:
       print "Number of iterations:", iter
       print "Last fractional change:", conv

   # Later
   return x

def coth(x):

   """
   Returns the hyperbolic cotangent. 
   """

   # Imports
   from numpy import exp

   # Just do it
   return (exp(x)+exp(-x))/(exp(x)-exp(-x))

def Cross(a, b):

   """
   Calculates the cross product of two vectors

      Date     Author  Remarks
   ---------- -------- -------
   2007-12-04 K. Ganga First version
   """

   # Imports
   from numpy import array

   # Just do it
   return array([ a[1]*b[2]-a[2]*b[1],
                 -a[0]*b[2]+a[2]*b[0],
                  a[0]*b[1]-a[1]*b[0]])

def DateTime2JulianDate(dt):

   """
   Converts a datetime, given in python format, to a Julian date/time.

      Date     Author  Remarks
   ---------- -------- -------
   2009-06-02 K. Ganga First version
   """

   from numpy import atleast_1d # Imports

   jd = atleast_1d(dt) # Make sure we have an array

   for i in range(jd.size):
       month = jd[i].month
       year  = jd[i].year
       day   = jd[i].day \
               + (jd[i].hour \
                  + (jd[i].minute \
                     + (jd[i].second+1e-6*jd[i].microsecond)/60.0)/60.0)/24.0
       if month < 3:
           year  = year - 1
           month = month + 12
       julian = int(365.25*year) + int(30.6001*(month+1)) + day + 1720994.5
       tmp = year + month / 100.0 + day / 10000.0
       if tmp >= 1582.1015:
           A = year / 100
           B = 2 - A + A/4
           julian = julian + B
       jd[i] = 1.0*julian

   if jd.size == 1: jd = jd[0] # Return to scalar if we have a 1-d array

   # Later
   return jd

def DateTime2MatplotlibTime(dt):

   """
   Converts a datetime, given in python format, to a Matplotlib time,
   given in days since the beginning of year 1. WARNING: it looks like
   the day number for 0001-01-01T00:00:00 is actually 1. 

      Date     Author  Remarks
   ---------- -------- -------
   2009-07-24 K. Ganga Replaced my own function with the matplotlib function.
   2009-05-21 K. Ganga First version
   """

   #from datetime import datetime # Import python datetime
   #diff = t - datetime(1, 1, 1, 0, 0, 0, 0) # Find the python format diff
   # Later; Return difference as float days. 
   #return diff.days + (diff.seconds+1.0e-6*diff.microseconds)/(24.0*3600.0)

   from matplotlib.dates import date2num

   # Later
   return date2num(dt)

def dT2flux(nu, dT, FWHM):

   """
   # Converts a CMB fluctuation in uK_CMB to flux in Janskys
   #
   # Usage
   #  flux = dT2flux(nu, dT, FWHM)
   # Inputs:
   #  nu  : Frequency in Hz
   #  dT  : Temperature fluctuation in uK_cmb
   #  FWHM: The beam full-width-at-half-max in degrees
   # Outputs:
   #  flux: The equivalent point-source flux in Janskys
   #
   #    Date    Programmer Remarks
   # ---------- ---------- -------
   # 2001-12-20 K. Ganga   First version.
   """

   from numpy import exp, sqrt, log, pi

   # Remove the beam dilution
   # Gaussian beam width in radians
   sigma = FWHM*(1.0/sqrt(8.0*log(2.0)))*pi/180.0 
   dT = dT*2.0*pi*sigma*sigma       # Temperature over a beam in uK_cmb
   dT = 1.0e-6*dT                   # Convert to K_cmb
   #
   # Convert to flux
   x = h*nu/(k*Tcmb)
   expx = exp(x)
   expx_1 = expx - 1.0   
   flux = dT*((2.0*h*nu*nu*nu/(c*c))*x*expx/(Tcmb*expx_1*expx_1))
   #
   # Convert to Janskys
   flux = 1.0e26*flux
   #
   # Later
   return flux

def dT2RJ(dT, nu):

   """
   Finds the multiplicative factor which converts a temperature in
   CMB units to R-J units
   """

   # Imports
   from numpy import exp

   x = h*nu/(k*Tcmb)
   expx = exp(x)
   T_RJ = dT*expx*(x/(expx-1.0))**2

   # Later
   return T_RJ

def ecl2equ(elon, elat, jd = None, degrees=False):

   """
   Converts from equatorial to ecliptic coordinates
   """

   # Imports
   from numpy import array, cos, double, sin, zeros

   qR = Q_Ecl2Equ(jd = jd)

   q0 = zeros(4, dtype=double)
   q0[1:] = ll2uv(elon, elat, degrees=degrees)
   q = QuatRot(qR, q0)
   ra, dec = v2ll(q[1:], degrees=degrees)

   # Later
   return ra, dec

def ecl2gal(elon, elat, jd = None, degrees = False):

   """
   Converts from equatorial to ecliptic coordinates

      Date     Author  Remarks
   ---------- -------- -------
   2009-01-21 K. Ganga Modified to accept array arguments
   """

   # Imports
   from numpy import array, cos, double, sin, zeros

   qR = Q_Ecl2Gal(jd = jd)

   n = array(elon).size

   q0 = zeros((4, n), dtype=double)
   q0[1:] = ll2uv(elon, elat, degrees=degrees).reshape((3, n))
   q = QuatRot(qR, q0)
   l, b = v2ll(q[1:], degrees=degrees)

   # Later
   return l, b

def ecl2gal_uv(uv, jd=None):

   """
   Converts unit vectors from equatorial to ecliptic coordinates

      Date     Author  Remarks
   ---------- -------- -------
   2009-04-16 K. Ganga Modified 'ecl2gal' to make this.
   """

   # Convert the input into a quaternion
   from numpy import double, zeros
   q0 = zeros((4, uv[0].size), dtype=double)
   q0[1:] = uv.copy()

   # Later
   return (QuatRot(Q_Ecl2Gal(jd=jd), q0))[1:]

def EnsureDir(mydir, mymode = 0700, verbose = False):

   """
   Checks to make sure that a directory exists and if not creates it.

      Date     Author  Remarks
   ---------- -------- -------
   2009-06-05 K. Ganga First version
   """

   from os import makedirs
   from os.path import dirname, exists
   if not exists(mydir):
       if verbose: print "Creating", mydir
       makedirs(mydir, mymode)

   # Later
   return

def equ2ecl(ra, dec, jd = None, degrees=False):

   """
   Converts from equatorial to ecliptic coordinates

      Date     Author  Remarks
   ---------- -------- -------
   2009-01-21 K. Ganga Modified to accept array arguments
   2007-12-05 K. Ganga Added time dependence
   """

   # Imports
   from numpy import array, cos, double, sin, zeros

   s2r = (pi/180.0)/3600.0
   e = s2r*84381.448
   if jd != None:
       T = (jd-2451545.0)/36525.0
       e += -(s2r*46.84024)*T - (s2r*59.0e-5)*T**2 + (s2r*1813.0e-6)*T**3
       T = 0.0

   qR = array([cos(0.5*e), -sin(0.5*e), 0.0, 0.0])

   n = array(ra).size
   q0 = zeros((4, n), dtype=double)
   q0[1:] = ll2uv(ra, dec, degrees=degrees)
   q = QuatRot(qR, q0)
   elon, elat = v2ll(q[1:], degrees=degrees)

   # Later
   return elon, elat

def equ2gal(ra, dec, degrees=False):

   """
   Converts from equatorial to Galactic coordinates.

   I got the rotation matrix from slalib. I convert this to a
   quaternion using the recipe in
   http://en.wikipedia.org/wiki/Rotation_representation_%28mathematics%29
   """

   # Imports
   from numpy import array, double, empty, sqrt, zeros

   q0 = zeros(4, dtype=double)
   q0[1:] = ll2uv(ra, dec, degrees=degrees)
   q = QuatEqu2Gal(q0)
   l, b = v2ll(q[1:], degrees=degrees)

   # Later
   return l, b

def equ2hor(ra, dec, geolat, lst):

   """
   Convert from ra/dec to az/el
   """

   # Imports
   from numpy import arccos, arcsin, cos, pi, sin, where

   d2r = pi/180.0
   r2d = 180.0/pi
   sin_dec = sin(dec*d2r)
   phi_rad = geolat*d2r
   sin_phi = sin(phi_rad)
   cos_phi = cos(phi_rad)
   ha = 15.0*ra2ha(ra, lst)
   sin_el  = sin_dec*sin_phi + cos(dec*d2r)*cos_phi*cos(ha*d2r)
   el = arcsin(sin_el)*r2d

   az = arccos( (sin_dec-sin_phi*sin_el)/(cos_phi*cos(el*d2r)))
   az = where(sin(ha*d2r) > 0.0, 2.0*pi-az, az)*r2d

   # Later
   return az, el

def euler(ai, bi, select, FK4 = 0):

  """
  NAME:
      EULER
  PURPOSE:
      Transform between Galactic, celestial, and ecliptic coordinates.
  EXPLANATION:
      Use the procedure ASTRO to use this routine interactively

  CALLING SEQUENCE:
       EULER, AI, BI, AO, BO, [ SELECT, /FK4, SELECT = ] 

  INPUTS:
        AI - Input Longitude in DEGREES, scalar or vector.  If only two 
                parameters are supplied, then  AI and BI will be modified
                to contain the output longitude and latitude.
        BI - Input Latitude in DEGREES

  OPTIONAL INPUT:
        SELECT - Integer (1-6) specifying type of coordinate
                 transformation.

       SELECT   From          To        |   SELECT      From         To
        1     RA-Dec (2000)  Galactic   |     4       Ecliptic     RA-Dec
        2     Galactic       RA-DEC     |     5       Ecliptic    Galactic
        3     RA-Dec         Ecliptic   |     6       Galactic    Ecliptic

       If not supplied as a parameter or keyword, then EULER will prompt
       for the value of SELECT
       Celestial coordinates (RA, Dec) should be given in equinox J2000 
       unless the /FK4 keyword is set.
  OUTPUTS:
        AO - Output Longitude in DEGREES
        BO - Output Latitude in DEGREES

  INPUT KEYWORD:
        /FK4 - If this keyword is set and non-zero, then input and output 
              celestial and ecliptic coordinates should be given in
              equinox B1950.
        /SELECT  - The coordinate conversion integer (1-6) may
                   alternatively be specified as a keyword
  NOTES:
        EULER was changed in December 1998 to use J2000 coordinates as the
        default, ** and may be incompatible with earlier versions***.
  REVISION HISTORY:
        Written W. Landsman,  February 1987
        Adapted from Fortran by Daryl Yentis NRL
        Converted to IDL V5.0   W. Landsman   September 1997
        Made J2000 the default, added /FK4 keyword
         W. Landsman December 1998
        Add option to specify SELECT as a keyword W. Landsman March 2003
  """

  # Imports
  from numpy import arcsin, arctan2, cos, fmod, pi, sin

  # npar = N_params()
  #  if npar LT 2 then begin
  #     print,'Syntax - EULER, AI, BI, A0, B0, [ SELECT, /FK4, SELECT= ]'
  #     print,'    AI,BI - Input longitude,latitude in degrees'
  #     print,'    AO,BO - Output longitude, latitude in degrees'
  #     print,'    SELECT - Scalar (1-6) specifying transformation type'
  #     return
  #  endif

  PI = pi
  twopi   =   2.0*PI
  fourpi  =   4.0*PI
  deg_to_rad = 180.0/PI
  # 
  # ;   J2000 coordinate conversions are based on the following constants
  # ;   (see the Hipparcos explanatory supplement).
  # ;  eps = 23.4392911111 # Obliquity of the ecliptic
  # ;  alphaG = 192.85948d           Right Ascension of Galactic North Pole
  # ;  deltaG = 27.12825d            Declination of Galactic North Pole
  # ;  lomega = 32.93192d            Galactic longitude of celestial equator  
  # ;  alphaE = 180.02322d           Ecliptic longitude of Galactic North Pole
  # ;  deltaE = 29.811438523d        Ecliptic latitude of Galactic North Pole
  # ;  Eomega  = 6.3839743d          Galactic longitude of ecliptic equator
  # 
  if FK4 == 1:

     equinox = '(B1950)' 
     psi   = [ 0.57595865315, 4.9261918136,
               0.00000000000, 0.0000000000,
               0.11129056012, 4.7005372834]     
     stheta =[ 0.88781538514,-0.88781538514,
               0.39788119938,-0.39788119938,
               0.86766174755,-0.86766174755]    
     ctheta =[ 0.46019978478, 0.46019978478,
               0.91743694670, 0.91743694670,
               0.49715499774, 0.49715499774]    
     phi  = [ 4.9261918136,  0.57595865315,
              0.0000000000, 0.00000000000,
              4.7005372834, 0.11129056012]
  else:

     equinox = '(J2000)'
     psi   = [ 0.57477043300, 4.9368292465,  
               0.00000000000, 0.0000000000,  
               0.11142137093, 4.71279419371]     
     stheta =[ 0.88998808748,-0.88998808748, 
               0.39777715593,-0.39777715593, 
               0.86766622025,-0.86766622025]    
     ctheta =[ 0.45598377618, 0.45598377618, 
               0.91748206207, 0.91748206207, 
               0.49714719172, 0.49714719172]    
     phi  = [ 4.9368292465,  0.57477043300, 
              0.0000000000, 0.00000000000, 
              4.71279419371, 0.11142137093]
  # 
  i  = select - 1                         # IDL offset
  a  = ai/deg_to_rad - phi[i]
  b = bi/deg_to_rad
  sb = sin(b)
  cb = cos(b)
  cbsa = cb * sin(a)
  b  = -stheta[i] * cbsa + ctheta[i] * sb
  #bo    = math.asin(where(b<1.0, b, 1.0)*deg_to_rad)
  bo    = arcsin(b)*deg_to_rad
  #
  a = arctan2( ctheta[i] * cbsa + stheta[i] * sb, cb * cos(a) )
  ao = fmod( (a+psi[i]+fourpi), twopi) * deg_to_rad
  return ao, bo

def G():

   """
   Returns Newton's gravitational constant in m^3/kg/s^2. See
   http://en.wikipedia.org/wiki/Gravitational_constant

      Date    Programmer Remarks
   ---------- ---------- -------
   2008-07-24 K. Ganga   First version.
   """

   # Just do it
   return 6.67428e-11

def gal2ecl(l, b, jd = None, degrees = False):

   """
   Converts from Galactic to ecliptic coordinates

      Date     Author  Remarks
   ---------- -------- -------
   2009-01-21 K. Ganga Modified to accept array inputs
   """

   # Imports
   from numpy import array, cos, double, sin, zeros

   qR = QuatInv(Q_Ecl2Gal(jd = jd))

   n = array(l).size
   q0 = zeros((4, n), dtype=double)
   q0[1:] = ll2uv(l, b, degrees=degrees).reshape((3, n))
   q = QuatRot(qR, q0)
   elon, elat = v2ll(q[1:], degrees=degrees)

   # Later
   return elon, elat

def gal2equ(l, b, degrees=False):

   """
   Converts from equatorial to Galactic coordinates.

   I got the rotation matrix from slalib. I convert this to a
   quaternion using the recipe in
   http://en.wikipedia.org/wiki/Rotation_representation_%28mathematics%29
   """

   # Imports
   from numpy import array, double, empty, sqrt, zeros

   q0 = zeros(4, dtype=double)
   q0[1:] = ll2uv(l, b, degrees=degrees)
   q = QuatGal2Equ(q0)
   ra, dec = v2ll(q[1:], degrees=degrees)

   # Later
   return ra, dec

def gst2lst(gst, geolon):

   # gst: Greenwich standard time in hours
   # geolon: Geographic longitude EAST in degrees. 
   #
   # Later
   return (gst + geolon/15.0)%24.

def hor2equ(az, el, geolat, lst):

   """
   Convert from az/el to ra/dec. 
   """

   # Imports
   from numpy import arccos, arcsin, cos, pi, sin, where

   d2r = pi/180.0
   r2d = 180.0/pi
   az_r     = az*d2r
   el_r     = el*d2r
   geolat_r = geolat*d2r

   # Convert to equatorial coordinates
   cos_el  = cos(el_r)
   sin_el  = sin(el_r)
   cos_phi = cos(geolat_r)
   sin_phi = sin(geolat_r)
   cos_az  = cos(az_r)
   sin_dec = sin_el*sin_phi + cos_el*cos_phi*cos_az
   dec     = arcsin(sin_dec)*r2d
   cos_ha  = (sin_el-sin_phi*sin_dec)/(cos_phi*cos(dec*d2r))
   cos_ha  = where(cos_ha <= -1.0, -0.99999, cos_ha)
   cos_ha  = where(cos_ha >=  1.0,  0.99999, cos_ha)
   ha      = arccos(cos_ha)
   ha      = where( sin(az_r) > 0.0 , 2.0*pi-ha, ha)*r2d

   ra      = lst*15.0-ha
   ra = where(ra >= 360.0, ra-360.0, ra)
   ra = where(ra <    0.0, ra+360.0, ra)

   # Later
   return ra, dec

def jd2tick(jd):

   """
   Takes a vector of Julian dates on input and comes up with tick locations
   and marks. 
   """

   # Imports
   from Ganga import jd2ymd, ymd2jd

   months = ['X', 'J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

   y, m, d = jd2ymd(jd.min())
   y = y[0]
   m = m[0]
   mark = []
   locs = []
   thisjd = ymd2jd(y, m, 0.0)
   while thisjd < jd.max():
       thismark = months[m]
       if m == 1: thismark = str(y)
       mark.append(thismark)
       locs.append(thisjd)
       m = m + 1
       if m > 12:
           m = 1
           y = y + 1
       thisjd = ymd2jd(y, m, 0.0)

   # Later
   return locs, mark

def jd2ymd(jd):

   """
   Given a Julian date, calculate the year, month and (decimal) day

   Usage:
    year, month, day = jd2ymd(jd)
   Inputs:
    jd : The Julian date
   Outputs:
    year : The year  (integer)
    month: The month (integer)
    day  : The day   (float)


      Date    Programmer Remarks
   ---------- ---------- -------
   2002-05-22 K. Ganga   First version. Adapted from Duffett-Smith. 
   2002-05-26 K. Ganga   Moved to separate file.
   """

   # Imports
   from numpy import array, where

   year = 0
   month = 0
   day = 0

   jd = array([jd])

   tmp = jd + 0.5
   I = tmp.astype(int)
   F = tmp - I

   B = I
   index = where(I > 2299160)
   if len(index) > 0:
       A = ((I[index]-1867216.25)/36524.25).astype(int)
       B = I[index] + 1 + A - (0.25*A).astype(int)

   C = B + 1524
   D = ((C-122.1)/365.25).astype(int)
   E = (365.25*D).astype(int)
   G = ((C-E)/30.6001).astype(int)
   d = C - E + F - (30.6001*G).astype(int)
   m = where(G > 13, G-13, G-1)
   y = where(m >  2, D-4716, D-4715)

   # Later
   return y, m, d

def ll2uv(lon, lat, degrees = False):

   """
   Convert a longitude/latitude pair to unit vector.

   Usage:
uv = ll2uv(lon, lat [, degrees=True])

   Inputs:
    lon: a vector or longitudes in radians (unless degrees is set to true)
    lat: a vector or latitudes in radians (unless degrees is set to true)

   Optional Inputs:
    degrees: Set to true if the inputs are in degrees rather than
             radians (radians are the default)

   Outputs:
    uv: Set of unit vectors. 

      Date     Author  Remarks
   ---------- -------- -------
   2007-10-25 K. Ganga First version.
   """

   # Imports
   from numpy import array, cos, pi, sin

   # Convert from degrees, if necessary, and make it an array
   factor = (pi/180.0)

   if degrees:
       lat = (pi/180.0)*array(lat)
       lon = (pi/180.0)*array(lon)
   else:
       lat = array(lat)
       lon = array(lon)

   # Later
   return array([cos(lat)*cos(lon), cos(lat)*sin(lon), sin(lat)])

def MatplotlibTime2DateTime(mpt):

   """
   Converts a Matplotlib time, given in days since the beginning of
   year 1 to a datetime, given in python format. WARNING: it looks
   like the day number for 0001-01-01T00:00:00 is actually 1.

      Date     Author  Remarks
   ---------- -------- -------
   2009-07-24 K. Ganga First version. 
   """

   from matplotlib.dates import num2date

   # Later
   return num2date(mpt)

def medfit(x, y):

   """
   Does a median fit
   """

   from numpy import median, sort

   n = x.size
   if y.size != x.size: exit("Problem with sizes.")

   sorted = sort(x)
   x0 = median(sorted[:n/2])
   x1 = median(sorted[n/2:])

   sorted = sort(y)
   y0 = median(sorted[:n/2])
   y1 = median(sorted[n/2:])

   m = (y1-y0)/(x1-x0)
   b = (x1*y0-x0*y1)/(x1-x0)

   # Later
   return m, b    

def Month2Num(m):

   """
   Takes a month in string format and returns the month of the year
   as a number.

   Example:
    % python
from Ganga import Month2Num
Month2Num('May')
    5

      Date     Author  Remarks
   ---------- -------- -------
   2009-09-12 K. Ganga First version.
   """

   # Later
   return month_numbers[m]

def nlf_func(x, c_in, derivative = False):

   c = array(c_in, dtype=double) # Make sure that 'c' is an array
   nx = x.size # Get the data set size
   nc = c.size # Get the number of fit parameters

   # Create the fitting function
   f = c[0] + c[1]*x**c[3] + c[2]*x**c[4]
   if not derivative: return f

   # Calculate the fit matrix 
   df = empty((nc, nx), dtype=double)
   df[0] = ones(nx, dtype=double)
   df[1] = x**c[3]
   df[2] = x**c[4]
   df[3] = c[1]*log(x)*x**c[3]
   df[4] = c[2]*log(x)*x**c[4]

   # Later
   return f, df

def nllsf(y, function, guess, x, limit = 1.0e-7, maxiter = 100000, 
         verbose = False, doplot = False):

   """
   Does a nonlinear least squared fit. 

   Inputs:
    y: The data to fit
    function:
     Here is an example:

      def nlf_func(x, c_in, derivative = True):

          c = array(c_in, dtype=double) # Make sure that 'c' is an array
          nx = x.size # Get the data set size
          nc = c.size # Get the number of fit parameters

          # Create the fitting function
          f = c[0] + c[1]*x**c[3] + c[2]*x**c[4]
          if not derivative: return f

          # Calculate the fit matrix 
          df = empty((nc, nx), dtype=double)
          df[0] = ones(nx, dtype=double)
          df[1] = x**c[3]
          df[2] = x**c[4]
          df[3] = c[1]*log(x)*x**c[3]
          df[4] = c[2]*log(x)*x**c[4]

          # Later
          return f, df
    guess: An initial guess at the parameters
    x : The abcissa values

   Outputs:

   Uses the algorithm presented at
   http://mathworld.wolfram.com/NonlinearLeastSquaresFitting.html

      Date     Author  Remarks
   ---------- -------- -------
   2008-05-26 K. Ganga First version.
   """

   # Imports
   from numpy import array, dot, double, linalg

   # Print out the setup, if desired
   if verbose:
       print "nllsf setup:"
       print "Number of parameters:", array(guess).size
       print "Number of points:", x.size
       print "Tolerance:", limit
       print "Do Plot?", doplot

   # Define the level we need the residual to go to in order to finish
   std = limit*y.std()
   r = y.std()

   # Do the fit
   params = array(guess, dtype=double).copy() # Copy the guess
   for iter in range(maxiter):                # Begin iterating

       # Create the values and deriv.
       f, df = function(x, params, derivative=True)
       residual = y - f                       # Create residual
       if abs(residual.std()-r) <= std: break        # Bail if we're okay
       r = residual.std()

       # Find the change in parameters needed
       dparam = linalg.solve(dot(df, df.transpose()), dot(df, residual))
       params += dparam

   # Print some info, if desired
   if verbose:
       print "Input relative standard dev. limit:", limit
       print "Input maximum number of iterations:", maxiter
       print "Output relative standard dev.:", residual.std()/y.std()
       print "Output number of iterations:", iter

   # Make a plot, if desired
   if doplot:
       import pylab
       pylab.loglog(x, y, 'o', label = "data")
       pylab.loglog(x, function(x, guess), label = "guess")
       pylab.loglog(x, function(x, params), label = "fit")
       pylab.xlabel("x")
       pylab.ylabel("y")
       pylab.legend()
       pylab.show()

   # Later
   return params

def npix2nside(npix):

   """
   npix2nside

   Given the number of pixels in a full-sky HEALPix map, this returns the
   corresponding 'nside'.
   """

   if   npix ==         12 : nside =     1
   elif npix ==         48 : nside =     2
   elif npix ==        192 : nside =     4
   elif npix ==        768 : nside =     8
   elif npix ==       3072 : nside =    16
   elif npix ==      12288 : nside =    32
   elif npix ==      49152 : nside =    64
   elif npix ==     196608 : nside =   128
   elif npix ==     786432 : nside =   256
   elif npix ==    3145728 : nside =   512
   elif npix ==   12582912 : nside =  1024
   elif npix ==   50331648 : nside =  2048
   elif npix ==  201326592 : nside =  4096
   elif npix ==  805306368 : nside =  8192
   elif npix == 3221225472 : nside = 16384
   else:
       exit("Disallowed number of pixels")

   # Later
   return nside

def nside2npix(nside):

   """
   nside2npix

   Given a HEALPix 'nside', this returns the number of pixels in the full sky.
   """

   # Later
   return 12*long(nside)*long(nside)

def nside2pixsize(nside):

   """
   Converts a HEALPix 'nside' to a pixel width in arcminutes
   """

   # Imports
   from numpy import pi, sqrt

   # Just do it
   return sqrt(4.0*pi/nside2npix(nside))*180.0*60/pi

def Parsec(units = 'm'):

   """
   Returns the value of a parsec in meters. See
   http://en.wikipedia.org/wiki/Parsec. 

      Date     Author  Remarks
   ---------- -------- -------
   2008-07-24 K. Ganga First version.
   """

   out = 30.857e15
   if units == 'km': out *= 1.0e-3

   # Just do it
   return out

def parseDateTime(s):

   """
   Create datetime object representing date/time
   expressed in a string

   Takes a string in the format produced by calling str()
   on a python datetime object and returns a datetime
   instance that would produce that string.

   Acceptable formats are:
   "YYYY-MM-DD HH:MM:SS.ssssss+HH:MM",
   "YYYY-MM-DD HH:MM:SS.ssssss",
   "YYYY-MM-DD HH:MM:SS+HH:MM",
   "YYYY-MM-DD HH:MM:SS"
   Where ssssss represents fractional seconds. The timezone
   is optional and may be either positive or negative
   hours/minutes east of UTC.
   """

   import re
   from datetime import datetime

   if s is None: return None

   # Split string in the form 2007-06-18 19:39:25.3300-07:00
   # into its constituent date/time, microseconds, and
   # timezone fields where microseconds and timezone are
   # optional.
   m = re.match(r'(.*?)(?:\.(\d+))?(([-+]\d{1,2}):(\d{2}))?$', str(s))
   datestr, fractional, tzname, tzhour, tzmin = m.groups()

   # Create tzinfo object representing the timezone
   # expressed in the input string.  The names we give
   # for the timezones are lame: they are just the offset
   # from UTC (as it appeared in the input string).  We
   # handle UTC specially since it is a very common case
   # and we know its name.
   if tzname is None: tz = None
   else:
       tzhour, tzmin = int(tzhour), int(tzmin)
       if tzhour == tzmin == 0: tzname = 'UTC'
       tz = FixedOffset(timedelta(hours=tzhour, minutes=tzmin), tzname)

   # Convert the date/time field into a python datetime
   # object.
   x = datetime.strptime(datestr, "%Y-%m-%d %H:%M:%S")

   # Convert the fractional second portion into a count
   # of microseconds.
   if fractional is None: fractional = '0'
   fracpower = 6 - len(fractional)
   fractional = float(fractional) * (10 ** fracpower)

   # Return updated datetime object with microseconds and
   # timezone information.
   return x.replace(microsecond=int(fractional), tzinfo=tz)            

def pcg(x, b, MultFunc, InvFunc, args=None, tol=1.0e-7, update=False, 
       verbose=False):

   """
   pcg

   Solves a linear system using a conjugate gradient method with
   preconditioning.

   Inputs:
   x: Initial guess at solution vector. This *is* changed. 
   b: The right-hand-side of the equation Ax=b being solved.
   MultFunc: The name of the function that will perform the multiplicaiton by A
   InvFunc: The name of the function that will perform approximate 
    multiplication by the inverse of A
   args: Additional args to give to MultFunc
   tol: The tolerance. |d|/|x|<tol.
   update: If true, report after each iteration. 

      Date     Author  Remarks
   ---------- -------- -------
   2009-08-12 K. Ganga Ported the C from libkmg to Python. 
   """

   # Imports
   from numpy import dot, inf, sqrt, zeros

   # Calculate r=b-Ax
   r = b - MultFunc(x, args)
   p = zeros(b.size, dtype=b.dtype)

   iter = 0
   conv = inf
   rho1 = inf
   rho2 = 0.0
   while conv > tol:

       # Calculate z=M^{-1}r
       z = InvFunc(r, args); 

       # Calculate the 'rho's
       rho2 = rho1
       rho1 = dot(r, z)

       beta = rho1/rho2
       p = z - beta*p

       q = MultFunc(p, args)

       alpha = rho1/dot(p, q)

       ftmp = alpha*p
       x += ftmp
       r -= alpha*q
       conv = sqrt((ftmp**2).sum()/(x**2).sum())
       iter += 1

   if verbose:
       print "Number of iterations:", iter
       print "Last fractional change:", conv

   # Later
   return x

def Phatmm(x, m):

   """
   Calculates legendre 
   """
   from numpy import array, sqrt, pi

   f = 0.25/pi
   for twom in range(2, 2*m+1, 2): f *= (twom+1.0)/twom
   return sqrt(f)*(array(x)**2-1.0)**m

def PolyFit(x, y, o):

   """
   Fits a function to a polynomial

   To do:
   - Make x optional, and assume uniform spacing if not given. 
   """

   # Imports
   from numpy import dot, dtype, empty, power, size
   from numpy.linalg import inv

   # Check inputs
   if size(y) < o+1: exit("Not enough points for this order fit.")

   # Create an x array, if not given
   if x == None: x = arange(size(y), dtype=dtype(y))

   # Check array sizes
   if size(x) != size(y): exit("Problem with array sizes.")

   # Do the fit and return the answer
   F = empty((o+1,size(x)), dtype=dtype(y))
   for i in range(o+1): F[i] = power(x, i)
   return dot(inv(dot(F, F.transpose())), dot(F, y))

def Q_Ecl2Equ(jd = None):

   """
   Returns the quaternion which will rotate from Ecliptic to
   Equatorial coordinates.

   if jd=None, this assumes J2000.

   See http://en.wikipedia.org/wiki/Axial_tilt for a discussion of the axial
   tilt.

      Date     Author  Remarks
   ---------- -------- -------
   2009-01-21 K. Ganga Modified to accept array inputs
   2007-12-05 K. Ganga Added time dependence
   """

   # Imports
   from numpy import array, cos, double, empty, pi, sin, zeros

   s2r = (pi/180.0)/3600.0
   e = s2r*84381.448
   if jd != None:
       T = (jd-2451545.0)/36525.0
       e += -(s2r*46.84024)*T - (s2r*59.0e-5)*T**2 + (s2r*1813.0e-6)*T**3
       T = 0.0

   #qR = array([cos(0.5*e), sin(0.5*e), 0.0, 0.0])
   n = array(e).size
   qR = zeros((4, n), dtype=double)
   qR[0] = cos(0.5*e)
   qR[1] = sin(0.5*e)

   # Later
   return qR

def Q_Ecl2Gal(jd = None):

   """
   Returns the quaternion which will rotate from Ecliptic to
   Galactic coordinates.
   """

   # Just do it
   return QuatMul(QuatInv(Q_Gal2Equ()), Q_Ecl2Equ(jd = jd))

def Q_Gal2Equ():

   """
   Returns the quaternion which will rotate from Galactic to
   Equatorial coordinates.
   """

   # Imports
   from numpy import double, empty, sqrt

   R = R_gal2equ()

   qR = empty(4, dtype=double)
   qR[0] = 0.5*sqrt(1.0+R[0,0]+R[1,1]+R[2,2])
   qR[1] = (0.25/qR[0])*(R[1,2]-R[2,1])
   qR[2] = (0.25/qR[0])*(R[2,0]-R[0,2])
   qR[3] = (0.25/qR[0])*(R[0,1]-R[1,0])

   # Later
   return qR

def QuatAngle(qA, qE):

   """
   Calculate a bolometer angle on the sky
   """

   from Ganga import Cross
   from numpy import double, sqrt, zeros

   # Find the y-axis. It points East. See page 16 of the HEALPix Primer
   # Then find the x-axis using the cross product of the detector and the
   # y-axis. With this, get the tangent of the angle and others stuff.
   norm = sqrt(qE[1]**2+qE[2]**2)
   y = zeros((3, qE[0].size), dtype=double)
   y[0] = -qE[2]/norm
   y[1] =  qE[1]/norm
   x = Cross(y,  qE[1:]) # Use Cosmo convention by default
   if False: x = -x # IAU Convention
   tanpsi = (y[0]*qA[1] + y[1]*qA[2] + y[2]*qA[3])/ \
            (x[0]*qA[1] + x[1]*qA[2] + x[2]*qA[3])
   sin2ang =      2.0*tanpsi/(1.0+tanpsi**2)
   cos2ang = (1.0-tanpsi**2)/(1.0+tanpsi**2)

   # Later
   return cos2ang, sin2ang

def QuatCon(q):

   """
   Finds the conjugate of a quaternion

   Taken from http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q49
   """

   c = q.copy()
   c[1] = -c[1]
   c[2] = -c[2]
   c[3] = -c[3]

   return c

def QuatEcl2Equ(q, jd = None):

   """
   Rotates a quaternion from Ecliptic to Equatorial coordinates
   """

   # Just do it
   return QuatRot(Q_Ecl2Equ(jd = jd), q)

def QuatEcl2Gal(q, jd = None):

   """
   Rotates a quaternion from Ecliptic to Galactic coordinates
   """

   # Just do it
   return QuatRot(Q_Ecl2Gal(jd = jd), q)

def QuatEqu2Gal(q):

   """
   Rotates a quaternion from Equatorial to Galactic coordinates
   """

   # Just do it
   return QuatRot(QuatInv(Q_Gal2Equ()), q)

def QuatGal2Equ(q):

   """
   Rotates a quaternion from Galactic to Equatorial coordinates
   """

   # Just do it
   return QuatRot(Q_Gal2Equ(), q)

def QuatInv(q):

   """
   Finds the inverse of a quaternion

   Taken from http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q50
   """

   return QuatCon(q)/QuatMag(q)

def QuatMag(q):

   """
   Finds the magnitude of a quaternion.

   Taken from http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q51
   """

   # Imports
   from numpy import sqrt

   # Later
   return sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])

def QuatMul(q1, q2):

   """
   Multiplies two quaternions, q1 and q2. Result is q1.q2.

   Taken from http:/www.j3d.org/matrix_faq/matrfaq_latest.html#Q53
   """

   # Imports
   from numpy import array

   return array([
       q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3], 
       q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2], 
       q1[0]*q2[2] + q1[2]*q2[0] + q1[3]*q2[1] - q1[1]*q2[3], 
       q1[0]*q2[3] + q1[3]*q2[0] + q1[1]*q2[2] - q1[2]*q2[1]])

def QuatRot(q, v):

   """
   Rotates a vector v, represented as a quaternion, by the rotation
   represented by the quaternion q.

   Taken from http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q63
   """

   # Just do it
   return QuatMul(q, QuatMul(v, QuatInv(q)))

def QuatSqr(q):

   """
   Finds the square root of a quaternion
   """

   from numpy import sqrt

   s = q.copy()
   s[0] = 0.5*sqrt(q[0]+1.0)
   s[1:] = 0.5*q[1:]/s[0]

   # Later
   return s

def R_gal2equ():

   """
   Returns the rotation matrix to convert from galactic to equatorial
   coordinates.
   """

   # Imports
   from numpy import array

   # Just do it
   return array([[-0.054875539726, -0.873437108010, -0.483834985808],
                 [ 0.494109453312, -0.444829589425,  0.746982251810], 
                 [-0.867666135858, -0.198076386122,  0.455983795705]])

def ra2ha(ra, lst):

   """
   Converts a right ascension to an hour angle. 
   """

   # Imports
   from numpy import where

   ha = lst - ra/15.0
   ha = where(ha < 0.0, ha+24.0, ha)

   # Later
   return ha

def rll2v(r, lon, lat, degrees = False):

   """
   Convert a longitude/latitude pair to a vector.

   Usage:
v = rll2v(r, lon, lat [, degrees=True])

   Inputs:
    r  : A magnitude. 
    lon: a vector or longitudes in radians (unless degrees is set to true)
    lat: a vector or latitudes in radians (unless degrees is set to true)

   Optional Inputs:
    degrees: Set to true if the inputs are in degrees rather than
             radians (radians are the default)

   Outputs:
    v: Set of vectors. 

      Date     Author  Remarks
   ---------- -------- -------
   2007-10-29 K. Ganga First version.
   """

   # Just do it
   return r*ll2uv(lon, lat, degrees = degrees)

def myrtbisfunc(x):
   return 4.0*x+3.0

def rtbis(rtbisfunc, x1, x2, xacc, JMAX=40, nullval = -1.6375e30):

   """
   Using bisection, find the root of a function 'rtbisfunc' known to lie
   between x1 and x2. The root returned will be refined until its
   accuracy is +- xacc.

   See Section 9.1, p. 354 of _Numerical Recipes_.

      Date     Author  Remarks
   ---------- -------- -------
   2009-01-14 K. Ganga Changed failure exit null value to be set-able.
   2008-10-16 K. Ganga Changed failure exit to null value return. 
   2008-09-01 K. Ganga First version
   """

   f    = rtbisfunc(x1)
   fmid = rtbisfunc(x2)
   if (f*fmid >= 0.0):
       print "Min./Max. values:", f, fmid
       return nullval

   # Orient the search so that f>0 lies at x+dx
   dx = x1-x2
   rtb = x2
   if f < 0.0:
       dx = -dx
       rtb = x1

   # Bisection loop
   for j in range(JMAX):
       dx *= 0.5
       xmid = rtb + dx
       fmid = rtbisfunc(xmid)
       if fmid <= 0.0: rtb = xmid
       if abs(dx) < xacc or fmid == 0.0: return rtb

   # You should never get here. If you do, increase JMAX
   exit("rtbis: Too many bisections.")

def smooth(x,window_len=10,window='hanning'):
   """smooth the data using a window with requested size.

   This method is based on the convolution of a scaled window with the signal.
   The signal is prepared by introducing reflected copies of the signal
   (with the window size) in both ends so that transient parts are minimized
   in the begining and end part of the output signal.

   input:
   x: the input signal
   window_len: the dimension of the smoothing window
   window: the type of window from 'flat', 'hanning', 'hamming',
    'bartlett', 'blackman'
    flat window will produce a moving average smoothing.

   output:
   the smoothed signal

   example:

   t=linspace(-2,2,0.1)
   x=sin(t)+randn(len(t))*0.1
   y=smooth(x)

   see also:

   numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
   numpy.convolve
   scipy.signal.lfilter

   TODO: the window parameter could be the window itself if an array instead
         of a string
   """

   import numpy

   if x.ndim != 1:
       raise ValueError, "smooth only accepts 1 dimension arrays."

   if x.size < window_len:
       raise ValueError, "Input vector needs to be bigger than window size."

   if window_len<3:
       return x

   if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
       raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

   s=numpy.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
   if window == 'flat': #moving average
       w=ones(window_len,'d')
   else:
       w=eval('numpy.'+window+'(window_len)')

   y=numpy.convolve(w/w.sum(),s,mode='same')
   return y[window_len-1:-window_len+1]

def StringTime2DateTime(s):

   """
   Convert a date/time in string format to a python datetime object.

      Date     Author  Remarks
   ---------- -------- -------
   2009-09-30 K. Ganga First version. 
   """

   # Replace months in string format to numerical format
   st = s # Make a copy of the input
   for k in month_numbers.keys():
       st = st.replace(k, "%02d" % (month_numbers[k]))

   # Later
   return parseDateTime(st)

def StringTime2MatplotlibTime(st):

   # Later
   return DateTime2MatplotlibTime(StringTime2DateTime(st))

def StringTime2YMDhms(str):

   """
   Converts a string time in format YYYY-MM-DDThh:mm:ss.sssss to a
   python datetime object
   """

   parts = str.split('T')
   if len(parts) != 2: exit("Problem splitting date and time.")
   date = parts[0]
   time = parts[1]
   Y, M, D = [int(p) for p in parts[0].split('-')]
   h, m, s = parts[1].split(':')
   h = int(h)
   m = int(m)
   if s[-1]=='Z': s = float(s[:-1])

   # Later
   return Y, M, D, h, m, float(s)

def ut2gst(ut, jd):

   # jd0 = Julian date at 0h on this day
   S   = jd-2451545.0
   T   = S/36525.0
   T0  = 6.697374558 + 2400.051336*T + 0.000025862*T*T
   while T0 >= 24.0: T0 = T0 - 24.0
   while T0 <   0.0: T0 = T0 + 24.0
   gst = T0 + ut*1.002737909
   while gst >= 24.0: gst = gst - 24.0
   while gst <   0.0: gst = gst + 24.0

   # Later
   return gst

def uv2ll(uv, degrees=False, co=False):
   return v2ll(uv, degrees=degrees, co=co)

def v2ll(v, degrees = False, co=False):

   """
   Converts a three-dimensional vector to a longitude and a latitude

   Usage:
lon, lat = v2ll(v [, degrees=True] [, co=True])

   Inputs:
    v: Set of vectors. Should be a (3, n) array. 

   Optional Inputs:
    degrees: Set to true if the outputs are to be degrees rather than
             radians (radians are the default)
    co: Set to true if output latitude is to be COlatitude, rather than
        latitude. 

   Outputs:
    lon: a vector of longitudes in radians (unless degrees is set to true).
    lat: a vector of latitudes in radians (unless degrees and/or 'co'
         is set to true,)

   Requirements:
   - numpy

      Date     Author  Remarks
   ---------- -------- -------
   2009-03-12 K. Ganga Added 'co' option for colatitude instead of latitude
   2009-01-21 K. Ganga Modified to ensure that longitude is always >= 0.0.
   2007-10-25 K. Ganga First version.
   """

   # Imports
   from numpy import arctan2, pi, sqrt, where

   # Convert to degrees, if necessary, and make it an array
   factor = 1.0
   if degrees: factor = (180.0/pi)

   lon = arctan2(v[1], v[0])
   lon = where(lon < 0.0, lon + 2.0*pi, lon) # Make sure long. is positive.
   lon *= factor

   lat = arctan2(v[2], sqrt(v[0]*v[0]+v[1]*v[1])) # Find the latitude in rad.
   if co: lat = 0.5*pi-lat # Convert to colatitude, if desired
   lat *= factor           # Convert to degrees, if necessary

   # Later
   return lon, lat

def vsun(T68):

   """
   returns position of the SUN for time T68 seconds since 5/24/68
   very approximate
   T0 is T68 for the autumnal equinox in 2000

   Stolen fron Ned Wright's 'Qgen.for'

   Returns the ecliptic angle elon. 
   """
   #REAL*8 T68
   #REAL*8 YEAR,T0,TH
   #REAL*4 C,S
   #COMPLEX*16 QGEN
   return 2*9.954932891146e-8*(T68-1020384.0e3)

def WMAPdipole(csys = 'G', units = 'mK', radians = False):

   """
   Returns the CMB dipole magnitude and direction

   Taken from Hinshaw et al. 2009

      Date     Author  Remarks
   ---------- -------- -------
   2009-09-11 K. Ganga Update to use Hinshaw09 values. Added velocity output.
   """

   # Imports
   from numpy import pi

   amp =   3.355 # Amplitude in mK_CMB
   l   = 263.99  # Galactic Longitude in degrees
   b   =  48.26  # Galactic Latitude in degrees

   # Convert to other units
   if   units ==   'mK': amp *= 1.0
   elif units ==    'K': amp *= 1.0e-3
   elif units ==   'uK': amp *= 1.0e3
   elif units ==  'm/s': amp *= 1.0e-3*c/Tcmb
   elif units == 'km/s': amp *= 1.0e-6*c/Tcmb
   else: exit("Problem with requested dipole units")

   # Convert to other coordinate systems
   if csys == 'G':
       lon = l
       lat = b
   elif csys == 'E': lon, lat = euler(l, b, 6)
   elif csys == 'Q': lon, lat = euler(l, b, 2)
   else: exit("Problem with requested dipole system.")

   # Convert to radians, if desired
   if radians:
       lon *= pi/180.0
       lat *= pi/180.0

   # Later
   return amp, lon, lat

def ymd2jd(y, m, d):

   """
   Finds the Julian date given the year, month and day. The day can
   be a decimal.

   Usage:
jd = ymd2jd(y, m, d)

   Inputs:
    y: year. Integer
    m: month. Integer
    d: day. Float.

   Outputs:
    jd: Julian date. Float.

   Requires the 'Julian' package. 

      Date     Author  Remarks
   ---------- -------- -------
   2007-10-25 K. Ganga First version.
   """

   # Imports
   from Julian import JulianAstro

   # Later
   return JulianAstro(m, d, y)

def MyFunc(x, L):

   from numpy import dot, transpose

   # Later
   return dot(dot(L, transpose(L)), x)

def MyInv(r, L):

   from numpy import diagonal, dot, double, transpose, zeros
   from numpy.linalg import inv

   n = len(L[0])
   Ainv = zeros((n, n), dtype=double)
   d = diagonal(dot(L, transpose(L)))
   for i in range(n): Ainv[i, i] = 1.0/d[i]

   # Later
   return dot(Ainv, r)

if __name__ == "__main__":

   from numpy import arange, arccos, double, empty, pi, random, sin, sqrt, \
        zeros

   # equ2gal
   print ""
   print "Checking 'equ2gal'..."
   ra = -39.0
   dec = 41.0
   l, b = equ2gal(ra, dec, degrees=True)
   print "These two pairs should be very close:"
   print euler(ra, dec, select=1), l, b

   print ""
   print "Checking 'gal2equ'..."
   ra, dec = gal2equ(l, b, degrees=True)
   print "These two pairs should be very close:"
   print euler(l, b, select = 2), ra, dec

   # ecl2equ
   print ""
   print "Checking 'ecl2equ'..."
   elon = 30.0
   elat = 10.0
   ra, dec = ecl2equ(elon, elat, degrees=True)
   print "These two pairs should be very close:"
   print euler(elon, elat, select=4), ra, dec

   # equ2ecl
   print ""
   print "Checking 'equ2ecl'..."
   print "These two pairs should be very close:"
   print euler(ra, dec, select=3), equ2ecl(ra, dec, degrees=True)

   # ecl2gal
   print ""
   print "Checking 'ecl2gal'..."
   l2, b2 = ecl2gal(elon, elat, degrees=True)
   print "These two pairs should be very close:"
   print euler(elon, elat, select=5), l2, b2

   # gal2ecl
   print ""
   print "Checking 'gal2ecl'..."
   print "These two pairs should be very close:"
   print euler(l2, b2, select=6), gal2ecl(l2, b2, degrees=True)

   # ll2uv
   print ""
   print "ll2uv:",ll2uv(0.0, 0.0),"should be 1.0, 0.0, 0.0"
   print "ll2uv:",ll2uv(0.0, 0.0, degrees = True),"should be 1.0, 0.0, 0.0"
   print "ll2uv:",ll2uv(0.5*pi, 0.0),"should be 0.0, 1.0, 0.0"
   print "ll2uv:",ll2uv(90.0, 0.0, degrees = True),"should be 0.0, 1.0, 0.0"
   print "ll2uv:",ll2uv(0.0, -0.5*pi),"should be 0.0, 0.0, -1.0"
   print "ll2uv:",ll2uv(0.0, -90.0, degrees = True),"should be 0.0, 0.0, -1.0"

   # v2ll
   print ""
   print "v2ll:",v2ll([10.0, 0.0, 0.0]), "should be 0.0, 0.0"
   print "v2ll:",v2ll([ 0.1, 0.0, 0.0]), "should be 0.0, 0.0"
   print "v2ll:",v2ll([ 0.0, 1.0, 0.0]), "should be", 0.5*pi,", 0.0"
   print "v2ll:",v2ll([ 0.0, 0.0, -1.0], degrees=True), "should be 0.0 -90.0"

   # ymd2jd
   print ""
   print "ymd2jd:", ymd2jd(1990, 1,   1.0), "should be 2447892.5"
   print "ymd2jd:", ymd2jd(1985, 2, 17.25), "should be 2446113.75"

   # Phatmm
   m = 1000
   print ""
   print "Phatmm:", Phatmm([1.0, 0.0], m), "should be 0.0"

   # dT2flux
   print ""
   freqs = [100.0e9, 143.0e9, 217.0e9, 353.0e9, 545.0e9, 857.0e9]
   dToT  = [    2.5,     2.2,     4.8,    14.7,   147.0,  6700.0]
   fwhm  = [9.5, 7.1, 5.0, 5.0, 5.0, 5.0]
   ifreq = 0
   for freq in freqs:
       print freq, dT2flux(freq, dToT[ifreq]*Tcmb, fwhm[ifreq]/60.0)
       ifreq += 1

   # pix2ang_ring
   #print ""
   #print pix2ang_ring(512, [100000, 300000])
   #print pix2ang_ring(512, 300000)

   # nside2pixsize
   print ""
   print "Pixel sizes for given 'nside':"
   for res in range(14):
       nside = 2**res
       amin = nside2pixsize(nside)
       print "%04d: %8.3f arcmin. (=%9.5f deg.=%9.5f rad.)" \
             % (nside, amin, amin/60.0, (pi/180.0)*amin/60.0)
       #print nside, amin, "arcmin. (=", amin/60.0, "deg.)"

   # dT2RJ
   print ""
   print dT2RJ(1.0, 217.0e9)

   # vec2pix_ring/pix2vec_ring
   #print ""
   #fact  = 1
   #nside = 32
   #ipix = arange(long(nside)*long(nside)/fact, dtype=long)
   #for i in range(12*fact):
   #    uv = pix2vec_ring(nside, ipix)
   #    jpix = vec2pix_ring(nside, uv)
   #    print i, abs(ipix-jpix).max(), ipix
   #    ipix += long(nside)*long(nside)/fact

   #print ""
   #nside = 8192
   #n = 10000
   #uv = empty((3, n), dtype=double)
   #nrep = 5
   #for irep in range(nrep):
   #    uv[0] = random.uniform(low=-1.0, high=1.0, size=(n))
   #    uv[1] = random.uniform(low=-1.0, high=1.0, size=(n))
   #    uv[2] = random.uniform(low=-1.0, high=1.0, size=(n))
   #    norm = sqrt(uv[0]**2+uv[1]**2+uv[2]**2)
   #    uv[0] /= norm
   #    uv[1] /= norm
   #    uv[2] /= norm
   #    ipix = vec2pix_ring(nside, uv)
   #    uv2 = pix2vec_ring(nside, ipix)
   #    print irep, abs(arccos(uv[0]*uv2[0]+uv[1]*uv2[1]+uv[2]*uv2[2])*(180.0/pi)*60.0).max()

   # Check the size of precession
   print ""

   ra0, dec0 = ecl2equ(0.0, 90.0, jd = None, degrees=True)
   ra1, dec1 = ecl2equ(0.0, 90.0, jd = ymd2jd(2010, 1, 1.0), degrees=True)
   print ra0, dec0
   print ra1, dec1
   print (dec1 - dec0)*60.0*60.0
   ra0, dec0 = ecl2equ(0.0, -90.0, jd = None, degrees=True)
   ra1, dec1 = ecl2equ(0.0, -90.0, jd = ymd2jd(2010, 1, 1.0), degrees=True)
   print ra0, dec0
   print ra1, dec1
   print (dec1 - dec0)*60.0*60.0
   ra0, dec0 = ecl2equ(90.0, 0.0, jd = None, degrees=True)
   ra1, dec1 = ecl2equ(90.0, 0.0, jd = ymd2jd(2010, 1, 1.0), degrees=True)
   print ra0, dec0
   print ra1, dec1
   print (dec1 - dec0)*60.0*60.0
   ra0, dec0 = ecl2equ(-90.0, 0.0, jd = None, degrees=True)
   ra1, dec1 = ecl2equ(-90.0, 0.0, jd = ymd2jd(2010, 1, 1.0), degrees=True)
   print ra0, dec0
   print ra1, dec1
   print (dec1 - dec0)*60.0*60.0
   ra0, dec0 = ecl2equ(45.0, 0.0, jd = None, degrees=True)
   ra1, dec1 = ecl2equ(45.0, 0.0, jd = ymd2jd(2010, 1, 1.0), degrees=True)
   print ra0, dec0
   print ra1, dec1
   print (dec1 - dec0)*60.0*60.0

   # Get some blackbody numbers
   print " "
   print "nu T Blackbody"
   T = 20.0
   B1 = BlackBody(857.0e9, T)
   B2 = BlackBody(c/100.0e-6, T)
   print T, B1/B2

   T = 25.0
   B1 = BlackBody(857.0e9, T)
   B2 = BlackBody(c/100.0e-6, T)
   print T, B1/B2

   T = 30.0
   B1 = BlackBody(857.0e9, T)
   B2 = BlackBody(c/100.0e-6, T)
   print T, B1/B2

   # Check the root finding
   print ""
   print rtbis(myrtbisfunc, -1.0, 0.0, 1.0e-5)

   # Check the conversion between MJy and Kcmb*sr
   #print "\nBright2dT"
   #from numpy import array
   #Jy = True
   #nu = array([100.0e9, 150.0e9])
   #bb, dbb = BlackBody(nu, Tcmb, derivative=True)
   #conv = 1.0/dbb
   #for i in range(nu.size): print nu[i], dbb[i], conv[i]

   from numpy import dot, sqrt, transpose, zeros

   x = [1.1, 3.3, 5.5]
   L = [[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]]
   print dot(L, transpose(L))
   print x
   b = MyFunc(x, L)
   x = zeros(b.size, dtype=float)
   print x

   cg(x, b, MyFunc, (L), verbose=True)
   print x

   x = zeros(b.size, dtype=float)
   print x
   pcg(x, b, MyFunc, MyInv, (L), verbose=True)
   print x
