import user
from glob import glob
import numpy as np
import os
import time

def utc(tstr):
    tfmt = '%Y-%m-%d %H:%M:%S'
    TZ = os.environ.get('TZ', None)
    os.environ['TZ'] = 'UTC'
    time.tzset()
    res = time.mktime(time.strptime(tstr, tfmt))
    if TZ is None:
        del os.environ['TZ']
    else:
        os.environ['TZ'] = TZ
    time.tzset()
    return res

def unix2jd(unixtime):
    """Convert unix timestamp to julian day
    """
    # Jan, 1st 1970, 00:00:00 is jd = 2440587.5
    return unixtime/86400. + 2440587.5

def jd2gst(jd, ut=None):
    """From Practical Astronomy With Your Calculator.
    """
    jd2 = np.floor(jd-0.5)
    jd2 += 0.5
    #print jd2
    S = jd2 - 2451545.0
    T = S/36525.0
    T0 = 6.697374558 + 2400.051336*T+0.000025862*T**2
    T0 %= 24.0
    if ut is None:
        ut = (jd-jd2)*24.
    #print ut
    UT = ut*1.002737909
    T0 += UT
    T0 %= 24.
    return T0
    

def conv_azel2equ(az, el, geolon, geolat, utc):
    import Ganga as G
    import time
    import os
    jd = unix2jd(utc)
    gst = jd2gst(jd)
    #print gst.min(), gst.max()
    lst = G.gst2lst(gst, geolon)
    #print lst.min(), lst.max()
    ra, dec = G.hor2equ(az, el, geolat, lst)
    return ra, dec

def conv_equ2azel(ra, dec, geolon, geolat, utc):
    import Ganga as G
    import time
    import os
    jd = unix2jd(utc)
    gst = jd2gst(jd)
    lst = G.gst2lst(gst, geolon)
    az, el = G.equ2hor(ra, dec, geolat, lst) 
    return az, el


def butterfilt(x, Fs, Fc = 1., order = 4, mode = 'low'):    
    # baseline = low pass filter of xfill
    N2 = 2**(int(np.ceil(np.log2(x.size))))
    ftx = np.fft.fft(x-x.mean(), n=N2)
    freq = np.fft.fftfreq(N2, d=1./Fs)
    butter_freq = Fc
    butter_order = order
    filt = 1./np.sqrt(1. + (freq/butter_freq)**(2*butter_order))
    ftx *= filt
    baseline = np.float64(np.fft.ifft(ftx)[:x.size]) + x.mean()
    if mode == 'low':
        return baseline
    elif mode == 'high':
        return x-baseline


def butterfilter(N, Fs, bands, order = 4):
    freq = np.fft.fftfreq(N, d=1./Fs)
    filt = np.zeros(freq.size, dtype=np.complex128)
    for band in bands:
        Fc, w = band
        filt1 = 0.5*(1+np.cos(2*np.pi*(freq-Fc)/w))
        filt1[np.absolute(freq-Fc) > w/2] = 0.0
        filt2 = 0.5*(1+np.cos(2*np.pi*(freq+Fc)/w))
        filt2[np.absolute(freq+Fc) > w/2] = 0.0
        print filt1.max()
        filt1 /= filt1.max()
        filt2 /= filt2.max()
        #print np.isnan(filt1).sum()
        filt += filt1+filt2
    return filt, freq
    
    
def peigne_filter(x, Fs, bands, mode = 'pass'):
    N2 = x.size#2**(int(np.ceil(np.log2(x.size))))
    ftx = np.fft.fft(x, N2)
    freq = np.fft.fftfreq(N2, d=1./Fs)
    filt = np.zeros_like(freq)
    for band in bands:
        filt[(np.absolute(freq)>band[0]) & (np.absolute(freq)<band[1])] = 1.
    if mode == 'pass':
        ftx *= filt
    elif mode == 'stop':
        ftx *= (1-filt)
    x2 = np.float64(np.fft.ifft(ftx)[:x.size])
    return x2, freq, filt
