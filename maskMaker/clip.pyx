from math import sin,cos,tan,pi,floor,log10,sqrt,pow,radians, fabs
import numpy as np
import pyfits
import copy
from astropy import wcs
import sys
import os
from subprocess import call
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
import ConfigParser

def replaceAfterLastDot(fileName, str2Rep):
    splitted = fileName.split('.')
    splitted[len(splitted)-1] = str2Rep
    return ".".join(splitted)

#inputs:
#    python maskmaker_wcut.py filename parallelIndex ra dec stampsize petroRad rowc colc
def main():
    item = sys.argv[1]                  # filename
    pIndex = sys.argv[2]                # parallelIndex
    ra = float(sys.argv[3])             # ra
    dec = float(sys.argv[4])            # dec
    halfSize = float(sys.argv[5])/2.0   # stampsize / 2 ?
    petroRad=float(sys.argv[6])         # petroRad
    petroMag=float(sys.argv[7])         # petroMag
    rowc=float(sys.argv[8])             # rowc
    colc=float(sys.argv[9])             # colc
    infilename = item#item[0:len(item) - 1]
    ptsInside2Rp = []
#    print("File: " + item)
    header = pyfits.getheader(infilename, 0)
    data = pyfits.getdata(infilename, 0)
    ylen, xlen = data.shape[0], data.shape[1]
 #   print("Tamanho:", ylen,xlen)
    sizy = np.min(np.array([fabs(rowc),(halfSize)*fabs(petroRad), fabs(data.shape[0]-rowc)]))
    sizx = np.min(np.array([fabs(colc),(halfSize)*fabs(petroRad), fabs(data.shape[1]-colc)]))  
    siz = int(np.min(np.array([sizy,sizx])))
    x=int(colc)
    y=int(rowc)
    data = data[y-siz:y+siz+1,x-siz:x+siz+1]  
        #os.remove('out_sex_large'+pIndex+'.cat')
        #os.remove('default'+pIndex+'.sex')
    if(siz < int((halfSize)*fabs(petroRad)) ):
        error = 2
    else:
        error = 0
    
    maskFileName = '../cutted/'+str(pIndex)+"_mask.fit"
    infilename = '../cutted/'+str(pIndex)+".fit"
    with open("../"+pIndex+"_log.txt","w") as l:
        l.write(str(error))
    pyfits.writeto('../cutted/'+str(pIndex)+"_wout-cleaning_stamp"+str(int(halfSize*2))+".fit",data,clobber=True) # save original fit
    pyfits.writeto('../cutted/'+str(pIndex)+".fit",data,clobber=True) # save fit to clean
