import astropy.io.fits as fits
import numpy
import gridAlg as gio
import os
import math
import ConfigParser
from scipy.optimize import curve_fit

def plotFITS(mat,fileName):
    hdu = fits.PrimaryHDU(mat)
    hdu.writeto(fileName, clobber=True)

def saveCSV(pts,fileName):
    
    numpy.savetxt(fileName,numpy.array(pts).T)

def saveConcentrationDev(dists, conc, nsample, foldSize=15):
    dconc = numpy.array([0.0 for i in range(len(conc)-2)])
    dists2 =  numpy.array([0.0 for i in range(len(conc)-2)])
    foldDists = []
    mean, median, maxi, mini,qmax, qmin = [],[],[],[],[],[]
    for i in range(1,len(conc)-1):
        dconc[i-1] = (conc[i+1] - conc [i-1]) / (dists[i+1]-dists [i-1])
        dists2[i-1] = dists[i]

    dists2 = numpy.asarray(dists2)
    dconc = numpy.asarray(dconc)

    for i in range(0,len(dconc),foldSize+1):
        soma = 0.0
        count = 0
        fold = []
        for j in range(i,min(i+foldSize,len(dconc))):
            fold.append(dconc[j])
        fold = numpy.array(fold)
        qqmax = numpy.percentile(fold, 75)
        qqmin = numpy.percentile(fold,25)
        cutted = fold[numpy.where((fold>qqmin-1.5*(qqmax-qqmin)) & (fold<qqmax+1.5*(qqmax-qqmin)))]
        fold =numpy.array(fold)
        mean.append(numpy.average(fold))
        median.append(numpy.median(fold))
        maxi.append(numpy.max(cutted))
        mini.append(numpy.min(cutted))
        qmax.append(qqmax)
        qmin.append(qqmin)
        foldDists.append(dists[i])
            
    s1 = numpy.array([dists,conc]).T
    s2 = numpy.array([dists2,dconc]).T
    box = numpy.array([mean,median,maxi,mini,qmax,qmin,foldDists]).T
    header = "mean median max min qmax qmin dist\n"
    numpy.savetxt(fileName+"stddev.csv",s2,fmt='%.5f')
    with open(fileName+"boxplot.csv", 'wb') as f:
        f.write(header)
        numpy.savetxt(f,box,fmt='%.5f')


def readFITSIMG(string):
    return numpy.array(fits.open(string)[0].data, numpy.float32)

def runSextractor(filePath, filename,xtraID,par=[],value=[]):
    configFile = ConfigParser.ConfigParser()
    configFile.read('cfg/paths.ini')
    sexPath = configFile.get("Path","Sextractor")
    cmd = sexPath+" "+filePath+filename
    for i in range(len(par)):
        cmd = cmd+" -"+par[i]+" "+str(value[i])
    cmd = cmd+" -CATALOG_NAME "+str(xtraID)+".cat  -CHECKIMAGE_TYPE SEGMENTATION,BACKGROUND -CHECKIMAGE_NAME "+str(xtraID)+"_seg.fits,"+str(xtraID)+"_bcg.fits -VERBOSE_TYPE QUIET"
    #print(cmd)
    process = os.popen(cmd)
    log = process.read()
    return log

def readSextractorOutput(filename):
    fileArx = open(filename, 'r')
    dicionario = []
    dados = [[]]
    for line in fileArx:
        splitList = line.split()
        if(splitList[0] == '#'):
            dicionario.append(splitList[2])
        else:
            dados.append(splitList)
    dados = [ [float(dados[i][j]) for j in range(len(dados[i])) ] for i in range(1,len(dados))]
    return (numpy.array(dicionario),numpy.array(dados))

def getFirstValue(dictionary, data, string):
    ind = getIndex(dictionary, string)
    if (data and ind < len(data[0])):
        return data[0][ind]
    else:
        return -1.0

def getValues(dictionary, data, string):
    ind = getIndex(dictionary, string)
    return data[:, ind]

def getIndex(dictionary, string):
    return numpy.where(map(lambda x: (x==string), dictionary))[0][0]

def filterSextractorData(mat, dicionario, dados):
    px = getIndex(dicionario, 'X_IMAGE')
    py = getIndex(dicionario, 'Y_IMAGE')
    cx,cy = float(len(mat))/2.0, float(len(mat[0]))/2.0
    possibleGalaxies= []
    for line in dados:
        possibleGalaxies.append(line)
    if(len(possibleGalaxies)>0):
        possibleGalaxies = sorted(possibleGalaxies,key=lambda l:(l[px]-cx)**2.0+(l[py]-cy)**2.0)
        return (dicionario, [possibleGalaxies[0]])
    else:
        return (dicionario, [])



############################################
#     Funcao sigmoidF:
# entrada:
#    x - numpy.array com as coordenadas x
#    k - parametro de ajuste 
# retorna:
#    y - numpy.array com os valores correspondentes a x
# Observacoes:
#    esta funcao eh similar a funcao Fermi-Dirac, porem  nao eh utilizado a constante de bolztman

def sigmoidF(x,k):
    temp = numpy.array([1.0/i if (i != 0.0) else 0.0  for i in x])
    return 1.0/(numpy.exp(k*temp)+1.0)

############################################
#     Funcao sigmoidF:
# entrada:
#    x - numpy.array com as coordenadas x
#    l - parametro do valor maximo da curva (equivalente ao brilho total)
#    k - parametro da curvatura 
#    c - desvio da curva (especificamente o valor medio)
# retorna:
#    y - numpy.array com os valores correspondentes a x
def logisticF(x, l, k, c):
    return l/(numpy.exp(-k*x-c)+1.0)

def logisticR(x, l, k, c, r):
    return l/(numpy.exp(-k*x-c)+1.0)-l/(numpy.exp(-c)+1.0)

def fittingParameters(dists,lum):
    maxLum = max(numpy.array(lum))
    parametersL, c2 = curve_fit(logisticF, dists, numpy.array(lum), p0 = [maxLum, 1.0, 1.0])
    parametersR, c2 = curve_fit(logisticR, dists, numpy.array(lum), p0 = [maxLum, 1.0, 1.0,maxLum])
    print(parametersR)
    parametersS, c1 = curve_fit(sigmoidF, dists, numpy.array(lum)/maxLum, sigma=0.1)
    return parametersL,parametersR, parametersS

