from ellipse import *
import scipy.interpolate as interpolate
import mahotas as mh
import numpy
import cmath
import operator
from numpy.random import normal as distN
import sys
import math

from scipy import fftpack

from libc.math cimport fabs, atan2, floor, sqrt, pow,cos, sin, exp
from libc.stdlib cimport rand, RAND_MAX
from libc cimport bool
import indexes as par

cimport numpy
cimport cython


cdef pi():
    return float(3.14159265359)
#########################################################################
#    applyMask
#########################################################################
# Objetivo:
#    Aplicar a mascara a matriz
# Entrada:
#    img - matriz de entrada - numpy.array(list of list)
#    mask - mascara de entrada - numpy.array(list of list)
#    hiddenValue - valor a ser removido - float
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float[:,:] applyMask(float[:,:] img,float[:,:] mask):
    cdef:
        int i, j
        int w, h
        float[:,:] mat
    w, h = len(img[0]), len(img)
    mat = numpy.array([[float(0.0) for j in range(w)] for i in range(h)],dtype=numpy.float32)
    for i in range(w):
        for j in range(h):
            if(mask[j, i] == 1.0):
                mat[j, i] = 0.0
            else:
                mat[j, i] = img[j, i]
    return mat


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float[:,:] removeSky(float[:,:] img, float sky):
    cdef:
        int i, j
        int w, h
        float[:,:] mat
    w, h = len(img[0]), len(img)
    mat = numpy.array([[float(0.0) for j in range(w)] for i in range(h)],dtype=numpy.float32)
    for i in range(w):
        for j in range(h):
            mat[j, i] = img[j, i]-sky
    return mat

#########################################################################
#    ellipseCut
#########################################################################
# Objetivo:
#    Remover os pontos fora da elipse;
# Entrada:
#    mat - matriz de entrada - numpy.array(list of list)
#    ellipse - objeto do tipo ellipse
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float[:,:] ellipseCut(float[:,:] mat, ellipse):
    cdef:
        float x, y
        int w, h
        float d
        float[:,:] cpyMat
    w, h = len(mat[0]), len(mat)
    cpyMat = numpy.array([[float(0.0) for j in range(w)] for i in range(h)],dtype=numpy.float32)
    for i in range(w):
        for j in range(h):
            x, y = (float(i), float(j))
            d = ellipse.findScale(x, y)
            if d > 1.0:
                cpyMat[j, i] = 0.0
            else:
                cpyMat[j, i] = mat[j,i]
    return cpyMat

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float[:,:] removeOtherGalaxies(float[:,:]  mat,float[:,:] segMask, float idGalaxy):
    cdef: 
        int w, h , i, j
        float[:,:] cpyMat
    w, h= len(mat[0]), len(mat)
    cpyMat = numpy.array([[float(0.0) for j in range(w)] for i in range(h)],dtype=numpy.float32)
    for i in range(w):
        for j in range(h):
            if((segMask[j, i] != idGalaxy) & (segMask[j, i] != 0)):
                cpyMat[j, i] = 0.0
            else:
                cpyMat[j, i] = mat[j, i]
    return cpyMat

#########################################################################
#    interpolateEllipse
#########################################################################
# Objetivo:
#    Rotacionar e interpolar a grade nos pontos com o valor 0.0
# Entrada:
#    mat - matriz de entrada - numpy.array(list of list)
#    nRot - numero de rotacoes - int
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def interpolateEllipse(float[:,:]  mat, ellipse):
    cdef:  
        float[:,:] cpyMat
        float[:] deltaRadius, 
        float sc, rho, mean
        float auxx, auxy
        int nx, ny, dRho
        int w, h, i, j, sameEllipse, epos1, epos2 
    w, h = len(mat[0]), len(mat)
    cpyMat = numpy.array([[float(0.0) for j in range(w)] for i in range(h)],dtype=numpy.float32)
    mask = numpy.array([[float(0.0) for j in range(w)] for i in range(h)],dtype=numpy.float32)
    deltaRadius = numpy.array([0.0 for i in range(500)],dtype=numpy.float32)
    for i in range(0,500):
        deltaRadius[i] = float(2.0)*float(i)*pi()/500.0-pi()

    epos1, epos2 = ellipse.posy, ellipse.posx
    for i in range(w):
        for j in range(h):
            if(mat[j, i]!=0.0):
                cpyMat[j, i] = mat[j,i]
                mask[j,i] = 0.0
                continue
            mask[j,i] = 1.0
            sc = ellipse.findScale(i, j)
            rho = atan2(j-epos1, i-epos2)
            sameEllipse = 0 
            mean = 0.0
            lstPts = []
            for dRho in range(len(deltaRadius)):
                auxx, auxy = ellipse.findPoint(sc, rho+deltaRadius[dRho])
                if (not numpy.isnan(auxx) and not numpy.isnan(auxy)):
                    nx, ny = int(auxx), int(auxy)
                else:
                    nx, ny = -1, -1
                if (nx>=0) & (ny>=0) & (nx<w) & (ny<h):
                    if (mat[ny, nx]!= 0.0):
                        sameEllipse = sameEllipse + 1
                        lstPts.append(mat[ny, nx])
            if sameEllipse < 1:
                cpyMat[j, i] = numpy.average(mat)
                continue
            if( len(lstPts)>0):
                stddev = numpy.std(lstPts)
                avg = numpy.average(lstPts)
                cpyMat[j, i] = distN(avg,stddev) 
    return(cpyMat, mask)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def filterOtsu(float[:,:] Fmat, ellipse, int nbins):
    cdef:
        float[:,:] mat, output
        int[:,:] watershed
        int px, py
        float minimo, maximo, maxDist
        int w, h, wit, hit
        object pts
    w, h = len(Fmat[0]), len(Fmat)
    minimo, maximo= -1.0,-1.0
    mat = numpy.array([[float(0.0) for j in range(w)] for i in range(h)],dtype=numpy.float32)
    #calculando maximo e minimo
    for wit in range(w):
        for hit in range(h):
            if(Fmat[hit, wit] < minimo) or ((wit == 0) and (hit==0)):
                minimo = Fmat[hit, wit]
            if(Fmat[hit, wit] > maximo) or ((wit == 0) and (hit==0)):
                maximo = Fmat[hit, wit]
    if(maximo <= minimo):
        raise Exception("Otsu filter says that only one point is the object!")

    #normalizando a matriz
    for wit in range(w):
        for hit in range(h):
            mat[hit, wit] = float(nbins)*(Fmat[hit, wit]-minimo)/(maximo-minimo)

    limiar =  mh.thresholding.otsu( numpy.array(mat,dtype=numpy.uint), ignore_zeros=True)

    #iniciando o watershed
    watershed = numpy.array([[float(0.0) for j in range(w)] for i in range(h)],dtype=numpy.int32)
    output= numpy.array([[float(0.0) for j in range(w)] for i in range(h)],dtype=numpy.float32)
    for wit in range(w):
        for hit in range(h):
            watershed[hit, wit] = 0
            output[hit, wit] =   1.0

    maxDist = 0.0
    pts = [(int(ellipse.posy),int(ellipse.posx))]
    while(len(pts) > 0):
        px, py = numpy.array(pts.pop(0), dtype=numpy.int32)
        if (watershed[px,py] == 0):
            found=False
            if(px+1<len(watershed)):
                if mat[px+1,py]> limiar:
                    pts.append((px+1,py))
                    found=True
            if(px>0):
                if mat[px-1,py]> limiar:
                    pts.append((px-1,py))
                    found=True
            if(py+1<len(watershed[px])):
                if mat[px,py+1]> limiar:
                    pts.append([px,py+1])
                    found=True
            if(py>0):
                if mat[px,py-1]> limiar:
                    pts.append([px,py-1])
                    found=True
            if(mat[px,py] > limiar) | (found==True):
                maxDist = max(maxDist, ellipse.findScale(float(px),float(py)) )
                watershed[px,py] = 1
    for wit in range(w):
        for hit in range(h):
            if(watershed[hit, wit] == 1):
                output[hit, wit] =   0.0
    return output, limiar

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def interpolatePoint(float[:,:]  mat, int x,int y,float hiddenValue):
    cdef:
        float sumDists,result
        int w, h, i, j, sameEllipse 
    w, h = len(mat[0]), len(mat)
    sumDists = 0.0
    result = 0.0
    if (floor( x + 1.0) < w) & (floor( x + 1.0) > -1) & (floor(y+.10)< h) & (floor(y-.10)> -1):
        for i in range(-1,2,1):
            for j in range(-1,2,1):
                dist =-sqrt(pow(float(i)-y,2.0)+pow(float(i)-y,2.0))
                if (dist == 0.0):
                    return mat[i][j]
                if (mat[i][j] == hiddenValue):
                    return mat[y][x]
                sumDists += dist
                result += mat[i][j]/dist
        return (result/sumDists)
    return mat[y, x]

###############################################
#
###############################################
#
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef float findRadiusLuminosity(float[:] dists,float[:] concentrations, float totalLum, float percentage):
    cdef:
        float soma, acc, accAnterior, dy, dx
        int lenDists, i    
    lenDists = len(dists)
    if(percentage < 0.0) or (percentage > 1.0):
        raise Exception("Percentage Invalid for Concentration! Got"+ str(percentage))
    if(totalLum == 0.0):
        raise Exception("Invalid Total Concentration! Got "+str(totalLum))
    if(percentage == 0.0):
        return 0.0

    accAnterior = concentrations[0]/totalLum
    for i in range(1,lenDists):
        acc = concentrations[i]/totalLum
            
        # interpolate the distance
        if(acc >= percentage):
            dy = acc - accAnterior
            dx = float(dists[i]) - float(dists[i-1])
            # if found a baseline, return the minimum baseline distance (also avoid overflow)
            if(fabs(dy) < 1.e-08):
                return float(dists[i])
            return float(dists[i-1])+dx*(percentage-accAnterior)/dy
        accAnterior = acc
    return float(max(dists))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef float butterworth(float d,float d0,float n):
    return 1.0/(1.0+pow(d/d0,2.0*n))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def filterButterworth2D2(float[:,:] img, float degradation, float n):
    # frac -> max smoothing at center
    cdef:
        int heigth,width,i, j, maxD0,dcut
        float d0 # cutoff freq
        float  minD0
        float[:,:] smoothed, temp, zeros
        numpy.complex64_t[:,:] newFreq, freq

    heigth,width = len(img),len(img[0])

    smoothed = numpy.array([[0.0 for i in range(width)] for j in range(heigth)],dtype=numpy.float32)
    zeros = numpy.array([[0.0 for i in range(width)] for j in range(heigth)],dtype=numpy.float32)
    temp = numpy.array([[0.0 for i in range(width)] for j in range(heigth)],dtype=numpy.float32)
    freq =fftpack.fftshift(fftpack.fft2(img))
    maxD0 = int(sqrt(pow(float(width),2.0)+pow(float(heigth),2.0))/4.0)    
    d0 = float(maxD0)*degradation
    newFreq = numpy.array([[freq[j][i] for i in range(width)] for j in range(heigth)], dtype=numpy.complex64)

    for i in range(heigth):
        for j in range(width):
            newFreq[i][j] = freq[i][j]*butterworth(sqrt(pow(float(i)-float(heigth)/2.0,2.0)+pow(float(j)-float(width)/2.0,2.0)),d0,n)
    smoothed = numpy.real(fftpack.ifft2(fftpack.ifftshift(newFreq))).astype(numpy.float32)
    return smoothed

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def filterButterworth2D(float[:,:] img, float smD, float n, ell):
    # frac -> max smoothing at center
    cdef:
        int heigth,width,cx,cy, i, j
        float maxD0
        float[:,:] smoothed
        numpy.complex64_t[:,:]  freq

    heigth,width = len(img),len(img[0])
    cx, cy = width/2, heigth/2

    smoothed = numpy.array([[0.0 for i in range(width)] for j in range(heigth)],dtype=numpy.float32)
    for i in range(heigth):
        for j in range(width):
            smoothed[i][j] = img[i][j]
            

    freq = fftpack.fft2(img)
    freq = fftpack.fftshift(freq)
    maxD0 = ell.maxRad  
    for i in range(heigth):
        for j in range(width):
            freq[i][j] = freq[i][j]*butterworth(sqrt(pow(float(i)-float(heigth)/2.0,2.0)+pow(float(j)-float(width)/2.0,2.0)),smD*maxD0,n)
    smoothed = numpy.real(fftpack.ifft2(fftpack.ifftshift(freq))).astype(numpy.float32)
    return smoothed


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float getCartesianDist(float x1, float y1, float x2, float y2):
     return sqrt(pow(x1-x2,2.0)+pow(y1-y2,2.0))   

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float samplingMatrixPoint(float epointx,float epointy, float pointx, float pointy, float dist, int nsamples):
     cdef: 
          float px, py, rx, ry
          int ninside, i
          float[:]  dists
     px, py = float(pointx), float(pointy)
     # caso pixel esteja totalmente dentro da galaxia
     if(getCartesianDist(px+0.5, py+0.5, epointx, epointy) <= dist) and \
     	(getCartesianDist(px-0.5, py+0.5, epointx, epointy) <= dist) and \
        (getCartesianDist(px+0.5, py-0.5, epointx, epointy) <= dist) and \
        (getCartesianDist(px-0.5, py-0.5, epointx, epointy) <= dist):
                 return 1.0
     # caso pixel esteja totalmente fora da galaxia
     if(getCartesianDist(px+0.5, py+0.5, epointx, epointy) > dist) and \
     	(getCartesianDist(px-0.5, py+0.5, epointx, epointy) > dist) and \
        (getCartesianDist(px+0.5, py-0.5, epointx, epointy) > dist) and \
        (getCartesianDist(px-0.5, py-0.5, epointx, epointy) > dist):
                 return 0.0
     # tratamento do caso de pixel na borda da galaxia
     ninside = 0
     for i in range(nsamples):  # para todos os pontos chutados dentro do pixel
          rx = (float(rand())/float(RAND_MAX))-0.5      # valor aleatorio normalizado de 0 a 1 (-0.5?) 
          ry = (float(rand())/float(RAND_MAX))-0.5
          # verifica se a distancia do centro da elipse ao ponto está dentro da galaxia
          if(getCartesianDist(px+rx ,py+ry, epointx, epointy) <= dist):
               ninside = ninside + 1        # incrementa contador de pontos dentro da galaxia
     return float(ninside)/float(nsamples)  # retorna razão de pontos dentro da galaxia pelo número de pontos "chutados"

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef samplingMatrixPoint2(ellipse, float pointx,float  pointy,float dist,int nsamples, sampledPts=False):
     cdef: 
          float px, py, rx, ry
          int ninside, i
          float[:]  dists
     if(nsamples==0):
        raise Exception("Invalid number of samples!(sampling Matrix)")
     px, py = float(pointx), float(pointy)
     if(ellipse.getCartesianDist(px+0.5,py+0.5)<=dist) and (ellipse.getCartesianDist(px-0.5,py+0.5)<=dist):
         if(ellipse.getCartesianDist(px+0.5,py-0.5)<=dist) and (ellipse.getCartesianDist(px-0.5,py-0.5)<=dist):
             if(sampledPts):
                 return 1.0, [[], []], [[], []]
             else:
                 return 1.0
     if(ellipse.getCartesianDist(px+0.5,py+0.5)>dist) and (ellipse.getCartesianDist(px-0.5,py+0.5)>dist):
         if(ellipse.getCartesianDist(px+0.5,py-0.5)>dist) and (ellipse.getCartesianDist(px-0.5,py-0.5)>dist):
             if(sampledPts):
                 return 0.0, [[], []], [[], []]
             else:
                 return 0.0
     dists = numpy.array([0.0 for i in range(nsamples)], dtype=numpy.float32)
     inside = []
     outside = []
     ninside = 0
     for i in range(nsamples):
          rx = (float(rand())/float(RAND_MAX))-0.5
          ry = (float(rand())/float(RAND_MAX))-0.5
          dists[i] = ellipse.getCartesianDist(px+rx ,py+ry)
          if(dists[i] < dist):
               ninside = ninside + 1
               inside.append((px+rx,py+ry))
          else:
               outside.append((px+rx,py+ry))

     if(sampledPts):
         return float(ninside)/float(nsamples), inside, outside
     else:
         return float(ninside)/float(nsamples)

@cython.boundscheck(False)
@cython.wraparound(False)
def getManySample( float[:,:] mat,ellipse,float dist,int nsamples=10):
    cdef float tp
    p1, p2 =[],[] 
    px,p2x=[],[]
    py,p2y=[],[]
    for y in range(len(mat)):
        for x in range(len(mat[y])):
            tp, p1, p2 = samplingMatrixPoint2(ellipse,x,y,dist,nsamples,True)
            if not(not(p1)):
                    px.append(p1[0][0])
                    py.append(p1[0][1])
            if not(not(p2)):
                    p2x.append(p2[0][0])
                    p2y.append(p2[0][1])
    return [px,py], [p2x,p2y]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)         # without mask
cpdef tuple getAccumulatedLuminosity(float[:,:] mat, object ellipse,int nsamples=50, int mindist = 0):
    cdef: 
        float frac, epx, epy
        float[:] dists, acc
        float isky, percentage
        int w, h, wit, hit, i, d, ndists
    if(nsamples<=0):
        raise Exception("Invalid number of samples!(Accumulated Lum). Got"+str(nsamples))
    w, h = len(mat[0]), len(mat)            # largura e altura
    epx, epy = ellipse.posx, ellipse.posy   # posicao central da ellipse
    ndists = len(mat)/2 -mindist            # metade da dimensão da matriz 
    dists = numpy.array([0.0 for i in range(ndists)],dtype=numpy.float32)   # distancias para raios
    acc = numpy.array([0.0 for i in range(ndists)],dtype=numpy.float32)     # luminosidade acumulada p/ cd raio
    for i in range(mindist,len(mat)/2):
        dists[i-mindist] = float(i)         # dists de indice i recebe i (indices para raios)
    isky = ellipse.sky
    for d in range(ndists):                 # até o meio da matriz
        acc[d] = 0.0                        # luminosidade acumulada para raio de distancia d
        for wit in range(w):                # iterador da largura
            for hit in range(h):            # itarador da altura
                percentage = samplingMatrixPoint(epx, epy,wit,hit,dists[d],nsamples)
                # mat[hit, wit] in [0,1] -> sky/mask or not
                acc[d] = acc[d] + (mat[hit, wit]-isky)*percentage # acc += (mat[hit,wit]-bcg[hit,wit])*percentage
    return (dists, acc)                     # retorna raios e luminosidade acumulada neles

@cython.boundscheck(False)
@cython.wraparound(False)   
@cython.cdivision(True)         # with mask
cpdef float F(float r, float[:,:] mat, float bcg, object ellipse):
    cdef: 
        float epx, epy, acc, percentage
        int w, h, wit, hit, i, j
    
    w, h = len(mat[0]), len(mat)            # largura e altura
    epx, epy = ellipse.posx, ellipse.posy   # posicao central da ellipse

    acc = 0.0                        	# luminosidade acumulada para r de distancia d
    for wit in range(w):                # iterador da largura
        for hit in range(h):            # iterador da altura
            percentage = samplingMatrixPoint(epx, epy, wit, hit, r, 1000)
            # considerando coordenada do pixel na matrix como centro do pixel 
            # acc = acc + (mat[hit, wit]-bcg[hit, wit])*percentage # acc += (mat[hit,wit]-bcg[hit,wit])*percentage
            acc = acc + (mat[hit, wit]-bcg)*percentage # acc += (mat[hit,wit]-bcg[hit,wit])*percentage
    return acc 		                    # retorna raios e luminosidade acumulada até eles

@cython.boundscheck(False)
@cython.wraparound(False)   
@cython.cdivision(True)         # with mask
cpdef tuple etaFunction(float[:,:] mat, object ellipse, int nsamples=50, int mindist = 0):
    cdef: 
        float[:] dists, acc, eta
        int ndists, w, h, i, j
        float bcgMean
    
    if(nsamples<=0):
        raise Exception("Invalid number of samples!(Accumulated Lum). Got"+str(nsamples))
    ndists = len(mat)/2 -mindist            # metade da dimensão da matriz

    w, h = len(mat[0]), len(mat)            # largura e altura

    dists = numpy.array([0.0 for i in range(ndists)],dtype=numpy.float32)   # distancias para r
    acc = numpy.array([0.0 for i in range(ndists)],dtype=numpy.float32)     # luminosidade acumulada
    eta = numpy.array([0.0 for i in range(ndists)],dtype=numpy.float32)     # eta para cada r

    # bcg = getSkyMedian()
    bcg_array = [] 
    for i in range(int(w/5)):
        for j in range (int(h/5)):
            # print("appending "+str(mat[i, j]))
            bcg_array.append(mat[i, j])
            bcg_array.append(mat[(w/5)*4 + i, j])
            bcg_array.append(mat[i, (h/5)*4 + j])
            bcg_array.append(mat[(w/5)*4 + i, (h/5)*4 + j])

    # print("matrix dimensions: " + str(w) + " x " + str(h) + " = " + str(w*h))
    # print("len(bcg_array) = " + str(len(bcg_array)))
    # for x in range(len(bcg_array)):
    #     print(bcg_array[i])

    bcgMean = numpy.median(bcg_array)
    # bcgMean = bcgMean - 0.1
    # print("bcgMean = "+str(bcgMean))

    for i in range(1,len(mat)/2):
        dists[i-mindist] = float(i)
        # F(r)
        acc[i-mindist] = F(i-mindist, mat, bcgMean, ellipse)
        # F(1.25*r)
        eta[i-mindist] = F(1.25*(i-mindist), mat, bcgMean, ellipse)
        # F(1.25*r)-F(0.8*r)
        eta[i-mindist] = eta[i-mindist] - F(0.8*(i-mindist), mat, bcgMean, ellipse)
        # divisao no numerador = 
        #(F(1.25*r)-F(0.8*r)) / pi*(1.25^2-0.8^2)*r^2
        eta[i-mindist] = eta[i-mindist]/(pi()*(pow(1.25,2.0)-pow(0.8,2.0))*pow(i-mindist,2.0))
        # divisao pelo denominador = 
        #((F(1.25*r)-F(0.8*r))/pi*(1.25^2-0.8^2)*r^2)/((F(r)/pi()*r^2))
        eta[i-mindist] = eta[i-mindist]/((acc[i-mindist])/(pi()*pow(i-mindist, 2.0)))

    return (dists, acc, eta)                     # retorna raios e luminosidade acumulada até eles

@cython.boundscheck(False)
@cython.wraparound(False)   
@cython.cdivision(True)         # with mask
cpdef float getPetroRad(float[:] raios, float[:] etas, float epx, int dimension):
    cdef: 
        float etaMin, numerator, denominator
        int r

    etaMin = 0.3

    iterRadius = iter(raios)
    next(iterRadius) # jumping r = 0
    for r in iterRadius:
        if etas[r] < etaMin:
            etaMin = etas[r]

    if (etaMin == 0.3):
        return 0 # set error flag

    iterRadius = iter(raios)
    next(iterRadius) # jumping r = 0
    for r in iterRadius:
        if (etas[r] < 0.2):
    		# linear interpolation
            numerator = 0.2 - etas[r] + r * (etas[r-1] - etas[r])
            denominator = etas[r-1] - etas[r]

            if (numerator > 0.0 and denominator > 0.0 and (epx + 2 * (numerator / denominator) <= dimension) and denominator > 0.0):
                return numerator / denominator
            else:
                return 0

@cython.boundscheck(False)
@cython.wraparound(False)   
@cython.cdivision(True)         # with mask
cpdef tuple getAccumulatedLuminosityM(float[:,:] mat,float[:,:] bcg, object ellipse,int nsamples=50, int mindist = 0):
    cdef: 
        float frac, epx, epy
        float[:] dists, acc
        float percentage
        int w, h, wit, hit, i, d, ndists

    if(nsamples<=0):
        raise Exception("Invalid number of samples!(Accumulated Lum). Got"+str(nsamples))
    w, h = len(mat[0]), len(mat)            # largura e altura
    epx, epy = ellipse.posx, ellipse.posy   # posicao central da ellipse
    ndists = len(mat)/2 -mindist            # metade da dimensão da matriz 
    dists = numpy.array([0.0 for i in range(ndists)],dtype=numpy.float32)   # distancias para aneis
    acc = numpy.array([0.0 for i in range(ndists)],dtype=numpy.float32)     # luminosidade acumulada
    for i in range(mindist,len(mat)/2):
        dists[i-mindist] = float(i)         # dists de indice i recebe i
    for d in range(ndists):                 # até o meio da matriz
        acc[d] = 0.0                        # luminosidade acumulada para anel de distancia d
        for wit in range(w):                # iterador da largura
            for hit in range(h):            # iterador da altura
                percentage = samplingMatrixPoint(epx, epy,wit+0.5,hit+0.5,dists[d],nsamples)
                acc[d] = acc[d] + (mat[hit, wit]-bcg[hit, wit])*percentage # acc += (mat[hit,wit]-bcg[hit,wit])*percentage

    return (dists, acc)                     # retorna raios e luminosidade acumulada até eles

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple filterSegmentedMask(float[:,:] mask, ellipse):
    cdef:
        float[:,:] output
        int[:] point
        float galID
        int epos1, epos2
    epos1, epos2 = int(ellipse.posy),int(ellipse.posx)
    output = numpy.array([[1.0 for n in m] for m in mask], dtype=numpy.float32)
    pts = [[epos1,epos2]]
    dists =[]
    galID=mask[epos1, epos2]
    while(len(pts) > 0):
        point = numpy.array(pts.pop(0), dtype=numpy.int32)
        #print(point,mask[point],output[point])
        if(mask[point[0],point[1]] == galID) & (output[point[0],point[1]] == 1.0):
            output[point[0],point[1]] = 0.0
            dists.append(sqrt(pow(float(point[0])-ellipse.posy,2.0)+pow(float(point[1])-ellipse.posx,2.0)))
            if(point[0]+1<len(output)):
                pts.append([point[0]+1,point[1]])
            if(point[0]>0):
                pts.append([point[0]-1,point[1]])
            if(point[1]+1<len(output[point[0]])):
                pts.append([point[0],point[1]+1])
            if(point[1]>0):
                pts.append([point[0],point[1]-1])
    if (dists):
        return (output, galID, max(dists))
    else:
        return (output, galID, 0.0)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef float[:,:] transformPolarCoord( mat, ellipse, int rbins, int tbins):
    cdef:
        float[:] rad, angles
        int ny, nx, i, j
        float x, y
    if(rbins==0):
        raise Exception("Invalid number of bins!(PolarCoord)")
    rad =  numpy.arange(0.0, ellipse.maxRad, ellipse.maxRad/float(rbins),dtype=numpy.float32)
    angles = numpy.arange(0.0, 2.0*pi(), 2.0*pi()/float(tbins), dtype=numpy.float32)
    output = numpy.array([[0.0 for j in range(len(angles))] for i in range(len(rad))],dtype=numpy.float32)
    ny, nx = len(mat), len(mat[0])
    interpolador = interpolate.interp2d([i for i in range(nx)],[j for j in range(ny)], mat, kind = 'linear')
    for ia in range(len(angles)):
        for ir in range(len(rad)):
            x, y = rad[ir]*cos(angles[ia])+ellipse.posx, rad[ir]*sin(angles[ia])+ellipse.posy
            output[ir][ia]= interpolador(x,y)[0]
    return (output)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef float[:,:] transformEllipseCoord( mat, ellipse, int rbins,int tbins):
    cdef:
        float[:] rad, angles
        int ny, nx, i, j
        float x, y
    if(rbins==0):
        raise Exception("Invalid number of bins!(EllipseCoord)")
    rad =  numpy.arange(0.0, 1.0, 1.0/float(rbins),dtype=numpy.float32)
    angles = numpy.arange(-ellipse.angle, 2.0*pi()-ellipse.angle, 2.0*pi()/float(tbins),dtype=numpy.float32)
    output = numpy.array([[0.0 for j in range(len(angles))] for i in range(len(rad))],dtype=numpy.float32)
    ny, nx = len(mat), len(mat[0])
    interpolador = interpolate.interp2d([i for i in range(nx)], [j for j in range(ny)], mat, kind = 'linear')
    for ia in range(len(angles)):
        for ir in range(len(rad)):
            x, y = ellipse.findPoint(rad[ir],angles[ia])
            output[ir][ia]= interpolador(x,y)[0]
    return (output)

def measureSky(mat,bx,by, kernelSize=256):
    halfKernel = int(kernelSize/2)
    print("Measuring Sky...Kernel size:",kernelSize)
    tl = []
    for i in range(-halfKernel,halfKernel+1):
        for j in range(-halfKernel,halfKernel+1):
            y, x = int(by+i), int(bx+j)
            if(y >= 0)and (y<len(mat)) and (x >= 0) and (x<len(mat[0])):
                tl.append(mat[y][x])
    tl=numpy.array(tl)
    if (len(tl)>1):
        median = numpy.median(tl)
        stddev = (sum((tl-median)**2.0)/(len(tl)-1))**0.5
        #print("Mean,Median,stddev without clippi()ng: ",numpy.mean(tl), median, stddev)
        
        stddevAnterior = 2.*stddev
        while((stddevAnterior-stddev)/stddev > 0.2):
            ntl =[]
            #clippi()ng the distribution:
            for i in range(len(tl)):
                if(tl[i] < median+3.*stddev) and (tl[i] > median-3.*stddev):
                    ntl.append(tl[i])
            tl = numpy.array(ntl)
            median = numpy.median(tl)
            stddevAnterior = stddev
            stddev = (sum((tl-median)**2.0)/(len(tl)-1))**0.5
            print("DSigma",(stddevAnterior-stddev)/stddev)

        median = numpy.median(tl)
        stddev = (sum((tl-median)**2.0)/(len(tl)-1))**0.5
        print("Mean,Median,stddev clipped: ",numpy.mean(tl), median, stddev)
        return (numpy.mean(tl), median, stddev)
    else:
        print("Insufficient points")

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def sampleSameDistance(float[:,:] mat,int nsample,float dist,ellipse):
    cdef:
        int count, i, j
        float angle, x, y  
    output = []
    mat2 = numpy.array([[0.0 for j in range(len(mat[i]))] for i in range(len(mat))],dtype=numpy.float32)
    mat3 = numpy.array([[0 for j in range(len(mat[i]))] for i in range(len(mat))],dtype=numpy.int32)
    for count in range(nsample):
        angle = 2.0*pi()*float(count)/float(nsample)
        x, y = ellipse.findPoint(dist,angle)
        if (x < 0.0) or (y<0.0) or (x>=len(mat[0])) or (y>=len(mat)):
            return numpy.array([],dtype=numpy.float32), mat2
        i, j = int(y), int(x)
        if(mat3[i,j] != 0.0):
            continue
        output.append(mat[i,j])
        mat2[i,j] = mat[i,j]
        mat3[i,j] = 1
    return numpy.array(output,dtype=numpy.float32), mat2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef tuple gradient(float[:,:] mat):
    cdef:
        float dist
        int w, h
        int p1x,p1y, p2x,p2y, x, y
        float[:,:] dx, dy
    w, h = len(mat[0]), len(mat)
    dx = numpy.array([[0.0 for i in range(w)] for j in range(h)], dtype=numpy.float32)
    dy = numpy.array([[0.0 for i in range(w)] for j in range(h)], dtype=numpy.float32)
    for y in range(h):
        for x in range(w):
            #dy gradient:
            dist = 2.0
            p1x, p1y = x, y 
            p2x,p2y = x, y

            if (y-1<0):
                dist = 1.0
            else:
                p1y = y-1
            if (y+1 >= h):
                dist = 1.0
            else:
                p2y = y+1
            dy[y, x] = (mat[ p2y, p2x ] - mat[ p1y, p1x ])/dist

            #dx gradient:
            dist = 2.0
            p1x, p1y = x, y 
            p2x,p2y = x, y
            if (x-1 < 0):
                dist = 1.0
            else:
                p1x = x-1
            if (x+1 >= w ):
                dist = 1.0
            else:
                p2x = x+1
            dx[y, x] = (mat[ p2y, p2x ] - mat[ p1y, p1x ])/dist
    return dy,dx
