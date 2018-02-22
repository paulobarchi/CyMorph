from libc.math cimport log10,sqrt, atan2, pow, fabs, ceil, floor
import numpy
import gridAlg
import scipy.ndimage as ndimage
import scipy.signal as signal
import math
from scipy.optimize import curve_fit
from scipy import fftpack
from random import shuffle

cimport numpy
cimport cython
cdef float pi = 3.14159265

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef float[:] ravel(float[:,:] mat):
    cdef:
          int w, h, countNotMasked, j, i,it
          float[:] line
    w, h = len(mat[0]), len(mat)
    countNotMasked = 0
    for i in range(w):
        for j in range(h):
            if(mat[j, i] != 0.0):
               countNotMasked = countNotMasked + 1
    line = numpy.array([0.0 for i in range(countNotMasked)], dtype=numpy.float32)
    it = 0
    for i in range(w):
        for j in range(h):
            if(mat[j, i] != 0.0):
                line[it] = mat[j, i]
                it = it + 1 
    return line
     

############################################
#     Funcoes de entropia:
# entrada:
#    mat - matriz do tipo numpy(list(list))
#    bins - numero de patamares
# retorna:
#    coeficiente encontrado,
#    matriz
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def entropyFunction(float[:,:] mat,int bins):
    cdef:
        float[:] freq, line
        double[:] binagem
        long[:] temp
        float somatorio,coef
        tuple x
        list entropies
        int w, h, i
    w, h = len(mat[0]), len(mat)
    line = ravel(mat)
    freq = numpy.array([0.0 for i in range(bins)], dtype=numpy.float32)
    temp, binagem = numpy.histogram(line,bins)
    somatorio = 0.0
    for i in range(bins):
        somatorio = somatorio + temp[i]
    for i in range(bins):
        freq[i] = float(temp[i])/float(somatorio)
    somatorio = 0.0
    for i in range(bins):
        if freq[i]>0.0:
            somatorio = somatorio - freq[i]*log10(freq[i])
    coef = somatorio/log10(bins)
    return coef


############################################
#     Translating and rotating
def rotateImage(float[:,:] img,float angle):
    imgR = ndimage.rotate(img, angle, reshape=False,mode='nearest')
    return imgR

############################################
#     Funcoes de assimetria:
# entrada:
#    mat - matriz do tipo numpy(list(list))
#    corrFunction - funcao de correlacao (stats.pearsonr, stats.spearmanr, ....)
# retorna:
#    coeficiente encontrado,
#    matrizRotacionada
#    pontos de correlacao (I,I_h)
#    mascara de pontos considerados (rotacionado e nao rotacionado)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def asymmetryFunction(float[:,:] mat,float[:,:] mask, corrFunct, ell):
    cdef:
         float minimo
         float[:,:]  matInverse, maskInverse,matWBCG
         int w, h, countNotMasked, it
    w, h = len(mat[0]), len(mat)
 
    maskInverse = rotateImage(mask, 180.0)
    matInverse = rotateImage(mat, 180.0)
    
    countNotMasked = 0
    for i in range(w):
        for j in range(h):
            if (mask[j,i] < 0.5) and (maskInverse[j,i] < 0.5) and (sqrt((i-w)**2+(j-h)**2) > 0.2*ell.fwhm): 
                countNotMasked = countNotMasked + 1
    v1 = []
    v2 = []
    it = 0
    for i in range(w):
        for j in range(h):
            if (mask[j,i] <= 0.5) and (maskInverse[j,i] <= 0.5) and (sqrt((i-w)**2+(j-h)**2) > 0.2*ell.fwhm): 
                v1.append(mat[j,i])
                v2.append(matInverse[j,i])
                it = it + 1

    #mv1 = numpy.max(v1)
    #mv2 = numpy.max(v2)
    #for it in range(countNotMasked):
    #    v1[it] = v1[it]/mv1
    #    v2[it] = v2[it]/mv2
    

    coef = corrFunct(v1, v2)[0]

    return coef, matInverse, (v1, v2)

#   OLD Funcao Concentration : OLD
############################################
# entrada:
#    mat - matriz do tipo numpy(list(list))
#    calibratedData - booleano indicando se os dados estao calibrados
#    p1 - porcentagem de luminosidade do numerador (0.8->c1,  0.9 -> c2,)
#    p2 - porcentagem de luminosidade do denominador (0.2 -> c1,  0.5 -> c2)
#    ell - elipse (com o centro definido)
#    nbins - variacao do raio avaliado
# retorna:
#    coeficiente encontrado,
#    matriz,
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def concentrationFunction(dists,concSeq, p1, p2, minCut):
    total, cutDist = getTotalLum(dists,concSeq,1.0,minCut)
    rn = gridAlg.findRadiusLuminosity(dists,concSeq,total, p1)
    rd = gridAlg.findRadiusLuminosity(dists,concSeq,total, p2)
    
    if(rd > 0.0):
        return log10(rn/rd),total
    else:
        raise Exception("Zero Division Error in concentration!")

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def getTotalLum(dists,conc, k,minD):

    # dconc é a concentração por anéis
    dconc = numpy.array([0.0 for i in range(len(conc)-2)])
    for i in range(1,len(conc)-1):      # para cada anel
        dconc[i-1] = (conc[i] - conc [i-1]) / (dists[i]-dists [i-1])

    # numpy.gradient faz (conc[i+1] - conc[i-1]) / (dists[i+1]-dists[i-1])
    dconc = numpy.gradient(conc)

    # Finding distance that dconc < 1%
    for cutDist in range(2,len(dconc)):        
        temp = conc[i] if conc[i] != 0.0 else 1.0
        if(abs(dconc[cutDist]/conc[cutDist])<minD):
            break 

    if(len(dconc)-1 == cutDist):
        raise Exception("Not Convergent Concentration!")

    median = numpy.median(dconc[cutDist:len(dconc)])
    # median = numpy.median(dconc)

    sigma = numpy.median(abs(dconc[cutDist:len(dconc)] - median))
    avg = 0.0
    n = 0.0

    for i in range(1,len(dconc)):
        if(abs(dconc[i])<k*sigma):
            n = n + 1.0
            avg += conc[i]
    if n > 10:
        return avg/n, cutDist
    else:
        raise Exception("Not Convergent Concentration!")


#   CURRENT Funcao Concentration : CURRENT
############################################
# entrada:
#    acc - acumulo de luminosidade por aneis (raio varia de 1 em 1)
#    Rp - raio petrosiano encontrado    
#    p1 - porcentagem de luminosidade do numerador (0.8->c1,  0.9 -> c2,)
#    p2 - porcentagem de luminosidade do denominador (0.2 -> c1,  0.5 -> c2)
# retorna:
#    coeficiente encontrado
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def getConcentration(acc, Rp, p1, p2):

    # get L_rp -- petrosian luminosity -- acumulo até 2 Rp
    if (Rp != 0 and not numpy.isnan(Rp) and len(acc) != 0 and (int(ceil(2*Rp)) < len(acc) and int(floor(2*Rp)) < len(acc)) ):
        L_rp = ( acc[int(ceil(2*Rp))] - acc[int(floor(2*Rp))] )
        L_rp = L_rp * (2*Rp - int(floor(Rp))) + acc[int(floor(2*Rp))]

        L_r1 = p1*L_rp
        L_r2 = p2*L_rp

        r1 = r2 = 0.0

        for r in range(1,len(acc)):
            if (r >= len(acc)):
                return numpy.nan
            if (r < len(acc) and acc[r] > L_r1):
                # linear interpolation
                r1 = ((L_r1 - acc[r-1]) / (acc[r] - acc[r-1])) + r -1
                break

        for r in range(1,len(acc)):
            if (r >= len(acc)):
                return numpy.nan
            if (acc[r] > L_r2):
                # linear interpolation
                r2 = ((L_r2 - acc[r-1]) / (acc[r] - acc[r-1])) + r -1
                break

        if (r1 > 0.0 and r2 > 0.0):
            return log10(r1 / r2)
        else:
            return numpy.nan
    else: 
        return numpy.nan

############################################ 
#     Funcao Smoothness (Suavizacao):
# entrada:
#    mat - matriz do tipo numpy(list(list))
#    kernel - matriz de convolucao
#    corrFunction - funcao de correlacao (stats.pearsonr, stats.spearmanr, ....)
# retorna:
#    coeficiente encontrado (valor e prob. da hipotese nula),
#    matriz,
#    matrizRotacionada
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def smoothnessFunction(float[:,:] mat, float[:,:]smMat, float[:,:] mask,float[:,:] bcg, corrFunct, ell):
    cdef:
        int w, h, countPts,it,i,j,countNotMasked
    w, h = len(mat[0]), len(mat)

    
    # counting the number of segmented pixels 
    countPts=0
    for i in range(w):
        for j in range(h):
            # if(mask[j, i] <= 0.00001) and (sqrt((i-w)**2+(j-h)**2) > 0.2*ell.fwhm):
            if(mask[j, i] == 0.0) and (sqrt((i-w/2)**2+(j-h/2)**2) > 0.2*ell.fwhm):
                countPts += 1

    if(countPts<6):
        raise Exception("Invalid number of smoothing pixels")
    it = 0 
    
    v1,v2= [0.0 for i in range(countPts)],[0.0 for i in range(countPts)]

    for i in range(w):
        for j in range(h):
            if (mask[j,i] == 0.0) and (sqrt((i-w/2)**2+(j-h/2)**2) > 0.2*ell.fwhm): 
                v1[it] = mat[j,i]
                v2[it] = smMat[j,i]
                it = it + 1

    # if (len(numpy.nonzero(v1)[0]) > 0.666*len(v1) or len(numpy.nonzero(v1)[0]) > 0.666*len(v2)):
        # coef = 1.0 - corrFunct(v1, v2)[0]
    coef = corrFunct(v1, v2)[0]
    # else:
    # coef = numpy.nan
    return coef,(v1, v2)


############################################
#     Funcao Spirality (Espiralidade):
# entrada:
#    mat - matriz do tipo numpy(list(list))
#    nPhase - numero de angulos na matriz  em coordenadas polares
#    nRadius - numero de distancias na matriz  em coordenadas polares
#    ellipse - elipse que contem galaxia
# retorna:
#    coeficiente encontrado ,
#    matriz,
#    matriz em Coordenadas Polares
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def spiralityFunction(float[:,:] mat,float[:,:]  mask, int nPhases,int  nRadius, ellipse):
    cdef:
        float[:,:] transformed, transformedMask,x, y
        float phase, mod, spirality, sumPhase, nphase
        int yIt, xIt
    transformed = gridAlg.transformPolarCoord(mat, ellipse, nPhases, nRadius)
    transformedMask = gridAlg.transformPolarCoord(mask, ellipse, nPhases, nRadius)
    y, x = gridAlg.gradient(transformed)

    ## y -> phase
    ## x -> module
    sumPhase = float(0.0)
    nphase = 1
    for yIt in range(len(y)):
        for xIt in range(len(y[yIt])):
            if(transformedMask[yIt, xIt] >= 0.5):
                continue
            mod = sqrt(pow(y[yIt, xIt],2.0)+pow(x[yIt, xIt],2.0))
            phase = atan2(y[yIt, xIt],x[yIt, xIt])
            phase = phase if phase > 0.0 else phase + 2.0*pi
            y[yIt, xIt] = phase
            x[yIt, xIt] = mod
            sumPhase = sumPhase + phase
            nphase = nphase + 1
    if(nphase > 0):
        mean = sumPhase / float(nphase)
    else:
        raise Exception("No points found in transformed polar coordinates masked!")
    spirality = 0.0
    for yIt in range(len(y)):
        for xIt in range(len(y[yIt])):
            if(transformedMask[yIt, xIt] >= 0.5):
                continue
            spirality += pow(y[yIt, xIt] - mean,2.0)
    if(nphase > 1):
        spirality = sqrt(spirality / float(nphase-1))
    else:
        raise Exception("No points found in transformed polar coordinates masked!")
    return (spirality, transformed, x, y)

############################################
#     Funcao Spirality (Espiralidade):
# entrada:
#    mat - matriz do tipo numpy(list(list))
#    nPhase - numero de angulos na matriz  em coordenadas polares
#    nRadius - numero de distancias na matriz  em coordenadas polares
#    ellipse - elipse que contem galaxia
# retorna:
#    coeficiente encontrado ,
#    matriz,
#    matriz em Coordenadas Polares
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def spiralityEllipsoidalFunction(float[:,:] mat,float[:,:]  mask,int nPhases,int nRadius, ellipse):
    cdef:
        float[:,:] transformed, transformedMask, x, y
        float phase, mod, spirality, sumPhase, nphase
        int yIt, xIt
    transformed = gridAlg.transformEllipseCoord(mat, ellipse, nPhases, nRadius)
    transformedMask = gridAlg.transformEllipseCoord(mask, ellipse, nPhases, nRadius)
    
    y, x = gridAlg.gradient(transformed)

    ## y -> phase
    ## x -> module
    sumPhase = float(0.0)
    nphase = 1
    for yIt in range(len(y)):
        for xIt in range(len(y[yIt])):
            if(transformedMask[yIt, xIt] >= 0.5):
                continue
            mod = sqrt(pow(y[yIt, xIt],2.0)+pow(x[yIt, xIt],2.0))
            phase = atan2(y[yIt, xIt],x[yIt, xIt])
            phase = phase if phase > 0.0 else phase + 2.0*pi
            y[yIt, xIt] = phase
            x[yIt, xIt] = mod
            sumPhase = sumPhase + phase
            nphase = nphase + 1
    if(nphase > 0):
        mean = sumPhase / float(nphase)
    else:
        raise Exception("No points found in transformed spiral coordinates masked!")
    spirality = 0.0
    for yIt in range(len(y)):
        for xIt in range(len(y[yIt])):
            if(transformedMask[yIt, xIt] >= 0.5):
                continue
            spirality += pow(y[yIt, xIt] - mean,2.0)
    if(nphase > 1):
        spirality = sqrt(spirality / float(nphase-1))
    else:
        raise Exception("No points found in transformed spiral coordinates masked!")
    return (spirality, transformed, x, y)
    


def spirality3(float[:,:] mat,float[:,:] bcg, ellipse):
    tseq = []
    nsamplePerDist=1000
    distances=numpy.array([0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75],dtype=numpy.float32)
    mat2 = numpy.array([[ mat[i][j]-bcg[i][j] for i in range(len(mat[i]))] for j in range(len(mat))], dtype=numpy.float32)
    for d in distances:
        seq, trash = gridAlg.sampleSameDistance(mat2,nsamplePerDist,d,ellipse)
        avg = numpy.average(seq)
        fft = fftpack.fft(seq/avg)
        tseq=numpy.concatenate([tseq,numpy.real(fft[1:(len(fft)-1)])])
        tseq=numpy.concatenate([tseq,numpy.imag(fft[1:(len(fft)-1)])])
    return max(numpy.abs(tseq))
    

############################################
#     Funcao Hamming kernel:
# entrada:
#    dim - dimensao da matriz
# retorna:
#    matriz resultante
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def hammingKernel(int dim):
    line = signal.hamming(float(dim))
    mat = numpy.sqrt(numpy.outer(line,line))
    mat = mat /mat.sum()
    mat = (mat).astype('float32')
    return mat

############################################
#     Funcao box car kernel:
# entrada:
#    dim - dimensao da matriz
# retorna:
#    matriz resultante
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def boxCarKernel(int dim):
    mat = numpy.array([[1.0 for i in range(dim)] for j in range(dim)])
    mat = mat / mat.sum()
    return mat

############################################
#    AsymmetryConselice
# entrada:
#    mat - matriz do tipo numpy(list(list))
#    corrFunction - funcao de correlacao (stats.pearsonr, stats.spearmanr, ....)
# retorna:
#    coeficiente encontrado,
#    matrizRotacionada
#    pontos de correlacao (I,I_h)
#    mascara de pontos considerados (rotacionado e nao rotacionado)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def asymmetryConselice(float[:,:] mat, angle, Rp):
    cdef:
         float[:,:]  matInverse
         float numerator, denominator
         int w, h
    w, h = len(mat[0]), len(mat)
 
    # angle = 90 or 180???
    matInverse = rotateImage(mat, angle)
    
    numerator = 0.0
    denominator = 0.0

    for i in range(w):
        for j in range(h):
            # considering only pixels inside 1.5 * Petrossian_radius
            if ( ( sqrt((i-w/2)**2 + (j-h/2)**2) < 1.5 * Rp )  and mat[j,i] > 0.0):
                # one sum for numerator and other for the denominator
                numerator = numerator + abs(mat[j,i] - matInverse[j,i])
                denominator = denominator + abs(mat[j,i])

    return (numerator) / (denominator), matInverse

#################################################
#     SmoothnessConselice OLD - from Conselice (2003)
# input:
#    mat - numpy matrix (list(list))
#    smMat - smoothed matrix
#    smoothedSky - skyMedian from smoothed matrix
#    Rp - petrosian radius
#
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def smoothnessConseliceOLD(float[:,:] mat, float[:,:] smMat, sky, Rp):
    cdef:
        int w, h, i, j
        float total, current
    
    w, h = len(mat[0]), len(mat)

    total = 0.0

    for i in range(w):
        for j in range(h):
            # considering only pixels inside 1.5 * petrossian radius
            # and outside 0.25 * petrossian radius
            if ( (sqrt((i-w/2)**2 + (j-h/2)**2) <= 1.5 * Rp) and \
                (sqrt((i-w/2)**2 + (j-h/2)**2) > 0.25 * Rp) and mat[j,i] > 0.0):
                if ( ( mat[j,i]-smMat[j,i] ) - sky  > 0.0 ):
                    # total = total + (( mat[j,i]-smMat[j,i] ) - sky ) / (mat[j,i])
                    total = total + ( mat[j,i]-smMat[j,i] ) / (mat[j,i])

    return 10*(total)


#################################################
#     SmoothnessConselice - from Conselice (2003)
# input:
#    mat - numpy matrix (list(list))
#    smMat - smoothed matrix
#    smoothedSky - skyMedian from smoothed matrix
#    Rp - petrosian radius
#
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def smoothnessConselice1(float[:,:] mat, float[:,:] smMat, sky, Rp):
    cdef:
        int w, h, i, j
        float total, current
    
    w, h = len(mat[0]), len(mat)

    numerator = 0.0
    denominator = 0.0

    for i in range(w):
        for j in range(h):
            # considering only pixels inside 1.5 * petrossian radius
            # and outside 0.25 * petrossian radius
            if ( (sqrt((i-w/2)**2 + (j-h/2)**2) <= 1.5 * Rp) and \
                (sqrt((i-w/2)**2 + (j-h/2)**2) > 0.25 * Rp)):
                if ( ( mat[j,i]-smMat[j,i] ) - sky  > 0.0 ):
                    numerator = numerator + ( mat[j,i]-smMat[j,i] - sky)
                    denominator =  denominator + (mat[j,i])

    if (denominator > 0.0):
        return 10*(numerator/denominator)
    else:
        return numpy.nan


#################################################
#     SmoothnessConselice - from Conselice (2003)
# input:
#    mat - numpy matrix (list(list))
#    smMat - smoothed matrix
#    smoothedSky - skyMedian from smoothed matrix
#    Rp - petrosian radius
#
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def smoothnessConselice2(float[:,:] mat, float[:,:] smMat, sky, Rp):
    cdef:
        int w, h, i, j
        float total, current
    
    w, h = len(mat[0]), len(mat)

    total = 0.0
    terms = []

    for i in range(w):
        for j in range(h):
            # considering only pixels inside 1.5 * petrossian radius
            # and outside 0.25 * petrossian radius
            if ( (sqrt((i-w/2)**2 + (j-h/2)**2) <= 1.5 * Rp) and \
                (sqrt((i-w/2)**2 + (j-h/2)**2) > 0.25 * Rp)):
                if ( (mat[j,i] - sky)  > 0.0  and ( mat[j,i]-smMat[j,i]) / (mat[j,i] - sky) > 0.0 ):
                    total = total + ( mat[j,i]-smMat[j,i]) / (mat[j,i] - sky)
                    terms.append(( mat[j,i]-smMat[j,i]) / (mat[j,i] - sky))

    return 10*(total), terms

#################################################
#     SmoothnessConselice - from Conselice (2003)
# input:
#    mat - numpy matrix (list(list))
#    smMat - smoothed matrix
#    smoothedSky - skyMedian from smoothed matrix
#    Rp - petrosian radius
#
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def smoothnessConselice3(float[:,:] mat, float[:,:] smMat, float[:,:] noBCG, Rp):
    cdef:
        int w, h, i, j
        float total, current
    
    w, h = len(mat[0]), len(mat)

    numerator = 0.0
    denominator = 0.0

    for i in range(w):
        for j in range(h):
            # considering only pixels inside 1.5 * petrossian radius
            # and outside 0.25 * petrossian radius
            if ( (sqrt((i-w/2)**2 + (j-h/2)**2) > 0.25 * Rp) ):
                if ( ( mat[j,i]-smMat[j,i] ) > 0.0 ):
                    numerator = numerator + ( mat[j,i]-smMat[j,i])
                denominator =  denominator + (noBCG[j,i]) # other version with tab in this line

    if (denominator > 0.0 and numerator > 0.0):
        return 10*(numerator/denominator)
    else:
        return numpy.nan


#################################################
#     SmoothnessConselice - from Conselice (2003)
# input:
#    mat - numpy matrix (list(list))
#    smMat - smoothed matrix
#    smoothedSky - skyMedian from smoothed matrix
#    Rp - petrosian radius
#
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def smoothnessConselice4(float[:,:] mat, float[:,:] smMat, sky):
    cdef:
        int w, h, i, j
        float numerator, denominator, smCons
    
    w, h = len(mat[0]), len(mat)

    numerator = 0.0
    denominator = 0.0

    nums = []
    dens = []

    for i in range(w):
        for j in range(h):
            # we consider only pixels outside the central 5x5 matrix, so...
            if ( i < w/2 - 2 or i > w/2 + 2 or j < h/2 - 2 or j > h/2 + 2 ):
                # only positive values
                if ( ( mat[j,i]-smMat[j,i] ) > 0.0 and mat[j,i] - sky > 0.0):
                    numerator = numerator + ( mat[j,i] - smMat[j,i] )
                    nums.append(mat[j,i]-smMat[j,i])
                    denominator =  denominator + (mat[j,i] - sky)
                    dens.append(mat[j,i]-sky)

    if (denominator > 0.0 and numerator > 0.0):
        # smCons = 10*(numerator/denominator)
        smCons = (numerator/denominator)
    else:
        smCons = numpy.nan
    return smCons, nums, dens


############################################
#     SmoothnessLotz - from Lotz (2004)
# input:
#    mat - numpy matrix (list(list))
#    smMat - smoothed matrix
#    smoothedSky - skyMedian from smoothed matrix
#    Rp - petrosian radius
#
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def smoothnessLotz(float[:,:] mat, float[:,:] smMat, smoothedSky, Rp):
    cdef:
        int w, h, i, j
        float numerator, denominator
    
    w, h = len(mat[0]), len(mat)

    numerator = 0.0
    denominator = 0.0

    for i in range(w):
        for j in range(h):
            # considering only pixels inside 1.5 * petrossian radius
            # and outside 0.25 * petrossian radius
            if ( (sqrt((i-w/2)**2 + (j-h/2)**2) <= 1.5 * Rp) and \
                (sqrt((i-w/2)**2 + (j-h/2)**2) > 0.25 * Rp) ):                
                numerator = numerator + abs( ( mat[j,i]-smMat[j,i] ) )
                denominator = denominator + abs(mat[j,i])

    if (denominator > 0.0):
        return (numerator/denominator)-smoothedSky
    else:
        return numpy.nan

############################################
#     SmoothnessTakamiya - from Takamiya (1999)
# input:
#    mat - numpy matrix (list(list))
#    smMat - smoothed matrix
#    Rp - petrosian radius
#
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def smoothnessTakamiya(float[:,:] mat, float[:,:] smMat, Rp):
    cdef:
        int w, h, i, j
        float numerator, denominator
    
    w, h = len(mat[0]), len(mat)

    numerator = 0.0
    denominator = 0.0

    for i in range(w):
        for j in range(h):
            # considering only pixels inside 1.5 * petrossian radius
            # and outside 0.25 * petrossian radius
            if ( (sqrt((i-w/2)**2 + (j-h/2)**2) <= 1.5 * Rp) and \
                (sqrt((i-w/2)**2 + (j-h/2)**2) > 0.25 * Rp) ):
                numerator = numerator + smMat[j,i]
                denominator = denominator + mat[j,i]

    if (denominator > 0.0):
        return (numerator/denominator)
    else:
        return numpy.nan
