
import scipy.stats as stats
from GPA import *
from scipy.ndimage.filters import convolve
from scipy.ndimage.filters import median_filter
import numpy
import gridAlg
import ellipse
import indexes
import galaxyIO
import sys
import os
import ConfigParser
import fileinput
import time

cimport numpy
from libc.math cimport sqrt, pow

verbose = 1

cdef float pi = 3.14159265

def printIfVerbose(string):
    if (verbose > 0):
        print string

cdef class CyMorph:
    cdef int Spirality_Ny, Spirality_Nx, Entropy_KFolds, Concentration_Density
    cdef float Ga_Tolerance, Ga_Angular_Tolerance, Ga_Position_Tolerance 
    cdef float Ga_sexThreshold, Entropy_sexThreshold, sky, 
    cdef float smoothingDegradation, Smoothness_sexThreshold
    cdef float Asymmetry_sexThreshold, butterOrder, minCut, stampSize
    cdef float d1, d2, petroRad, rowc, colc
    cdef int[:] indexes2Evaluate 
    cdef int verbose
    cdef int onlySegmentation
    cdef int errorVar

    def __init__(self):
        self.verbose = True
        self.sky = -1.0

        self.Ga_Tolerance = 0.03
        self.Ga_Angular_Tolerance = 0.03
        self.Ga_Position_Tolerance =  1.0
        self.Ga_sexThreshold = 1.4 # TODO: ASSERT BEST DEFAULT THRESHOLD

        self.Spirality_Ny, self.Spirality_Nx = 200, 400

        self.Entropy_KFolds = 150
        self.Entropy_sexThreshold = 1.4 # TODO: ASSERT BEST DEFAULT THRESHOLD

        self.Concentration_Density = 100
        
        self.smoothingDegradation = 0.9
        self.Smoothness_sexThreshold = 1.4 # TODO: ASSERT BEST DEFAULT THRESHOLD

        self.Asymmetry_sexThreshold = 1.4 # TODO: ASSERT BEST DEFAULT THRESHOLD

        self.butterOrder = 3
        self.minCut = 0.09
        self.d1,self.d2 = -1.0, -1.0
        self.stampSize = 5.0
        self.onlySegmentation = 0
        self.errorVar = 0
        self.indexes2Evaluate = numpy.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=numpy.int32)
        
    
    # Sets
    def setSky(self, float value):
        self.sky = value    

    def setGa_Tolerance(self, float value):
        self.Ga_Tolerance = value         
    def setGa_Angular_Tolerance(self, float value):
        self.Ga_Angular_Tolerance = value
    def setGa_Position_Tolerance(self, float value):
        self.Ga_Position_Tolerance = value
    def setGa_sexThreshold(self, float value):
        self.Ga_sexThreshold = value

    def setSpirality_Ny(self, int value):
        self.Spirality_Ny = value 
    def setSpirality_Nx(self, int value):
        self.Spirality_Nx = value

    def setEntropy_KFolds(self, int value):
        self.Entropy_KFolds = value
    def setEntropy_sexThreshold(self, float value):
        self.Entropy_sexThreshold = value

    def setConcentration_Density(self, int value):
        self.Concentration_Density = value

    def setSmoothnessOrder(self, float value):
        self.butterOrder = value
    def setSmoothnessDegree(self,float value):
        self.smoothingDegradation = value
    def setSmoothness_sexThreshold(self, float value):
        self.Smoothness_sexThreshold = value

    def setConcentrationDists(self,float d1, float d2):
        self.d1 = d1
        self.d2 = d2
    
    def setAsymmetry_sexThreshold(self, float value):
        self.Asymmetry_sexThreshold = value

    def setOnlySegmentation(self,int value):
        self.onlySegmentation = value
    def setStampSize(self, float value):
        self.stampSize = value

    def setOnlyIndex(self,c):
        self.indexes2Evaluate = numpy.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=numpy.int32)
        self.setIndexV(c,int(1))

    def isSetIndex(self,indx):
        if(indx=='C') and (self.indexes2Evaluate[0] == 1):
           return True
        elif(indx=='A') and (self.indexes2Evaluate[1] == 1):
           return True
        elif(indx=='S') and (self.indexes2Evaluate[2] == 1):
           return True
        elif(indx=='H') and (self.indexes2Evaluate[3] == 1):
           return True
        elif(indx=='Sp') and (self.indexes2Evaluate[4] == 1):
           return True
        elif(indx=='Ga') and (self.indexes2Evaluate[5] == 1):
           return True
        elif(indx=='C1') and (self.indexes2Evaluate[6] == 1):
           return True
        elif(indx=='C2') and (self.indexes2Evaluate[7] == 1):
           return True
        elif(indx=='CN') and (self.indexes2Evaluate[8] == 1):
           return True
        elif(indx=='A2') and (self.indexes2Evaluate[9] == 1):
           return True
        elif(indx=='A3') and (self.indexes2Evaluate[10] == 1):
           return True
        elif(indx=='S2') and (self.indexes2Evaluate[11] == 1):
           return True
        elif(indx=='S3') and (self.indexes2Evaluate[12] == 1):
           return True
        elif(indx=='G1') and (self.indexes2Evaluate[13] == 1):
           return True
        elif(indx=='Rp') and (self.indexes2Evaluate[14] == 1):
           return True
        elif(indx=='nO') and (self.indexes2Evaluate[15] == 1):
           return True
        else:
           return False

    def setIndexV(self, indx ):
        if(indx=='C'):
           self.indexes2Evaluate[0] = 1
        elif(indx=='A'):
           self.indexes2Evaluate[1] = 1
        elif(indx=='S'):
           self.indexes2Evaluate[2] = 1
        elif(indx=='H'):
           self.indexes2Evaluate[3] = 1
        elif(indx=='Sp'):
           self.indexes2Evaluate[4] = 1
        elif(indx=='Ga'):
           self.indexes2Evaluate[5] = 1
        elif(indx=='C1'):
           self.indexes2Evaluate[6] = 1
        elif(indx=='C2'):
           self.indexes2Evaluate[7] = 1
        elif(indx=='CN'):
           self.indexes2Evaluate[8] = 1
        elif(indx=='A2'):
           self.indexes2Evaluate[9] = 1
        elif(indx=='A3'):
           self.indexes2Evaluate[10] = 1
        elif(indx=='S2'):
           self.indexes2Evaluate[11] = 1
        elif(indx=='S3'):
           self.indexes2Evaluate[12] = 1
        elif(indx=='G1'):
           self.indexes2Evaluate[13] = 1
        elif(indx=='Rp'):
           self.indexes2Evaluate[14] = 1           
        elif(indx=='nO'):
           self.indexes2Evaluate[14] = 1
        else:
           raise Exception("Unknown index: "+str(indx))

    def _replaceAfterLastDot(self,char* fileName,char* str2Rep):
        splitted = fileName.split('.')
        splitted[len(splitted)-1] = str2Rep
        return ".".join(splitted)

    def clearIt(self,fileName,xtraID):
        # os.remove("cutted/"+str(xtraID)+".fit")
        os.remove(str(xtraID)+".cat")
        os.remove(str(xtraID)+"_seg.fits")
        os.remove(str(xtraID)+"_bcg.fits")

 
    def _runMaskMaker(self,char* path,char* fileName,char* xtraID,float ra, float dec,float petroRad,float petroMag,float rowc,float colc):
        cuttedFile = str(xtraID)+'.fit'

        configFile = ConfigParser.ConfigParser()
        configFile.read('cfg/paths.ini')
        pythonPath = configFile.get("Path","Python")

        #Change directory, execute maskMaker and get back
        localPath = os.getcwd()
        newPath = os.getcwd()+'/maskMaker'
        os.chdir(newPath)
        cmd = pythonPath+" -W\"ignore\" maskmaker_wcut.py ../"+path+fileName+" "+\
            str(xtraID)+" "+str(ra)+" "+str(dec)+" "+str(self.stampSize)+\
            " "+str(petroRad)+" "+str(petroMag)+" "+str(rowc)+" "+str(colc)+" >> logMaskMaker.txt"
        pr = os.popen(cmd)
        pr.read()
        os.chdir(localPath)
        with open(str(xtraID)+"_log.txt",'r') as eF:
            self.errorVar = int(eF.read())
        os.remove(str(xtraID)+"_log.txt")
        return cuttedFile

    def maskAndClean(self,char* path,char* fileName,char* image,char* xtraID,char* maskFile,float ra,float dec,float petroRad,float petroMag,float rowc,float colc,calibratedData,saveFig,clear,float segThreshold):
    	
    	# sExtractor 1st run to obtain fit segmentation regions
        printIfVerbose("Running Sextractor")
        notMasked = galaxyIO.readFITSIMG(path+fileName)
        height, width = len(notMasked), len(notMasked[0])
        galaxyIO.runSextractor(path,fileName,xtraID,["DETECT_THRESH"],[segThreshold])

        if (maskFile == <bytes> ''):
            printIfVerbose("No mask... Considering every point in the image")
            mask = numpy.array([[0.0 for j in range(width)] for i in range(height)],dtype=numpy.float32)
        else:
            printIfVerbose("Reading mask file "+path+maskFile)
            mask = galaxyIO.readFITSIMG(path+maskFile)
        
        segmentation = galaxyIO.readFITSIMG(str(xtraID)+"_seg.fits")
        bcg = galaxyIO.readFITSIMG(str(xtraID)+"_bcg.fits")
        dic, data = galaxyIO.readSextractorOutput(str(xtraID)+".cat")
        dicFiltered, dataFiltered = galaxyIO.filterSextractorData(mask, dic, data)
        if clear:
            self.clearIt(fileName,xtraID)
        printIfVerbose("Interpolating ellipse")
        e = ellipse.ellipse(dic,dataFiltered,calibratedData,self.sky)
        segmentationMask, idGalaxy, segMax = gridAlg.filterSegmentedMask(segmentation,e)
        removedGalaxies = gridAlg.removeOtherGalaxies(notMasked, segmentation, idGalaxy)
        newMat, holes = gridAlg.interpolateEllipse(removedGalaxies,e)

        # save clean stamp (fit)
        galaxyIO.plotFITS(newMat,"cutted/"+str(xtraID)+".fit")

        # sExtractor 2nd run (with clean image as input) to calibrate background
        printIfVerbose("Running Sextractor again")
        bcgW = 32
        galaxyIO.runSextractor("cutted/", str(xtraID)+".fit", xtraID,["BACK_SIZE","DETECT_THRESH"],[bcgW,segThreshold])
        segmentation = galaxyIO.readFITSIMG(str(xtraID)+"_seg.fits")
        bcg = galaxyIO.readFITSIMG(str(xtraID)+"_bcg.fits")
        dic, data = galaxyIO.readSextractorOutput(str(xtraID)+".cat")
        # here we have a dictionary and data of identified objects in image
        dicFiltered, dataFiltered = galaxyIO.filterSextractorData(mask, dic, data)
        if clear:
            self.clearIt(fileName,xtraID)
        if saveFig:
            galaxyIO.plotFITS(bcg,"imgs/bcg"+str(bcgW)+".fit")

        e = ellipse.ellipse(dic,dataFiltered,calibratedData,self.sky)
        segmentationMask, idGalaxy, segMax = gridAlg.filterSegmentedMask(segmentation,e)
        printIfVerbose("Starting enhancing")
        noBCG = numpy.array([[ newMat[i][j]-bcg[i][j] for j in range(len(newMat[i]))] for i in range(len(newMat))], dtype=numpy.float32)
        dx,dy = convolve(noBCG,numpy.array([[-1,-2,-1],\
                                            [ 0, 0, 0],\
                                            [ 1, 2, 1]])),\
                convolve(noBCG,numpy.array([[-1,0,1],\
                                            [-2,0,2],\
                                            [-1,0,1]])) 
        dp,di = convolve(noBCG,numpy.array([[-2,-1,0],\
                                            [-1, 0,1],\
                                            [ 0, 1,2]])),\
                convolve(noBCG,numpy.array([[ 0, 1, 2],\
                                            [-1, 0, 1],\
                                            [-2,-1, 0]])) 
        gradMod = numpy.array([[sqrt(pow(dp[i][j],2.0)+pow(di[i][j],2.0)+pow(dx[i][j],2.0)+pow(dy[i][j],2.0)) for j in range(len(dx[i]))] for i in range(len(dx))],dtype=numpy.float32)
        maxGrad = numpy.max(gradMod)
        gradModF = numpy.array([[ noBCG[i,j]*(gradMod[i,j])/maxGrad for j in range(len(dx[i]))] for i in range(len(dx))], dtype=numpy.float32)

        # save images from processing steps
        if saveFig:
             galaxyIO.plotFITS(noBCG,"imgs/noBCG.fit")
             galaxyIO.plotFITS(segmentationMask,"imgs/mask.fit")
             galaxyIO.plotFITS(gradModF,"imgs/gradientMod.fit")
        
        printIfVerbose("Making mask")
        segmentationMask, idGalaxy, segMax = gridAlg.filterSegmentedMask(segmentation,e)
        mat = gridAlg.applyMask(notMasked, mask)
        matSexSeg = gridAlg.applyMask(notMasked, segmentationMask)
        return data, newMat, e, gradModF, noBCG, segmentationMask, mask, mat, matSexSeg, bcg, width, height

    #@profile
    def run(self,char *fpath,char * image,mask_File='',char *saveResult="",float petroRad=0.0,float petroMag=0.0,float rowc=0.0,float colc=0.0,float ra=-1.0,float dec=-1.0,calibratedData=False,char* xtraID='', saveFig=True, clear=False,clip=False):
        cdef:
                #image proprieties 
                int width, height, it#, r

                #matrices
                float[:,:] notMasked, mask, segmentation, scaleMatrix, segmentationMask, removedGalaxies, zeros
                float[:,:] matInverse, matSmoth, matSmoothness, transformed, transformedEll, transformed2, transformedEll2

                #indexes:
                float a2, a3, s2, s3,h, sp2, sp3, ga, c1, c2
                float sa2, sa3, ss2, ss3,sh, ssp2, ssp3
                float oa2, oa3, os2, os3,oh, osp2, osp3, yToFind, petroRadByEta

                float[:] radius, acc, etas

        # processing objId = xtraID
        results = [xtraID]
        labels = ["Id"]

        printIfVerbose("Running File: "+fpath+image)
        maskFile = mask_File
        fileName = image
        path = fpath

        defaultThreshold = 1.8

        t0 = time.time()*1000.0

        if (clip == True):
            path = 'Field/'

        # segmentation, mask and cleaning stamp        
        if (ra != -1.0) and (dec != -1.0) and (clip==True):
            fileName = self._runMaskMaker(path,image,xtraID,ra,dec,petroRad,petroMag,rowc,colc)
            path = 'cutted/'

        data, newMat, e, gradModF, noBCG, segmentationMask, mask, mat, matSexSeg, bcg, width, height = self.maskAndClean(path,fileName,image,xtraID,maskFile,ra,dec,petroRad,petroMag,rowc,colc,calibratedData,saveFig,clear,defaultThreshold)
        
        t1 = time.time()*1000.0

        labels.append('segMaskAndCleanTime')
        results.append(t1-t0)


        # errorVar flag 1: too many objects in 2*petroRad
        numberOfObjects = len(data)
        galX, galY = data[0][8], data[0][9]

        numberOfObjectsIn2Rp = 0
        for obj in range(1, numberOfObjects):
            # get cartesian dist
            objDist = sqrt(pow(galX-data[obj][8],2.0)+pow(galY-data[obj][9],2.0))
            if (objDist < 2 * petroRad):
                numberOfObjectsIn2Rp = numberOfObjectsIn2Rp + 1
        
        if(self.isSetIndex("nO")):
            printIfVerbose("Saving number of objects in 2Rp to output file")
            labels.append("num_objects_in_2Rp")
            results.append(numberOfObjectsIn2Rp)

        if (numberOfObjectsIn2Rp > 5):
            self.errorVar = 1

        # errorVar flag 2: If petroRad not found
        printIfVerbose("Calculating petroRad")
        
        t0 = time.time()*1000.0
        
        try:
            radius, acc, etas = gridAlg.etaFunction(newMat, e, self.Concentration_Density)
            petroRadByEta = gridAlg.getPetroRad(radius, etas, e.posx, len(newMat[0]))

            # if petroRad not found, error = 2 (not possible to calc concetration)
            if (petroRadByEta == 0 or numpy.isnan(petroRadByEta)):
                self.errorVar = 2

        except Exception:
            self.errorVar = 2

        t1 = time.time()*1000.0

        labels.append('RpTime')
        results.append(t1-t0)

        if(self.isSetIndex("Rp")):
            printIfVerbose("Saving Rp to output file")
            labels.append("Rp")
            results.append(petroRadByEta)
        

        ### STARTING NON-PARAMETRIC INDEXES ###        
        printIfVerbose("Starting Indexes")

        if (self.isSetIndex("S") or self.isSetIndex("S2") or self.isSetIndex("S3")):

            printIfVerbose("Smoothing image")

            t0 = time.time()*1000.0

            matSmoothness = gridAlg.filterButterworth2D(gradModF,self.smoothingDegradation, self.butterOrder,e)
            diffMat = numpy.array([[(matSmoothness[i][j]-gradModF[i][j]) for j in range(len(newMat[i]))] for i in range(len(newMat))], dtype=numpy.float32)
            maximo = numpy.max(diffMat)
            diffSmoothed = numpy.array([[(matSmoothness[i][j]-noBCG[i][j])/maximo for j in range(len(newMat[i]))] for i in range(len(newMat))], dtype=numpy.float32)
            rotated = indexes.rotateImage(newMat,180)
            diffRotated = numpy.array([[ (newMat[i][j]-rotated[i][j]) for j in range(len(newMat[i]))] for i in range(len(newMat))], dtype=numpy.float32)
        
            if(saveFig):
                galaxyIO.plotFITS(matSmoothness,"imgs/smoothed.fits")
                galaxyIO.plotFITS(diffSmoothed,"imgs/smoothDiff.fits")
                galaxyIO.plotFITS(diffRotated,"imgs/rotateDiff.fits")
        
            t1 = time.time()*1000.0

            labels.append('smoothTime')
            results.append(t1-t0)
	        
            printIfVerbose("Smoothed")

        if(self.onlySegmentation == 0):
            printIfVerbose("Without masking")

            if(self.isSetIndex("A") or self.isSetIndex("A2")):

                t0 = time.time()*1000.0                

                a2, matInverse, a2Corr = indexes.asymmetryFunction(gradModF, mask, stats.pearsonr, e)
                
                if(saveFig):
                    numpy.savetxt("imgs/asymmetry.txt",numpy.array(a2Corr).T)

                t1 = time.time()*1000.0
               
                labels.append('a2Time')
                results.append(t1-t0)

                labels.append("A2")
                results.append(a2)

            if(self.isSetIndex("A") or self.isSetIndex("A3")):

                t0 = time.time()*1000.0                

                a3, matInverse, a3Corr = indexes.asymmetryFunction(gradModF, mask, stats.spearmanr, e)
                
                if(saveFig):
                    numpy.savetxt("imgs/asymmetry.txt",numpy.array(a3Corr).T)

                t1 = time.time()*1000.0

                labels.append('A3Time')
                results.append(t1-t0)

                labels.append("A3")
                results.append(a3)
                

            if(self.isSetIndex("S") or self.isSetIndex("S2")):

                t0 = time.time()*1000.0

                s2, s22RpCorr = indexes.smoothnessFunction(gradModF,matSmoothness, mask, bcg, stats.pearsonr,e)

                t1 = time.time()*1000.0

                labels.append('S2Time')
                results.append(t1-t0)
            
                if (numpy.isnan(s2)):
                    self.errorVar = 7
                if(saveFig):
                    numpy.savetxt("imgs/smoothness.txt",numpy.array(s22RpCorr).T)

                labels.append("S2")
                results.append(s2)

    
            if(self.isSetIndex("S") or self.isSetIndex("S3")):
                
                t0 = time.time()*1000.0                

                s3, s32RpCorr = indexes.smoothnessFunction(gradModF,matSmoothness, mask, bcg, stats.spearmanr,e)

                t1 = time.time()*1000.0

                labels.append('S3Time')
                results.append(t1-t0)
            
                if (numpy.isnan(s3)):
                    self.errorVar = 7
                if(saveFig):
                    numpy.savetxt("imgs/smoothness.txt",numpy.array(s32RpCorr).T)

                labels.append("S3")
                results.append(s3)


            if(self.isSetIndex("H")):

                t0 = time.time()*1000.0

                h = indexes.entropyFunction(mat, self.Entropy_KFolds)

                t1 = time.time()*1000.0
                
                labels.append('HTime')
                results.append(t1-t0)

                if (h>1.0) or (h < 0.0):
                    self.errorVar = 6
                    raise Exception("Unexpected Entropy value:"+str(h))
                
                labels.append("H")
                results.append(h)


            if(self.isSetIndex("Ga")):
                printIfVerbose("Starting GPA (G2)")

                t0 = time.time()*1000.0

                gpaObject = GPA(noBCG)
                gpaObject.setPosition(e.posx,e.posy)
                gpaObject.r = numpy.min(numpy.array([e.maxRad, float(width)/2.0,float(height)/2.0]))
                ga = gpaObject.evaluate(mtol=self.Ga_Tolerance, ftol=self.Ga_Angular_Tolerance, ptol=self.Ga_Position_Tolerance,mask=mask,gversion=2)
                
                t1 = time.time()*1000.0

                labels.append('GaTime')
                results.append(t1-t0)

                if (ga > 2.0) or (ga < 0.0):
                    self.errorVar = 5
                    raise Exception('Unexpected Ga value:'+str(ga))
                
                labels.append("Ga")
                results.append(ga)


            if(self.isSetIndex("G1")):
                printIfVerbose("Starting GPA (G1)")
                
                t0 = time.time()*1000.0

                gpaObject = GPA(noBCG)
                gpaObject.setPosition(e.posx,e.posy)
                gpaObject.r = numpy.min(numpy.array([e.maxRad, float(width)/2.0,float(height)/2.0]))
                g1 = gpaObject.evaluate(mtol=self.Ga_Tolerance, ftol=self.Ga_Angular_Tolerance, ptol=self.Ga_Position_Tolerance,mask=mask,gversion=1)
                
                t1 = time.time()*1000.0

                labels.append('G1Time')
                results.append(t1-t0)

                if (g1 > 2.0) or (g1 < 0.0):
                    self.errorVar = 4
                    raise Exception('Unexpected Ga value:'+str(g1))
                
                labels.append("G1")
                results.append(g1)


        if(self.isSetIndex("C") or self.isSetIndex("C1") or self.isSetIndex("C2") or self.isSetIndex("CN")):
            printIfVerbose("Starting Concentration (C)")
        
        if(self.isSetIndex("C") or self.isSetIndex("C1")):

            labels.append('C1Time')

            if(self.errorVar != 2):
                t0 = time.time()*1000.0
            	
                c1 = indexes.getConcentration(acc, petroRadByEta, 0.8, 0.2)
                
                if (numpy.isnan(c1)):
                    self.errorVar = 3
                
                t1 = time.time()*1000.0
                results.append(t1-t0)
            
            else:
                c1 = numpy.nan
                results.append(numpy.nan)

            labels.append("C1")
            results.append(c1)


        if(self.isSetIndex("C") or self.isSetIndex("C2")):

            labels.append('C2Time')

            if(self.errorVar != 2):
                t0 = time.time()*1000.0
            
                c2 = indexes.getConcentration(acc, petroRadByEta, 0.9, 0.5)
                
                if (numpy.isnan(c2)):
                    self.errorVar = 3
                
                t1 = time.time()*1000.0
                results.append(t1-t0)
            
            else:
                c2 = numpy.nan
                results.append(numpy.nan)

            labels.append("C2")
            results.append(c2)

        
        if (self.isSetIndex("C") or self.isSetIndex("CN")) and (self.d1>0.0) and (self.d2>0.0) and (self.d1<1.0) and (self.d2<1.0):
            
            labels.append('CNTime')

            if(self.errorVar != 2):
                t0 = time.time()*1000.0
            
                cn = indexes.getConcentration(acc, petroRadByEta, self.d1, self.d2)
                
                if (numpy.isnan(cn)):
                    self.errorVar = 3
                
                t1 = time.time()*1000.0
                results.append(t1-t0)
            
            else:
                cn = numpy.nan
                results.append(numpy.nan)

            labels.append("CN")
            results.append(cn)


        printIfVerbose("Sextractor segmentation")
        # Com segmentacao do sextractor:
        if(self.isSetIndex("A") or self.isSetIndex("A2") or self.isSetIndex("A3")):

            printIfVerbose("Starting Asymmetry (A)")

            t0 = time.time()*1000.0

            if (clip == True):
                path = 'Field/'

            if (ra != -1.0) and (dec != -1.0) and (clip==True):
                fileName = self._runMaskMaker(path,image,xtraID,ra,dec,petroRad,petroMag,rowc,colc)
                path = 'cutted/'

            data, newMat, e, gradModF, noBCG, segmentationMask, mask, mat, matSexSeg, bcg, width, height = self.maskAndClean(path,fileName,image,xtraID,maskFile,ra,dec,petroRad,petroMag,rowc,colc,calibratedData,saveFig,clear,self.Asymmetry_sexThreshold)
            
            t1 = time.time()*1000.0

            labels.append('AsegMaskAndCleanTime')
            results.append(t1-t0)


        if(self.isSetIndex("A") or self.isSetIndex("A2")):

            t0 = time.time()*1000.0

            sa2, matInverse, a2SexCorr = indexes.asymmetryFunction(gradModF, segmentationMask, stats.pearsonr, e)
            if(saveFig):
                numpy.savetxt("imgs/sasymmetry.txt",numpy.array(a2SexCorr).T)

            t1 = time.time()*1000.0

            labels.append('sA2Time')
            results.append(t1-t0)
            
            labels.append("sA2")
            results.append(sa2)
            

        if(self.isSetIndex("A") or self.isSetIndex("A3")):

            t0 = time.time()*1000.0

            sa3, matInverse, a3SexCorr = indexes.asymmetryFunction(gradModF, segmentationMask,  stats.spearmanr, e)
            
            if(saveFig):
                numpy.savetxt("imgs/sasymmetry.txt",numpy.array(a3SexCorr).T)
            
            t1 = time.time()*1000.0

            labels.append('sA3Time')
            results.append(t1-t0)
            
            labels.append("sA3")
            results.append(sa3)
            

        if(self.isSetIndex("S") or self.isSetIndex("S2") or self.isSetIndex("S3")):

            printIfVerbose("Starting Smoothing Index (S)")
            
            t0 = time.time()*1000.0

            if (clip == True):
                path = 'Field/'

            if (ra != -1.0) and (dec != -1.0) and (clip==True):
                fileName = self._runMaskMaker(path,image,xtraID,ra,dec,petroRad,petroMag,rowc,colc)
                path = 'cutted/'

            data, newMat, e, gradModF, noBCG, segmentationMask, mask, mat, matSexSeg, bcg, width, height = self.maskAndClean(path,fileName,image,xtraID,maskFile,ra,dec,petroRad,petroMag,rowc,colc,calibratedData,saveFig,clear,self.Smoothness_sexThreshold)
            
            t1 = time.time()*1000.0

            labels.append('SsegMaskAndCleanTime')
            results.append(t1-t0)            
            

        if(self.isSetIndex("S") or self.isSetIndex("S2")):

            t0 = time.time()*1000.0
                
            ss2, s2SexCorr = indexes.smoothnessFunction(noBCG,matSmoothness, segmentationMask, bcg, stats.pearsonr,e)

            t1 = time.time()*1000.0

            labels.append('sS2Time')
            results.append(t1-t0)

            if (numpy.isnan(ss2)):
                self.errorVar = 1
            if(saveFig):
                numpy.savetxt("imgs/ssmoothness.txt",numpy.array(s2SexCorr).T)
            
            labels.append("sS2")
            results.append(ss2)


        if(self.isSetIndex("S") or self.isSetIndex("S3")):
            
            t0 = time.time()*1000.0

            ss3, s3SexCorr = indexes.smoothnessFunction(noBCG,matSmoothness, segmentationMask, bcg, stats.spearmanr,e)
            
            t1 = time.time()*1000.0

            labels.append('sS3Time')
            results.append(t1-t0)

            if (numpy.isnan(ss3)):
                self.errorVar = 1
            if(saveFig):
                numpy.savetxt("imgs/ssmoothness.txt",numpy.array(s3SexCorr).T)
	        
            labels.append("sS3")
            results.append(ss3)


        if(self.isSetIndex("H")):
            
            printIfVerbose("Starting Entropy (H)")

            t0 = time.time()*1000.0

            if (clip == True):
                path = 'Field/'

            if (ra != -1.0) and (dec != -1.0) and (clip==True):
                fileName = self._runMaskMaker(path,image,xtraID,ra,dec,petroRad,petroMag,rowc,colc)
                path = 'cutted/'

            data, newMat, e, gradModF, noBCG, segmentationMask, mask, mat, matSexSeg, bcg, width, height = self.maskAndClean(path,fileName,image,xtraID,maskFile,ra,dec,petroRad,petroMag,rowc,colc,calibratedData,saveFig,clear,self.Entropy_sexThreshold)
            
            t1 = time.time()*1000.0
            
            labels.append('HsegMaskAndCleanTime')
            results.append(t1-t0)

            t0 = time.time()*1000.0

            sh = indexes.entropyFunction(matSexSeg,self.Entropy_KFolds)

            t1 = time.time()*1000.0

            labels.append('sHTime')
            results.append(t1-t0)

            if (sh>1.0) or (sh < 0.0):
                raise Exception("Unexpected Sextractor Segmentation Entropy value:"+str(sh))
            
            labels.append("sH")
            results.append(sh)


        if(self.isSetIndex("Ga") or self.isSetIndex("G1")):
            
            t0 = time.time()*1000.0

            if (clip == True):
                path = 'Field/'

            if (ra != -1.0) and (dec != -1.0) and (clip==True):
                fileName = self._runMaskMaker(path,image,xtraID,ra,dec,petroRad,petroMag,rowc,colc)
                path = 'cutted/'

            data, newMat, e, gradModF, noBCG, segmentationMask, mask, mat, matSexSeg, bcg, width, height = self.maskAndClean(path,fileName,image,xtraID,maskFile,ra,dec,petroRad,petroMag,rowc,colc,calibratedData,saveFig,clear,self.Ga_sexThreshold)
            
            t1 = time.time()*1000.0

            labels.append('GsegMaskAndCleanTime')
            results.append(t1-t0)            
            

        if(self.isSetIndex("Ga")):
            
            printIfVerbose("Starting Ga (sG2)")
            
            t0 = time.time()*1000.0

            gpaObject = GPA(newMat)
            gpaObject.setPosition(e.posx,e.posy)
            gpaObject.r = numpy.min(numpy.array([e.maxRad, float(width)/2.0,float(height)/2.0]))
            ga = gpaObject.evaluate(mtol=self.Ga_Tolerance, ftol=self.Ga_Angular_Tolerance, ptol=self.Ga_Position_Tolerance, mask=segmentationMask, gversion=2)

            t1 = time.time()*1000.0

            if (ga>2.0) or (ga < 0.0):
                raise Exception('Unexpected Ga value:'+str(ga))
            

            labels.append('sGaTime')
            results.append(t1-t0)
            
            labels.append("sGa")
            results.append(ga)


        if(self.isSetIndex("G1")):

            printIfVerbose("Starting GPA (sG1)")

            t0 = time.time()*1000.0

            gpaObject = GPA(newMat)
            gpaObject.setPosition(e.posx,e.posy)
            gpaObject.r = numpy.min(numpy.array([e.maxRad, float(width)/2.0,float(height)/2.0]))
            g1 = gpaObject.evaluate(mtol=self.Ga_Tolerance, ftol=self.Ga_Angular_Tolerance, ptol=self.Ga_Position_Tolerance,mask=segmentationMask,gversion=1)
            
            t1 = time.time()*1000.0

            if (g1>2.0) or (g1 < 0.0):
                raise Exception('Unexpected Ga value:'+str(ga))
                        
            labels.append('sG1Time')
            results.append(t1-t0)
            
            labels.append("sG1")
            results.append(g1)
        
        labels.append("Error")
        results.append(self.errorVar)
        
        if (len(saveResult) != 0):
            outputVector= [results]
            
            if not(os.path.isfile(saveResult)):
                numpy.savetxt(saveResult, numpy.array([labels]), delimiter=',', fmt="%s")
            
            with open(saveResult,'a') as f_handle:
                numpy.savetxt(f_handle, numpy.array(outputVector), delimiter=',', fmt="%s")
            
            printIfVerbose("File "+fileName+" done.")
        
        else:
            outputVector= [results]
            numpy.savetxt("Result.csv", numpy.array([labels]), delimiter=',', fmt="%s")
            
            with open("Result.csv",'a') as f_handle:
                numpy.savetxt(f_handle, numpy.array(outputVector), delimiter=',', fmt="%s")
            
            printIfVerbose("Results saved in Result.csv")
