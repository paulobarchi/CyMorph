import numpy
import sys
import os
import ConfigParser
import CyMorph

def printError():
    print('Example of configuration File:')
    print('')
    print('\t[File_Configuration]')
    print('\tPath: tests/input/elliptical_Simulated/')
    print('\tFilename: img400_n6_re10_band_rNoise_sim_0.fits')
    print('\tSky: 1000.0 ; Optional | if not set will be considered sextractor measured sky')
    print('\t; Ra_Dec: 0, 0 ; Optional |  Object Position, if not set will be considered the image center') 
    print('')
    print('\t[Output_Configuration]')
    print('\tVerbose:False')
    print('\tSaveFigure: False')
    print('')
    print('\t[Mask_Configuration]')
    print('\t;Mask: file.fits ; Optional | this will set a mask from a file if not used our mask maker')
    print('\tRun_MaskMaker: False ; if this flag is set, it will run and use our mask (this flag has priority)')
    print('')
    print('\t[Indexes_Configuration]')
    print('\tEntropy_Bins: 159 ; Number of bins used in entropy')
    print('\tGa_Tolerance: 0.03 ; Ga module tolerance (between 0.0 and 1.0)')
    print('\tGa_Angular_Tolerance: 0.03 ; Ga angular tolerance in radians (between 0.0 and PI)')
    print('\tGa_Position_Tolerance: 1.0 ; Ga positioning tolerance in pixel distance')
    print('\tSpirality_Interpolation: 200, 400 ; y and x Coordinate Systems Dimension')
    print('\tConcentration_Density: 100 ;  Sampling density in each boundary distance pixel ')
    print('')
    exit()

def runParallel(configFileName,path,fileName,ra,dec,xtraID,saveResult,petroRad,rowc,colc,PetroMag):
    configFile = ConfigParser.ConfigParser()
    configFile.read(configFileName)
    m = CyMorph.CyMorph()
    
    try:
        saveFig = configFile.getboolean('Output_Configuration','SaveFigure')
        listOfIndexes = configFile.get('File_Configuration','Indexes').replace(',',' ').split(' ')
        clip=True
        for c in listOfIndexes:
            if(len(c)<1):
                continue
            m.setIndexV(c)
        if(m.isSetIndex("H")):
            m.setEntropy_KFolds(int(configFile.get('Indexes_Configuration','Entropy_Bins')))
            m.setEntropy_sexThreshold(float(configFile.get('Indexes_Configuration', 'Entropy_sexThreshold')))


        if(m.isSetIndex("Ga")):
            m.setGa_Tolerance(float(configFile.get('Indexes_Configuration','Ga_Tolerance')))
            m.setGa_Angular_Tolerance(float(configFile.get('Indexes_Configuration','Ga_Angular_Tolerance')))
            m.setGa_Position_Tolerance(float(configFile.get('Indexes_Configuration','Ga_Position_Tolerance')))
            m.setGa_sexThreshold(float(configFile.get('Indexes_Configuration', 'Ga_sexThreshold')))

        if(m.isSetIndex("C") or m.isSetIndex("C1")or m.isSetIndex("C2") or m.isSetIndex("CN")):
            m.setConcentration_Density(int(configFile.get('Indexes_Configuration','Concentration_Density')))

        if(m.isSetIndex("S") or m.isSetIndex("S2") or m.isSetIndex("S3")):
            m.setSmoothnessOrder(float(configFile.get('Indexes_Configuration','butterworth_order')))
            m.setSmoothnessDegree(float(configFile.get('Indexes_Configuration','smooth_degree')))
            m.setSmoothness_sexThreshold(float(configFile.get('Indexes_Configuration', 'Smoothness_sexThreshold')))

        if(m.isSetIndex("A") or m.isSetIndex("A2") or m.isSetIndex("A3")):
            m.setAsymmetry_sexThreshold(float(configFile.get('Indexes_Configuration', 'Asymmetry_sexThreshold')))


        m.setOnlySegmentation(0)
        if configFile.getboolean('Output_Configuration','Verbose') == True :
            CyMorph.verbose = 1
        else:
            CyMorph.verbose = 0           
    except Exception as inst:
        print('Wrong Configuration File',inst.args[0])
    maskFile = ''
    try:
        m.setSky(float(configFile.get('Indexes_Configuration','Sky')))
    except ConfigParser.NoOptionError:
        m.setSky(-1.0)
    try:
        d1, d2 = configFile.get('Indexes_Configuration','Concentration_Distances').split(',')
        d1, d2 = float(d1), float(d2)
        m.setConcentrationDists(d1, d2)
    except:
        print("No concentration distance specified")
    try:
        ssize = float(configFile.get('File_Configuration','stamp_size'))
        m.setStampSize(ssize)
    except Exception as ex:
        print("No stamp size defined, using default value", ex)
    m.run(path,fileName,maskFile,saveResult,petroRad=petroRad,rowc=rowc,colc=colc,ra=ra,dec=dec, saveFig=saveFig, clear=True, clip=clip, xtraID=xtraID)

def runIt(fileName): # ainda precisa incorpara petroRad rowc,colc nas chamadas a m.run
    configFile = ConfigParser.ConfigParser()
    configFile.read(fileName)
    m = CyMorph.CyMorph()
    
    try:
        path = configFile.get('File_Configuration','Path')
        fileName = configFile.get('File_Configuration','Filename')
        saveFig = configFile.getboolean('Output_Configuration','SaveFigure')
        clip = configFile.getboolean('File_Configuration','Clip')
        listOfIndexes = configFile.get('File_Configuration','Indexes').replace(',',' ').split(' ')
        for c in listOfIndexes:
            if(len(c)<1):
                continue
            m.setIndexV(c)
        
        if(m.isSetIndex("H")):
            m.setEntropy_KFolds(int(configFile.get('Indexes_Configuration','Entropy_Bins')))
        if(m.isSetIndex("Ga") or m.isSetIndex("G1")):
            m.setGa_Tolerance(float(configFile.get('Indexes_Configuration','Ga_Tolerance')))
            m.setGa_Angular_Tolerance(float(configFile.get('Indexes_Configuration','Ga_Angular_Tolerance')))
            m.setGa_Position_Tolerance(float(configFile.get('Indexes_Configuration','Ga_Position_Tolerance')))
        if(m.isSetIndex("C")or m.isSetIndex("C1")or m.isSetIndex("C2") or m.isSetIndex("CN")):
            m.setConcentration_Density(int(configFile.get('Indexes_Configuration','Concentration_Density')))
        if(m.isSetIndex("S") or m.isSetIndex("S2") or m.isSetIndex("S3")):
            m.setSmoothnessOrder(float(configFile.get('Indexes_Configuration','butterworth_order')))
            m.setSmoothnessDegree(float(configFile.get('Indexes_Configuration','smooth_degree')))
        saveResult = ''
        if configFile.getboolean('File_Configuration','Only_segmentation') == True :
            m.setOnlySegmentation(1)
        else:
            m.setOnlySegmentation(0)
        if(configFile.getboolean('Output_Configuration','Verbose')):
            CyMorph.verbose = 1
        else:
            CyMorph.verbose = 0           
    except Exception as exc:
        print('Wrong Configuration File',exc)
        printError()
    try:
        maskFile = configFile.get('Mask_Configuration','Mask')
    except:
        maskFile = ''
    try:
        ra, dec = configFile.get('File_Configuration','Ra_Dec').split(',')
        ra,dec=float(ra),float(dec)
        radec = True
    except ConfigParser.NoOptionError:
        ra,dec=-1.0,-1.0
        radec = False
    try:
        print(configFile.get('File_Configuration','Sky'))
        m.setSky(float(configFile.get('File_Configuration','Sky')))
    except ConfigParser.NoOptionError:
        m.setSky(-1.0)
    try:
        xtraID = configFile.get('File_Configuration','Id')
    except:
        xtraID = 'x'
    try:
        d1, d2 = configFile.get('Indexes_Configuration','Concentration_Distances').split(',')
        d1, d2 = float(d1), float(d2)
        m.setConcentrationDists(d1, d2)
    except:
        print("No concentration distance specified")
    try:
        ssize = float(configFile.get('File_Configuration','stamp_size'))
        m.setstampSize(ssize)
    except:
        print("No stamp size defined, using default value")
    if (radec):
        m.run(path,fileName,maskFile,saveResult, saveFig=saveFig, ra=ra, dec=dec, clear=True, clip=clip, xtraID=xtraID)
    else:
        m.run(path,fileName,maskFile,saveResult, saveFig=saveFig, clear=True, clip=clip, xtraID=xtraID)
        

    ############################################
    #     Main temporario:
if __name__ == "__main__":
# python morfometryka2.py config.ini
    if len(sys.argv)!=2:
        print("Configuration File Needed!")
        printError()
    # print(sys.argv[1])
    runIt(sys.argv[1])
   
