import sys
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import shutil

from numpy.random import RandomState
from scipy.io import loadmat, savemat
from configparser import *
from datetime import datetime
from data_preproc import DataPreproc

class testDataset:
    def __init__(self, refDir, expConfigFilename):
        # directory containing all the configuration files for the experiment
        self.refDir = refDir
        # file with configuration details for the launched experiment
        self.expConfigFilename = refDir + '/' + expConfigFilename
        # data pre-processing object
        self.dpp = DataPreproc()
        # loading details from configuration files
        self.loadExpConfig()
    
    def loadExpConfig(self):
        """
        Function loading the configuration details for the experiment & data pre-processing flags
        """
        config = ConfigParser()
        config.read(self.expConfigFilename)

        self.npRandSeed = config.getint('PARAMETERS', 'npRandSeed')
        self.npRandState = config.getint('PARAMETERS', 'npRandState')

        self.dataDir = config.get('EXP_DETAILS', 'dsetDir')
        self.expsDir = config.get('EXP_DETAILS', 'expsDir')
        self.expName = config.get('EXP_DETAILS', 'expID')
        self.dSetName = config.get('EXP_DETAILS', 'dSetName')
        self.statesFile = config.get('EXP_DETAILS', 'statesFile')
        self.modelDirName = config.get('EXP_DETAILS', 'modelDirName')
        self.modelName = config.get('EXP_DETAILS', 'modelName')
        self.logFlag = config.getboolean('EXP_DETAILS', 'logFlag')
        self.meanSubtructionFlag = config.getboolean('EXP_DETAILS', 'meanSubtructionFlag')
        self.scaleFlag = config.getboolean('EXP_DETAILS', 'scaleFlag')
        self.scaling = config.get('EXP_DETAILS', 'scaling')
        self.doPCA = config.getboolean('EXP_DETAILS', 'doPCA')
        self.whitenFlag = config.getboolean('EXP_DETAILS', 'whitenFlag')
        self.rescaleFlag = config.getboolean('EXP_DETAILS', 'rescaleFlag')
        self.rescaling = config.get('EXP_DETAILS', 'rescaling')

        self.dataFilename = self.dataDir + self.dSetName
        self.saveDir = self.expsDir + self.expName

        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)
        # shutil.copy2(self.expConfigFilename, self.saveDir)
        # shutil.copy2(self.modelConfigFilename, self.saveDir)

    def loadData(self):
        """
        Function loading the data
        """
        if not os.path.exists(self.saveDir + '/dataDetails/'):
            os.makedirs(self.saveDir + '/dataDetails/')

        # load data file:
        if self.dataFilename.split('.')[1] == 'npz':
            dLoad = np.load(self.dataFilename)
        elif self.dataFilename.split('.') == 'mat':
            dLoad = loadmat(self.dataFilename)
        else:
            print("error! Unrecognized data file")
        
        self.d = dLoad['d']
        self.obsKeys = dLoad['epochsLinked']
        self.epochTime = dLoad['epochTime']
        
        
        """
        If you want to keep only EEG features, uncomment next line.
		"""

        # self.d = self.d[:, :self.d.shape[1]-1]

        self.d = np.array(self.d, dtype=np.float32)
        self.obsKeys = np.array(self.obsKeys, dtype=np.float32)
        print(("initial size: ", self.d.shape))
        # print("FrameIDs : ", self.obsKeys, "of shape : ", self.obsKeys.shape)

        with open(self.saveDir + '/dataDetails/' + 'initialData.txt', 'w') as f:
            f.write("\n Modeling: %s " % self.dataFilename)
            f.write("\n Dataset size: %s " % str(self.d.shape))
            f.write("\n Dataset type: %s " % str(self.d.dtype))
            f.write("\n \n d_min: %s " % str(np.min(self.d, axis=0)))
            f.write("\n \n d_max: %s " % str(np.max(self.d, axis=0)))
            f.write("\n \n d_mean: %s " % str(np.mean(self.d, axis=0)))
            f.write("\n \n d_std: %s " % str(np.std(self.d, axis=0)))
            f.close()

    def prepareFile(self):
        self.d, self.obsKeys, self.dMean, self.dStd, self.dMinRow, self.dMaxRow, self.dMin, self.dMax = \
        self.dpp.preprocAndScaleData(self.d, self.obsKeys, self.logFlag, self.meanSubtructionFlag, self.scaleFlag,
                                    self.scaling, self.doPCA, self.whitenFlag, self.rescaleFlag, self.rescaling,
                                    'minmaxFileInit', self.saveDir)
        
        d = self.d.astype(np.float32)
        visData = self.d
        np.savez(f'{self.expsDir}{self.expName}/visData.npz', data=d, obsKeys=self.obsKeys, epochTime=self.epochTime)

        with open(f'{self.expsDir}{self.expName}/visData.txt', 'w') as f:
            f.write("\n visData size: %s " % str(visData.shape))
            f.write("\n visData type: %s " % str(visData.dtype))
            f.write("\n \n visData Range: %s " % str(np.max(visData, axis=0) - np.min(visData, axis=0)))
            f.write("\n \n visData min: %s " % str(np.min(visData, axis=0)))
            f.write("\n \n visData max: %s " % str(np.max(visData, axis=0)))
            f.write("\n \n visData mean: %s " % str(np.mean(visData, axis=0)))
            f.write("\n \n visData std: %s " % str(np.std(visData, axis=0)))
            f.close()

        