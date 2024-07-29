"""
@:description
Class implementing the pre-processing steps for the analysis of electrophysiological data.
In our experiments, data were pre-processed before modeling them with the mean-covariance RBM.
"""

import numpy as np
from sklearn.decomposition import PCA
import pickle
import pickle
import shutil
import os
from scipy.io import loadmat, savemat
from io import open

class DataPreproc:
    '''
    This class is aimed at performing pre-processing of electrophysiological data before
    feeding them to the models used for analysis.
    '''

    def __init__(self):
        self.name = 'dpp'

    def trimForGPU(self, d, obsKeys, epochTime, batch_size):
        '''
        This function extracts a subset of the rows of the data matrix. The objective
        is obtaining a data matrix which can be divided in batches of the selected size
        with no row left out

        inputs:
            d           data matrix
            obsKeys		matrix with epochsIDs and the corresponding label given by manual scoring
            epochTime	matrix with epochsIDs and the corresponding day-time
            batch_size  size of the batch using while training the model

        output:
            subset of each data matrix
        '''

        totnumcases = d.shape[0]

        return d[0:int(np.floor(totnumcases / batch_size) * batch_size), :].copy(), obsKeys[0:int(
            np.floor(totnumcases / batch_size) * batch_size)].copy(), \
            epochTime[0:int(np.floor(totnumcases / batch_size) * batch_size), :].copy()

    def preprocAndScaleData(self, d, obsKeys, logFlag, meanSubtractionFlag, scalingFlag, scaling, pcaFlag, whitenFlag,
                            rescalingFlag, rescaling, minmaxFile, saveDir):
        '''
        This function scales the data according to what is written in the configuration file

        inputs:
            d               	data matrix
            obsKeys         	matrix with epochsIDs and the corresponding label given by manual scoring
            logFlag         	flag indicating whether to take the log of the data matrix
            meanSubtractionFlag flag indicating whether to subtract the mean from each feature in the data matrix
            scalingFlag			flag indicating whether to scale the data matrix
            scaling         	string indicating what kind of data scaling has to be applied
            pcaFlag				flag indicating whether to apply pca to the data
            whitenFlag			flag indicating whether to apply whitening to the data
            rescalingFlag		flag indicating whether to re-scale the data matrix after pre-processing
            rescaling			string indicating what kind of data scaling has to be applied
            minmaxFile      	string indicating the name of the file storing important aspects of the data matrix
            saveDir				experiment directory for storing stuff

        outputs:
            subset of data matrix and of the matrix with epochsIDs
        '''
        with open(saveDir + '/dataDetails/' + 'preprocDetails.txt', 'a') as f:
            f.write("Pre-processing steps: ")

        # --- Taking the log of the features which have only positive values:
        if logFlag:
            print("Taking the natural logarithm...")
            # d = np.log(d)
            for feat in range(d.shape[1]):
                if d[:, feat].min() >= 0:
                    print(("Taking the log of feature: ", feat))
                    d[:, feat] = np.log(d[:, feat] + np.spacing(1))

            d = np.array(d, dtype=np.float32)

            with open(saveDir + '/dataDetails/' + 'preprocDetails.txt', 'a') as f:
                f.write("\n Taking the natural logarithm of non-negative features.")

            with open(saveDir + '/dataDetails/' + 'logData.txt', 'w') as f:
                f.write("\n Dataset size: %s " % str(d.shape))
                f.write("\n Dataset type: %s " % str(d.dtype))
                f.write("\n \n d_min: %s " % str(np.min(d, axis=0)))
                f.write("\n \n d_max: %s " % str(np.max(d, axis=0)))
                f.write("\n \n d_mean: %s " % str(np.mean(d, axis=0)))
                f.write("\n \n d_std: %s " % str(np.std(d, axis=0)))
                f.close()

        # Compute mean, std, min, max
        dMean = np.mean(d, axis=0)
        dStd = np.std(d, axis=0)

        dMin = np.min(d)
        dMax = np.max(d)

        dMinRow = np.min(d, axis=0)
        dMaxRow = np.max(d, axis=0)

        np.savez_compressed(saveDir + '/dataDetails/' + minmaxFile, dMin=dMin, dMax=dMax, dMinRow=dMinRow,
                            dMaxRow=dMaxRow, dMean=dMean, dStd=dStd)

        # from operator import sub
        # r = map(sub, dMaxRow, dMinRow)

        # --- Data Scaling:
        # different type of scalings can be allied

        if meanSubtractionFlag:
            print("Subtracting each feature's mean..")
            d = d - dMean

            with open(saveDir + '/dataDetails/' + 'preprocDetails.txt', 'a') as f:
                f.write("\n Subtracting each feature's mean...")

            with open(saveDir + '/dataDetails/' + 'meanSubtraction.txt', 'w') as f:
                f.write("\n Dataset size: %s " % str(d.shape))
                f.write("\n Dataset type: %s " % str(d.dtype))
                f.write("\n \n Data Range: %s " % str(np.max(d, axis=0) - np.min(d, axis=0)))
                f.write("\n \n Data min: %s " % str(np.min(d, axis=0)))
                f.write("\n \n Data max: %s " % str(np.max(d, axis=0)))
                f.write("\n \n Data mean: %s " % str(np.mean(d, axis=0)))
                f.write("\n \n Data std: %s " % str(np.std(d, axis=0)))
                f.close()

            """
            Re-compute min, max, mean, std
            """
            dMean = np.mean(d, axis=0)
            dStd = np.std(d, axis=0)

            dMin = np.min(d)
            dMax = np.max(d)

            dMinRow = np.min(d, axis=0)
            dMaxRow = np.max(d, axis=0)

        if scalingFlag:
            print("Scaling...")
            if 'global' in scaling:
                # centering and scaling the data matrix
                d = 10. * ((d - dMin) / (dMax - dMin) - 0.5)
            elif 'single' in scaling:
                # centering and scaling each column of the data matrix independently
                d = 10. * ((d - dMinRow) / (dMaxRow - dMinRow) - 0.5)
            elif 'baseZeroG' in scaling:
                # scaling the data matrix to be in the [0, 1] interval
                d = (d - dMin) / (dMax - dMin)
            elif 'baseZeroS' in scaling:
                # scaling each column of the data matrix independently to be in the [0, 10] interval
                d = 10. * (d - dMinRow) / (dMaxRow - dMinRow)
            elif 'baseZeroCol' in scaling:
                # centering and scaling each column of the data matrix independently to be in the [0, 1] interval
                d = (d - dMinRow) / (dMaxRow - dMinRow)
            elif 'stdz' in scaling:
                # standardise each column of the data matrix independently
                d = (d - dMean) / dStd
            elif 'minZero' in scaling:
                # translate each column of the data matrix to have 0 as the minimum value
                d = d - dMinRow
            elif 'NOscaling' in scaling:
                print("NO Scaling has been applied..")

            with open(saveDir + '/dataDetails/' + 'preprocDetails.txt', 'a') as f:
                f.write("\n Scaling...%s" % scaling)

            with open(saveDir + '/dataDetails/' + 'scaledData.txt', 'w') as f:
                f.write("\n Dataset size: %s " % str(d.shape))
                f.write("\n Dataset type: %s " % str(d.dtype))
                if logFlag:
                    f.write("\n \n Pre-processing: Log of non negative features & scaling %s" % scaling)
                else:
                    f.write("\n \n Pre-processing: scaling %s" % scaling)
                f.write("\n \n Data Range: %s " % str(np.max(d, axis=0) - np.min(d, axis=0)))
                f.write("\n \n Data min: %s " % str(np.min(d, axis=0)))
                f.write("\n \n Data max: %s " % str(np.max(d, axis=0)))
                f.write("\n \n Data mean: %s " % str(np.mean(d, axis=0)))
                f.write("\n \n Data std: %s " % str(np.std(d, axis=0)))
                f.close()

        # --- PCA part
        if pcaFlag:

            print("Applying PCA transform...")
            with open(saveDir + '/dataDetails/' + 'preprocDetails.txt', 'a') as f:
                f.write("\n Applying pca")
            d = self.pca(d, saveDir + '/dataDetails/', 'pca_obj.save', 'minmaxFilePCA', whitenFlag)

            if rescalingFlag:
                print("Rescaling after PCA...")
                d = self.rescalingFunct(d, rescaling)

                with open(saveDir + '/dataDetails/' + 'preprocDetails.txt', 'a') as f:
                    f.write("\n Scaling...%s" % rescaling)

            # np.savez_compressed('minmaxFilePCAresc', dMin = dMin, dMax = dMax, dMinRow = dMinRow, dMaxRow = dMaxRow, dMean = dMean, dStd = dStd)

        return d, obsKeys, dMean, dStd, dMinRow, dMaxRow, dMin, dMax

    def pca(self, d, refDir, pcaFile, minmaxFilePCA, whitenFlag=False):
        '''
        This function performs pca of the data. If the computation has already
        been implemented and saved, the object is loaded from disk.

        inputs:
            d               data matrix
            refDir          directory containing the configuration files of the experiment
            pcaFile         string indicating which file is associated to the serialised PCA object
            minmaxFilePCA   string indicating the name of the file storing important aspects of the transformed data matrix
            whitenFlag      boolean variables indicating whether to perform whitening of the data or not

        output:
            d               transformed data matrix

        '''
        if os.path.isfile(pcaFile):
            print()
            'Loading PCA file..'
            with open(pcaFile) as pcaPklFile:
                pca = pickle.load(pcaPklFile)
            shutil.copy2(pcaFile, refDir)
            with open(refDir + 'regardingPCAobject.txt', 'w') as f:
                f.write("Re-used pca object from previous experiment.")

            print("Subtracting each feature's mean..")
            dFeatMean = np.mean(d, axis=0)

            d = d - dFeatMean

            with open(refDir + 'preprocDetails.txt', 'a') as f:
                f.write("\n PCA Object loaded...so...Subtracting each feature's mean..")

            with open(refDir + 'meanSubtraction.txt', 'w') as f:
                f.write("\n Dataset size: %s " % str(d.shape))
                f.write("\n Dataset type: %s " % str(d.dtype))
                f.write("\n \n d_min: %s " % str(np.min(d, axis=0)))
                f.write("\n \n d_max: %s " % str(np.max(d, axis=0)))
                f.write("\n \n d_mean: %s " % str(np.mean(d, axis=0)))
                f.write("\n \n d_std: %s " % str(np.std(d, axis=0)))
                f.close()

            # d = np.dot(d, pca.components_.T)
            d = pca.transform(d)

        else:
            pca = PCA(d.shape[1], whiten=whitenFlag)
            pca.fit(d)

            pca_obj = open(refDir + pcaFile, 'wb')
            pickle.dump(pca.fit(d), pca_obj, protocol=pickle.HIGHEST_PROTOCOL)
            pca_obj.close()

            with open(refDir + 'regardingPCAobject.txt', 'w') as f:
                f.write("New pca rotation has been applied.")

            d = pca.transform(d)

        dMinPCA = np.min(d)
        dMaxPCA = np.max(d)
        dMeanPCA = np.mean(d, axis=0)
        dStdPCA = np.std(d, axis=0)
        dMinRowPCA = np.min(d, axis=0)
        dMaxRowPCA = np.max(d, axis=0)

        # after pca, the transformed data matrix is centered around 0 and scaled
        # d = 10.*((d - dMinPCA) / (dMaxPCA - dMinPCA) - 0.5)
        np.savez_compressed(refDir + minmaxFilePCA, dMinPCA=dMinPCA, dMaxPCA=dMaxPCA, dMinRowPCA=dMinRowPCA,
                            dMaxRowPCA=dMaxRowPCA, dMeanPCA=dMeanPCA, dStdPCA=dStdPCA)

        if whitenFlag:
            with open(refDir + 'whitenedData.txt', 'w') as f:
                f.write("\n Dataset size: %s " % str(d.shape))
                f.write("\n Dataset type: %s " % str(d.dtype))
                f.write("\n \n d_min: %s " % str(np.min(d, axis=0)))
                f.write("\n \n d_max: %s " % str(np.max(d, axis=0)))
                f.write("\n \n d_mean: %s " % str(np.mean(d, axis=0)))
                f.write("\n \n d_std: %s " % str(np.std(d, axis=0)))
                f.write("\n \n Variance ratio : %s " % str(pca.explained_variance_ratio_))
                f.write("\n \n Components with maximum variance : %s " % str(pca.components_))
                f.write("\n \n The estimated number of components : %s " % str(pca.n_components_))
                f.close()
        else:
            with open(refDir + 'pcaData.txt', 'w') as f:
                f.write("\n Dataset size: %s " % str(d.shape))
                f.write("\n Dataset type: %s " % str(d.dtype))
                f.write("\n \n d_min: %s " % str(np.min(d, axis=0)))
                f.write("\n \n d_max: %s " % str(np.max(d, axis=0)))
                f.write("\n \n d_mean: %s " % str(np.mean(d, axis=0)))
                f.write("\n \n d_std: %s " % str(np.std(d, axis=0)))
                f.write("\n \n Variance ratio : %s " % str(pca.explained_variance_ratio_))
                f.write("\n \n Components with maximum variance : %s " % str(pca.components_))
                f.write("\n \n The estimated number of components : %s " % str(pca.n_components_))
                f.close()

        return d

    def rescalingFunct(self, d, scaling):
        # Scaling_function :
        # print("d.shape: ", d.shape)
        dMaxRow = np.max(d, axis=0)
        dMinRow = np.min(d, axis=0)

        # Compute mean, std, min, max
        dMean = np.mean(d, axis=0)
        dStd = np.std(d, axis=0)

        dMin = np.min(d)
        dMax = np.max(d)

        # --- Data Scaling:
        # different type of scalings can be allied
        print("Re-Scaling...")

        if 'global' in scaling:
            # centering and scaling the data matrix between [-5, 5]
            d = 10. * ((d - dMin) / (dMax - dMin) - 0.5)
        elif 'single' in scaling:
            # centering and scaling each column of the data matrix independently between [-5, 5]
            d = 10. * ((d - dMinRow) / (dMaxRow - dMinRow) - 0.5)
        elif 'baseZeroG' in scaling:
            # scaling the data matrix to be in the [0, 1] interval
            d = (d - dMin) / (dMax - dMin)
        elif 'baseZeroS' in scaling:
            # scaling each column of the data matrix independently to be in the [0, 10] interval
            d = 10. * (d - dMinRow) / (dMaxRow - dMinRow)
        elif 'baseZeroCol' in scaling:
            # centering and scaling each column of the data matrix independently to be in the [0, 1] interval
            d = (d - dMinRow) / (dMaxRow - dMinRow)
        elif 'stdz' in scaling:
            # standardise each column of the data matrix independently
            d = (d - dMean) / dStd
        elif 'minZero' in scaling:
            # translate each column of the data matrix to have 0 as the minimum value
            d = d - dMinRow
        elif 'NOscaling' in scaling:
            print("NO Scaling has been applied..")

        return d
