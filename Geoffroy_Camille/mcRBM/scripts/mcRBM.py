"""
@:description
Class implementing the mean-covariance Restricted Boltzmann Machine (mcRBM) by Marc'Aurelio Ranzato. It is based on the
original code with minor modifications according to the needs of our experiments.

@:refer
"Modeling Pixel Means and Covariances Using Factorized Third-Order Boltzmann Machines"

@:original_code
http://www.cs.toronto.edu/~ranzato/publications/mcRBM/code/mcRBM_04May2010.zip
"""

import sys
import numpy as np
import os
import cudamat as cmt
import pickle
import matplotlib.pyplot as plt
import shutil

from numpy.random import RandomState
from scipy.io import loadmat, savemat
from configparser import *
from datetime import datetime
from data_preproc import DataPreproc



class mcRBM:
    def __init__(self, refDir, expConfigFilename, modelConfigFilename, gpuId=0):
        # directory containing all the configuration files for the experiment
        self.refDir = refDir
        # file with configuration details for the launched experiment
        self.expConfigFilename = refDir + '/' + expConfigFilename
        # file with configuration details for the model to be trained
        self.modelConfigFilename = refDir + '/' + modelConfigFilename
        # data pre-processing object
        self.dpp = DataPreproc()
        # loading details from configuration files
        self.loadExpConfig()
        self.loadModelConfig()
        # id of the GPU which will be used for computation
        self.gpuId = int(gpuId)

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

    def loadModelConfig(self):
        """
        Function loading the configuration details for the model to be trained
        """
        config = ConfigParser()
        config.read(self.modelConfigFilename)

        self.verbose = config.getint('VERBOSITY', 'verbose')

        self.num_epochs = config.getint('MAIN_PARAMETER_SETTING', 'num_epochs')
        self.batch_size = config.getint('MAIN_PARAMETER_SETTING', 'batch_size')
        self.startFH = config.getint('MAIN_PARAMETER_SETTING', 'startFH')
        self.startwd = config.getint('MAIN_PARAMETER_SETTING', 'startwd')
        self.doPCD = config.getint('MAIN_PARAMETER_SETTING', 'doPCD')

        # model parameters
        self.num_fac = config.getint('MODEL_PARAMETER_SETTING', 'num_fac')
        self.num_hid_cov = config.getint('MODEL_PARAMETER_SETTING', 'num_hid_cov')
        self.num_hid_mean = config.getint('MODEL_PARAMETER_SETTING', 'num_hid_mean')
        self.apply_mask = config.getint('MODEL_PARAMETER_SETTING', 'apply_mask')
        self.epsilon = config.getfloat('OPTIMIZER_PARAMETERS', 'epsilon')
        self.weightcost_final = config.getfloat('OPTIMIZER_PARAMETERS', 'weightcost_final')
        self.hmc_step_nr = config.getint('HMC_PARAMETERS', 'hmc_step_nr')
        self.hmc_target_ave_rej = config.getfloat('HMC_PARAMETERS', 'hmc_target_ave_rej')

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

    # Function taken from original code
    def compute_energy_mcRBM(self, data, normdata, vel, energy, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1, t2,
                             t6, feat, featsq, feat_mean, length, lengthsq, normcoeff, small, num_vis):
        # normalize input data vectors
        data.mult(data, target=t6)  # DxP (nr input dims x nr samples)
        t6.sum(axis=0, target=lengthsq)  # 1xP
        lengthsq.mult(0.5, target=energy)  # energy of quadratic regularization term
        lengthsq.mult(1. / num_vis)  # normalize by number of components (like std)
        lengthsq.add(small)  # small prevents division by 0
        cmt.sqrt(lengthsq, target=length)
        length.reciprocal(target=normcoeff)  # 1xP
        data.mult_by_row(normcoeff, target=normdata)  # normalized data
        ## potential
        # covariance contribution
        cmt.dot(VF.T, normdata, target=feat)  # HxP (nr factors x nr samples)
        feat.mult(feat, target=featsq)  # HxP
        cmt.dot(FH.T, featsq, target=t1)  # OxP (nr cov hiddens x nr samples)
        t1.mult(-0.5)
        t1.add_col_vec(bias_cov)  # OxP
        cmt.exp(t1)  # OxP
        t1.add(1, target=t2)  # OxP
        cmt.log(t2)
        t2.mult(-1)
        energy.add_sums(t2, axis=0)
        # mean contribution
        cmt.dot(w_mean.T, data, target=feat_mean)  # HxP (nr mean hiddens x nr samples)
        feat_mean.add_col_vec(bias_mean)  # HxP
        cmt.exp(feat_mean)
        feat_mean.add(1)
        cmt.log(feat_mean)
        feat_mean.mult(-1)
        energy.add_sums(feat_mean, axis=0)
        # visible bias term
        data.mult_by_col(bias_vis, target=t6)
        t6.mult(-1)  # DxP
        energy.add_sums(t6, axis=0)  # 1xP
        # kinetic
        vel.mult(vel, target=t6)
        energy.add_sums(t6, axis=0, mult=.5)

    # same as the previous function. Needed only if the energy has to be computed
    # and stored to check the training process
    def compute_energy_mcRBM_visual(self, data, normdata, energy, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1, t2,
                                    t6, feat, featsq, feat_mean, length, lengthsq, normcoeff, small, num_vis):
        # normalize input data vectors
        data.mult(data, target=t6)  # DxP (nr input dims x nr samples)
        t6.sum(axis=0, target=lengthsq)  # 1xP
        lengthsq.mult(0.5, target=energy)  # energy of quadratic regularization term
        lengthsq.mult(1. / num_vis)  # normalize by number of components (like std)
        lengthsq.add(small)  # small prevents division by 0
        cmt.sqrt(lengthsq, target=length)
        length.reciprocal(target=normcoeff)  # 1xP
        data.mult_by_row(normcoeff, target=normdata)  # normalized data
        ## potential
        # covariance contribution
        cmt.dot(VF.T, normdata, target=feat)  # HxP (nr factors x nr samples)
        feat.mult(feat, target=featsq)  # HxP
        cmt.dot(FH.T, featsq, target=t1)  # OxP (nr cov hiddens x nr samples)
        t1.mult(-0.5)
        t1.add_col_vec(bias_cov)  # OxP
        cmt.exp(t1)  # OxP
        t1.add(1, target=t2)  # OxP
        cmt.log(t2)
        t2.mult(-1)
        energy.add_sums(t2, axis=0)
        # mean contribution
        cmt.dot(w_mean.T, data, target=feat_mean)  # HxP (nr mean hiddens x nr samples)
        feat_mean.add_col_vec(bias_mean)  # HxP
        cmt.exp(feat_mean)
        feat_mean.add(1)
        cmt.log(feat_mean)
        feat_mean.mult(-1)
        energy.add_sums(feat_mean, axis=0)
        # visible bias term
        data.mult_by_col(bias_vis, target=t6)
        t6.mult(-1)  # DxP
        energy.add_sums(t6, axis=0)  # 1xP
        # kinetic
        data.mult(data, target=t6)
        energy.add_sums(t6, axis=0, mult=.5)

    # Function taken from original code
    #################################################################
    # compute the derivative if the free energy at a given input
    def compute_gradient_mcRBM(self, data, normdata, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1, t2, t3, t4, t6,
                               feat, featsq, feat_mean, gradient, normgradient, length, lengthsq, normcoeff, small,
                               num_vis):
        # normalize input data
        data.mult(data, target=t6)  # DxP
        t6.sum(axis=0, target=lengthsq)  # 1xP
        lengthsq.mult(1. / num_vis)  # normalize by number of components (like std)
        lengthsq.add(small)
        cmt.sqrt(lengthsq, target=length)
        length.reciprocal(target=normcoeff)  # 1xP
        data.mult_by_row(normcoeff, target=normdata)  # normalized data
        cmt.dot(VF.T, normdata, target=feat)  # HxP
        feat.mult(feat, target=featsq)  # HxP
        cmt.dot(FH.T, featsq, target=t1)  # OxP
        t1.mult(-.5)
        t1.add_col_vec(bias_cov)  # OxP
        t1.apply_sigmoid(target=t2)  # OxP
        cmt.dot(FH, t2, target=t3)  # HxP
        t3.mult(feat)
        cmt.dot(VF, t3, target=normgradient)  # VxP
        # final bprop through normalization
        length.mult(lengthsq, target=normcoeff)
        normcoeff.reciprocal()  # 1xP
        normgradient.mult(data, target=gradient)  # VxP
        gradient.sum(axis=0, target=t4)  # 1xP
        t4.mult(-1. / num_vis)
        data.mult_by_row(t4, target=gradient)
        normgradient.mult_by_row(lengthsq, target=t6)
        gradient.add(t6)
        gradient.mult_by_row(normcoeff)
        # add quadratic term gradient
        gradient.add(data)
        # add visible bias term
        gradient.add_col_mult(bias_vis, -1)
        # add MEAN contribution to gradient
        cmt.dot(w_mean.T, data, target=feat_mean)  # HxP
        feat_mean.add_col_vec(bias_mean)  # HxP
        feat_mean.apply_sigmoid()  # HxP
        gradient.subtract_dot(w_mean, feat_mean)  # VxP

    # Function taken from original code
    ############################################################3
    # Hybrid Monte Carlo sampler
    def draw_HMC_samples(self, data, negdata, normdata, vel, gradient, normgradient, new_energy, old_energy, VF, FH,
                         bias_cov, bias_vis, w_mean, bias_mean, hmc_step, hmc_step_nr, hmc_ave_rej, hmc_target_ave_rej,
                         t1, t2, t3, t4, t5, t6, t7, thresh, feat, featsq, batch_size, feat_mean, length, lengthsq,
                         normcoeff, small, num_vis):
        vel.fill_with_randn()
        negdata.assign(data)
        self.compute_energy_mcRBM(negdata, normdata, vel, old_energy, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1,
                                  t2, t6, feat, featsq, feat_mean, length, lengthsq, normcoeff, small, num_vis)
        self.compute_gradient_mcRBM(negdata, normdata, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1, t2, t3, t4,
                                    t6, feat, featsq, feat_mean, gradient, normgradient, length, lengthsq, normcoeff,
                                    small, num_vis)
        # half step
        vel.add_mult(gradient, -0.5 * hmc_step)
        negdata.add_mult(vel, hmc_step)
        # full leap-frog steps
        for ss in range(hmc_step_nr - 1):
            ## re-evaluate the gradient
            self.compute_gradient_mcRBM(negdata, normdata, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1, t2, t3,
                                        t4, t6, feat, featsq, feat_mean, gradient, normgradient, length, lengthsq,
                                        normcoeff, small, num_vis)
            # update variables
            vel.add_mult(gradient, -hmc_step)
            negdata.add_mult(vel, hmc_step)
        # final half-step
        self.compute_gradient_mcRBM(negdata, normdata, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1, t2, t3, t4,
                                    t6, feat, featsq, feat_mean, gradient, normgradient, length, lengthsq, normcoeff,
                                    small, num_vis)
        vel.add_mult(gradient, -0.5 * hmc_step)
        # compute new energy
        self.compute_energy_mcRBM(negdata, normdata, vel, new_energy, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1,
                                  t2, t6, feat, featsq, feat_mean, length, lengthsq, normcoeff, small, num_vis)
        # rejecton
        old_energy.subtract(new_energy, target=thresh)
        cmt.exp(thresh)
        t4.fill_with_rand()
        t4.less_than(thresh)
        #    update negdata and rejection rate
        t4.mult(-1)
        t4.add(1)  # now 1's detect rejections
        t4.sum(axis=1, target=t5)
        t5.copy_to_host()
        rej = t5.numpy_array[0, 0] / batch_size
        data.mult_by_row(t4, target=t6)
        negdata.mult_by_row(t4, target=t7)
        negdata.subtract(t7)
        negdata.add(t6)
        hmc_ave_rej = 0.9 * hmc_ave_rej + 0.1 * rej
        if hmc_ave_rej < hmc_target_ave_rej:
            hmc_step = min(hmc_step * 1.01, 0.25)
        else:
            hmc_step = max(hmc_step * 0.99, .001)
        return hmc_step, hmc_ave_rej

    def saveLsq(self):
        '''
        Function saving the sum of the square of the data
        (needed for training as well as for post-analysis)
        '''
        d = self.d.astype(np.float32)

        dsq = np.square(d)
        lsq = np.sum(dsq, axis=0)
        with open(self.refDir + 'lsqComplete.pkl', 'wb') as pklFile:
            pickle.dump(lsq, pklFile)

    def train(self):
        """
        @:description:
        Main train function; modified version of the original train function.
        @:addition
        GPU selection (useful for multi-GPU machines) Saving the sum of the square of the data for post-processing
        - Visible data are saved
        - Data samples are permuted for training
        - Weights are saved every 100 training epochs
        - Training energy is visualized every 100 training epochs

        NOTE : anneal learning rate used in the initial code, is NOT used here!
        """

        # plt.ion()

        f1 = plt.figure()
        ax1 = f1.add_subplot(111)
        # ax2 = f1.add_subplot(122)
        # plt.show()

        cmt.cuda_set_device(self.gpuId)
        cmt.cublas_init()
        cmt.CUDAMatrix.init_random(1)

        np.random.seed(self.npRandSeed)
        prng = RandomState(self.npRandState)

        ################################################################
        ##################### CHANGE PATH ##############################
        # Move to current experiment path:
        os.chdir(self.saveDir)
        # Get current path:
        os.getcwd()

        self.plotsDir = 'plots'
        # self.probabilitiesDir = 'p_all'
        if not os.path.isdir(self.plotsDir):
            os.makedirs(self.plotsDir)
        if not os.path.isdir(self.plotsDir + '/energy'):
            os.makedirs(self.plotsDir + '/energy')
        # if not os.path.isdir(self.probabilitiesDir):
        #	os.makedirs(self.probabilitiesDir)
        if not os.path.isdir('weights'):
            os.makedirs('weights')

        d = self.d.astype(np.float32)
        print(("visible size: ", d.shape))

        dsq = np.square(d)
        lsq = np.sum(dsq, axis=0)
        with open('lsqComplete.pkl', 'wb') as pklFile:
            pickle.dump(lsq, pklFile)

        del dsq, lsq

        # Save visible data :
        visData = d
        np.savez('visData.npz', data=d, obsKeys=self.obsKeys, epochTime=self.epochTime)

        with open('visData.txt', 'w') as f:
            f.write("\n Dataset : %s" % (self.dataFilename))
            f.write("\n visData size: %s " % str(visData.shape))
            f.write("\n visData type: %s " % str(visData.dtype))
            f.write("\n \n visData Range: %s " % str(np.max(visData, axis=0) - np.min(visData, axis=0)))
            f.write("\n \n visData min: %s " % str(np.min(visData, axis=0)))
            f.write("\n \n visData max: %s " % str(np.max(visData, axis=0)))
            f.write("\n \n visData mean: %s " % str(np.mean(visData, axis=0)))
            f.write("\n \n visData std: %s " % str(np.std(visData, axis=0)))
            f.close()

        del visData  # if not needed for computing the latent states

        permIdx = prng.permutation(d.shape[0])

        d = d[permIdx, :]

        # subsetting train and test datasets
        # trainPerc = 0.7
        # trainSampNum = int(np.ceil(trainPerc*d.shape[0]))
        # trainSampNum = int(np.floor(trainSampNum/self.batch_size)*self.batch_size)
        # testSampNum = int(d.shape[0]-trainSampNum-1)

        # The test dataset is not used at the moment, it can be used as
        # a validation set to check for overfitting. To use it, uncomment
        # all the variables with 'test' in their name

        # ~ d_test = d[trainSampNum+1:,:]
        # d = d[:trainSampNum,:]
        # obsKeys = self.obsKeys[:trainSampNum]

        totnumcases = d.shape[0]
        num_vis = d.shape[1]

        num_batches = int(totnumcases / self.batch_size)
        print(("num_batches: ", num_batches))
        dev_dat = cmt.CUDAMatrix(d.T)  # VxP
        # ~ test_dat = cmt.CUDAMatrix(d_test.T)

        del d, self.d, self.epochTime, self.obsKeys

        # training parameters (as in the original code by Ranzato)
        epsilon = self.epsilon
        epsilonVF = 2 * epsilon
        epsilonFH = 0.02 * epsilon
        epsilonb = 0.02 * epsilon
        epsilonw_mean = 0.2 * epsilon
        epsilonb_mean = 0.1 * epsilon
        weightcost_final = self.weightcost_final

        # HMC setting
        hmc_step_nr = self.hmc_step_nr
        hmc_step = 0.01
        hmc_target_ave_rej = self.hmc_target_ave_rej
        hmc_ave_rej = hmc_target_ave_rej

        # initialize weights
        VF = cmt.CUDAMatrix(np.array(0.02 * prng.randn(num_vis, self.num_fac), dtype=np.float32, order='F'))  # VxH
        if self.apply_mask == 0:
            FH = cmt.CUDAMatrix(np.array(np.eye(self.num_fac, self.num_hid_cov), dtype=np.float32, order='F'))  # HxO
        else:
            dd = loadmat(
                'your_FHinit_mask_file.mat')  # see CVPR2010paper_material/topo2D_3x3_stride2_576filt.mat for an example
            FH = cmt.CUDAMatrix(np.array(dd["FH"], dtype=np.float32, order='F'))
        bias_cov = cmt.CUDAMatrix(np.array(2.0 * np.ones((self.num_hid_cov, 1)), dtype=np.float32, order='F'))
        bias_vis = cmt.CUDAMatrix(np.array(np.zeros((num_vis, 1)), dtype=np.float32, order='F'))
        w_mean = cmt.CUDAMatrix(
            np.array(0.05 * prng.randn(num_vis, self.num_hid_mean), dtype=np.float32, order='F'))  # VxH
        bias_mean = cmt.CUDAMatrix(np.array(-2.0 * np.ones((self.num_hid_mean, 1)), dtype=np.float32, order='F'))

        # initialize variables to store derivatives
        VFinc = cmt.CUDAMatrix(np.array(np.zeros((num_vis, self.num_fac)), dtype=np.float32, order='F'))
        FHinc = cmt.CUDAMatrix(np.array(np.zeros((self.num_fac, self.num_hid_cov)), dtype=np.float32, order='F'))
        bias_covinc = cmt.CUDAMatrix(np.array(np.zeros((self.num_hid_cov, 1)), dtype=np.float32, order='F'))
        bias_visinc = cmt.CUDAMatrix(np.array(np.zeros((num_vis, 1)), dtype=np.float32, order='F'))
        w_meaninc = cmt.CUDAMatrix(np.array(np.zeros((num_vis, self.num_hid_mean)), dtype=np.float32, order='F'))
        bias_meaninc = cmt.CUDAMatrix(np.array(np.zeros((self.num_hid_mean, 1)), dtype=np.float32, order='F'))

        # initialize temporary storage
        data = cmt.CUDAMatrix(np.array(np.empty((num_vis, self.batch_size)), dtype=np.float32, order='F'))  # VxP
        normdata = cmt.CUDAMatrix(np.array(np.empty((num_vis, self.batch_size)), dtype=np.float32, order='F'))  # VxP
        negdataini = cmt.CUDAMatrix(np.array(np.empty((num_vis, self.batch_size)), dtype=np.float32, order='F'))  # VxP
        feat = cmt.CUDAMatrix(np.array(np.empty((self.num_fac, self.batch_size)), dtype=np.float32, order='F'))
        featsq = cmt.CUDAMatrix(np.array(np.empty((self.num_fac, self.batch_size)), dtype=np.float32, order='F'))
        negdata = cmt.CUDAMatrix(np.array(prng.randn(num_vis, self.batch_size), dtype=np.float32, order='F'))
        old_energy = cmt.CUDAMatrix(np.array(np.zeros((1, self.batch_size)), dtype=np.float32, order='F'))
        new_energy = cmt.CUDAMatrix(np.array(np.zeros((1, self.batch_size)), dtype=np.float32, order='F'))
        energy = cmt.CUDAMatrix(np.array(np.zeros((1, self.batch_size)), dtype=np.float32, order='F'))
        gradient = cmt.CUDAMatrix(np.array(np.empty((num_vis, self.batch_size)), dtype=np.float32, order='F'))  # VxP
        normgradient = cmt.CUDAMatrix(
            np.array(np.empty((num_vis, self.batch_size)), dtype=np.float32, order='F'))  # VxP
        thresh = cmt.CUDAMatrix(np.array(np.zeros((1, self.batch_size)), dtype=np.float32, order='F'))
        feat_mean = cmt.CUDAMatrix(
            np.array(np.empty((self.num_hid_mean, self.batch_size)), dtype=np.float32, order='F'))
        vel = cmt.CUDAMatrix(np.array(prng.randn(num_vis, self.batch_size), dtype=np.float32, order='F'))
        length = cmt.CUDAMatrix(np.array(np.zeros((1, self.batch_size)), dtype=np.float32, order='F'))  # 1xP
        lengthsq = cmt.CUDAMatrix(np.array(np.zeros((1, self.batch_size)), dtype=np.float32, order='F'))  # 1xP
        normcoeff = cmt.CUDAMatrix(np.array(np.zeros((1, self.batch_size)), dtype=np.float32, order='F'))  # 1xP

        # commented to avoid computing the energy on test data
        # ~ data_test = cmt.CUDAMatrix( np.array(np.empty((num_vis, testSampNum)), dtype=np.float32, order='F')) # Vxtest_batch
        # ~ normdata_test = cmt.CUDAMatrix( np.array(np.empty((num_vis, testSampNum)), dtype=np.float32, order='F')) # Vxtest_batch
        # ~ length_test = cmt.CUDAMatrix( np.array(np.zeros((1, testSampNum)), dtype=np.float32, order='F')) # 1xtest_batch
        # ~ lengthsq_test = cmt.CUDAMatrix( np.array(np.zeros((1, testSampNum)), dtype=np.float32, order='F')) # 1xtest_batch
        # ~ normcoeff_test = cmt.CUDAMatrix( np.array(np.zeros((1, testSampNum)), dtype=np.float32, order='F')) # 1xtest_batch
        # ~ vel_test = cmt.CUDAMatrix( np.array(prng.randn(num_vis, testSampNum), dtype=np.float32, order='F'))
        # ~ feat_test = cmt.CUDAMatrix( np.array(np.empty((self.num_fac, testSampNum)), dtype=np.float32, order='F'))
        # ~ featsq_test = cmt.CUDAMatrix( np.array(np.empty((self.num_fac, testSampNum)), dtype=np.float32, order='F'))
        # ~ feat_mean_test = cmt.CUDAMatrix( np.array(np.empty((self.num_hid_mean, testSampNum)), dtype=np.float32, order='F'))
        # ~ energy_test = cmt.CUDAMatrix( np.array(np.zeros((1, testSampNum)), dtype=np.float32, order='F'))

        if self.apply_mask == 1:  # this used to constrain very large FH matrices only allowing to change values in a neighborhood
            dd = loadmat('your_FHinit_mask_file.mat')
            mask = cmt.CUDAMatrix(np.array(dd["mask"], dtype=np.float32, order='F'))
        normVF = 1
        small = 0.5

        # other temporary vars
        t1 = cmt.CUDAMatrix(np.array(np.empty((self.num_hid_cov, self.batch_size)), dtype=np.float32, order='F'))
        t2 = cmt.CUDAMatrix(np.array(np.empty((self.num_hid_cov, self.batch_size)), dtype=np.float32, order='F'))
        t3 = cmt.CUDAMatrix(np.array(np.empty((self.num_fac, self.batch_size)), dtype=np.float32, order='F'))
        t4 = cmt.CUDAMatrix(np.array(np.empty((1, self.batch_size)), dtype=np.float32, order='F'))
        t5 = cmt.CUDAMatrix(np.array(np.empty((1, 1)), dtype=np.float32, order='F'))
        t6 = cmt.CUDAMatrix(np.array(np.empty((num_vis, self.batch_size)), dtype=np.float32, order='F'))
        t7 = cmt.CUDAMatrix(np.array(np.empty((num_vis, self.batch_size)), dtype=np.float32, order='F'))
        t8 = cmt.CUDAMatrix(np.array(np.empty((num_vis, self.num_fac)), dtype=np.float32, order='F'))
        t9 = cmt.CUDAMatrix(np.array(np.zeros((self.num_fac, self.num_hid_cov)), dtype=np.float32, order='F'))
        t10 = cmt.CUDAMatrix(np.array(np.empty((1, self.num_fac)), dtype=np.float32, order='F'))
        t11 = cmt.CUDAMatrix(np.array(np.empty((1, self.num_hid_cov)), dtype=np.float32, order='F'))

        # commented to avoid computing the energy on test data
        # ~ t1_test = cmt.CUDAMatrix( np.array(np.empty((self.num_hid_cov, testSampNum)), dtype=np.float32, order='F'))
        # ~ t2_test = cmt.CUDAMatrix( np.array(np.empty((self.num_hid_cov, testSampNum)), dtype=np.float32, order='F'))
        # ~ t3_test = cmt.CUDAMatrix( np.array(np.empty((self.num_fac, testSampNum)), dtype=np.float32, order='F'))
        # ~ t4_test = cmt.CUDAMatrix( np.array(np.empty((1,testSampNum)), dtype=np.float32, order='F'))
        # ~ t5_test = cmt.CUDAMatrix( np.array(np.empty((1,1)), dtype=np.float32, order='F'))
        # ~ t6_test = cmt.CUDAMatrix( np.array(np.empty((num_vis, testSampNum)), dtype=np.float32, order='F'))

        meanEnergy = np.zeros(self.num_epochs)
        minEnergy = np.zeros(self.num_epochs)
        maxEnergy = np.zeros(self.num_epochs)
        # ~ meanEnergy_test = np.zeros(self.num_epochs)
        # ~ minEnergy_test = np.zeros(self.num_epochs)
        # ~ maxEnergy_test = np.zeros(self.num_epochs)

        # start training
        for epoch in range(self.num_epochs):

            print("Epoch " + str(epoch))

            # anneal learning rates as found in the original code -
            # uncomment if you wish to use annealing!
            # ~ epsilonVFc    = epsilonVF/max(1,epoch/20)
            # ~ epsilonFHc    = epsilonFH/max(1,epoch/20)
            # ~ epsilonbc    = epsilonb/max(1,epoch/20)
            # ~ epsilonw_meanc = epsilonw_mean/max(1,epoch/20)
            # ~ epsilonb_meanc = epsilonb_mean/max(1,epoch/20)

            # no annealing is used in our experiments because learning
            # was stopping too early
            epsilonVFc = epsilonVF
            epsilonFHc = epsilonFH
            epsilonbc = epsilonb
            epsilonw_meanc = epsilonw_mean
            epsilonb_meanc = epsilonb_mean

            weightcost = weightcost_final

            if epoch <= self.startFH:
                epsilonFHc = 0
            if epoch <= self.startwd:
                weightcost = 0

            # commented to avoid computing the energy on test data
            # ~ data_test = test_dat

            # ~ data_test.mult(data_test, target = t6_test) # DxP
            # ~ t6_test.sum(axis = 0, target = lengthsq_test) # 1xP
            # ~ lengthsq_test.mult(1./num_vis) # normalize by number of components (like std)
            # ~ lengthsq_test.add(small) # small avoids division by 0
            # ~ cmt.sqrt(lengthsq_test, target = length_test)
            # ~ length_test.reciprocal(target = normcoeff_test) # 1xP
            # ~ data_test.mult_by_row(normcoeff_test, target = normdata_test) # normalized data

            for batch in range(num_batches):

                # get current minibatch
                data = dev_dat.slice(batch * self.batch_size,
                                     (batch + 1) * self.batch_size)  # DxP (nr dims x nr samples)

                # normalize input data
                data.mult(data, target=t6)  # DxP
                t6.sum(axis=0, target=lengthsq)  # 1xP
                lengthsq.mult(1. / num_vis)  # normalize by number of components (like std)
                lengthsq.add(small)  # small avoids division by 0
                cmt.sqrt(lengthsq, target=length)
                length.reciprocal(target=normcoeff)  # 1xP
                data.mult_by_row(normcoeff, target=normdata)  # normalized data
                ## compute positive sample derivatives
                # covariance part
                cmt.dot(VF.T, normdata, target=feat)  # HxP (nr facs x nr samples)
                feat.mult(feat, target=featsq)  # HxP
                cmt.dot(FH.T, featsq, target=t1)  # OxP (nr cov hiddens x nr samples)
                t1.mult(-0.5)
                t1.add_col_vec(bias_cov)  # OxP
                t1.apply_sigmoid(target=t2)  # OxP
                cmt.dot(featsq, t2.T, target=FHinc)  # HxO
                cmt.dot(FH, t2, target=t3)  # HxP
                t3.mult(feat)
                cmt.dot(normdata, t3.T, target=VFinc)  # VxH
                t2.sum(axis=1, target=bias_covinc)
                bias_covinc.mult(-1)
                # visible bias
                data.sum(axis=1, target=bias_visinc)
                bias_visinc.mult(-1)
                # mean part
                cmt.dot(w_mean.T, data, target=feat_mean)  # HxP (nr mean hiddens x nr samples)
                feat_mean.add_col_vec(bias_mean)  # HxP
                feat_mean.apply_sigmoid()  # HxP
                feat_mean.mult(-1)
                cmt.dot(data, feat_mean.T, target=w_meaninc)
                feat_mean.sum(axis=1, target=bias_meaninc)

                # HMC sampling: draw an approximate sample from the model
                if self.doPCD == 0:  # CD-1 (set negative data to current training samples)
                    hmc_step, hmc_ave_rej = self.draw_HMC_samples(data, negdata, normdata, vel, gradient, normgradient,
                                                                  new_energy, old_energy, VF, FH, bias_cov, bias_vis,
                                                                  w_mean, bias_mean, hmc_step, hmc_step_nr, hmc_ave_rej,
                                                                  hmc_target_ave_rej, t1, t2, t3, t4, t5, t6, t7,
                                                                  thresh, feat, featsq, self.batch_size, feat_mean,
                                                                  length, lengthsq, normcoeff, small, num_vis)
                else:  # PCD-1 (use previous negative data as starting point for chain)
                    negdataini.assign(negdata)
                    hmc_step, hmc_ave_rej = self.draw_HMC_samples(negdataini, negdata, normdata, vel, gradient,
                                                                  normgradient, new_energy, old_energy, VF, FH,
                                                                  bias_cov, bias_vis, w_mean, bias_mean, hmc_step,
                                                                  hmc_step_nr, hmc_ave_rej, hmc_target_ave_rej, t1, t2,
                                                                  t3, t4, t5, t6, t7, thresh, feat, featsq,
                                                                  self.batch_size, feat_mean, length, lengthsq,
                                                                  normcoeff, small, num_vis)

                # compute derivatives at the negative samples
                # normalize input data
                negdata.mult(negdata, target=t6)  # DxP
                t6.sum(axis=0, target=lengthsq)  # 1xP
                lengthsq.mult(1. / num_vis)  # normalize by number of components (like std)
                lengthsq.add(small)
                cmt.sqrt(lengthsq, target=length)
                length.reciprocal(target=normcoeff)  # 1xP
                negdata.mult_by_row(normcoeff, target=normdata)  # normalized data
                # covariance part
                cmt.dot(VF.T, normdata, target=feat)  # HxP
                feat.mult(feat, target=featsq)  # HxP
                cmt.dot(FH.T, featsq, target=t1)  # OxP
                t1.mult(-0.5)
                t1.add_col_vec(bias_cov)  # OxP
                t1.apply_sigmoid(target=t2)  # OxP
                FHinc.subtract_dot(featsq, t2.T)  # HxO
                FHinc.mult(0.5)
                cmt.dot(FH, t2, target=t3)  # HxP
                t3.mult(feat)
                VFinc.subtract_dot(normdata, t3.T)  # VxH
                bias_covinc.add_sums(t2, axis=1)
                # visible bias
                bias_visinc.add_sums(negdata, axis=1)
                # mean part
                cmt.dot(w_mean.T, negdata, target=feat_mean)  # HxP
                feat_mean.add_col_vec(bias_mean)  # HxP
                feat_mean.apply_sigmoid()  # HxP
                w_meaninc.add_dot(negdata, feat_mean.T)
                bias_meaninc.add_sums(feat_mean, axis=1)

                # update parameters
                VFinc.add_mult(VF.sign(), weightcost)  # L1 regularization
                VF.add_mult(VFinc, -epsilonVFc / self.batch_size)
                # normalize columns of VF: normalize by running average of their norm
                VF.mult(VF, target=t8)
                t8.sum(axis=0, target=t10)
                cmt.sqrt(t10)
                t10.sum(axis=1, target=t5)
                t5.copy_to_host()
                normVF = .95 * normVF + (.05 / self.num_fac) * t5.numpy_array[0, 0]  # estimate norm
                t10.reciprocal()
                VF.mult_by_row(t10)
                VF.mult(normVF)
                bias_cov.add_mult(bias_covinc, -epsilonbc / self.batch_size)
                bias_vis.add_mult(bias_visinc, -epsilonbc / self.batch_size)

                if epoch > self.startFH:
                    FHinc.add_mult(FH.sign(), weightcost)  # L1 regularization
                    FH.add_mult(FHinc, -epsilonFHc / self.batch_size)  # update
                    # set to 0 negative entries in FH
                    FH.greater_than(0, target=t9)
                    FH.mult(t9)
                    if self.apply_mask == 1:
                        FH.mult(mask)
                    # normalize columns of FH: L1 norm set to 1 in each column
                    FH.sum(axis=0, target=t11)
                    t11.reciprocal()
                    FH.mult_by_row(t11)
                w_meaninc.add_mult(w_mean.sign(), weightcost)
                w_mean.add_mult(w_meaninc, -epsilonw_meanc / self.batch_size)
                bias_mean.add_mult(bias_meaninc, -epsilonb_meanc / self.batch_size)

            if self.verbose == 1:
                print("VF: " + '%3.2e' % VF.euclid_norm() + ", DVF: " + '%3.2e' % (VFinc.euclid_norm() * (
                        epsilonVFc / self.batch_size)) + ", FH: " + '%3.2e' % FH.euclid_norm() + ", DFH: " + '%3.2e' % (
                              FHinc.euclid_norm() * (
                              epsilonFHc / self.batch_size)) + ", bias_cov: " + '%3.2e' % bias_cov.euclid_norm() + ", Dbias_cov: " + '%3.2e' % (
                              bias_covinc.euclid_norm() * (
                              epsilonbc / self.batch_size)) + ", bias_vis: " + '%3.2e' % bias_vis.euclid_norm() + ", Dbias_vis: " + '%3.2e' % (
                              bias_visinc.euclid_norm() * (
                              epsilonbc / self.batch_size)) + ", wm: " + '%3.2e' % w_mean.euclid_norm() + ", Dwm: " + '%3.2e' % (
                              w_meaninc.euclid_norm() * (
                              epsilonw_meanc / self.batch_size)) + ", bm: " + '%3.2e' % bias_mean.euclid_norm() + ", Dbm: " + '%3.2e' % (
                              bias_meaninc.euclid_norm() * (
                              epsilonb_meanc / self.batch_size)) + ", step: " + '%3.2e' % hmc_step + ", rej: " + '%3.2e' % hmc_ave_rej)
                with open('terminal.txt', 'a') as f:
                    f.write('\n' + "epoch: %s" % str(
                        epoch) + ", VF: " + '%3.2e' % VF.euclid_norm() + ", DVF: " + '%3.2e' % (VFinc.euclid_norm() * (
                            epsilonVFc / self.batch_size)) + ", FH: " + '%3.2e' % FH.euclid_norm() + ", DFH: " + '%3.2e' % (
                                    FHinc.euclid_norm() * (
                                    epsilonFHc / self.batch_size)) + ", bias_cov: " + '%3.2e' % bias_cov.euclid_norm() + ", Dbias_cov: " + '%3.2e' % (
                                    bias_covinc.euclid_norm() * (
                                    epsilonbc / self.batch_size)) + ", bias_vis: " + '%3.2e' % bias_vis.euclid_norm() + ", Dbias_vis: " + '%3.2e' % (
                                    bias_visinc.euclid_norm() * (
                                    epsilonbc / self.batch_size)) + ", wm: " + '%3.2e' % w_mean.euclid_norm() + ", Dwm: " + '%3.2e' % (
                                    w_meaninc.euclid_norm() * (
                                    epsilonw_meanc / self.batch_size)) + ", bm: " + '%3.2e' % bias_mean.euclid_norm() + ", Dbm: " + '%3.2e' % (
                                    bias_meaninc.euclid_norm() * (
                                    epsilonb_meanc / self.batch_size)) + ", step: " + '%3.2e' % hmc_step + ", rej: " + '%3.2e' % hmc_ave_rej)
                sys.stdout.flush()

            # commented to avoid computing the energy on trainig data
            self.compute_energy_mcRBM_visual(data, normdata, energy, VF, FH, bias_cov, bias_vis, w_mean, bias_mean, t1,
                                             t2, t6, feat, featsq, feat_mean, length, lengthsq, normcoeff, small,
                                             num_vis)
            energy.copy_to_host()
            meanEnergy[epoch] = np.mean(energy.numpy_array)
            minEnergy[epoch] = np.min(energy.numpy_array)
            maxEnergy[epoch] = np.max(energy.numpy_array)

            # commented to avoid computing the energy on test data
            # ~ self.compute_energy_mcRBM_visual(data_test,normdata_test,energy_test,VF,FH,bias_cov,bias_vis,w_mean,bias_mean,t1_test,t2_test,t6_test,feat_test,featsq_test,feat_mean_test,length_test,lengthsq_test,normcoeff_test,small,num_vis)
            # ~ energy_test.copy_to_host()
            # ~ meanEnergy_test[epoch] = np.mean(energy_test.numpy_array)
            # ~ minEnergy_test[epoch] = np.min(energy_test.numpy_array)
            # ~ maxEnergy_test[epoch] = np.max(energy_test.numpy_array)

            ax1.cla()
            ax1.plot(list(range(epoch)), meanEnergy[0:epoch])
            ax1.plot(list(range(epoch)), maxEnergy[0:epoch])
            ax1.plot(list(range(epoch)), minEnergy[0:epoch])

            if np.mod(epoch, 100) == 0:
                # f1.savefig(output_folder + str(epoch)+'_'+'fig.png')
                f1.savefig(self.plotsDir + '/energy/energyAt_%s.png' % str(epoch))

            # back-up every once in a while
            if np.mod(epoch, 100) == 0:
                VF.copy_to_host()
                FH.copy_to_host()
                bias_cov.copy_to_host()
                w_mean.copy_to_host()
                bias_mean.copy_to_host()
                bias_vis.copy_to_host()
                savemat("./weights/ws_temp%s" % str(epoch),
                        {'VF': VF.numpy_array, 'FH': FH.numpy_array, 'bias_cov': bias_cov.numpy_array,
                         'bias_vis': bias_vis.numpy_array, 'w_mean': w_mean.numpy_array,
                         'bias_mean': bias_mean.numpy_array, 'epoch': epoch})

                # uncomment if computing the energy in order to store its evolution throghout training
                # ~ savemat(self.refDir + '/' + "training_energy_" + str(self.num_fac) + "_cov" + str(self.num_hid_cov) + "_mean" + str(self.num_hid_mean), {'meanEnergy':meanEnergy,'meanEnergy_test':meanEnergy_test,'maxEnergy': maxEnergy, 'maxEnergy_test': maxEnergy_test, 'minEnergy': minEnergy, 'minEnergy_test': minEnergy_test, 'epoch':epoch})
                # savemat("training_energy_" + str(self.num_fac) + "_cov" + str(self.num_hid_cov) + "_mean" + str(self.num_hid_mean), {'meanEnergy':meanEnergy, 'maxEnergy': maxEnergy, 'minEnergy': minEnergy, 'epoch':epoch})

            # in order to stop the training gracefully, create an empty file
            # named 'stop_now' in the folder containing the experiment
            # configuration file
            if os.path.isfile('stop_now'):
                break

        # final back-up
        VF.copy_to_host()
        FH.copy_to_host()
        bias_cov.copy_to_host()
        bias_vis.copy_to_host()
        w_mean.copy_to_host()
        bias_mean.copy_to_host()
        savemat("ws_fac%s" % str(self.num_fac) + "_cov%s" % str(self.num_hid_cov) + "_mean%s" % str(self.num_hid_mean),
                {'VF': VF.numpy_array, 'FH': FH.numpy_array, 'bias_cov': bias_cov.numpy_array,
                 'bias_vis': bias_vis.numpy_array, 'w_mean': w_mean.numpy_array, 'bias_mean': bias_mean.numpy_array,
                 'epoch': epoch})

        # uncomment if computing the energy in order to store its evolution throghout training
        # ~ savemat(self.refDir + '/' + "training_energy_" + str(self.num_fac) + "_cov" + str(self.num_hid_cov) + "_mean" + str(self.num_hid_mean), {'meanEnergy':meanEnergy,'meanEnergy_test':meanEnergy_test,'maxEnergy': maxEnergy, 'maxEnergy_test': maxEnergy_test, 'minEnergy': minEnergy, 'minEnergy_test': minEnergy_test, 'epoch':epoch})
        savemat(
            "training_energy_" + str(self.num_fac) + "_cov" + str(self.num_hid_cov) + "_mean" + str(self.num_hid_mean),
            {'meanEnergy': meanEnergy, 'maxEnergy': maxEnergy, 'minEnergy': minEnergy, 'epoch': epoch})

        # Compute states if desired:
        # normalise data for covariance hidden:
        # dsq = np.square(visData)
        # lsq = np.sum(dsq, axis=0)
        # lsq /= visData.shape[1]
        # lsq += np.spacing(1)
        # l = np.sqrt(lsq)
        # normD = visData/l

        # logisticArg_c = (-0.5*np.dot(FH.numpy_array.T, np.square(np.dot(VF.numpy_array.T, normD.T))) + bias_cov.numpy_array).T
        # p_hc = logisticFunc(logisticArg_c)

        # logisticArg_m = np.dot(visData, w_mean.numpy_array) + bias_mean.numpy_array.T
        # p_hm = logisticFunc(logisticArg_m)

        # p_all = np.concatenate((p_hc, p_hm), axis=1)
        # savemat(self.probabilitiesDir + '/pAll_%i.mat' % epoch, mdict={'p_all':p_all})

        with open('done', 'w') as doneFile:
            doneFile.write(datetime.strftime(datetime.now(), '%d/%m/%Y %H:%M:%S'))
        # doneFile.closed
