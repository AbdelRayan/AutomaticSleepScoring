import os
import sys
import numpy as np
import pandas as pd
from numpy.random import RandomState
from scipy.io import loadmat, savemat
from configparser import ConfigParser
import PIL.Image
import matplotlib.pyplot as plt
from scipy.io import loadmat

class GetStates(object):
    def __init__(self, refDir, expDoneFlag, modelDir, finalModel):
        self.refDir = refDir
        self.expDoneFlag = expDoneFlag
        print(f"finalModel: {finalModel}")
        self.model = finalModel
        self.modelDir = modelDir

        np.random.seed(124)
        self.prng = RandomState(123)

        self.saveDir = self.refDir

    def loadData(self, statesFilePath, statesFile):
        os.chdir(self.saveDir)
        print("Analysing experiment : ", os.getcwd())

        visData = 'visData.npz'
        dataFile = np.load(visData)
        self.d = dataFile['data']
        self.obsKeys = dataFile['obsKeys'].astype(int)
        self.states = loadmat(f'{statesFilePath}{statesFile}')
        self.states['downsampledStates'] = self.states.pop('states')

    def computeStates(self):
        p_hc, p_hm = self.hidden_activation()
        self.p_all = np.concatenate((p_hc, p_hm), axis=1)
        # print("p_hc :", p_hc)
        # np.savez("p_hc", p_hc)
        # np.savez("p_hm", p_hm)
        if not os.path.isdir('analysis'):
            os.makedirs('analysis')
        os.chdir('analysis')
        

        if not os.path.isdir('epoch%d' % self.epochID):
            os.makedirs('epoch%d' % self.epochID)

        os.chdir('epoch%d' % self.epochID)
        print("Storing in...", os.getcwd())

        if not os.path.isdir('hcActivation'):
            os.makedirs('hcActivation')
        if not os.path.isdir('hmActivation'):
            os.makedirs('hmActivation')
        if not os.path.isdir('binaryActivation'):
            os.makedirs('binaryActivation')

        image = PIL.Image.fromarray(np.uint8(p_hc * 255.))
        resized_image = image.resize((1200, 1200))        
        image.save('./hcActivation/%i.png' % self.epochID)

        image = PIL.Image.fromarray(np.uint8(p_hm * 255.))
        resized_image = image.resize((1200, 1200))
        resized_image.save('./hmActivation/%i.png' % self.epochID)

        self.binary_latentActivation = (self.p_all >= 0.5).astype(int)
        plt.figure(figsize=(10, 50))
        plt.imshow(self.binary_latentActivation, cmap='gray')
        plt.title('Binary Latent Activations')
        plt.xlabel('Hidden units')
        plt.ylabel('Epoch')
        plt.savefig('./binaryActivation/%i.png' % self.epochID)

        str_repr = np.array([''.join(map(str, row)) for row in self.binary_latentActivation])
        
        unique_bin, uniqueFramesID, ic = np.unique(str_repr, return_index=True, return_inverse=True)
        uniqueAct = self.binary_latentActivation[uniqueFramesID]
        uniqueCount = np.array([np.sum(ic == i) for i in range(len(uniqueFramesID))])
        p_unique = self.p_all[uniqueFramesID]
        uniqueStates = np.zeros((len(uniqueAct), len(uniqueAct[0]) + 2))
        inferredStates = np.column_stack((
            np.zeros(len(self.binary_latentActivation)),
            self.states['downsampledStates'].astype(int).flatten()[:len(self.binary_latentActivation)]))

        for i in range(len(uniqueAct)):
            uniqueStates[i, 0] = i + 1
            uniqueStates[i, 1] = uniqueCount[i]
            uniqueStates[i, 2:] = uniqueAct[i]

            row_indices = np.where((self.binary_latentActivation == uniqueAct[i]).all(axis=1))[0]

            inferredStates[row_indices, 0] = i + 1 

        np.savez_compressed('latentStates.npz', 
                            probabilities=self.p_all, 
                            binary=self.binary_latentActivation, 
                            inferredStates=inferredStates,
                            uniqueStates=uniqueStates)

    def computeUniqueStates(self):
        uniqueAct, p_unique = self.compute_uniques(self.binary_latentActivation, self.p_all)
        print("np.shape(uniqueAct) : ", np.shape(uniqueAct))
        print("np.unique(uniqueAct) :", np.unique(uniqueAct))
        print("uniqueAct", uniqueAct)
        del self.p_all

        print("Checking if there are hidden_units that are always off..")
        print("The sum of the unique latent activations' columns is : ", np.sum(uniqueAct, axis=0))

        with open('latentStatesInfo.txt', 'w') as f:
            f.write("\n The number of the unique latent activations is : %s" % uniqueAct.shape[0])
            f.write("\n The sum of the unique latent activations' columns is : %s" % np.sum(uniqueAct, axis=0))
            f.close()

        uniqueAct2 = np.insert(uniqueAct, 0, 0, axis=1)
        uniqueAct2 = np.insert(uniqueAct2, 0, 0, axis=1)

        self.obsKeys = np.insert(self.obsKeys, 1, 0, axis=1)
        for i in range(uniqueAct.shape[0]):
            temp_idx = np.where(np.all(self.binary_latentActivation == uniqueAct[i, :], axis=1))[0]

            uniqueAct2[i, 0] = i
            uniqueAct2[i, 1] = len(temp_idx)

            self.obsKeys[temp_idx, 1] = i

        np.savez_compressed('uniqueStates.npz', uniqueStates=uniqueAct2, probabilities=p_unique)
        np.savez('obsKeys.npz', obsKeys=self.obsKeys)

    def compute_uniques(self, p_h_bin, p_h):
        tmpUnique = np.unique(p_h_bin.view(np.dtype((np.void, p_h_bin.dtype.itemsize * p_h_bin.shape[1]))),
                              return_index=True, return_counts=True)
        uniqueAct = tmpUnique[0].view(p_h_bin.dtype).reshape(-1, p_h_bin.shape[1])
        uniqueFramesID = tmpUnique[1]
        uniqueFramesID = uniqueFramesID.reshape(-1, 1)
        uniqueCount = tmpUnique[2]
        p_unique = p_h[uniqueFramesID[:, 0], :]

        print("The number of the unique latent activations is :", uniqueAct.shape[0])

        return uniqueAct, p_unique

    def logisticFunc(self, x):
        return 1. / (1. + np.exp(-x))

    def hidden_activation(self):
        if self.expDoneFlag == 'True':
            print(self.modelDir + self.model)
            ws_temp = loadmat(self.modelDir + self.model)

        else:
            temp_model = input("Please enter the training epoch you want to analyze: ")
            ws_temp = loadmat('./weights/ws_temp%d.mat' % int(temp_model))
        w_mean = ws_temp['w_mean']
        FH = ws_temp['FH']
        VF = ws_temp['VF']
        bias_cov = ws_temp['bias_cov']
        bias_mean = ws_temp['bias_mean']
        self.epochID = ws_temp['epoch']
        dsq = np.square(self.d)
        lsq = np.sum(dsq, axis=0)
        lsq /= self.d.shape[1]
        lsq += np.spacing(1)
        l = np.sqrt(lsq)
        normD = self.d / l

        logisticArg_c = (-0.5 * np.dot(FH.T, np.square(np.dot(VF.T, normD.T))) + bias_cov).T

        print("logisticArg_c min : ", np.min(logisticArg_c))
        print("logisticArg_c max : ", np.max(logisticArg_c))
        print("logisticArg_c mean : ", np.mean(logisticArg_c))
        print(np.isfinite(logisticArg_c).all())

        p_hc = self.logisticFunc(logisticArg_c)

        logisticArg_m = np.dot(self.d, w_mean) + bias_mean.T

        print("logisticArg_m min : ", np.min(logisticArg_m))
        print("logisticArg_m max : ", np.max(logisticArg_m))
        print("logisticArg_m mean : ", np.mean(logisticArg_m))
        print(np.isfinite(logisticArg_m).all())

        p_hm = self.logisticFunc(logisticArg_m)

        return p_hc, p_hm


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='Experiment path', default='/teamspace/studios/this_studio/mouse-sleep-analysis/sample_data/experiments')
    parser.add_argument('-done', help='Experiment done flag', default=False)
    parser.add_argument('-input', help = 'Input (dataset + manual scoring) path', default='/teamspace/studios/this_studio/mouse-sleep-analysis/sample_data/input')
    parser.add_argument('-md', help = 'model Dir')
    parser.add_argument('-m', help='Saved model name')
    parser.add_argument('-s', help='Saved states file name')
    args = parser.parse_args()
    print('Initialization...')
    print("args.f :", args.f)
    print("args.done :" , args.done)
    print("args.md", args.md)
    print("args.input", args.input)
    print("args.m :", args.m)
    print("args.s :", args.s)
    model = GetStates(args.f, args.done, args.md, args.m)


    print('Loading data...')
    model.loadData(args.input, args.s)

    print('Computing latent states...')
    model.computeStates()

    print('Computing the unique binary latent states...')
    model.computeUniqueStates()
