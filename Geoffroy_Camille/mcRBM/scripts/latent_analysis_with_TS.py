import os
import sys
import pickle
import math
import numpy as np
from numpy.random import RandomState
from scipy.io import loadmat, savemat
from configparser import *
from matplotlib import use
from pylab import *
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import linkage, dendrogram

from scipy.stats import ttest_ind

from sklearn import manifold
from sklearn.metrics import DistanceMetric

import sys; 
from data_preproc import DataPreproc


class StatesAnalysis(object):   
    '''
    Object performing the analysis of the observed latent states in terms
    of sleep physiology.
    '''
    
    def __init__(self, refDir, expConfigFilename, epochID, threshold, multi, norm, features, groupNames):
        # directory containing all the configuration files for the experiment
        self.refDir = refDir
        # file with configuration details for the launched experiment
        self.expConfigFilename = expConfigFilename
        # data pre-processing object
        self.dpp = DataPreproc()
        # loading details from configuration files
        self.loadExpConfig()
        # id of the epoch to be analysed
        self.epochID = int(epochID)
        # threshold for latent states to be kept : latent states with 
        # population smaller than this threshold, are not used for most
        # of the produced graphs
        self.threshold = int(threshold)
        # true if multi-subject experiment
        self.multi = multi
        # normalization technique to be used for scaling the histogram
        # over the observed latent states
        self.norm = norm
        # which features were used for the analysis : EEG bands,
        # EEG ratios, pca
        self.features = features
        
        if self.multi:
            print("A multi-subject experiment is being analyzed..")
            # in case of multi subject analysis, the names of mouse groups
            # need to be given in the same order as they have been stored
            # in the dataset
            self.groupNames = [k for k in groupNames.split(',')]
            #print(self.groupNames)
        
        np.random.seed(124)
        self.prng = RandomState(123)
    
    def loadExpConfig(self):
        '''
        Method loading the configuration details of the current experiment.
        '''
        
        config = ConfigParser()
        config.read(self.refDir + "configuration_files/" + self.expConfigFilename)
                
        #-- Experiment details:        
        self.dataDir = config.get('EXP_DETAILS','dsetDir')
        self.expsDir = config.get('EXP_DETAILS','expsDir')
        self.expName = config.get('EXP_DETAILS','expID')        
        self.dSetName = config.get('EXP_DETAILS','dSetName')
        
        self.logFlag = config.getboolean('EXP_DETAILS','logFlag')
        self.meanSubtructionFlag = config.getboolean('EXP_DETAILS','meanSubtructionFlag')
        self.scaleFlag = config.getboolean('EXP_DETAILS','scaleFlag')
        self.scaling = config.get('EXP_DETAILS','scaling')
        self.doPCA = config.getboolean('EXP_DETAILS','doPCA')
        self.whitenFlag = config.getboolean('EXP_DETAILS','whitenFlag')
        self.rescaleFlag = config.getboolean('EXP_DETAILS','rescaleFlag')
        self.rescaling = config.get('EXP_DETAILS','rescaling')
        
        self.saveDir = self.expsDir + self.expName


    #-- Data Loading function:
    def loadData(self):
        '''
        Method loading the visible data.
        '''

        os.chdir(self.saveDir)
        # Get current path:
        print(("Analysing experiment : ", os.getcwd()))

        """
        Load visible data
        """
        visData = 'visData.npz'
        dataFile = np.load(visData)
        self.d = dataFile['data']
        obsKeys = dataFile['obsKeys'].astype(int)
        self.epochTime = dataFile['epochTime']

        self.sleepStages = ['Wake', 'NREM', 'TS', 'REM']

        """
        Back-project data to the log space for visualization
        """
        print("Backprojecting the data to the log space..")
        self.dinit = self.backProjection(self.d)
        np.savez('backProjectedData.npz', d=self.dinit, obsKeys=obsKeys, epochTime=self.epochTime)

        del obsKeys, dataFile
    
    #-- Function for projecting back the pre-processed data:
    def backProjection(self, d):
        '''
        Method projecting the pre-processed data back to the log-space.
        '''
        
        if self.doPCA:
            with open('./dataDetails/pca_obj.save') as pcaPklFile:
                pca = pickle.load(pcaPklFile)
            if self.rescaleFlag:
                print("Scaling back to PCA data..")
                minMax = np.load('./dataDetails/minmaxFilePCA.npz')

                d = self.backProjectionScaling(self.rescaling, d, minMax['dMaxRowPCA'], minMax['dMinRowPCA'], 
                                    minMax['dMinPCA'], minMax['dMaxPCA'], minMax['dMeanPCA'], minMax['dStdPCA'])

            print("Inverting PCA data..")
            d = pca.inverse_transform(d)

        if self.scaleFlag:
            print("Scaling back to Log data..")
            minMax = np.load('./dataDetails/minmaxFileInit.npz')
            d = self.backProjectionScaling(self.scaling, d, minMax['dMaxRow'], minMax['dMinRow'], minMax['dMin'], 
                                    minMax['dMax'], minMax['dMean'], minMax['dStd'])

        return d            
    
    def backProjectionScaling(self, scaling, d, dMaxRow, dMinRow, dMin, dMax, dMean, dStd):
        '''
        Method for projecting back the scaled data.
        '''

        if 'single' in scaling:         
            d = ( (d+5.) * (dMaxRow - dMinRow) ) /10. + dMinRow
        elif 'global' in scaling:
            d = ( (d+5.) * (dMax - dMin) ) /10. + dMin
        elif 'baseZeroG' in scaling:
            d = d * (dMax - dMin) + dMin
        elif 'baseZeroS' in scaling:
            d = ( d * (dMaxRow - dMinRow) ) / 10. + dMinRow
        elif 'baseZeroCol' in scaling:
            d = d * (dMaxRow - dMinRow) + dMinRow
        elif 'stdz' in scaling:
            d = d * dStd + dMean
        elif 'minZero' in scaling:
            d = d + dMinRow         

        return d            

    #-- Latent States Analysis --#
    def analyzeStates(self):
        '''
        Method performing basic visualization of the input data
        associated with the observed latent states.
        '''

        # Move in the current training epoch's folder       
        os.chdir('analysis/epoch%d' %self.epochID)  
        # Get current path:
        os.getcwd()

        if not os.path.isdir('boxPlotsBackProjectedData'):
            os.makedirs('boxPlotsBackProjectedData')    

        # Load unique latent activations
        fileName = 'uniqueStates.npz'
        fload = np.load(fileName)
        self.uniqueStates = fload['uniqueStates']
        # Load obsKeys
        fileName = 'obsKeys.npz'
        fload = np.load(fileName)
        self.obsKeys = fload['obsKeys']

        del fload

        """
        In case of multi-subject analysis:
        insert a column to give to each epoch the label of the strain (e.g. 1,2,3)
        it belongs to.
        """     
        if self.multi:
            self.obsKeys = np.insert(self.obsKeys, self.obsKeys.shape[1], 0, axis=1)
            for i in range(self.obsKeys.shape[0]):
                self.obsKeys[i, self.obsKeys.shape[1]-1] = int(str(self.obsKeys[i, self.obsKeys.shape[1]-2])[0])
            self.mouseGroups = np.unique(self.obsKeys[:, self.obsKeys.shape[1]-1])
            self.subjects = np.unique(self.obsKeys[:, self.obsKeys.shape[1]-2])

            print(("mouseGroups: ", self.mouseGroups))
            print(("subjects: ", self.subjects))

            for strain in np.unique(self.obsKeys[:, self.obsKeys.shape[1]-1]):
                print(("Strain %d: of shape: %d" %(strain, self.obsKeys[self.obsKeys[:, self.obsKeys.shape[1]-1]==strain, :].shape[0])))


        #labels = ['f%s' %i for i in range(self.d.shape[1])]
        # EEG_labels = ['Theta', 'Delta', 'Ratio', 'Slope', 'Complexity']
        EEG_labels = ['', 'IndexW', 'IndexR', 'IndexN', 'Index1', 'Index2', 'Index3', 'Index4', '0-0.5hz', 'Theta', 'Delta']
        # EEG_labels = ['IndexW', 'IndexR', 'IndexN', 'Index1', 'Index2', 'Index3', 'Index4', 'Theta', 'Delta']
        
        # if self.features=='bands':
        #     EEG_labels = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        # elif self.features=='ratios':
        #     Bands = [r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$']
        #     EEG_labels = []       
        #     for i in range(5):    
        #         for j in range(i+1, 5):
        #             EEG_labels.append(Bands[i] + '$/$' + Bands[j])
        # else:
        #     EEG_labels = ['f%d' %(i+1) for i in range( self.dinit.shape[1] -1 )]

        EEG_range = [math.floor(self.dinit[:,:self.dinit.shape[1]-1].min()), math.ceil(self.dinit[:,:self.dinit.shape[1]-1].max())]
        print("self.uniqueStates :", self.uniqueStates)
        for i in self.uniqueStates[:, 0]:
            id_bin = self.uniqueStates[i,13:]
            idx = np.where( self.obsKeys[:, 1] == i )[0]            
            latent_frames = self.obsKeys[idx, :]

            length_awake = round((len(np.where((latent_frames[:,3]==1))[0])/float(len(latent_frames))),3)
            length_nrem = round((len(np.where((latent_frames[:,3]==3))[0])/float(len(latent_frames))),3)
            length_TS = round((len(np.where((latent_frames[:,3]==4))[0])/float(len(latent_frames))),3)
            length_rem = round((len(np.where((latent_frames[:,3]==5))[0])/float(len(latent_frames))),3)
            print(f"length_awake : {length_awake}")
            #dPlot = [self.d[idx, j] for j in range(self.d.shape[1])]
            # print(f"self.dinit: {self.dinit}")
            print(f"self.dinit.shape[1]: {self.dinit.shape[1]}")
            # dPlotEEG = [self.dinit[idx, j] for j in range(self.dinit.shape[1]-1)] 
            dPlotEEG = [self.dinit[idx, j] for j in range(self.dinit.shape[1])]    
            # print(f"dPlotEEG: {dPlotEEG}")
            # visualize boxplots per latent state
            self.BoxPlotsSimple(dPlotEEG, './boxPlotsBackProjectedData/', len(idx), i, EEG_labels, 
                                        length_awake, length_nrem, length_TS, length_rem, EEG_range, id_bin)
       
    def groupStatistics(self):
        '''
        Method performing some statistical analysis over the
        under analysis mouse groups.

        1: bar plot per latent state displaying the number of epochs
           each latent state falls in each subject.
        2: visualization of the p-values of the 2-sample independent t-test.
           This test aims at helping us find the mouse group specific latent
           states.
        '''

        """
        Create output directory
        """
        self.groupStatistics = 'groupStatistics'
        if not os.path.isdir(self.groupStatistics):
            os.makedirs(self.groupStatistics)
        os.chdir(self.groupStatistics)
        os.getcwd()

        if not os.path.isdir('groupBoxPlots'):
            os.makedirs('groupBoxPlots')    
        if not os.path.isdir('ttest'):
            os.makedirs('ttest')
        if not os.path.isdir('barPlots'):
            os.makedirs('barPlots')     

        """
        Find the unique latent-states' IDs
        """
        self.lstatesIDs = np.unique(self.obsKeys[:, 1])
        self.lstatesPopulation = np.bincount(self.obsKeys[:, 1])

        subjectsIDs = np.unique(self.obsKeys[:, self.obsKeys.shape[1]-2])
        self.strainIDs = np.unique(self.obsKeys[:, self.obsKeys.shape[1]-1])

        '''
        Sort subject IDs so that we have the strains in order:
        '''
        self.subjectsIDs_ordered = []

        '''
        and create a dictionary : the strainID and the number of animals it contains:
        '''
        self.num_subjs_per_strain = {}
        for str_ID in self.strainIDs:
            self.num_subjs_per_strain[str_ID] = 0
            for subject in subjectsIDs:
                if subject not in self.subjectsIDs_ordered:
                    if int(str(subject)[0])==str_ID:
                        self.subjectsIDs_ordered.append(subject)
                        self.num_subjs_per_strain[str_ID] = self.num_subjs_per_strain[str_ID]+1

        """
        Compute the per subject distribution: a discrete distribution per subject 
        (i.e., number of frames each subject appers in each latent state)
        """
        #--- Create Array : (M_animals * N_lstates)
        self.subjectsDistr = np.zeros((len(self.subjectsIDs_ordered), len(self.lstatesIDs)), dtype=np.int32)
        #-- Insert a labels column : label = MouseID
        self.subjectsDistr = np.insert(self.subjectsDistr, self.subjectsDistr.shape[1], self.subjectsIDs_ordered, axis=1)
        #-- Insert a strain column :
        self.subjectsDistr = np.insert(self.subjectsDistr, self.subjectsDistr.shape[1], 0, axis=1)
        #-- Iterate through columns==animals and define the strainID :
        for i in range(self.subjectsDistr.shape[0]):
            self.subjectsDistr[i, self.subjectsDistr.shape[1]-1] = int(str(self.subjectsDistr[i, self.subjectsDistr.shape[1]-2])[0])


        #-- Create Subjects' Distribution : for how many epochs each subject appears in each latent-state
        for i in range(self.subjectsDistr.shape[0]):            
            subjectFrames = self.obsKeys[self.obsKeys[:,4]==self.subjectsDistr[i, self.subjectsDistr.shape[1]-2],:]
            print(("Subject:", self.subjectsDistr[i, self.subjectsDistr.shape[1]-2], "of latent size:", subjectFrames.shape))

            for lstate in self.lstatesIDs:
                self.subjectsDistr[i, lstate] = len(np.where(subjectFrames[:, 1]==lstate)[0])


        self.strainIDs1 = self.strainIDs

        savemat('subjectsDistribution.mat', mdict={'subjectsDistribution':self.subjectsDistr})

        '''
        Computing and visualizing statistics
        '''

        '''
        Split subjectsDistr into strains:
        '''
        strainsDistr_dict = {}
        for strainID in self.strainIDs:
            strainsDistr_dict[strainID] = self.subjectsDistr[np.where(self.subjectsDistr[:, self.subjectsDistr.shape[1]-1]==strainID)[0], :self.subjectsDistr.shape[1]-2]
            print(("Strain: ", strainID, " of shape: ", strainsDistr_dict[strainID].shape))

        '''
        Create a dictionary to save for each latent-state the t-test scores:
        '''
        latentDict = {}
        latentDict_pvalues = {}
        latentDict_statistics = {}

        '''
        Create dictionary : Latent - Counts
        '''
        LatentCounts = {}

        '''
        Iterate through latent states:
        ########## TO DO: simplify loop! ##########
        '''
        for lstate in self.lstatesIDs:

            ##########################################################
            """
            Bar-plot part: how many times each subject appears in the current latent-state
            """
            latent_frames = self.obsKeys[np.where(self.obsKeys[:, 1] == lstate)[0], :]

            length_awake = round((len(np.where((latent_frames[:, latent_frames.shape[1]-3]==1))[0])/float(len(latent_frames))),3)
            length_nrem = round((len(np.where((latent_frames[:, latent_frames.shape[1]-3]==3))[0])/float(len(latent_frames))),3)
            length_TS = round((len(np.where((latent_frames[:, latent_frames.shape[1]-3]==4))[0])/float(len(latent_frames))),3)
            length_rem = round((len(np.where((latent_frames[:, latent_frames.shape[1]-3]==5))[0])/float(len(latent_frames))),3)

            '''
            Initialize subjects' counts to 0:
            '''
            subjects_counts = {}
            for subject in self.subjectsIDs_ordered:
                #print subject
                subjects_counts[subject] = 0

            '''
            Iterate through frames in the current latent state &
            count how many times each subject appears in:
            '''
            for l_frame in range(latent_frames.shape[0]):
                subjects_counts[latent_frames[l_frame, latent_frames.shape[1]-2]] = subjects_counts[latent_frames[l_frame, latent_frames.shape[1]-2]] + 1

            '''
            Bar plot
            '''
            N = len(self.subjectsIDs_ordered)
            ind = np.arange(N)

            C_matrix = np.zeros((len(subjects_counts),2), dtype=np.int32)
            i = 0
            for k in list(subjects_counts.keys()):
                C_matrix[i, 0] = k
                C_matrix[i, 1] = subjects_counts[k]
                i = i+1

            C_matrix = C_matrix[C_matrix[:, 0].argsort()]

            #-- Sort subject IDs so that we have the strains in order:
            Counts = []
            Lb = []
            for str_ID in self.strainIDs1:
                for subject in C_matrix[:,0]:
                    idx_posit = np.where(C_matrix[:,0]==subject)[0][0]
                    if subject not in Lb:
                        if int(str(subject)[0])==str_ID:
                            Lb.append(subject)
                            Counts.append(C_matrix[idx_posit,1])        

            LatentCounts['l_state_' + str(lstate)] = {}
            LatentCounts['l_state_' + str(lstate)]['counts'] = np.asarray(Counts)
            LatentCounts['l_state_' + str(lstate)]['subjects'] = np.asarray(Lb)

            """
            Statistical test part: Indipendent 2 sample ttest
            """

            '''
            Create a square array (Number_of_strains * Number_of_strains)
            '''         
            tt_statistic = np.zeros((len(self.strainIDs), len(self.strainIDs)), dtype=float32)
            tt_pvalues = np.zeros((len(self.strainIDs), len(self.strainIDs)), dtype=float32)

            '''
            Iterate through strains & compute the pairwise tests:
            '''
            for i in range(len(self.strainIDs)-1):
                for j in range(i+1,len(self.strainIDs)):

                    strain_i = strainsDistr_dict[self.strainIDs[i]]
                    strain_j = strainsDistr_dict[self.strainIDs[j]]

                    [tt_statist, tt_p] = ttest_ind(strain_i[:,lstate], strain_j[:,lstate], equal_var = True)

                    tt_statistic[i,j] = tt_statist
                    tt_statistic[j,i] = tt_statist
                    tt_pvalues[i,j] = tt_p
                    tt_pvalues[j,i] = tt_p

            for i_el in range(tt_pvalues.shape[0]):
                tt_pvalues[i_el, i_el] = 1.


            """
            Visualization part
            """

            # Bar plot
            fig = plt.figure(figsize=(20,15))
            ax1 = fig.add_subplot(111)
            #fig.suptitle('Number of epochs associated with this latent state: ' + str(len(latent_frames)) + '\nAwake: ' + str(length_awake*100) + '%, Nrem: ' + str(length_nrem*100) + '%, Rem: ' + str(length_rem*100) + '%', fontsize=30, fontweight='bold')
            fig.suptitle('Latent State ' + str(lstate) + '\nWakefulness: ' + str(round(length_awake,3)*100) + '%, NREM Sleep: ' + str(round(length_nrem,3)*100) + '%, TS Sleep: ' + str(round(length_TS,3)*100)+ '%, REM Sleep: ' + str(round(length_rem,3)*100) + '%' + '\nTotal number of epochs: ' + str(len(latent_frames)), fontsize=30, fontweight='bold')

            ax1.spines['top'].set_visible(False)
            #ax.spines['top'].set_color('none')
            ax1.spines['right'].set_visible(False)
            #ax.spines['right'].set_color('none')
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            ax1.yaxis.set_ticks_position('left')
            ax1.xaxis.set_ticks_position('bottom')
            width = 0.35
            #id_start = 0
            strain_bar = {}
            colors = ['#b2182b', '#238b45', '#238b45', '#3690c0', '#3690c0', '#023858']

            idx_start = 0
            idx_end = self.num_subjs_per_strain[self.strainIDs1[0]]
            strain_bar[0] = ax1.bar(ind[idx_start:idx_end], Counts[idx_start:idx_end]/self.lstatesPopulation[lstate], width, color=colors[0], edgecolor = "none")
            for i in range(len(self.num_subjs_per_strain)-1):
                idx_start = idx_start+self.num_subjs_per_strain[self.strainIDs1[i]]
                idx_end = idx_end+self.num_subjs_per_strain[self.strainIDs1[i+1]]
                strain_bar[i+1] = ax1.bar(ind[idx_start:idx_end], Counts[idx_start:idx_end]/self.lstatesPopulation[lstate], width, color=colors[i+1], edgecolor = "none")   

            ax1.set_ylabel('Count', fontweight='bold', fontsize=30)
            ax1.set_xlabel('Subjects', fontweight='bold', fontsize=30)
            xTickMarks = ['s%s' %str(j) for j in Lb]
            ax1.set_xticks(ind+width/2)
            xtickNames = ax1.set_xticklabels(xTickMarks, fontweight='bold')
            ax1.xaxis.set_ticks_position('none')
            ax1.yaxis.set_ticks_position('none')

            legend_properties = {'weight':'bold', 'size':25}

            if len(self.strainIDs)>2:
                ax1.legend((strain_bar[0], strain_bar[1], strain_bar[3]), self.groupNames, bbox_to_anchor=(.9, .9), loc=2, borderaxespad=0., prop=legend_properties)
            else:
                ax1.legend((strain_bar[0], strain_bar[1]), self.groupNames, bbox_to_anchor=(.9, .9), loc=2, borderaxespad=0., prop=legend_properties)
            plt.setp(xtickNames, rotation=45, fontsize=20)

            yint = []
            locs, l = plt.yticks()
            for each in locs:
                yint.append( round(each, 2) )
            plt.yticks(yint)

            ax1.set_yticklabels(ax1.get_yticks(), fontweight='bold', fontsize=20)           

            fname = 'lState_' + str(lstate) + '.png'
            fname = os.path.join('./barPlots/', fname)
            fig.savefig(fname, transparent=True, dpi=100)
            plt.close(fig)


            """
            T-test visualization
            """
            fig = plt.figure(figsize=(33, 33))
            ax4 = fig.add_subplot(111)
            fig.suptitle('Independent two-sample t-test: p-values', fontsize=60, fontweight='bold')

            #hmat = ax4.pcolor(tt_pvalues, cmap='RdYlGn_r', vmin=0., vmax=1.)
            hmat = ax4.pcolor(tt_pvalues, cmap='gray_r', vmin=0., vmax=1.)
            #ax4.set_title('Independent two-sample t-test: p-values', fontsize=25, fontweight='bold')


            # text portion
            if len(self.strainIDs)>2:
                ind_array = np.arange(0., 3., 1.)
            else:
                ind_array = np.arange(0., 2., 1.)
            x, y = np.meshgrid(ind_array, ind_array)

            for x_val, y_val in zip(x.flatten(), y.flatten()):
                c = tt_pvalues[x_val.astype(np.int32), y_val.astype(np.int32)]
                if c==1.0 :
                    ax4.text(x_val+0.5, y_val+0.5, c, va='center', ha='center', color='w', fontweight='bold', fontsize=50)
                else:
                    ax4.text(x_val+0.5, y_val+0.5, round(c, 3), va='center', ha='center', color='k', fontweight='bold', fontsize=50)

            ax4.set_xticks(np.arange(tt_pvalues.shape[0])+0.5, minor=False)
            ax4.set_yticks(np.arange(tt_pvalues.shape[1])+0.5, minor=False)

            # want a more natural, table-like display
            ax4.invert_yaxis()
            ax4.xaxis.tick_top()

            ax4.set_xticklabels(self.groupNames, minor=False, fontweight='bold', fontsize=50)
            ax4.set_yticklabels(self.groupNames, minor=False, fontweight='bold', fontsize=50)

            divider4 = make_axes_locatable(ax4)
            cax4 = divider4.append_axes("right", size="5%", pad=0.1)

            cb = plt.colorbar(hmat, cax=cax4)
            for l in cb.ax.yaxis.get_ticklabels():
                l.set_weight("bold")
                l.set_fontsize(40)

            fname = 'lState_' + str(lstate) + '.png'
            fname = os.path.join('./ttest/', fname)
            fig.savefig(fname, transparent=True, dpi=100)
            plt.close(fig)

            """
            Group-boxPlot
            """
            data_to_plot = []
            for k in list(strainsDistr_dict.keys()):
                data_to_plot.append(strainsDistr_dict[k][:, lstate]/self.lstatesPopulation[lstate])

            fig = plt.figure(figsize=(15,12))
            fig.suptitle('LS ' + str(lstate) + ' - Total: ' + str(len(latent_frames)) + ' epochs' + '\nWAKE: ' + str(length_awake*100) + '%, NREM: ' + str(length_nrem*100) + '\nTS' + str(length_TS*100) +   '%, REM: ' + str(length_rem*100) + '%', fontsize=35, fontweight='bold')

            ax = fig.add_subplot(111)
            ax.grid(False)
            ax.patch.set_facecolor('0.85')
            ax.patch.set_alpha(0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(5)
            ax.spines['bottom'].set_color('k')
            ax.spines['left'].set_linewidth(5)
            ax.spines['left'].set_color('k')
            ax.yaxis.set_ticks_position('none')
            ax.xaxis.set_ticks_position('none')
            ax.set_ylabel('Frequency', fontweight='bold', fontsize=35)


            bp = plt.boxplot(data_to_plot, patch_artist=True)

            plt.setp(bp['boxes'],            # customise box appearance
                     edgecolor='k',          # outline colour
                     linewidth=5.,           # outline line width
                     facecolor='None')       # fill box with colour
            plt.setp(bp['whiskers'], color='k', linestyle='--', linewidth=4.5)

            plt.setp(bp['medians'],          # customize median lines
                     color='k',              # line colour
                     linewidth=5.)           # line thickness


            xtickNames = ax.set_xticklabels(self.groupNames)
            plt.setp(xtickNames, fontsize=35, fontweight='bold')

            yint = []
            locs, labels = plt.yticks()
            for each in locs:
                yint.append(round(each, 2))
            plt.yticks(yint)

            ax.set_yticklabels(ax.get_yticks(), fontweight='bold', fontsize=30)

            fname = 'lstate%d.jpeg' %lstate
            fname = os.path.join('./groupBoxPlots/', fname)
            fig.savefig(fname, format='jpeg', transparent=True, dpi=100)
            plt.close(fig)


    def visibleDistributions(self):
        '''
        Method computing & visualizing the input data distributions
        associated with the observed latent states.
        '''

        if not os.path.isdir('distributions'):
            os.makedirs('distributions')

        """
        Set features' labels for visualization part
        """     
        self.visibleFeatures = ['v%d' %(i+1) for i in range(self.d.shape[1])]
        self.initFeatures = ['IndexW', 'IndexR', 'IndexN', 'Index1', 'Index2', 'Index3', 'Index4', '0-0.5hz', 'Theta', 'Delta']

        # if self.features=='bands':
        #     self.initFeatures = ['Delta', 'Theta', 'Delta/Theta', 'EMG']
        # elif self.features=='ratios':
        #     Bands = [r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'$\gamma$']
        #     self.initFeatures = []        
        #     for i in range(5):    
        #         for j in range(i+1, 5):
        #             self.initFeatures.append(Bands[i] + '$/$' + Bands[j])     
        #     self.initFeatures.append('EMG')
        # else:
        #     self.initFeatures = ['f%d' %(i+1) for i in range( self.dinit.shape[1] )]      

        """
        Iterate through centroids/latent states and infer the visible
        data distributions
        """

        visibleDistributions = {}
        backProjectedDistributions = {}

        minCovVis = 1.
        maxCovVis = -1.
        minCovBack = 1.
        maxCovBack = -1.        
        for lstate in self.uniqueStates[:, 0]:
            if not os.path.isdir('./distributions/lState%d' %lstate):
                os.makedirs('./distributions/lState%d' %lstate)

            idx = np.where( self.obsKeys[:, 1]==lstate )[0]

            d_visible = self.d[idx, :]
            d_back = self.dinit[idx, :]

            visibleDistributions['lstate%d' %lstate] = {}
            backProjectedDistributions['lstate%d' %lstate] = {}

            visibleDistributions['lstate%d' %lstate]['mean'] = d_visible.mean(axis=0)
            visibleDistributions['lstate%d' %lstate]['cov'] = np.cov(d_visible.T)
            visibleDistributions['lstate%d' %lstate]['data'] = d_visible

            backProjectedDistributions['lstate%d' %lstate]['mean'] = d_back.mean(axis=0)
            backProjectedDistributions['lstate%d' %lstate]['cov'] = np.cov(d_back.T)
            backProjectedDistributions['lstate%d' %lstate]['data'] = d_back

            """
            Look for min/max values of all covariance matrices for scaling matrices
            for visualization if wished
            """         
            if np.cov(d_back.T).min() < minCovBack:
                minCovBack = np.cov(d_back.T).min()
            if np.cov(d_back.T).max() > maxCovBack:
                maxCovBack = np.cov(d_back.T).max()

            if np.cov(d_visible.T).min() < minCovVis:
                minCovVis = np.cov(d_visible.T).min()
            if np.cov(d_visible.T).max() > maxCovVis:
                maxCovVis = np.cov(d_visible.T).max()

            """
            Visualize distributions
            """
            self.visualizeDistribution(d_visible, minCovVis, maxCovVis, self.visibleFeatures, './distributions/lState%d/' %lstate, 'visible')
            self.visualizeDistribution(d_back, minCovVis, maxCovVis, self.initFeatures, './distributions/lState%d/' %lstate, 'backProjected')

        savemat('./distributions/visibleDistributions.mat', mdict={'visibleDistributions':visibleDistributions})
        savemat('./distributions/projecteddBackLogDistr.mat', mdict={'projecteddBackLogDistr':backProjectedDistributions})


    # Function for visualizing the distribution of each latent state over
    # the three sleep stages as a colorMat
    def stageDistribution(self):
        '''
        Method computing the distribution of each latent state over
        the three sleep stages & visualizing it as a colorMat.
        '''

        nonSingle = self.uniqueStates[self.uniqueStates[:, 1] > self.threshold, :]

        ids_NonSingle = np.unique(nonSingle[:, 0])      

        #--- Create save folder :
        if not os.path.isdir('heatMap'):
            os.makedirs('heatMap')

        """
        Compute each latent state's PDF according to how many epochs 
        were manually labeled as Wakefulness, NREM, REM. This can be
        visualized with an RGB color shade.
        """ 

        self.lstateColor, self.lstateCount = self.lstateStageDistribution(self.obsKeys, 1, 3)   
        np.savez_compressed('./heatMap/lstateColor.npz', lstateColor=self.lstateColor, lstateCount=self.lstateCount)

        #-- Re-order matrix for Visualization:
        self.C1 = self.reorderMat(self.lstateColor)
        #-- Visualize array:
        column_labels = ['Wakefulness', 'NREM', 'TS', 'REM']
        self.displayMat(self.lstateColor[self.C1, :], column_labels, './heatMap/heatMap')

        #--- Remove the singleton Latent-states:
        self.idxFramesKeep = []
        for i in range(self.obsKeys.shape[0]):
            if self.obsKeys[i, 1] in ids_NonSingle:
                self.idxFramesKeep.append(i)    

        self.lstateColorThresh, lstateCount2 = self.lstateStageDistribution(self.obsKeys[self.idxFramesKeep, :], 1, 3)

        # savemat:
        #savemat('./heatMap/thresholdedlstatesColor.mat', mdict={'lstateColor':lstateColor2})
        np.savez('./heatMap/thresholdedlstatesColor.npz', lstateColor=self.lstateColorThresh, lstateCount=lstateCount2)

        #-- Re-order matrix for Visualization:
        self.C2 = self.reorderMat(self.lstateColorThresh)
        #-- Visualize array:        
        self.displayMat(self.lstateColorThresh[self.C2,:], column_labels, './heatMap/heatMapThresholded')

        del nonSingle, ids_NonSingle, lstateCount2

    def computeTransitions(self):
        '''
        Method computing the transition probabilities array
        '''

        if not os.path.isdir('transMatrices'):
            os.makedirs('transMatrices')

        """ Compute overall experiment transitions """
        self.obsKeysGroup = self.obsKeys
        self.transitionsMatrix('./transMatrices/')

        del self.obsKeysGroup

        """ Iterate through videoIDs if mulit-videos experiment """     
        if self.multi:          
            for self.group in self.mouseGroups:
                self.obsKeysGroup = self.obsKeys[self.obsKeys[:, self.obsKeys.shape[1]-1] == self.group, :]
                self.transitionsMatrix('./transMatrices/')          

        # del self.obsKeysGroup

    def transitionsMatrix(self, saveDir):
        '''
        Method computing the transition probabilities from each
        latent state to the rest and putting them to a square matrix.
        '''

        """
        Find the unique centroids & their occurence
        """
        centroidsOccurence = np.bincount( self.obsKeys[:, 1] )

        """
        Create the transitions matrix : according to the frame by frame transition
        """
        transMat = np.zeros(( len(self.uniqueStates), len(self.uniqueStates) ), dtype=float32)
        for i in range(0, len( self.obsKeysGroup )-1):
            if ( self.obsKeysGroup[i+1, 0] - self.obsKeysGroup[i, 0] ) == 1:                
                a = self.obsKeysGroup[i, 1]
                b = self.obsKeysGroup[i+1, 1]
                transMat[a,b] = transMat[a,b] + 1

        if self.multi:
            #savemat(saveDir + 'countTransMat%d.mat' %self.group, mdict={'countTransMat':transMat})
            np.savez(saveDir + 'countTransMat%d.npz' %self.group, countTransMat=transMat)
        else:
            #savemat(saveDir + 'countTransMat.mat', mdict={'countTransMat':transMat})
            np.savez(saveDir + 'countTransMat.npz', countTransMat=transMat)

        transMat = transMat/transMat.sum(axis=1)[:,None]
        where_are_NaNs = np.isnan(transMat)
        transMat[where_are_NaNs] = 0
        if self.multi:
            #savemat(saveDir + 'transitionsMat%d.mat' %self.group, mdict={'transitionsMat':transMat})
            np.savez(saveDir + 'transitionsMat%d.npz' %self.group, transitionsMat=transMat)
        else:
            #savemat(saveDir + 'transitionsMat.mat', mdict={'transitionsMat':transMat})
            np.savez(saveDir + 'transitionsMat.npz', transitionsMat=transMat)

        """
        Matrix visualization
        """     
        if self.multi:
            self.displayTransitionsArray(transMat, './transMatrices/transitionsMat%d' %self.group)
        else:
            self.displayTransitionsArray(transMat, './transMatrices/transitionsMat')

        """
        Detect & remove singletons
        """
        idx =  np.where( centroidsOccurence <= self.threshold )[0]

        transMat = np.delete(transMat, idx, 0)
        transMat = np.delete(transMat, idx, 1)

        #savemat(saveDir + 'thresholdedStatesOnMatrix.mat', mdict={'thresholdedStatesOnMatrix': np.delete(self.uniqueStates[:, 0], idx)})
        np.savez(saveDir + 'thresholdedStatesOnMatrix.npz', thresholdedStatesOnMatrix=np.delete(self.uniqueStates[:, 0], idx))


        if self.multi:
            #savemat(saveDir + 'transitionsMatThresholded%d.mat' %self.group, mdict={'transitionsMat':transMat})
            np.savez(saveDir + 'transitionsMatThresholded%d.npz' %self.group, transitionsMat=transMat)
        else:
            #savemat(saveDir + 'transitionsMatThresholded.mat', mdict={'transitionsMat':transMat})
            np.savez(saveDir + 'transitionsMatThresholded.npz', transitionsMat=transMat)

        if self.multi:
            self.displayTransitionsArray(transMat, './transMatrices/transitionsMatThresholded%d' %self.group)
        else:
            self.displayTransitionsArray(transMat, './transMatrices/transitionsMatThresholded')

        del transMat, idx, centroidsOccurence, where_are_NaNs       

    # Function for computing and visualizing entropy & Mutual Information
    def entropyMIcontrol(self):
        '''
        Method computing the entropy and mutual information of the
        observed latent states.
        '''

        if not os.path.isdir('entropyMI'):
            os.makedirs('entropyMI')

        """ Latent states' PDF: 
                                lstatePDF = p(stage | lstate) """
        lstatePDF = np.concatenate((self.uniqueStates[:, :2], self.lstateColor), axis=1)
        countsArray = np.concatenate((self.uniqueStates[:, :2], self.lstateCount), axis=1)

        """ Compute the Marginal of each Stage (class marginal)
            ..where stagePDF = p(lstate | stage) """
        stagePDF = self.lstateCount
        stagePDF = stagePDF.astype(float32)     
        stagePDF = stagePDF/stagePDF.sum(axis=0)        
        stagePDF = np.concatenate((self.uniqueStates[:, :2], stagePDF), axis=1)

        np.savez_compressed('./entropyMI/lstatePDF.npz', lstatePDF=lstatePDF, lstateCount=countsArray)
        np.savez('./entropyMI/stagePDF.npz', stagePDF=stagePDF)

        #savemat('./entropy/lstatePDF.mat', mdict={'lstatePDF':lstatePDF})
        #savemat('./entropy/lstateCount.mat', mdict={'lstateCount':countsArray})
        #savemat('./entropy/stagePDF.mat', mdict={'stagePDF':stagePDF})

        """
        Remove latent states smaller than the desired threshold
        """     
        idx = np.where( self.uniqueStates[:, 1] <= self.threshold )[0]
        np.savez('./entropyMI/removedStates.npz', removedStates=self.uniqueStates[idx, 0])
        #savemat('./entropy/removedStates.mat', mdict={'removedStates':self.uniqueStates[idx, 0]})

        lstatePDF = np.delete(lstatePDF, idx, 0)
        countsArray = np.delete(countsArray, idx, 0)
        stagePDF = np.delete(stagePDF, idx, 0)

        """ Compute the Mutual-Information : I(lstates; stages) """
        MI = self.mutualInformation(countsArray[:, 2:])

        with open ('./entropyMI/mutualInformation.txt','w') as f:
            f.write("\n The mutual information is : %s bits" %MI)
            f.close()

        """ Compute the Entropy of each latent state """
        lstateEntropy = self.variableEntropy(lstatePDF[:, 2:])
        lstateEntropy = np.concatenate((lstatePDF[:, :2], lstateEntropy), axis=1)

        np.savez('./entropyMI/lstatesEntropy.npz', lstateEntropy=lstateEntropy)
        #savemat('./entropy/lstateEntropy.mat', mdict={'lstateEntropy':lstateEntropy})

        self.entropiesHistogram(lstateEntropy[:, lstateEntropy.shape[1]-1], 'entropyMI')

        """ Compute the Entropy of each stage """
        stagesH, Hx = self.stageEntropy(countsArray[:, 2:], './entropyMI/mutualInformation.txt')

        np.savez('./entropyMI/stageEntropy.npz', stageEntropy=stagesH)
        #savemat('./entropy/stageEntropy.mat', mdict={'stageEntropy':stagesH})

        """ Compute the mutual information for each stage """
        MI_stage = self.mutualInformation_perStage(countsArray[:, 2:])

        with open ('./entropyMI/mutualInformation.txt','a') as f:
            f.write("\n Stage Entropy:")
            f.write("\n Wakefulness = %s" %stagesH[0])
            f.write("\n NREM = %s" %stagesH[1])
            f.write("\n TS = %s" %stagesH[2])
            f.write("\n REM = %s\n" %stagesH[3])
            f.write("\n Stage Normalized MI:")
            f.write("\n The MI/H(wake) = %s" %(MI_stage[0]/stagesH[0]))
            f.write("\n The MI/H(nrem) = %s" %(MI_stage[1]/stagesH[1]))
            f.write("\n The MI/H(TS) = %s" %(MI_stage[2]/stagesH[2]))
            f.write("\n The MI/H(rem) = %s\n" %(MI_stage[3]/stagesH[3]))
            f.write("\n Overall Normalized MI:")
            f.write("\n The MI/H(Stage) = %s" %(MI/Hx))
            f.close()

        self.MI_stimulusH_barPlot(MI_stage, stagesH, MI/Hx, 'entropyMI')

    def mutualInformation(self, Count):
        '''
        Computes the mutual information

        MI(lstates; stages) = 
            sum( p(lstate, stage) * log2 ( p(lstate, stage) / ( p(lstate)*p*(stage) ) ) 
        '''
        Count = Count.astype(float32)

        """ total population """
        Total_sum = Count.sum()
        """ p(latent_state) = sum_over_lstate / total_population """
        p_latent = Count.sum(axis=1)/Total_sum
        """ p(latent_stage) = sum_over_stage / total_population """ 
        p_stage = Count.sum(axis=0)/Total_sum

        MI = 0              
        """ Iterate through latent states """
        for i in range(Count.shape[0]):
            for j in range(Count.shape[1]):
                if Count[i,j] != 0.:

                    """ Compute p(lstate, stage) """
                    p_li_sj = Count[i,j]/Total_sum

                    denominator = p_latent[i]*p_stage[j]

                    MI += p_li_sj*np.log2(p_li_sj/denominator)      

        return MI

    def mutualInformation_perStage(self, Count):
        '''
        Computes the mutual information

        MI(lstates; stages) = 
            sum( p(lstate, stage) * log2 ( p(lstate, stage) / ( p(lstate)*p*(stage) ) ) 
        '''
        Count = Count.astype(float32)

        """ total population """
        Total_sum = Count.sum()
        """ p(latent_state) = sum_over_lstate / total_population """
        p_latent = Count.sum(axis=1)/Total_sum
        """ p(latent_stage) = sum_over_stage / total_population """ 
        p_stage = Count.sum(axis=0)/Total_sum

        MI = []
        """ Iterate through stages """
        for j in range(Count.shape[1]):
            MI_j = 0
            """ Iterate through latent states """
            for i in range(Count.shape[0]):
                if Count[i,j] != 0.:
                    """ Compute p(lstate, stage) """
                    p_li_sj = Count[i,j]/Total_sum                  
                    denominator = p_latent[i]*p_stage[j]                    
                    MI_j += p_li_sj*np.log2(p_li_sj/denominator)
            MI.append(MI_j)

        return np.asarray(MI)

    def variableEntropy(self, latent_PDF):
        '''
        Computes the entropy of a discrete random variable in bits.

        H(X) = -sum( p(x)*log2(p(x)), for x in X.
        '''
        latent_PDF = np.insert(latent_PDF, latent_PDF.shape[1], 0, axis=1)

        """
        Iterate through latent states
        """
        for p in range(latent_PDF.shape[0]):
            Hx = 0.
            for i in range(latent_PDF.shape[1]):
                if latent_PDF[p, i] != 0.:
                    Hx += -latent_PDF[p,i]*np.log2(latent_PDF[p,i])

            latent_PDF[p, latent_PDF.shape[1]-1] = Hx           

        return latent_PDF

    def stageEntropy(self, Count, tetFile):
        '''
        Random variable = stage (wake, nrem, rem)

        Computes the Entropy: 
            H(stage) = -sum_over_stages(p(si)*log2(p(si)))
        '''

        Count = Count.astype(float32)

        """ total population """
        Total_sum = Count.sum()

        """ p(latent_stage) = sum_over_stage / total_population """ 
        p_stage = Count.sum(axis=0)/Total_sum
        #print("p_stage : ", p_stage)

        """
        Iterate through Stages (which represent our random variable)
        """
        Hx = 0.
        stagesH = []
        for j in range(len(p_stage)):
            if p_stage[j] != 0.:
                H_step = -p_stage[j]*np.log2(p_stage[j])
                Hx += H_step
            with open (tetFile, 'a') as f:
                f.write("\n Stage %d Entropy : %s" %(j, H_step))
                f.close()

            stagesH.append(H_step)
        with open (tetFile, 'a') as f:
            f.write("\n Variable Stage Entropy : %s" %Hx)
            f.close()

        return np.asarray(stagesH), Hx


    # HeatMap function : Computes the RGB color for each latent state
    def lstateStageDistribution(self, inputMat, columnState, columnStage):
        '''
        Method computing the probability of each latent state to 
        belong to each of the 3 sleep stages.
        Returns a matrix of discrete probability distributions associated
        with an RGB color.
        '''

        lstates = np.unique(inputMat[:, columnState])

        latent_color = []
        counts = []

        """
        Iterate through latent states
        """
        for i in lstates:
            """
            and find the corresponding to each one of them obsKeys
            """
            obsKeys = inputMat[inputMat[:, columnState] == i, :]

            # length current latent's :
            lstatePopulation = len(obsKeys)

            # find the per class samples :          
            length_awake = len(np.where(obsKeys[:, columnStage]==1)[0])         
            length_nrem = len(np.where(obsKeys[:, columnStage]==3)[0])  
            length_TS = len(np.where(obsKeys[:, columnStage]==4)[0])          
            length_rem = len(np.where(obsKeys[:, columnStage]==5)[0])

            vec_len = (length_awake/lstatePopulation, length_nrem/lstatePopulation, length_TS/lstatePopulation, length_rem/lstatePopulation)
            latent_color.append(vec_len)
            counts.append([length_awake, length_nrem, length_TS, length_rem])
        return np.asarray(latent_color), np.asarray(counts)

    def prototypesHistogram(self):
        '''
        Method computing the histogram over the desired latent states &
        reordering it for visualization.

        It does it for the data of the overall experiment, as well as 
        per mouse group and per mouse in case of multi-subject analysis.
        '''
        # we have from previous functions: self.C2, self.lstateColorThresh, self.idxFramesKeep
        if not os.path.isdir('statesHistogram'):
            os.makedirs('statesHistogram')

        lstatesOfInterest = self.uniqueStates[self.uniqueStates[:, 1] > self.threshold, 0]

        """
        Compute histogram over latent states of interest
        """
        self.centroidsHist = np.zeros(( len(lstatesOfInterest), 3 ), dtype=float32)
        i = 0
        for ci in lstatesOfInterest:
            idx = np.where(self.obsKeys[:, 1] == ci)[0]     

            self.centroidsHist[i,0] = len(idx)
            self.centroidsHist[i,1] = len(idx)
            self.centroidsHist[i,2] = ci
            i += 1

        """
        Normalize histogram
        """
        if self.norm == "L2":
            print("Normalizing histogram by L2 norm..")
            self.centroidsHist[:, 1] = self.centroidsHist[:, 1] / np.linalg.norm(self.centroidsHist[:, 1], 2)
        else:
            print("Normalizing histogram by strain length..")
            self.centroidsHist[:, 1] = self.centroidsHist[:, 1] / len(self.obsKeys)
        #savemat('./statesHistogram/centroidsHist.mat', mdict={'centroidsHist':self.centroidsHist})
        np.savez('./statesHistogram/thresholdedStatesHist.npz', thresholdedStatesHist=self.centroidsHist)

        self.plotHistogram('./statesHistogram/', 'Exp', .9)

        if self.multi:      
            """
            Compute the per strain histogram over latent states of interest
            """

            print("Computing the per strain histogram..")

            for strain in self.mouseGroups:

                print("Strain %d" %strain)

                self.centroidsHist = np.zeros(( len(lstatesOfInterest), 3 ), dtype=float32)
                strainFrames = self.obsKeys[np.where(self.obsKeys[:, self.obsKeys.shape[1]-1] == strain)[0], :]
                i = 0
                for ci in lstatesOfInterest:
                    idx = np.where(strainFrames[:, 1] == ci)[0]     

                    self.centroidsHist[i, 0] = len(idx)
                    self.centroidsHist[i, 1] = len(idx)
                    self.centroidsHist[i, 2] = ci
                    i += 1

                """
                Normalize histogram
                """
                if self.norm == "L2":
                    print("Normalizing histogram by L2 norm..")
                    self.centroidsHist[:,1] = self.centroidsHist[:,1] / np.linalg.norm(self.centroidsHist[:,1], 2)
                else:
                    print("Normalizing histogram by strain length..")
                    self.centroidsHist[:,1] = self.centroidsHist[:,1] / len(strainFrames)

                #savemat('./statesHistogram/centroidsHist%d.mat' %strain, mdict={'centroidsHist':self.centroidsHist})
                np.savez('./statesHistogram/thresholdedStatesHist%d.npz' %strain, thresholdedStatesHist=self.centroidsHist)

                self.plotHistogram('./statesHistogram/', 'Group%d' %strain, .9)

            """
            Create the per subject histogram
            """         
            if not os.path.isdir('./statesHistogram/perSubject'):
                os.makedirs('./statesHistogram/perSubject')

            print("Computing the per subject histogram..")

            for subject in self.subjects:

                print("Subject %d" %subject)

                self.centroidsHist = np.zeros(( len(lstatesOfInterest), 3 ), dtype=float32)
                subjFrames = self.obsKeys[np.where(self.obsKeys[:, self.obsKeys.shape[1]-2] == subject)[0], :]
                i = 0
                for ci in lstatesOfInterest:
                    idx = np.where(subjFrames[:, 1] == ci)[0]       

                    self.centroidsHist[i, 0] = len(idx)
                    self.centroidsHist[i, 1] = len(idx)
                    self.centroidsHist[i, 2] = ci
                    i += 1

                """
                Normalize histogram
                """
                if self.norm == "L2":
                    print("Normalizing histogram by L2 norm..")
                    self.centroidsHist[:,1] = self.centroidsHist[:,1] / np.linalg.norm(self.centroidsHist[:,1], 2)
                else:
                    print("Normalizing histogram by strain length..")
                    self.centroidsHist[:,1] = self.centroidsHist[:,1] / len(subjFrames)

                np.savez('./statesHistogram/perSubject/thresholdedStatesHist%d.npz' %subject, thresholdedStatesHist=self.centroidsHist)

                self.plotHistogram('./statesHistogram/perSubject/', 'S%d' %subject, 1.) 

    # Function for re-ordering a matrix with linkage:
    def reorderMat(self, matrixToCluster):
        '''
        Method re-organizing a matrix according to linkage.
        '''     
        aux_linkage = linkage(matrixToCluster, 'average', metric='euclidean')       
        R = dendrogram(aux_linkage, p=0, count_sort='ascending')

        return np.asarray(R['ivl']).astype(int)

    #-- Visualization Functions

    # Function of visualizing the histogram of the latent states
    def statesHistogram(self):
        '''
        Methos plotting the histogram of the latent states, i.o.w.
        how many samples fall in each unique latent state.
        '''

        #--- Create save folder :
        if not os.path.isdir('statesHistogram'):
            os.makedirs('statesHistogram')
        plt.style.use('bmh')
        colors = list(plt.rcParams['axes.prop_cycle']) #= cycler(color='bgrcmyk')

        f1 = plt.figure()
        plt.suptitle('Number of Latent States: %d' %self.uniqueStates.shape[0])
        ax = f1.add_subplot(111)

        width = .8
        ax.bar(self.uniqueStates[:, 0], self.uniqueStates[:, 1], width, color=colors[1]['color'], edgecolor = "none")   
        ax.set_xlabel('Latent States')
        ax.set_ylabel('Number of frames')
        #plt.legend(loc='best') 
        ax.set_xlim([0, self.uniqueStates.shape[0]])
        ax.set_ylim([0, np.max(self.uniqueStates[:, 1])])

        # yint = []
        # locs, l = plt.yticks()
        # for each in locs:
            # yint.append( round(each, 2) )
        # plt.yticks(yint)

        # ax1.set_yticklabels(ax1.get_yticks(), fontweight='bold', fontsize=30)

        filename = 'lStatesHistogram.png'
        filename = os.path.join('./statesHistogram/', filename)
        f1.savefig(filename)

    # Function for dispaying an array
    def displayMat(self, matrixToDisplay, column_labels, filename):
        '''
        Function for displaying an array.
        ''' 

        fig, ax = plt.subplots(figsize=(12,10))
        plt.style.use('bmh')
        heatmap = ax.pcolor(matrixToDisplay, cmap='RdBu_r')
        plt.axis([0, matrixToDisplay.shape[1], 0, matrixToDisplay.shape[0]])

        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(matrixToDisplay.shape[1])+0.5, minor=False)
        ax.set_yticks(np.linspace(0, matrixToDisplay.shape[0], num=10, dtype=np.int32), minor=False)

        plt.tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom='off',      # ticks along the bottom edge are off
                        top='off',         # ticks along the top edge are off
                        labelbottom='on') # labels along the bottom edge are off

        plt.tick_params(
                        axis='y',          # changes apply to the y-axis
                        which='both',      # both major and minor ticks are affected
                        left='off',      # ticks along the bottom edge are off
                        right='off',         # ticks along the top edge are off
                        labelleft='on') # labels along the bottom edge are off


        ax.set_xticklabels(column_labels, minor=False, fontsize=25, fontweight='bold')#, color='w')
        ax.set_yticklabels(ax.get_yticks(), fontweight='bold', fontsize=15)#, color='w')
        ax.set_ylabel("Latent-States", fontsize=25, fontweight='bold')#, color='w')


        divider4 = make_axes_locatable(ax)
        cax4 = divider4.append_axes("right", size="5%", pad=0.2)

        cb = plt.colorbar(heatmap, cax=cax4)
        for l in cb.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_fontsize(15)

        fig.savefig(filename + '.jpeg', format='jpeg', transparent=True, dpi=100)
        print("filename :", filename)

        plt.close(fig)

    def displayTransitionsArray(self, A, filename):
        '''
        Method displaying an array.
        '''

        f1 = plt.figure()
        ax1 = f1.add_subplot(1,1,1)
        ax1.set_aspect('equal')
        ax1.patch.set_facecolor('None')
        ax1.grid(False)

        norm = matplotlib.colors.Normalize(vmin=0., vmax=1.)    

        hmat = plt.pcolor(A, norm=norm, cmap='RdBu_r')
        plt.colorbar(hmat)

        plt.xlim(0, A.shape[0])
        plt.ylim(0, A.shape[1])
        f1.savefig(filename + '.png', transparent=True, dpi=100)

    def BoxPlotsSimple(self, d_to_plot_1, fig_path, population, i, labels_1, length_awake, length_nrem, length_TS, length_rem, range_1, id_bin):
        '''
        Method visualizing the boxplots of the LOG initial EEG
        data mapping to eacch latent state.
        '''
        def binary_to_decimal(binary):
            binary_string = ''.join(str(bit) for bit in binary)
            return int(binary_string, 2)
        plt.style.use('bmh')
        
        colors = list(plt.rcParams['axes.prop_cycle'])

        fig, ax1 = plt.subplots(figsize=(15, 10), constrained_layout=True)        
        id_dec_bin = binary_to_decimal(id_bin)
        plt.title(f'LS {i}_{id_dec_bin} - Total: {population} epochs\n'
             f'WAKE: {length_awake*100:.1f}%, NREM: {length_nrem*100:.1f}%\n'
             f'TS: {length_TS*100:.1f}%,REM: {length_rem*100:.1f}%',
             y=1.001, fontsize=35, fontweight='bold')
        #
        print(f"ax1.get_xticks(): {ax1.get_xticks()}")
        bp1 = ax1.boxplot(d_to_plot_1, patch_artist=True)

        print(f"ax1.get_xticks(): {ax1.get_xticks()}")
        ax1.grid(False)
        ax1.patch.set_facecolor('0.85')
        ax1.patch.set_alpha(0.5)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_linewidth(5)
        ax1.spines['bottom'].set_color('k')
        ax1.spines['left'].set_linewidth(5)
        ax1.spines['left'].set_color('k')
        ax1.set_ylabel('Log Power', fontweight='bold', fontsize=35)
        ax1.set_ylim(range_1)
        ax1.set_yticks(np.linspace(range_1[0], range_1[1], num=int(range_1[1]-range_1[0])+2, dtype=np.int32), minor=False)
        ax1.set_yticklabels(ax1.get_yticks(), fontweight='bold', fontsize=35)
        ax1.xaxis.set_ticks_position('none')
        ax1.yaxis.set_ticks_position('none')

        for tl in ax1.get_yticklabels():
            tl.set_color('#b2182b')

        
        print("Length of labels_1:", len(labels_1))
        tick_locs = range(len(labels_1))
        ax1.set_xticks(tick_locs)
        print("Number of tick locations on ax1:", len(ax1.get_xticks()))
        print("tick locations on ax1:", ax1.get_xticks())
        
        xtickNames = ax1.set_xticklabels(labels_1)
        plt.setp(xtickNames, rotation=90, fontsize=35, fontweight='bold')

        plt.setp(bp1['boxes'],            # customise box appearance
                 color='#b2182b',         # outline colour
                 linewidth=2.)

        plt.setp(bp1['whiskers'], color='#999999', linewidth=2.)
        plt.setp(bp1['fliers'], color='k', marker='+')

        plt.setp(bp1['medians'],          # customize median lines
                 color='#1a1a1a',         # line colour
                 linewidth=2.)            # line thickness


        fname = 'lstate%d_%d.jpeg' %(i, id_dec_bin)
        fname = os.path.join(fig_path, fname)
        fig.savefig(fname, format='jpeg', transparent=True, dpi=100)
        plt.close(fig)

    def MI_stimulusH_barPlot(self, MI_stage, stagesH, overAll, saveDir):
        '''
        Function for visualizing as bar plot the normalized mutual information.
        M.I. / stage_entropy
        '''

        ind = 0.75
        ind2 = np.arange(5)

        plt.style.use('bmh')
        colors = list(plt.rcParams['axes.prop_cycle'])

        fig = plt.figure(figsize=(25,20), frameon=False)
        ax1 = fig.add_subplot(111)
        plt.grid(False)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_linewidth(15)
        ax1.spines['bottom'].set_color('k')
        ax1.spines['left'].set_linewidth(15)
        ax1.spines['left'].set_color('k')
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        width = 0.5
        bar_width = 0.43
        ax1.bar(ind, MI_stage[0]/stagesH[0], bar_width, color=colors[9]['color'], edgecolor = "none", label="Wakefulness")
        ax1.bar(ind+width, MI_stage[1]/stagesH[1], bar_width, color=colors[8]['color'], edgecolor = "none", label="NREM")
        ax1.bar(ind+2*width, MI_stage[2]/stagesH[2], bar_width, color=colors[3]['color'], edgecolor = "none", label="TS")
        ax1.bar(ind+3*width, MI_stage[3]/stagesH[3], bar_width, color=colors[1]['color'], edgecolor = "none", label="REM")
        ax1.bar(ind+0.25+5*width, overAll, bar_width, color='#40e0d0', edgecolor = "none", label="OverAll")


        ax1.set_ylabel('M.I. / Stage Entropy', fontweight='bold', fontsize=70)
        xTickMarks = ['', 'Stages', '', 'All Stages', '']
        ax1.set_xlim([0, 5])
        ax1.set_xticks(ind2 + bar_width/2. + width/2.)
        xtickNames = ax1.set_xticklabels(xTickMarks, fontweight='bold')

        plt.setp(xtickNames, fontsize=70)
        ax1.xaxis.set_ticks_position('none')
        ax1.yaxis.set_ticks_position('none')        

        yint = []
        locs, l = plt.yticks()
        for each in locs:
            yint.append( round(each, 2) )
        plt.yticks(yint)

        ax1.set_yticklabels(ax1.get_yticks(), fontweight='bold', fontsize=50)

        ax1.xaxis.labelpad = 30
        ax1.yaxis.labelpad = 20
        ax1.tick_params(direction='out', pad=20)
        plt.draw()      

        legend_properties = {'weight':'bold', 'size':65}
        plt.legend(frameon=False, borderaxespad=0., prop=legend_properties, bbox_to_anchor=(.77, 1.05), loc=2)

        fname = 'MIstageH.jpeg'
        fname = os.path.join(saveDir, fname)
        fig.savefig(fname, format='jpeg', transparent=True, dpi=100)
        plt.close(fig)

    def entropiesHistogram(self, entropies, saveDir):
        '''
        Function for visualizing the entropies' histogram.
        '''

        plt.style.use('bmh')
        colors = list(plt.rcParams['axes.prop_cycle'])      

        f1 = plt.figure(figsize=(12,10), frameon=False)
        plt.grid(False)

        ax = f1.add_subplot(111)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('none')
        width = 0.8

        graph_maximum = entropies.max()
        graph_minimum = entropies.min()

        bin_size = .01; min_edge = graph_minimum; max_edge = graph_maximum

        N = (max_edge-min_edge)/bin_size; Nplus1 = N + 1
        print(f"calling np.linspace with {min_edge}, {max_edge}, {Nplus1}")
        bin_list = np.linspace(int(min_edge), int(max_edge), int(Nplus1))

        [n, bins, patches] = plt.hist(entropies, bin_list, range=(graph_minimum, graph_maximum), edgecolor = 'none', facecolor = colors[1]['color'])

        yint = []
        locs, l = plt.xticks()
        for each in locs:
            yint.append( round(each, 2) )
        plt.xticks(yint)
        ax.set_xticklabels(ax.get_xticks(), fontweight='bold', fontsize=17)
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20      

        legend_properties = {'weight':'bold', 'size':30}
        legend = plt.legend(prop=legend_properties)
        frame = legend.get_frame().set_alpha(0)

        plt.title('Histogram of Entropies', fontsize=25, fontweight='bold')
        plt.xlabel('Entropy', fontsize=25, fontweight='bold')
        plt.ylabel('Number of Latent States', fontsize=25, fontweight='bold')
        plt.legend()

        fname = 'entropiesHist.png'
        fname = os.path.join(saveDir, fname)
        plt.savefig(fname, transparent=True, dpi=100)
        plt.close(f1)       

    def plotHistogram(self, saveDir, name, yLim):       
        """
        Function visualizing the histogram over the latent states.
        """

        ind = np.arange( len(self.uniqueStates[self.uniqueStates[:, 1] > self.threshold, 0]) )

        fig = plt.figure(figsize=(12,10), frameon=False)
        ax1 = fig.add_subplot(111)
        fig.suptitle('Histogram over the Latent States', fontsize=20, fontweight='bold')

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)
        ax1.spines['left'].set_visible(False)

        width = 0.8

        colors = self.lstateColorThresh[self.C2, :]     
        counts = self.centroidsHist[:, 1][self.C2]      
        for i in range( len(self.uniqueStates[self.uniqueStates[:, 1] > self.threshold, 0]) ):
            ax1.bar(ind[i], counts[i], width, color=(colors[i, 2], colors[i, 1], colors[i, 0]), edgecolor = "none")

        if self.norm == "L2":
            ax1.set_ylabel('L2 Normalized Count', fontweight='bold', fontsize=20)
        else:
            ax1.set_ylabel('Normalized Count', fontweight='bold', fontsize=20)
        ax1.set_xlabel('Latent States', fontweight='bold', fontsize=20)
        ax1.xaxis.set_ticks_position('none')
        ax1.yaxis.set_ticks_position('none')    

        yint = []
        locs, l = plt.yticks()
        for each in locs:
            yint.append( round(each, 2) )
        plt.yticks(yint)    
        ax1.set_yticklabels(ax1.get_yticks(), fontweight='bold', fontsize=17)

        ax1.set_xlim([0, len(self.uniqueStates[self.uniqueStates[:, 1] > self.threshold, 0])-1])        
        ax1.set_xticks(np.linspace(0, len(self.uniqueStates[self.uniqueStates[:, 1] > self.threshold, 0]), num=10, dtype=np.int32), minor=False)
        ax1.set_xticklabels(ax1.get_xticks(), fontweight='bold', fontsize=17)       

        red_patch = mpatches.Patch(color=(0.0, 0.0, 1.0), label='Wakefulness')
        green_patch = mpatches.Patch(color=(0.0, 1.0, 0.0), label='NREM')
        grey_patch = mpatches.Patch(color=(0.5, 0.5, 0.5), label='TS')
        blue_patch = mpatches.Patch(color=(1.0, 0.0, 0.0), label='REM')
        legend_properties = {'weight':'bold'}
        legend = plt.legend(handles=[red_patch, green_patch, grey_patch, blue_patch], borderaxespad=0., fontsize=30, prop=legend_properties)
        frame = legend.get_frame().set_alpha(0)

        fname = 'coloredHistogram' + name + '.png'      
        fname = os.path.join(saveDir, fname)
        plt.savefig(fname, transparent=True, dpi=100)       
        plt.close(fig)

    def visualizeDistribution(self, d, A, B, labels, saveDir, filename):
        '''
        FMethod visualizing a data distribution as a covariance matrix
        together with its mean values.
        '''

        dCov = np.cov(d.T)

        f = plt.figure(figsize=(10,10))
        ax = f.add_subplot(111)

        hmat = plt.pcolor(dCov, cmap='RdBu_r')

        """
        set the limits of the plot to the limits of the data
        """
        plt.axis([0, dCov.shape[0], 0, dCov.shape[0]])

        ax.set_xticks(np.arange(dCov.shape[0])+0.5, minor=False)
        ax.set_yticks(np.arange(dCov.shape[1])+0.5, minor=False)

        """
        want a more natural, table-like display
        """
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(labels, minor=False, fontweight='bold', fontsize=20)
        ax.set_yticklabels(labels, minor=False, fontweight='bold', fontsize=20)

        plt.tick_params(
                            axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom='off',      # ticks along the bottom edge are off
                            top='off',         # ticks along the top edge are off
                            labelbottom='off') # labels along the bottom edge are off
        plt.tick_params(
                            axis='y',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            left='off',      # ticks along the bottom edge are off
                            right='off',         # ticks along the top edge are off
                            labelleft='on') # labels along the bottom edge are off

        divider4 = make_axes_locatable(ax)
        cax4 = divider4.append_axes("right", size="5%", pad=0.25)


        cb = plt.colorbar(hmat, cax=cax4)

        for l in cb.ax.yaxis.get_ticklabels():
            l.set_weight("bold")
            l.set_fontsize(15)

        M = d.mean(axis=0)
        M2 = []
        for mi in range(len(M)):
            M2.append(round(M[mi],2))

        if self.features=='bands':
            f.text(0.45, .07, "Mean:", fontweight='bold', fontsize=20)
            f.text(0.25, .04, "%s" %M2, fontweight='bold', fontsize=20)
        else:
            f.text(0.45, .07, "Mean:", fontweight='bold', fontsize=20)
            f.text(0.25, .04, "%s" %M2[:7], fontweight='bold', fontsize=20)
            f.text(0.3, .01, "...%s" %M2[7::], fontweight='bold', fontsize=20)

        fname = os.path.join(saveDir, filename)
        f.savefig(fname, transparent=True, dpi=100)
        plt.close(f)
    
