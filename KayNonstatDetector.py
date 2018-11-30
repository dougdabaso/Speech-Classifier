
def KayNonstatDetect(MySignalInTime,pfa):

    Nx = len(MySignalInTime) # Length of the original signal
    n = range(0,Nx) # time points that will be used for the analysis

    # Comments in the original paper [Skay2008] about the basis functions.
    # For basis functions we will choose those corresponding to a second - order polynomial or the set[1, n, n ^ 2].

    p0 = ones((Nx,1))/sqrt(Nx)
    mu1 = mean(n)
    mu2 = mean((n - mu1)**2)
    mu3 = mean((n - mu1)**3)
    p1 = (n - mu1)/linalg.norm(n - mu1)
    p2 = (n - mu1)**2 - (mu3 / mu2)*(n - mu1) - mu2
    p2 = p2/linalg.norm(p2)

    # Following the example in Section 3.2 of [Skay2008]:
    y = zeros([2, 1])
    rao_test_statistic = []

    for N in range(3,Nx):

        # Using the covariance method estimate we have
        ahat_times_c = -( sum(MySignalInTime[range(0,N-1)]*MySignalInTime[range(1,N)])/sum(MySignalInTime[range(0,N-1)]**2) )

        # Equation for u_hat[n]
        uhat = MySignalInTime[range(1,N)] + ahat_times_c*MySignalInTime[range(0,N-1)]

        # Equation for b0hat_times_c
        b0hat_times_c = sqrt(sum(uhat**2)/(N-1))

        y[0,:] = sum(p1[range(0,N-1)]*uhat*MySignalInTime[range(0,N-1)])/(b0hat_times_c**2)
        y[1,:] = sum(p2[range(0,N-1)]*uhat*MySignalInTime[range(0,N-1)])/(b0hat_times_c**2)

       # The Rao test statistics Tn(x) would be computed for each value of N, since N increased to assess the maximum
       # data record length possible for the stationarity to hold

        if (N == 3):
            rao_test_statistic.append(0)
            rao_test_statistic.append(0)
            rao_test_statistic.append(0)
            rao_test_statistic.append((1 - (ahat_times_c)**2)*(dot(y[:,0],y[:,0])))
        else:
            rao_test_statistic.append((1 - (ahat_times_c)**2)*(dot(y[:,0],y[:,0])))

        # Note: The expression for the Rao test statistic shown above is presented after the equation (20) in the paper.

    threshold = 2*log(1/pfa)
    outcome_vec = rao_test_statistic >= threshold

    if (sum(outcome_vec)==0):
       outcome = 0
    else:
       outcome = 1

    return outcome, outcome_vec, rao_test_statistic, threshold
















# # Importing standard modules
# import os
# import matplotlib.pyplot as plt
# from scipy.stats import mode
# from numpy import *
#
# # For this proof of concept, we can use pyhht package (https://pyhht.readthedocs.io) which offers implementations of EMD
# #  and time-frequency (TF) transforms (which are the two approaches we are going to use a priori)
# from pyhht import EMD # Importing EMD function from pyhht module
# from pyhht.visualization import plot_imfs
# from scipy.io.wavfile import read as readwav # We can use scipy to read WAVs
# from scipy import signal # getting downsample function from scipy module
# from supporting_functions import *
#
#
# # 1) Defining paths and variable names ---------------------------------------------------------------------------------
#
# MasterPath = 'C:\\Users\\Douglas\\Oya\\AVA Dataset\\python scripts'
# FolderWithAudio = 'C:\\Users\\Douglas\\Oya\\AVA Dataset\\audio'
# FileNameExampleNoSpeech = '5BDj0ow5hnA_NO_SPEECH_1.wav'
# FileNameExampleSpeechWithNoise = '5BDj0ow5hnA_SPEECH_WITH_NOISE_1.wav'
# FileNameExampleCleanSpeech = '5BDj0ow5hnA_CLEAN_SPEECH_1.wav'
#
# pfa = 0.05 # to define the threshold for Kay method
#
# # Required by now for downsampling audio signals to make the EMD calculation a little faster. The resampled signal x
# #  starts at the same value as x but is sampled with a spacing of len(x)/DownSamplingFactor*(spacing of x).
# NumberOfSamplesDownSampledSignal = 10000 # We can thing if it is worth to do so later
# thresh_range = linspace(0.01,1,1000)  # This is the range of threshold considered for computing the most relevant IMFs
#                                         # for more info check one of my papers published in ICASSP 2014 [2]
#
#
#
#
# # i) no speech
# NoSpeech = readwav(FileNameExampleNoSpeech) # Reading WAV signal (NoSpeech)
# NoSpeechSignal = array(NoSpeech[1],dtype=float) # Transforming data to a numpy array
# NoSpeechSignal = NoSpeechSignal[:,0] # WAV is being read as 2-channel signal, we can choose the first one by now
# NoSpeechSignalDownsampled = signal.resample(NoSpeechSignal, NumberOfSamplesDownSampledSignal) # Downsampling original signal
# NoSpeechSignalEMD = EMD(NoSpeechSignalDownsampled,t=None, threshold_1=0.05, threshold_2=0.5, alpha=0.05, ndirs=4, fixe=0, maxiter=5000, fixe_h=0, n_imfs=0, nbsym=2) # Performing the Empirical Mode Decomposition
# NoSpeechSignalIMFs = NoSpeechSignalEMD.decompose() # Gathering the IMFs obtained from the decomposition
# MaskRelevantIMFsNoSpeech = relevant_IMFs(NoSpeechSignalIMFs,NoSpeechSignalDownsampled,thresh_range)
# MaskRelevantIMFsNoSpeech = transpose(array(MaskRelevantIMFsNoSpeech.astype('bool')))
# NoSpeechSignalIMFs2 = NoSpeechSignalIMFs[MaskRelevantIMFsNoSpeech]
# DenoisedNoSpeechSignal = sum(NoSpeechSignalIMFs2,0) # Sum up the remaining IMFs to obtain denoised signal
# NoSpeechSignal_outcome, NoSpeechSignal_outcome_vec, NoSpeechSignal_rao_test_statistic, NoSpeechSignal_threshold = KayNonstatDetect(DenoisedNoSpeechSignal,pfa)
#
#
#
# # ii) speech with noise
# SpeechWithNoise = readwav(FileNameExampleSpeechWithNoise) # Reading WAV signal (SpeechWithNoise)
# SpeechWithNoiseSignal = array(SpeechWithNoise[1],dtype=float) # Transforming data to a numpy array
# SpeechWithNoiseSignal = SpeechWithNoiseSignal[:,0] # WAV is being read as 2-channel signal, we can choose the first one by now
# SpeechWithNoiseSignalDownsampled = signal.resample(SpeechWithNoiseSignal, NumberOfSamplesDownSampledSignal) # Downsampling original signal
# SpeechWithNoiseSignalEMD = EMD(SpeechWithNoiseSignalDownsampled,t=None, threshold_1=0.05, threshold_2=0.5, alpha=0.05, ndirs=4, fixe=0, maxiter=5000, fixe_h=0, n_imfs=0, nbsym=2) # Performing the Empirical Mode Decomposition
# SpeechWithNoiseSignalIMFs = SpeechWithNoiseSignalEMD.decompose() # Gathering the IMFs obtained from the decomposition
# MaskRelevantIMFsSpeechWithNoise = relevant_IMFs(SpeechWithNoiseSignalIMFs,SpeechWithNoiseSignalDownsampled,thresh_range)
# MaskRelevantIMFsSpeechWithNoise = transpose(array(MaskRelevantIMFsSpeechWithNoise.astype('bool')))
# SpeechWithNoiseSignalIMFs2 = SpeechWithNoiseSignalIMFs[MaskRelevantIMFsSpeechWithNoise]
# DenoisedSpeechWithNoiseSignal = sum(SpeechWithNoiseSignalIMFs2,0) # Sum up the remaining IMFs to obtain denoised signal
# SpeechWithNoiseSignal_outcome, SpeechWithNoiseSignal_outcome_vec, SpeechWithNoiseSignal_rao_test_statistic, SpeechWithNoiseSignal_threshold = KayNonstatDetect(DenoisedSpeechWithNoiseSignal,pfa)
#
#
# # iii) clean speech
# CleanSpeech = readwav(FileNameExampleCleanSpeech) # Reading WAV signal (CleanSpeech)
# CleanSpeechSignal = array(CleanSpeech[1],dtype=float) # Transforming data to a numpy array
# CleanSpeechSignal = CleanSpeechSignal[:,0] # WAV is being read as 2-channel signal, we can choose the first one by now
# CleanSpeechSignalDownsampled = signal.resample(CleanSpeechSignal, NumberOfSamplesDownSampledSignal) # Downsampling original signal
# CleanSpeechSignalEMD = EMD(CleanSpeechSignalDownsampled,t=None, threshold_1=0.05, threshold_2=0.5, alpha=0.05, ndirs=4, fixe=0, maxiter=5000, fixe_h=0, n_imfs=0, nbsym=2) # Performing the Empirical Mode Decomposition
# CleanSpeechSignalIMFs = CleanSpeechSignalEMD.decompose() # Gathering the IMFs obtained from the decomposition
# MaskRelevantIMFsCleanSpeech = relevant_IMFs(CleanSpeechSignalIMFs,CleanSpeechSignalDownsampled,thresh_range)
# MaskRelevantIMFsCleanSpeech = transpose(array(MaskRelevantIMFsCleanSpeech.astype('bool')))
# CleanSpeechSignalIMFs2 = CleanSpeechSignalIMFs[MaskRelevantIMFsCleanSpeech]
# DenoisedCleanSpeechSignal = sum(CleanSpeechSignalIMFs2,0) # Sum up the remaining IMFs to obtain denoised signal
# CleanSpeechSignal_outcome, CleanSpeechSignal_outcome_vec, CleanSpeechSignal_rao_test_statistic, CleanSpeechSignal_threshold = KayNonstatDetect(DenoisedCleanSpeechSignal,pfa)
#
#
# plt.plot(NoSpeechSignal_rao_test_statistic,label="no speech")
# plt.plot(SpeechWithNoiseSignal_rao_test_statistic,label="speech with noise")
# plt.plot(CleanSpeechSignal_rao_test_statistic,label="clean speech")
# plt.legend(loc='upper right')
#
#
