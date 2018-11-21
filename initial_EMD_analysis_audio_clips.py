# Draft of the script for implementing the framework of [1].

# Header ---------------------------------------------------------------------------------------------------------------

# Importing standard modules
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode

# For this proof of concept, we can use pyhht package (https://pyhht.readthedocs.io) which offers implementations of EMD
#  and time-frequency (TF) transforms (which are the two approaches we are going to use a priori)
from pyhht import EMD # Importing EMD function from pyhht module
from pyhht.visualization import plot_imfs
from scipy.io.wavfile import read as readwav # We can use scipy to read WAVs
from scipy import signal # getting downsample function from scipy module
from supporting_functions import relevant_IMFs

# 1) Defining paths and variable names ---------------------------------------------------------------------------------

MasterPath = 'C:\\Users\\Douglas\\Oya\\AVA Dataset\\python scripts'
FileNameExampleNoSpeech = '5BDj0ow5hnA_NO_SPEECH_1.wav'
FileNameExampleSpeechWithNoise = '5BDj0ow5hnA_SPEECH_WITH_NOISE_1.wav'
FileNameExampleCleanSpeech = '5BDj0ow5hnA_CLEAN_SPEECH_1.wav'

# Reaquired by now for downsampling audio signals to make the EMD calculation a little faster. The resampled signal x
#  starts at the same value as x but is sampled with a spacing of len(x)/DownSamplingFactor*(spacing of x).
NumberOfSamplesDownSampledSignal = 5000 # We can thing if it is worth to do so later
thresh_range = np.linspace(0.01,1,1000)  # This is the range of threshold considered for computing the most relevant IMFs
                                        # for more info check one of my papers published in ICASSP 2014 [2]


# [1] K. I. Molla, K. Hirose, and N. Minematsu, "Robust voiced/unvoiced speech
# classification using empirical mode decomposition and periodic correlation model",
# Conference: INTERSPEECH 2008, 9th Annual Conference of the International
# Speech Communication Association, Brisbane, Australia, September 22-26, 2008.

# [2] IEEE ICASSP 2014 - International Conference on Acoustic, Speech and
# Signal Processing, "On selecting relevant intrinsic mode functions in em-
# pirical mode decompositions: an energy-based approach",D. Baptista de Souza et al.


# 2) Reading audio signals and performing EMD --------------------------------------------------------------------------

os.chdir(MasterPath) # Going to MasterPath

# Gathering the IMFs for each signal: i) no speech, ii) speech with noise, and iii) clean speech

# i) no speech
NoSpeech = readwav(FileNameExampleNoSpeech) # Reading WAV signal (NoSpeech)
NoSpeechSignal = np.array(NoSpeech[1],dtype=float) # Transforming data to a numpy array
NoSpeechSignal = NoSpeechSignal[:,0] # WAV is being read as 2-channel signal, we can choose the first one by now
NoSpeechSignalDownsampled = signal.resample(NoSpeechSignal, NumberOfSamplesDownSampledSignal) # Downsampling original signal
NoSpeechSignalEMD = EMD(NoSpeechSignalDownsampled,t=None, threshold_1=0.05, threshold_2=0.5, alpha=0.05, ndirs=4, fixe=0, maxiter=5000, fixe_h=0, n_imfs=0, nbsym=2) # Performing the Empirical Mode Decomposition
NoSpeechSignalIMFs = NoSpeechSignalEMD.decompose() # Gathering the IMFs obtained from the decomposition
print('IMFs successfully built')



# ii) speech with noise
SpeechWithNoise = readwav(FileNameExampleSpeechWithNoise) # Reading WAV signal (SpeechWithNoise)
SpeechWithNoiseSignal = np.array(SpeechWithNoise[1],dtype=float) # Transforming data to a numpy array
SpeechWithNoiseSignal = SpeechWithNoiseSignal[:,0] # WAV is being read as 2-channel signal, we can choose the first one by now
SpeechWithNoiseSignalDownsampled = signal.resample(SpeechWithNoiseSignal, NumberOfSamplesDownSampledSignal) # Downsampling original signal
SpeechWithNoiseSignalEMD = EMD(SpeechWithNoiseSignalDownsampled,t=None, threshold_1=0.05, threshold_2=0.5, alpha=0.05, ndirs=4, fixe=0, maxiter=5000, fixe_h=0, n_imfs=0, nbsym=2) # Performing the Empirical Mode Decomposition
SpeechWithNoiseSignalIMFs = SpeechWithNoiseSignalEMD.decompose() # Gathering the IMFs obtained from the decomposition
print('IMFs successfully built')

# iii) clean speech
CleanSpeech = readwav(FileNameExampleCleanSpeech) # Reading WAV signal (CleanSpeech)
CleanSpeechSignal = np.array(CleanSpeech[1],dtype=float) # Transforming data to a numpy array
CleanSpeechSignal = CleanSpeechSignal[:,0] # WAV is being read as 2-channel signal, we can choose the first one by now
CleanSpeechSignalDownsampled = signal.resample(CleanSpeechSignal, NumberOfSamplesDownSampledSignal) # Downsampling original signal
CleanSpeechSignalEMD = EMD(CleanSpeechSignalDownsampled,t=None, threshold_1=0.05, threshold_2=0.5, alpha=0.05, ndirs=4, fixe=0, maxiter=5000, fixe_h=0, n_imfs=0, nbsym=2) # Performing the Empirical Mode Decomposition
CleanSpeechSignalIMFs = CleanSpeechSignalEMD.decompose() # Gathering the IMFs obtained from the decomposition
print('IMFs successfully built')




# 3) Signal denoising via selecting the most relevant IMFs -------------------------------------------------------------

# Considering that most relevant IMFs usually carry less noise, but it needs to be validated for this application

# i) no speech
MaskRelevantIMFsNoSpeech = relevant_IMFs(NoSpeechSignalIMFs,NoSpeechSignalDownsampled,thresh_range)
MaskRelevantIMFsNoSpeech = np.transpose(np.array(MaskRelevantIMFsNoSpeech.astype('bool')))
NoSpeechSignalIMFs2 = NoSpeechSignalIMFs[MaskRelevantIMFsNoSpeech]
DenoisedNoSpeechSignal = np.shape(np.sum(NoSpeechSignalIMFs2,0)) # Sum up the remaining IMFs to obtain denoised signal
plt.figure(0) # Plotting relevant IMFs + residue
plot_imfs(NoSpeechSignalDownsampled, NoSpeechSignalIMFs2)


# ii) speech with noise
MaskRelevantIMFsSpeechWithNoise = relevant_IMFs(SpeechWithNoiseSignalIMFs,SpeechWithNoiseSignalDownsampled,thresh_range)
MaskRelevantIMFsSpeechWithNoise = np.transpose(np.array(MaskRelevantIMFsSpeechWithNoise.astype('bool')))
SpeechWithNoiseSignalIMFs2 = SpeechWithNoiseSignalIMFs[MaskRelevantIMFsSpeechWithNoise]
DenoisedSpeechWithNoiseSignal = np.shape(np.sum(SpeechWithNoiseSignalIMFs2,0)) # Sum up the remaining IMFs to obtain denoised signal
plt.figure(1) # Plotting relevant IMFs + residue
plot_imfs(SpeechWithNoiseSignalDownsampled, SpeechWithNoiseSignalIMFs2)

# iii) clean speech
MaskRelevantIMFsCleanSpeech = relevant_IMFs(CleanSpeechSignalIMFs,CleanSpeechSignalDownsampled,thresh_range)
MaskRelevantIMFsCleanSpeech = np.transpose(np.array(MaskRelevantIMFsCleanSpeech.astype('bool')))
CleanSpeechSignalIMFs2 = CleanSpeechSignalIMFs[MaskRelevantIMFsCleanSpeech]
DenoisedCleanSpeechSignal = np.shape(np.sum(CleanSpeechSignalIMFs2,0)) # Sum up the remaining IMFs to obtain denoised signal
plt.figure(2) # Plotting relevant IMFs + residue
plot_imfs(CleanSpeechSignalDownsampled, CleanSpeechSignalIMFs2)









# Supporting functions -------------------------------------------------------------------------------------------------

def relevant_IMFs(current_IMFs, current_signal, thresh_range):
    # This function executes the energy based approach proposed in [IMFICASSP14] for
    # computing the group of most relevant IMFs obtained from the EMD algorithm

    # By Douglas David Baptista de Souza

    # [IMFICASSP14] IEEE ICASSP 2014 - International Conference on Acoustic, Speech and
    # Signal Processing, "On selecting relevant intrinsic mode functions in em-
    # pirical mode decompositions: an energy-based approach",D. Baptista de Souza et al.


    number_of_modes_obtained = current_IMFs.shape[0]  # Total number of IMFs obtained from the decomposition
    energy_vector = []  # To store the energy of each IMF
    epsilon_vector = []  # epsilon value of each IMF
    outside_array_relevant_IMFs_vector = []
    selected_mode = []

    # 1) Computing the energy of each IMF obtained from EMD and computing the
    # epsilon for the energy-based criterion

    # The residual is stored in the last row of the IMF array
    for i in range(0, (number_of_modes_obtained-1)):
        power_sig_imfs = current_IMFs[i,] ** 2
        energy_vector.append(power_sig_imfs.sum())  # Storing the energy value of each IMF
        cov_vector = []  # To store the covariance between the i-th and the j-th IMFs
        for j in range(0, number_of_modes_obtained - 1):
            if i != j: cov_matrix = np.cov(current_IMFs[i,], current_IMFs[j,])
            if i != j: cov_vector.append(cov_matrix[0, 1])  # covariance is the off-diag. terms

        cov_matrix_residual = np.cov(current_IMFs[i,], current_IMFs[
            number_of_modes_obtained - 1,])  # computing the covariance between the residual and the current IMF
        epsilon_vector.append(np.sum(cov_vector) + cov_matrix_residual[0, 1])

    # 2) Computing the threshold using the energy-criterion

    energy_x_p = np.sum((current_signal - np.mean(current_signal)) ** 2)
    epsilon_final = np.sum(epsilon_vector)

    # 3) For the collection of all possible thresh values, check the most frequent
    # modes, i.e., the most relevant IMFs

    # outside_array_relevant_IMFs_vector = np.zeros((np.size(thresh_range),number_of_modes_obtained-2))
    outside_array_relevant_IMFs_vector = []

    for i_thresh in range(0, np.size(thresh_range)):

        thresh = thresh_range[i_thresh]

        relevant_IMF_vector = []

        # Verifyign the condition for the i-th IMF
        for i in range(0, (number_of_modes_obtained-1)):

            if (np.sqrt(energy_vector[i]) * ((np.sqrt(energy_x_p) * thresh) - np.sqrt(
                energy_vector[i]))) < epsilon_final: relevant_IMF_vector.append(1)
            if (np.sqrt(energy_vector[i]) * ((np.sqrt(energy_x_p) * thresh) - np.sqrt(
                energy_vector[i]))) > epsilon_final: relevant_IMF_vector.append(0)

        outside_array_relevant_IMFs_vector.append(np.transpose(relevant_IMF_vector))

    # Computing the most frequent group of IMFs
    selected_mode = mode(outside_array_relevant_IMFs_vector)

    # Adding the residue to the decomposition
    selected_relevant_IMFs = np.append(selected_mode.mode, 1)

    return selected_relevant_IMFs
















