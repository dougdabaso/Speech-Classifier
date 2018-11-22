

# Supporting functions -------------------------------------------------------------------------------------------------



def PCStats(MySignalInTime,p,q,M):

    # This function computes the statistics describing periodical correlations , as defined in [1] in eq. (6). As input,
    #  it takes the signal in time, the value of p, q, and M.

    MySignalInTime = MySignalInTime - mean(MySignalInTime) # Subtracting the mean for ACF calculation
    # Computing the normalized autocorrelation function
    AutoCorrFunc = correlate(MySignalInTime, MySignalInTime, mode='full')
    NormAutoCorrFunc = AutoCorrFunc[int(ceil(AutoCorrFunc.size/2)):]/AutoCorrFunc[int(ceil(AutoCorrFunc.size/2))]

    # For computations we wil be using the PSD, so the FFT of the autocorrelation function is computed
    MyFFT = fft.fft(NormAutoCorrFunc)

    # Computing coherence and incoherence statistics
    NFFT=len(MyFFT) # Number of bins of the FFT
    N=len(NormAutoCorrFunc) # Length of the time series used for analysis
    tau = abs(q-p) # Required parameter for the calculations
    L = (N-1-tau)/M # Required parameter for the calculations
    print(L)
    print(N)
    print(tau)
    CoherentStat=zetaMag(0,tau,M,MyFFT)
    IncoherentStat=(1/(L+1))*zetaMag(p*M,p*M+tau,M,MyFFT)

    return CoherentStat, IncoherentStat




def zetaMag(p,q,M,MyFFT):
    # This function computes the sample coherence, as defined in [1] in eq. (6). As input, it takes the current FFT
    # of the desired time series (it could be normalized autocorrelation function), the value of p, q, and M.
    Num = []
    Den1 = []
    Den2 = []

    NFFT = len(MyFFT) # The number of bins used in FFT

    for m in range(0, M):
        Ind1FFT = p + m  # Indexes of FFT
        Ind2FFT = q + m  # Indexes of FFT
        # Taking into account the periodicity of the Discrete Fourier transform (check if it should be taken into consideration)
        if (Ind1FFT > NFFT):
            Ind1FFT = Ind1FFT - NFFT
        if (Ind2FFT > NFFT):
            Ind2FFT = Ind2FFT - NFFT
        # Computing components of the numerator and the denominator of (6)
        Num.append(abs(MyFFT[Ind1FFT] * conj(MyFFT[Ind2FFT])))
        Den1.append(abs(MyFFT[Ind1FFT]) ** 2)
        Den2.append(abs(MyFFT[Ind2FFT]) ** 2)

    return (abs(sum(Num)) ** 2) / (sum(Den1) * sum(Den2))





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
            if i != j: cov_matrix = cov(current_IMFs[i,], current_IMFs[j,])
            if i != j: cov_vector.append(cov_matrix[0, 1])  # covariance is the off-diag. terms

        cov_matrix_residual = cov(current_IMFs[i,], current_IMFs[
            number_of_modes_obtained - 1,])  # computing the covariance between the residual and the current IMF
        epsilon_vector.append(sum(cov_vector) + cov_matrix_residual[0, 1])

    # 2) Computing the threshold using the energy-criterion

    energy_x_p = sum((current_signal - mean(current_signal)) ** 2)
    epsilon_final = sum(epsilon_vector)

    # 3) For the collection of all possible thresh values, check the most frequent
    # modes, i.e., the most relevant IMFs

    # outside_array_relevant_IMFs_vector = zeros((size(thresh_range),number_of_modes_obtained-2))
    outside_array_relevant_IMFs_vector = []

    for i_thresh in range(0, size(thresh_range)):

        thresh = thresh_range[i_thresh]

        relevant_IMF_vector = []

        # Verifyign the condition for the i-th IMF
        for i in range(0, (number_of_modes_obtained-1)):

            if (sqrt(energy_vector[i]) * ((sqrt(energy_x_p) * thresh) - sqrt(
                energy_vector[i]))) < epsilon_final: relevant_IMF_vector.append(1)
            if (sqrt(energy_vector[i]) * ((sqrt(energy_x_p) * thresh) - sqrt(
                energy_vector[i]))) > epsilon_final: relevant_IMF_vector.append(0)

        outside_array_relevant_IMFs_vector.append(transpose(relevant_IMF_vector))

    # Computing the most frequent group of IMFs
    selected_mode = mode(outside_array_relevant_IMFs_vector)

    # Adding the residue to the decomposition
    selected_relevant_IMFs = append(selected_mode.mode, 1)

    return selected_relevant_IMFs












