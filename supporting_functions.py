

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


