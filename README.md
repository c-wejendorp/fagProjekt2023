Link to Google Drive: https://drive.google.com/drive/folders/1tGbLmWekWre0a-gSkYFZTbXflIspOndC?usp=sharing

This project performs Multimodal Multisubject (Multicondition) Archetypal Analysis on the Wakeman and Henson dataset.

preprocessing:
    Here you will find two folders: JesperScripts and ourScripts.
        
        JesperScripts provides all necessary preprocessing of the raw data as well as a guide to how to acces the data in "notes_on_data_JESPER". This folder should remain untampered.

        ourScripts provides the dataset creation with additional preprocessing steps
        (frobenius norm, highpass-filtering etc) prior to saving the data in an easy-to-read structure. checkPath is used to save the data to the correct directories.
        ourScripts also provde a plot_scripts subfolder which is used for manual plotting of the brain surface (after analysis has been performed). plot_matrix_on_brain is used for the spatial concatenation and the plot_mod_on_brain is used for plotting only modality-differences (ignoring conditions)

toyData:
    Clean-up still in the works

    Here you will find several iterations of the toyData process.

    toyDataAA.py is used for the simplest case in which there is no multi-anything.
    
    MMAA_ver2 was used most
    MMAA_verOutlier generates a single source acting as "noise" in the data. Ideally, it should show that outliers are downweighted in the optimized solution with a robust loss

MMAA:
    This is the main folder for the analysis itself and all later analysis-work and plotting.

        loadData.py loads the data from createDataset and sets the data up as a class
        
        MMAA_model_CUDA performs the MM(M)A analysis itself and returns the data-matrices and loss(es)

        trainModels utilzes the MMAA_model_CUDA and saves all data to paths with easily intepretable file names for different seeds and number of archestypes

        reconstructSms is used in case the subject-specific S-matrices are saved individually and needs to be fused together to the (m x s x k x v) matrix

        nmi contains the helper function that calculates the normalized mutual information between S-matrices. this file is used in the calc_NMI which saves the mean, std and max nmi to use for plotting

        calc_test_loss_Sms calculates the mean, std and minimum loss for each modality across seeds and saves it for later plotting of the loss curves

        plotFunctions is a folder containing the two major data analysis plots: The lossPlot and the reproducibilityPlot (nmi). 

        HPC contains all files used for setting up the HPC. 
            arguments contains all necessary input arguments for the given combination of modalities you wish to use in you analysis. Created from createArguments

            submit_scripts contains all scripts used for submitting to HPC in which you provide which split you wish to use and which combination of modalities. Created from createSubmitScripts

            All the other sh files?

Classifier:
    I'll let you explain all the clean-up you did, Danina <3

deprecated_scripts
    This folder contains scripts either used for the time-concatenation analysis or files that may be of use later should anything go horribly wrong. They have been kept in case we made a mistake during clean-up.

