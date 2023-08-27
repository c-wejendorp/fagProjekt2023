import numpy as np
import numpy.typing as npt

class Synthetic_Data:
    def __init__(self,T_eeg:int, T_meg:int, T_fmri:int, nr_subjects:int, nr_sources:int, arg_eeg_sources:tuple, arg_meg_sources:tuple, arg_fmri_sources:tuple, activation_timeidx_eeg:npt.ArrayLike, activation_timeidx_meg:npt.ArrayLike, activation_timeidx_fmri:npt.ArrayLike) -> None:
        self.nr_sources = nr_sources # assume default activation is 0
        self.nr_subjects = nr_subjects

        self.X_eeg = self.initialize_sources_diff_time_dimensions(T_eeg, arg_eeg_sources, activation_timeidx_eeg)
        self.X_meg = self.initialize_sources_diff_time_dimensions(T_meg, arg_meg_sources, activation_timeidx_meg)
        self.X_fmri = self.initialize_sources_diff_time_dimensions(T_fmri, arg_fmri_sources, activation_timeidx_fmri)

    def initialize_sources_diff_time_dimensions(self, T, modality_sources, timeidx_of_activation) -> np.ndarray:
        """
        initilize sources given time index and modality specifications. Time of activation of the 
        activated sources are assumed to be the same, but the activation level is normally distributed
        with 1 mean and 0.1 standard deviation
        
        T (int): the number of time steps recorded
        modality_sources (tuple of list of int): tuple with lists of source indices
        timeidx_of_actiavtion (arr of int): int marking the beginning of a source activation
        """
        
        X = np.zeros((self.nr_subjects, T, self.nr_sources)) # s x T x V

        # loop over the sources in a "cluster" and their corresponding time of activation
        for _, (time_idx_start, sources) in enumerate(zip(timeidx_of_activation, modality_sources)):
            
            # make each activation 5 time indicies long
            idx_of_activation = np.arange(time_idx_start, time_idx_start + 5) 
            
            # loop over each source in the "cluster"
            for source in sources:
                activation_time = np.zeros(T)
                
                # make the activation of the sources overlap with 2 time indicies 
                idx_of_activation += 3 
                activation_time[idx_of_activation] = 1

                # create outlier source for source 24
                if source == 24:
                    X[0, activation_time==1 , source] = np.random.normal(1, 0.1) # tweak the activation value
                else:
                    X[:, activation_time==1 , source] = np.random.normal(1, 0.1)

        return X


if __name__ == "__main__":

    # create index numbers for the modality sources as tuples with lists
    # each modality will, for each list, have one unique, one meg-shared, 
    # one fmri-shared and one all-sharing. 
    arg_eeg_sources = (np.arange(0,4), np.arange(7,11), np.arange(14,18))
    arg_meg_sources = (np.array([0+i*7, 1+i*7, 4+i*7, 5+i*7]) for i in range(3))
    arg_fmri_sources = (np.array([1+i*7, 2+i*7, 4+i*7, 6+i*7]) for i in range(3)) 

    # displace time of activation for meg and fmri
    activation_timeidx_eeg = np.array([0, 30, 60])
    activation_timeidx_meg = activation_timeidx_eeg + 10
    activation_timeidx_fmri = activation_timeidx_eeg + 50

    # create synthetic data
    X = Synthetic_Data(T_eeg=100, T_meg=100, T_fmri=500, 
                       nr_subjects=10, nr_sources=30, 
                       arg_eeg_sources=arg_eeg_sources, 
                       arg_meg_sources=arg_meg_sources, 
                       arg_fmri_sources=arg_fmri_sources, 
                       activation_timeidx_eeg = activation_timeidx_eeg, 
                       activation_timeidx_meg=activation_timeidx_meg, 
                       activation_timeidx_fmri=activation_timeidx_fmri)