import numpy as np
import matplotlib.pyplot as plt
from toyDataAAMulti import toyDataAA
from tqdm import tqdm
from collections import defaultdict 
'''
plot loss curve
'''
# from scipy.spatial import ConvexHull, convex_hull_plot_2d

#run the AA based on number of archetypes
numArchetypes=range(2,26)
seeds = range(0,101, 10)
arc_res = defaultdict(lambda: [])

for numArchetype in tqdm(numArchetypes):
    for seed in seeds:
        loss_Adam = toyDataAA(numArchetypes=numArchetype,
                loss_type='mle_rob',
                numpySeed=32,
                torchSeed=seed,
                plotDistributions=False,
                learningRate=1e-1,
                numIterations=5000, 
                T_eeg=100, 
                T_meg=100, 
                T_fmri=500, 
                nr_subjects=10, 
                nr_sources=25, 
                arg_eeg_sources=(np.arange(0,4), np.arange(7,11), np.arange(14,18)), 
                arg_meg_sources=(np.array([0+i*7, 1+i*7, 4+i*7, 5+i*7]) for i in range(3)), 
                arg_fmri_sources=(np.array([1+i*7, 2+i*7, 4+i*7, 6+i*7]) for i in range(3)), 
                activation_timeidx_eeg = np.array([0, 30, 60]), 
                activation_timeidx_meg=np.array([0, 30, 60]) + 10, 
                activation_timeidx_fmri=np.array([0, 30, 60]) + 50)
        arc_res[numArchetype].append(loss_Adam[-1])


loss_mean = np.array([np.mean(loss) for archetype, loss in arc_res.items()], dtype="float64")
loss_std = np.array([np.std(loss) for archetype, loss in arc_res.items()])


fig1, ax1 = plt.subplots()
#plot the loss as function of archetypes
ax1.plot(numArchetypes, loss_mean, '-', label = "Loss as a function of number of archetypes")
ax1.fill_between(numArchetypes, loss_mean - loss_std, loss_mean + loss_std, alpha=0.2)

#for idx,loss in enumerate(losses): 
#    ax1.plot(range(1,numIterations+1), loss, label=f'Loss with {numArchetypes[idx]} archetypes. Final loss: {loss[-1]:.2f}')

#print loss on plot for every other archetype
for a,b in zip(numArchetypes[::2], loss_mean[::2]): 
    plt.text(a, b, f"{b:.1f}")

#set the ticks to every other archetype
ax1.set_xticks(numArchetypes[::2])
ax1.set_title('Loss Comparisons')
ax1.set_xlabel('Number of archetypes')
ax1.set_ylabel('Loss')
# ax1.set_yscale('log')
ax1.legend()
fig1.savefig('toyData/plots/lossComparison_toyMMAA.png')


