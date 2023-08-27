import matplotlib.pyplot as plt
from toyDataAA import toyDataAA
from scipy.spatial import ConvexHull

#run the AA based on number of archetypes
numArchetypes = range(2,21)
archeTypesList = []
losses = []
numIterations = 10000

# append archetype coordinates and losses for each number of archetypes
for numArchetype in numArchetypes:
    data, archeTypes, loss_Adam = toyDataAA(numArchetypes=numArchetype,numIterations=numIterations)
    archeTypesList.append(archeTypes)
    losses.append(loss_Adam[-1])

# plot the loss as function of archetypes
fig1, ax1 = plt.subplots()
ax1.plot(numArchetypes, losses, label = "Loss as a function of number of archetypes")

# print loss on plot for every other archetype
for a,b in zip(numArchetypes[::2], losses[::2]): 
    plt.text(a, b, f"{b:.1f}")

# set the ticks to every other archetype
ax1.set_xticks(numArchetypes[::2])
ax1.set_title('Loss Comparison After 10.000 Iterations')
ax1.set_xlabel('Number of archetypes')
ax1.set_ylabel('Loss')
ax1.set_yscale('log')
ax1.legend()
fig1.savefig('toyData/plots/lossComparison.png')

# we want to plot for 3,5,9,10 archetypes. That corresponds to index [1,3,7,8] 
plottedArchetypes=[3,5,9,10]
indexForArcheType=[x - 2 for x in plottedArchetypes]

# plot the data together with the found archetypes. 
distributions=[data[:,0:100],data[:,100:200],data[:,200:]]

fig2, ax2 = plt.subplots()
matplotlibStandardcolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

# plot archetypes and the convex hull
for idx, archeTypes in enumerate([archeTypesList[i] for i in indexForArcheType]):    
    points=archeTypes.T    
    hull = ConvexHull(points)    
    for simplex in hull.simplices:        
        ax2.plot(points[simplex, 0], points[simplex, 1],'k-',color=matplotlibStandardcolors[idx])   
    ax2.plot(archeTypes[0,:], archeTypes[1,:], 'x', color=matplotlibStandardcolors[idx], alpha=1,label=f'{plottedArchetypes[idx]} archetypes')

oldIdx=idx
for idx,distribution in enumerate(distributions,oldIdx+1):
    ax2.plot(distribution[0,:], distribution[1,:], '.', color=matplotlibStandardcolors[idx], alpha=0.5)

ax2.set_title('Archetypes and their convex hull')
ax2.legend(loc='upper left', borderaxespad=0)
fig2.savefig('toyData/plots/convexHulls.png')   
plt.show()