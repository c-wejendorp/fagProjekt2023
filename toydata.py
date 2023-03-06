import torch 
import numpy as np
import matplotlib.pyplot as plt

###dim
V = 100
T = 2
k = 3

np.random.seed(32)
torch.manual_seed(0)

###initialize three normal distributions (mean, std)
norm1 = np.random.multivariate_normal([5, 0], [[1, 1], [1, 2]], size = V)
norm2 = np.random.multivariate_normal([17, 0], [[3, -1], [-1, 7]], size = V)
norm3 = np.random.multivariate_normal([15, 10], [[10, 0.2], [0.2, 5]], size = V)

###plot
plt.plot(norm1[:,0], norm1[:,1], '.', alpha=0.5)
plt.plot(norm2[:,0], norm2[:,1], '.', alpha=0.5)
plt.plot(norm3[:,0], norm3[:,1], '.', alpha=0.5)
plt.axis()
plt.grid()
#plt.show()

###create x
X = np.vstack([norm1, norm2, norm3]).T
X = torch.tensor(X)

###optimize

#create model
class MatrixMult(torch.nn.Module):
    def __init__(self, V, T, k, X):
        super(MatrixMult, self).__init__()
        self.C = torch.nn.Parameter(torch.nn.Softmax(dim = 0)(torch.rand((V*3, k), dtype=torch.double))) #softmax upon initialization
        self.S = torch.nn.Parameter(torch.nn.Softmax(dim = 0)(torch.rand((k, V*3), dtype=torch.double))) #-||-
        self.A = 0
        self.X = X
    
    def soft_fun(self, M):
        """Implements softmax along columns to respect matrix constraints"""
        
        softmax = torch.nn.Softmax(dim = 0)
        softM = softmax(M)
        return softM

    def forward(self):        
        XC = torch.matmul(self.X, self.soft_fun(self.C))
        self.A = XC
        XCS = torch.matmul(XC, self.soft_fun(self.S))
        return XCS

#hyperparameters
lr = 1e-3
n_iter = 10000

#loss function
model = MatrixMult(V, T, k, X)
lossCriterion = torch.nn.MSELoss(reduction = "sum")
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# Creating Dataloader object
loss_Adam = []
for i in range(n_iter):
    # zeroing gradients after each iteration
    optimizer.zero_grad()
    # making a prediction in forward pass
    Xrecon = model.forward()
    # calculating the loss between original and predicted data points
    loss = lossCriterion(Xrecon, X)
    # backward pass for computing the gradients of the loss w.r.t to learnable parameters
    loss.backward()
    # updating the parameters after each iteration
    optimizer.step()
    # store loss into list
    loss_Adam.append(loss.item())

print("final loss: ", loss_Adam[-1])

#plot archetype points as x's
A = model.A.detach().numpy()
print("archetype coordinates: \n", A)
plt.plot(A[0,:], A[1,:], 'x', alpha=1)
plt.fill(A[0,:], A[1,:], facecolor='none', edgecolor='purple', linewidth=3)
plt.show()