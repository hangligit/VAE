import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt

def create_dataset(data_size=4000):
    assert data_size % 4 == 0
    n = int(data_size/4)
    mu=np.array([[1,1],[1,-1],[-1,-1],[-1,1]])
    s=0.1
    X=np.random.randn(data_size,2)
    y=np.zeros(data_size).astype(int)
    for i in range(4):
        X[i*n:(i+1)*n,:]=s * X[i*n:(i+1)*n,:] + mu[i]
        y[i*n:(i+1)*n] += i
    return X,y


class IAF(nn.Module):
	def __init__(self,):
		super(IAF,self).__init__()
		self.autoreg=nn.Sequential(
				nn.Linear(32,32),
				nn.ELU(),
				nn.Linear(32,10),
			)
		self.fc=nn.Linear(10,10)

	def forward(self,x):
		x=self.autoreg(x)
		x=self.fc(x)
		return x

class VAE(nn.Module):
	def __init__(self,input_size=2, hidden_size=32,
		z_size=2, h_size=2, hout=False):
		super(VAE, self).__init__()
		self.hout=hout
		self.enc = nn.Sequential(
				nn.Linear(input_size, hidden_size),
				nn.ReLU(),
			)
		self.encz = nn.Linear(hidden_size, z_size)
		self.ench = nn.Linear(hidden_size, h_size)

		self.dec=nn.Sequential(
				nn.Linear(z_size,hidden_size),
				nn.ReLU(),
				nn.Linear(hidden_size, input_size)
			)
	def forward(self, x):
		x=self.enc(x)
		z=self.encz(x)
		h=self.ench(x) if self.hout else None
		x_hat=self.dec(z)
		return (z,h), x_hat

if __name__=='__main__':
	vae=VAE()
	optimizer=torch.optim.Adam(vae.parameters(), lr=0.01)
	xl,yl=create_dataset()

	plt.ion()
	fig = plt.figure(figsize=1.5*plt.figaspect(1)) 

	loss_history=list()
	for i in range(100):
		batch=np.random.choice(4000, 16)
		x=xl[batch]
		x=torch.from_numpy(x).type(torch.float32)
		_,x_hat = vae(x)
		loss=torch.mean((x-x_hat)**2)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		loss_history.append(loss.item())
		print("Iteration {} loss {}".format(i, loss.item()))

		plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=yl[batch],cmap=plt.cm.cool)
		plt.scatter(x_hat.data.numpy()[:,0], x_hat.data.numpy()[:,1])
		plt.xlim(-1.5,1.5)
		plt.ylim(-1.5,1.5)
		plt.draw(); plt.pause(0.1); plt.clf()

	plt.ioff()
	plt.plot(loss_history)
	plt.show()