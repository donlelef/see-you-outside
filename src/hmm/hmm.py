import torch
import numpy as np
import data_io
import torch.optim as optim
import matplotlib.pyplot as plt

class HMM(torch.nn.Module):
    """HMM model
    """
    def __init__(self):
        super(HMM,self).__init__()
        # a priori of the distribution
        self.prior = torch.nn.Parameter(torch.tensor([0.95,0.04,0.009,0.001]))
        # self.prior = torch.cat([self.prior,torch.nn.Parameter(torch.Tensor([0.,0.,0.]),requires_grad=False)])
        # from exposed to asymptomatic
        self.eta = torch.nn.Parameter(torch.tensor(1/2.34))
        # from asymptomatic to infected
        self.alpha = torch.nn.Parameter(torch.tensor([1/2.86]))
        # prob leaving infected
        self.mu = torch.nn.Parameter(torch.tensor(1/3.2))
        # conditional prob to icu
        self.gamma = torch.nn.Parameter(torch.tensor(0.02))
        # death rate (inverse of time in icu)
        self.phi = torch.nn.Parameter(torch.tensor(0.05))
        # prob death
        self.w = torch.nn.Parameter(torch.tensor(0.9))
        # prob recover from ICU conditioned on death rate
        self.xi = torch.nn.Parameter(torch.tensor(0.1))

        # parameter for control
        # infectivity of asymptomatic
        self.beta = torch.nn.Parameter(torch.tensor(0.06),requires_grad=False)
        # average number of contact
        self.k = torch.nn.Parameter(torch.tensor(13.3))
        # contact rate (only one group now)
        self.C = torch.nn.Parameter(torch.tensor(1.),requires_grad=False)
        # density factor
        self.eps = torch.nn.Parameter(torch.tensor(0.01))
        # effecitve population
        self.n_eff = torch.tensor(8570000)
        # area of the region
        self.s = torch.tensor(41285)
        #  household size
        self.sigma = torch.tensor(2.5)

        self.tc = 22
        self.kappa = torch.nn.Parameter(torch.tensor([1,13]),requires_grad=False)
        self.tr = 100
        self.obs = torch.tensor([[0.,0.,0.,1.,1.,1.,1.], # total case includes all recover, death, i and h
                                 [0,0,0,0,0.25,0,0],
                                 [0,0,0,0,0,0,1]])
    
    def clamp(self):
        # projection to constraints

        # project prior to simplex
        self.prior.data.clamp_(0) # make positive first
        temp = self.prior.data.numpy()+0.001 # avoid all susceptible
        self.prior.data = torch.tensor(temp/sum(temp)) # project to simplex
        # from exposed to asymptomatic
        self.eta.data.clamp_(0,1)
        # from asymptomatic to infected
        self.alpha.data.clamp_(0,1)
        # prob leaving infected
        self.mu.data.clamp_(0,1)
        # conditional prob to icu
        self.gamma.data.clamp_(0,1)
        # death rate (inverse of time in icu)
        self.phi.data.clamp_(0,1)
        # prob death
        self.w.data.clamp_(0,1)
        # prob recover from ICU
        self.xi.data.clamp_(0,1)

        # parameter for control
        # infectivity of asymptomatic
        # self.beta = torch.nn.Parameter(torch.tensor(0.06),requires_grad=False)
        # average number of contact
        self.k.data.clamp_(1)
        # contact rate (only one group now)
        # self.C = torch.nn.Parameter(torch.tensor(1.),requires_grad=False)
        # density factor
        self.eps.data.clamp_(0,0.05)
        
    def get_ch(self,x):
        return torch.pow((x[0] + x[4]),self.sigma)
    
    def f(self,x):
        return 2 - torch.exp(-self.eps*x)

    def infect_rate(self,x):
        z = 1/(self.f(self.n_eff/self.s))
        x_a = x[2]*self.n_eff
        x_i = x[3]*self.n_eff
        t = x[7]
        
        if t < self.tc:
            k = self.k
        else:
            ind = np.where(np.array(self.tr) > t.detach().numpy())[0]
            if len(ind) == 0:
                k = self.k
            else:
                k = self.kappa[min(ind)] * (self.sigma-1)
        
        return 1 - torch.pow(1-self.beta, z*k*self.f(self.n_eff/self.s)*self.C*x_a/self.n_eff)\
        * torch.pow(1-self.beta, z*k*self.f(self.n_eff/self.s)*self.C*x_i/self.n_eff)

    def forward(self,T):
        outputs = []
        hidden_log = []
        for i in range(T):
            if i == 0:
                hidden = torch.cat((self.prior.view(-1,1),torch.tensor([0.,0.,0.,1.]).view(-1,1)),0)
            else:
                hidden = self.innovate(hidden)
            
            hidden_log.append(hidden)
            outputs.append(torch.mm(self.obs,hidden[0:-1].view(-1,1)))

        return torch.cat(outputs, 0).view(T, -1), torch.cat(hidden_log,0).view(T,-1)
                

    def innovate(self,x):
        # evolve one step
        Gamma = self.infect_rate(x)
        t = x[7]
        if t == self.tc:
            CH = self.get_ch(x)
            # stack matrix, we have to use torch.stack
            line1 = torch.cat((((1-CH)*(1-Gamma)).view(1,-1),
                                torch.zeros(1,6)),dim=1)
            line2 = torch.cat(((1-(1-CH)*(1-Gamma)).view(1,-1),(1-self.eta).view(1,-1),
                                torch.zeros(1,5)),dim=1)
            line3 = torch.cat((torch.zeros(1,1),
                                self.eta.view(1,-1),(1-self.alpha).view(1,-1),
                                torch.zeros(1,4)),dim=1)
            line4 = torch.cat((torch.zeros(1,2),
                                self.alpha.view(1,1),(1-self.mu).view(1,1),
                                torch.zeros(1,3)),dim=1)
            line5 = torch.cat((torch.zeros(1,3),
                                (self.mu*self.gamma).view(1,-1),(self.w*(1-self.phi) + (1-self.w)*(1-self.xi)).view(1,-1),
                                torch.zeros(1,2)),dim=1)
            line6 = torch.cat((torch.zeros(1,3),
                                (self.mu*(1-self.gamma)).view(1,-1),((1-self.w)*self.xi).view(1,-1),
                                torch.ones(1,1),torch.zeros(1,1)),dim=1)
            line7 = torch.cat((torch.zeros(1,4),
                                (self.w*self.phi).view(1,1),
                                torch.zeros(1,1),torch.ones(1,1)),dim=1)
            trans = torch.cat((line1,line2,line3,line4,line5,line6,line7),dim=0)
            # trans = torch.tensor([[(1-CH)*(1-Gamma), 0, 0, 0, 0, 0, 0],
            #                 [1-(1-CH)*(1-Gamma), 1-self.eta, 0, 0, 0, 0, 0],
            #                 [0,self.eta, 1-self.alpha, 0, 0, 0, 0],
            #                 [0, 0, self.alpha, 1-self.mu, 0, 0, 0],
            #                 [0, 0, 0, self.mu*self.gamma, self.w*(1-self.phi) + (1-self.w)*(1-self.xi), 0, 0],
            #                 [0, 0, 0, self.mu*(1-self.gamma), (1-self.w)*self.xi, 1, 0],
            #                 [0, 0, 0, 0, self.w*self.phi, 0, 1]],requires_grad=True)
        else:
            line1 = torch.cat(((1-Gamma).view(1,-1),
                                torch.zeros(1,6)),dim=1)
            line2 = torch.cat((Gamma.view(1,-1),(1-self.eta).view(1,-1),
                                torch.zeros(1,5)),dim=1)
            line3 = torch.cat((torch.zeros(1,1),
                                self.eta.view(1,-1),(1-self.alpha).view(1,-1),
                                torch.zeros(1,4)),dim=1)
            line4 = torch.cat((torch.zeros(1,2),
                                self.alpha.view(1,1),(1-self.mu).view(1,1),
                                torch.zeros(1,3)),dim=1)
            line5 = torch.cat((torch.zeros(1,3),
                                (self.mu*self.gamma).view(1,-1),(self.w*(1-self.phi) + (1-self.w)*(1-self.xi)).view(1,-1),
                                torch.zeros(1,2)),dim=1)
            line6 = torch.cat((torch.zeros(1,3),
                                (self.mu*(1-self.gamma)).view(1,-1),((1-self.w)*self.xi).view(1,-1),
                                torch.ones(1,1),torch.zeros(1,1)),dim=1)
            line7 = torch.cat((torch.zeros(1,4),
                                (self.w*self.phi).view(1,1),
                                torch.zeros(1,1),torch.ones(1,1)),dim=1)
            trans = torch.cat((line1,line2,line3,line4,line5,line6,line7),dim=0)
            # trans = torch.tensor([[(1-Gamma), 0, 0, 0, 0, 0, 0],
            #                 [Gamma, 1-self.eta, 0, 0, 0, 0, 0],
            #                 [0, self.eta, 1-self.alpha, 0, 0, 0, 0],
            #                 [0, 0, self.alpha, 1-self.mu, 0, 0, 0],
            #                 [0, 0, 0, self.mu*self.gamma, self.w*(1-self.phi) + (1-self.w)*(1-self.xi), 0, 0],
            #                 [0, 0, 0, self.mu*(1-self.gamma), (1-self.w)*self.xi, 1, 0],
            #                 [0, 0, 0, 0, self.w*self.phi, 0, 1]],requires_grad=True)

        return torch.cat((torch.mm(trans,x[0:-1].view(-1,1)).view(-1,1),(t+1).view(-1,1)),dim=0)

data = data_io.load_data().transpose()
data = data[:,1:]

model = HMM()
model.tc = model.tc # we leave 10 days unused
optimizer = optim.Adam(model.parameters(),lr= 1e-3)
criterion = torch.nn.MSELoss()
scale = torch.tensor([[1e3],[1e6]]).float()

for i in range(250):
    optimizer.zero_grad()
    outputs,states_log = model(data.shape[0])
    # scaling the loss
    weighted_outputs = scale*outputs[:,1:].transpose(0,1)
    weighted_data = scale*torch.tensor(data).float().transpose(0,1)/model.n_eff.float()
    loss = criterion(weighted_outputs.transpose(0,1),weighted_data.transpose(0,1))
    loss.backward()
    optimizer.step()
    model.clamp()
    print('iter: {}, loss: {}'.format(i,loss))


plt.figure(1)
plt.clf()
plt.plot(np.arange(data.shape[0]),data[:,0]/model.n_eff.float().detach().numpy(),label='data hospitalized')
plt.plot(np.arange(data.shape[0]),data[:,1]/model.n_eff.float().detach().numpy(),label='data death')
plt.plot(np.arange(data.shape[0]),outputs.detach().numpy()[:,1],label='hospitalized')
plt.plot(np.arange(data.shape[0]),outputs.detach().numpy()[:,2],label='death')
plt.legend()
plt.show()





            