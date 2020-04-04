"""implement models in:
A mathematical model for the spatiotemporal epidemic spreading of COVID19
"""
import pyro
import pyro.distributions as dist
import numpy as np
from pyro import poutine
from pyro.infer import SVI,TraceTMC_ELBO,TraceEnum_ELBO,JitTraceEnum_ELBO
import torch
from torch.distributions import constraints
import data_io

def model_swiss(data, include_prior=True):
    assert not torch._C._get_tracing_state()
    length, data_dim = data.shape
    hidden_dim = 6
    with poutine.mask(mask=include_prior):
        # Our prior on transition probabilities will be:
        # stay in the same state with 90% probability; uniformly jump to another
        # state with 10% probability.
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet(torch.Tensor([[0.9,0.1,0,0,0,0],
                                              [0,0.8,0.2,0,0,0],
                                              [0,0,0.5,0.1,0.4,0],
                                              [0,0,0,0.9,0.08,0.02],
                                              [0,0,0,0,1,0],
                                              [0,0,0,0,0,1]])).to_event(1))
        # fixed observation matrix
        probs_y = lambda x: torch.mm(torch.tensor(np.asarray([[0.,0,1.,1,0,0],
                                                            [0,0,0,1,0,0],
                                                            [0,0,0,0,0,1]],dtype=np.float32)),x.view(hidden_dim,-1))
    # In this first model we'll sequentially iterate over sequences in a
    # minibatch; this will make it easy to reason about tensor shapes.
    obs_plate = pyro.plate("tones", data_dim, dim=-1)
    x = pyro.sample("x_init",dist.Dirichlet(torch.Tensor([0.90,0.05,0.05,0,0,0])))
    for t in pyro.markov(range(length)):
        # On the next line, we'll overwrite the value of x with an updated
        # value. If we wanted to record all x values, we could instead
        # write x[t] = pyro.sample(...x[t-1]...).
        x = torch.mm(probs_x,x.view(hidden_dim,1))
        with obs_plate:
            pyro.sample("y_{}".format(t), probs_y(x),
                        obs=data[t,:])

def guide(data):
    p_0s = pyro.param("p_0s",torch.tensor(0.1),constraint=constraints.positive)
    p_0e = pyro.param("p_0e",torch.tensor(0.1),constraint=constraints.positive)
    p_0i = pyro.param("p_0i",torch.tensor(0.1),constraint=constraints.positive)
    p_0h = pyro.param("p_0h",torch.tensor(0.1),constraint=constraints.positive)
    p_0r = pyro.param("p_0r",torch.tensor(0.1),constraint=constraints.positive)
    x = pyro.sample("x_init",dist.Dirichlet(torch.Tensor([p_0s,p_0e,p_0i,p_0h,p_0r,1-p_0s-p_0e-p_0i-p_0h-p_0r])))

    p_se = pyro.param("p_se",torch.tensor(0.1),constraint=constraints.positive)
    p_ei = pyro.param("p_ei",torch.tensor(0.2),constraint=constraints.positive)
    p_ih = pyro.param("p_ih",torch.tensor(0.2),constraint=constraints.positive)
    p_ir = pyro.param("p_ir",torch.tensor(0.2),constraint=constraints.positive)
    p_hd = pyro.param("p_hd",torch.tensor(0.2),constraint=constraints.positive)
    p_hr = pyro.param("p_hr",torch.tensor(0.2),constraint=constraints.positive)

    probs_x = pyro.sample("probs_x",
                        dist.Dirichlet(torch.Tensor([[1-p_se,p_se,0,0,0,0],
                                        [0,1-p_ei,p_ei,0,0,0],
                                        [0,0,1-p_ih-p_ir,p_ih,p_ir,0],
                                        [0,0,0,1-p_hr-p_hd,p_hr,p_hd],
                                        [0,0,0,0,1,0],
                                        [0,0,0,0,0,1]])).to_event(1))
    return probs_x, x

svi = pyro.infer.SVI(model=model_swiss,
                     guide=guide,
                     optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
                     loss=TraceTMC_ELBO())

losses, p_se,p_ei,p_ih,p_ir,p_hd,p_hr  = [], [], [],[],[],[],[]
num_steps = 2500
# load data
data = torch.Tensor(data_io.load_data())/8570000
for t in range(num_steps):
    losses.append(svi.step(data))
    p_se.append(pyro.param("p_se").item())
    p_ei.append(pyro.param("p_ei").item())
    p_ih.append(pyro.param("p_ih").item())
    p_ir.append(pyro.param("p_ir").item())
    p_hd.append(pyro.param("p_hd").item())
    p_hr.append(pyro.param("p_hr").item())


plt.plot(losses)
plt.title("ELBO")
plt.xlabel("step")
plt.ylabel("loss")
plt.show()


    