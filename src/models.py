"""implement models in:
A mathematical model for the spatiotemporal epidemic spreading of COVID19
"""
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro import SVI,TraceTMC_ELBO,TraceEnum_ELBO,JitTraceEnum_ELBO
import torch
from torch.distributions import constraints
import data_io

args = {'hidden_dim'：6}

def model_swiss(data, lengths, include_prior=True):
    assert not torch._C._get_tracing_state()
    length, data_dim = sequences.shape
    hidden_dim = 6
    with poutine.mask(mask=include_prior):
        # Our prior on transition probabilities will be:
        # stay in the same state with 90% probability; uniformly jump to another
        # state with 10% probability.
        probs_x = pyro.sample("probs_x",
                              dist.Dirichlet([[0.9,0.1,0,0,0,0],
                                              [0,0.8,0.2,0,0,0],
                                              [0,0,0.5,0.1,0.4,0],
                                              [0,0,0,0.9,0.08,0.02],
                                              [0,0,0,0,1,0],
                                              [0,0,0,0,0,1] ])
                                  .to_event(1))
        # We put a weak prior on the conditional probability of a tone sounding.
        # We know that on average about 4 of 88 tones are active, so we'll set a
        # rough weak prior of 10% of the notes being active at any one time.
        probs_y = lambda x: torch.mm(torch.tensor([[0,0,1,1,0,0],
                                            [0,0,0,0,1,0],
                                            [0,0,0,0,0,1]]),x.view(hidden_dim,-1))
    # In this first model we'll sequentially iterate over sequences in a
    # minibatch; this will make it easy to reason about tensor shapes.
    obs_plate = pyro.plate("tones", data_dim, dim=-1)
    x = 0
    for t in pyro.markov(range(length)):
        # On the next line, we'll overwrite the value of x with an updated
        # value. If we wanted to record all x values, we could instead
        # write x[t] = pyro.sample(...x[t-1]...).
        x = pyro.sample("x_{}".format(t), dist.Categorical(probs_x[x]),
                        infer={"enumerate": "parallel"})
        with obs_plate:
            pyro.sample("y_{}".format(t), probs_y(x):,
                        obs=sequence[t])

def guide():
    p_se = pyro.param("p_se",torch.tensor(0.1),constraints=constraints.positive)
    P_ei = pyro.param("p_ei",torch.tensor(0.2),constraints=constraints.positive)
    p_ih = pyro.param("p_ih",torch.tensor(0.2),constraints=constraints.positive)
    p_ir = pyro.param("p_ir",torch.tensor(0.2),constraints=constraints.positive)
    p_hd = pyro.param("p_hd",torch.tensor(0.2),constraints=constraints.positive)
    p_hr = pyro.param("p_hr",torch.tensor(0.2),constraints=constraints.positive)

    probs_x = pyro.sample("probs_x",
                        dist.Dirichlet([[1-p_se,p_se,0,0,0,0],
                                        [0,1-p_ei,p_ei,0,0,0],
                                        [0,0,1-p_ih-p_ir,p_ih,p_ir,0],
                                        [0,0,0,1-p_hr-p_hd,p_hr,p_hd],
                                        [0,0,0,0,1,0],
                                        [0,0,0,0,0,1] ]).to_event(1)))
    return torch.mm(torch.tensor([[0,0,1,1,0,0],
                                  [0,0,0,0,1,0],
                                  [0,0,0,0,0,1]]),x.view(hidden_dim,-1))

svi = pyro.infer.SVI(model=model,
                     guide=scale_parametrized_guide,
                     optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
                     loss=TraceTMC_ELBO())

losses, p_se,p_ei,p_ih,p_ir,p_hd,p_hr  = [], [], [],[],[],[],[]
num_steps = 2500
data = data_io.load_data()
for t in range(num_steps):
    losses.append(svi.step(guess))
    p_se.append(pyro.param("p_se").item())
    p_ei.append(pyro.param("p_ei").item())
    p_ih.append(pyro.param("p_ih").item())
    p_ir.append(pyro.param("p_ir").item())
    p_hd.append(pyro.param("p_hd").item())
    p_hr.append(pyro.param("p_hr").item())


    