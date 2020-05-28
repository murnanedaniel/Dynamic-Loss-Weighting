# Dynamic-Loss-Weighting

A small collection of tools to manage deep learning with multiple sources of loss. Based on the paper [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115) (Kendall, et al; 2017). 

## Context

Given $$e^5$$ 

## Usage

The model `MultiNoiseLoss()` is implemented as a torch module. A typical use case is 
```
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)  ## Net is some torch module, e.g. with an mlp layer Net.mlp
multi_loss = MultiNoiseLoss(n_losses=2).to(device)

optimizer = torch.optim.Adam([
    {'params': model.mlp.parameters()},
    {'params': multi_loss.noise_params}], lr = 0.001)

    
lambda1 = lambda ep: 1 / (2**(ep+11)//10)
lambda2 = lambda ep: 1
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
```

N.B. We have to include the dynamic weighting parameters in the optimizer, so that they are also updated, as well as handling their learning rate. We could of course increment that LR in the steps as the rest of the model, but this may not be desirable, and the LR of the weighting really depends on the situation. Anecdotally: The learning rate for the dynamic weights is not so important, and it can stay at LR=1e-2 and seems to work fine. 