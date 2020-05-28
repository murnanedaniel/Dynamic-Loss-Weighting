# Dynamic-Loss-Weighting

A small collection of tools to manage deep learning with multiple sources of loss. Based on the paper [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115) (Kendall, et al; 2017). 

## Context

Many deep learning situations may call for handling multiple sources of loss. These may be from independent tasks, or for loss from different parts of the same network. The naive solution is to simple take a sum ![\mathcal{L}=\mathcal{L}_1+...+\mathcal{L}_n](https://render.githubusercontent.com/render/math?math=%5Cmathcal%7BL%7D%3D%5Cmathcal%7BL%7D_1%2B...%2B%5Cmathcal%7BL%7D_n), however there's no reason to think that an equal weighting of losses should be optimal. The paper by Kendall, et al., derives a formula for combining losses from regression tasks and classification tasks in terms of "noise parameters" ![\sigma_i](https://render.githubusercontent.com/render/math?math=%5Csigma_i), which are to be variables in the loss minimisation. These are essentially a way to regularise the loss as:

![\mathcal{L}=\frac{1}{2(\sigma^{reg.}_1)^2}\mathcal{L}^{reg.}_1 + \frac{1}{2(\sigma^{reg.}_2)^2}\mathcal{L}^{reg.}_2 + ... + \frac{1}{(\sigma^{class.}_1)^2}\mathcal{L}^{class.}_1 + \frac{1}{(\sigma^{class.}_2)^2}\mathcal{L}^{class.}_2 + ... + \log{\sigma^{reg.}_1} + \log{\sigma^{reg.}_2} + ... + \log{\sigma^{class.}_1} + \log{\sigma^{class.}_2} + ...](https://render.githubusercontent.com/render/math?math=%5Cmathcal%7BL%7D%3D%5Cfrac%7B1%7D%7B2(%5Csigma%5E%7Breg.%7D_1)%5E2%7D%5Cmathcal%7BL%7D%5E%7Breg.%7D_1%20%2B%20%5Cfrac%7B1%7D%7B2(%5Csigma%5E%7Breg.%7D_2)%5E2%7D%5Cmathcal%7BL%7D%5E%7Breg.%7D_2%20%2B%20...%20%2B%20%5Cfrac%7B1%7D%7B(%5Csigma%5E%7Bclass.%7D_1)%5E2%7D%5Cmathcal%7BL%7D%5E%7Bclass.%7D_1%20%2B%20%5Cfrac%7B1%7D%7B(%5Csigma%5E%7Bclass.%7D_2)%5E2%7D%5Cmathcal%7BL%7D%5E%7Bclass.%7D_2%20%2B%20...%20%2B%20%5Clog%7B%5Csigma%5E%7Breg.%7D_1%7D%20%2B%20%5Clog%7B%5Csigma%5E%7Breg.%7D_2%7D%20%2B%20...%20%2B%20%5Clog%7B%5Csigma%5E%7Bclass.%7D_1%7D%20%2B%20%5Clog%7B%5Csigma%5E%7Bclass.%7D_2%7D%20%2B%20...)

Simply, we weight each regression loss by ![\frac{1}{2\sigma^2}](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D) and each classification loss by ![\frac{1}{\sigma^2}](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B1%7D%7B%5Csigma%5E2%7D) and add a regulariser of ![\log{\sigma}](https://render.githubusercontent.com/render/math?math=%5Clog%7B%5Csigma%7D). Without this, the optimiser would take all ![\sigma_i\rightarrow\infty](https://render.githubusercontent.com/render/math?math=%5Csigma_i%5Crightarrow%5Cinfty). 




## Usage

The model `MultiNoiseLoss()` is implemented as a torch module. A typical use case (with two classification losses, for example) is 
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

The loss is then called in the training loop as
```
...
loss_1 = cross_entropy(predictions_1, targets_1)
loss_2 = cross_entropy(predictions_2, targets_2)

loss = multi_loss([loss_1, loss_2])
loss.backward()
optimizer.step()
...
```

