# Dynamic-Loss-Weighting

A small collection of tools to manage deep learning with multiple sources of loss. Based on the paper [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115) (Kendall, et al; 2017). 

## Context

Many deep learning situations may call for handling multiple sources of loss. These may be from independent tasks, or for loss from different parts of the same network. The naive solution is to simple take a sum $\mathcal{L}=\mathcal{L}_1+...+\mathcal{L}_n$, however there's no reason to think that an equal weighting of losses should be optimal. A dominant approach of works is combining multi objective losses with weighted linear sum of the losses for each individual task, $\mathcal{L} = \sum_{i=1}^{n} w_i \mathcal{L}_i$, where $w_i$ are The paper by Kendall, et al., derives a formula for combining losses from regression tasks and classification tasks in terms of learnable "noise parameters" $\sigma_i$, which are to be variables in the loss minimisation. These are essentially a way to regularise the loss as:

$$\mathcal{L}=\frac{1}{2(\sigma^{reg.}_1)^2}\mathcal{L}^{reg.}_1 + \frac{1}{2(\sigma^{reg.}_2)^2}\mathcal{L}^{reg.}_2 + \ldots + \frac{1}{(\sigma^{class.}_1)^2}\mathcal{L}^{class.}_1 + \frac{1}{(\sigma^{class.}_2)^2}\mathcal{L}^{class.}_2 + \ldots + \log{\sigma^{reg.}_1} + \log{\sigma^{reg.}_2} + \ldots + \log{\sigma^{class.}_1} + \log{\sigma^{class.}_2} + \ldots $$

Simply, we weight each regression loss by $\frac{1}{2\sigma^2}$ and each classification loss by $\frac{1}{\sigma^2}$. Without this, the optimiser would take all $\sigma_i \rightarrow \infty$. 

Our implementation adapts Kendall's loss weighting approach using learnable parameters $\eta_i$, where for each loss term $\matchal{L}_i$, $\text{loss\_component}_i = \frac{1}{\sqrt{\eta_i}} \mathcal{L}_i + \log(\eta_i)$ and total loss $$\mathcal{L} = \sum_{i=1}^{k} \left\[ \frac{1}{\sqrt{\eta_i}} \cdot \mathcal{L}_i + \log(\eta_i) \right\]$$

This allows optimal learning of weights for any type of loss in a multi-task learning setup, without predefining specific treatment for regression or classification tasks.

## Usage

The model `MultiNoiseLoss()` is implemented as a torch module. A typical use case (with two classification losses, for example) is

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Net().to(device)  ## Net is some torch module, e.g. with an mlp layer Net.mlp
multi_loss = MultiNoiseLoss(n_losses=2).to(device)

optimizer = torch.optim.Adam([
    {'params': model.mlp.parameters()},
    {'params': multi_loss.noise_params}], lr = 0.001)

    
lambda1 = lambda ep: 1 / (2**((ep+11)//10))
lambda2 = lambda ep: 1
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
```

N.B. We have to include the dynamic weighting parameters in the optimizer, so that they are also updated, as well as handling their learning rate. We could of course increment that LR in the steps as the rest of the model, but this may not be desirable, and the LR of the weighting really depends on the situation. Anecdotally: The learning rate for the dynamic weights is not so important, and it can stay at LR=1e-2 and seems to work fine. 

The loss is then called in the training loop as

```python
...
loss_1 = cross_entropy(predictions_1, targets_1)
loss_2 = cross_entropy(predictions_2, targets_2)

loss = multi_loss([loss_1, loss_2])
loss.backward()
optimizer.step()
...
```

## Notes

In the original paper, regression tasks are weighted by $\frac{1}{2\sigma_i^2}$ and include a $\log(sigma_i)$ term. 
Our implementation does not apply the factor of $0.5$ (which is directed for regression tasks) but can be adapted to do so.
