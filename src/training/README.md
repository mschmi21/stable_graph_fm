# Trainer

This folder contains the flow-matching training routine.

At each training step, the trainer samples a time $t$, samples an initial point $x_0$ from the prior, constructs the conditional path sample $x_t$ and target vector field $u_t$, and trains the model using mean-squared error.

## Pseudocode

```text
for each batch z:

    sample t ~ Uniform(0, 1)
    sample x_0 ~ prior

    x_t      = path.sample_xt(x_0, z, t)
    u_target = path.target_vector_field(x_0, z, t)

    u_pred = model(x_t, t, L)

    loss = MSE(u_pred, u_target)

    backpropagate loss
    update model parameters