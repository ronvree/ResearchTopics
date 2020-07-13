
import argparse
import tqdm

from v2.util.experience_replay import ExperienceReplay


import torch
from torch.nn import Module

from torch.utils.data import DataLoader


class Trainer:

    # TODO -- data augmentation

    def __init__(self, dataset: ExperienceReplay, args: argparse.Namespace):

        # Store a reference to the dataset
        self._dataset = dataset

    def train(self,
              model: Module,
              optimizer: torch.optim.Optimizer,
              batch_size: int,
              num_iters: int = 1,
              ) -> tuple:

        info = dict()  # TODO

        # Convert the dataset to a torch.utils.data.TensorDataSet
        dataset = self._dataset.as_dataset()
        # Build an iterable over the dataset
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 )
        # Set the model to train mode
        model.train()
        # Do optimization loops
        for i_opt in range(num_iters):
            # Build progress bar
            loop = tqdm.tqdm(enumerate(dataloader),
                             total=len(dataloader),
                             desc='Train q-net')
            # Loop over the data
            for i_batch, (o, a, r, o_, a_) in loop:
                # Estimate q values
                qs = model.forward(o, a)
                # Compute targets from the dataset
                with torch.no_grad():
                    targets = r + model.forward(o_, a_)
                # Reset all gradients
                optimizer.zero_grad()
                # Compute the loss
                loss = torch.nn.functional.l1_loss(qs, targets)  # TODO -- l2 loss?
                # Compute the gradient
                loss.backward()
                # Use the gradient to optimize the parameters
                optimizer.step()

                # Update progress bar
                loop.set_postfix_str(
                    ''.join([
                        f'iter {i_opt}/{num_iters}',
                        f', batch {i_batch}/{len(dataloader)}',
                        f', loss: {loss.item():.3f} '
                    ])
                )

        return model, info








