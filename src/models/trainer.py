"""Trainign utilities for torch models."""


import torch

from tqdm import tqdm


def train(model, train_loader, criterion, device="cuda", n_epochs=50):

    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

    pbar = tqdm(range(n_epochs))

    for epoch_index in pbar:

        model.train()

        epoch_loss = 0.

        for batch in train_loader:

            # Read data
            img_1, img_2, label = batch

            img_1 = img_1.float().to(device)
            img_2 = img_2.float().to(device)
            label = label.to(device).long()

            # Predict with the model
            optimizer.zero_grad()
            output = model(img_1, img_2)

            # Compute loss
            loss = criterion(output, label)
            epoch_loss += loss.item()

            # Optimization step
            loss.backward()
            optimizer.step()

        epoch_loss /= len(train_loader)

        pbar.set_description(f"Epoch {epoch_index+1}/{n_epochs}; Loss {epoch_loss:.6f}")

        scheduler.step()

        model.eval()
