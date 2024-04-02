from utils import pre
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from model import VCModel

# use if training
#pre.process('dataset', 'logs/', 'test')

class SpectrogramDataset(Dataset):
    def __init__(self, specs_dir):
        self.specs_dir = specs_dir
        self.spec_files = [f for f in os.listdir(specs_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.spec_files)

    def __getitem__(self, idx):
        spec_file = self.spec_files[idx]
        spec_path = os.path.join(self.specs_dir, spec_file)
        spec = np.load(spec_path)

        if len(spec.shape) == 4:
            spec = spec.transpose(0, 3, 1, 2)
        spec = torch.tensor(spec, dtype=torch.float32)

        return spec

def train_model(model, dataloader, num_epochs, save_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, data in enumerate(dataloader, 0):
            inputs = data.unsqueeze(1).to(device)
            targets = data.to(device)
            if inputs.shape != targets.shape:
                inputs = inputs.view(targets.shape)
            inputs = inputs.squeeze(1)
            outputs = model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    torch.save(model.state_dict(), save_path)
    print('Finished Training')

specs_dir = 'logs/test/specs/'
dataset = SpectrogramDataset(specs_dir)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = VCModel()

train_model(model, dataloader, num_epochs=10, save_path='model.pth')
