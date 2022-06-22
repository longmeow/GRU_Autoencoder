import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy


class ModelTrainer():
    def __init__(self, model, train_data, device, config):
        self.model = model
        self.train_data = train_data
        self.device = device
        self.config = config
        self.train_loss_list = list()
        self.min_loss = float('inf')
        self.best_model = None
        self.best_optimizer = None

    def train_epoch(self, criterion, optimizer, epoch, device, h1, h2, h3, h4, h5):
        train_loss = 0.0
        self.model.train()
        for idx, x in enumerate(self.train_data):
            self.model.zero_grad()
            out = self.model(x.to(device).float(), h1, h2, h3, h4, h5)
            loss = criterion(out.float(), x.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss = train_loss / len(self.train_data)
        self.train_loss_list.append(train_loss)

        if train_loss < self.min_loss:
                self.min_loss = train_loss
                self.best_model = deepcopy(self.model.state_dict())
                self.best_optimizer = deepcopy(optimizer.state_dict())
                self.best_epoch_in_round = epoch
    
    def train(self):
        self.model.to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['lr'])
        criterion = nn.MSELoss()

        for epoch in range(1, self.config['n_epochs']+1):
            h1 = self.model.init_hidden(int(self.config['embed_dim'])*4).float()
            h2 = self.model.init_hidden(int(self.config['embed_dim'])*2).float()
            h3 = self.model.init_hidden(int(self.config['embed_dim'])).float()
            h4 = self.model.init_hidden(int(self.config['embed_dim'])*2).float()
            h5 = self.model.init_hidden(int(self.config['embed_dim'])*4).float()
            self.train_epoch(criterion, optimizer, epoch, self.device, h1, h2, h3, h4, h5)
        
        torch.save(self.best_model, self.config["checkpoint_dir"] + "best_model.pt")
        torch.save(self.best_optimizer, self.config["checkpoint_dir"] + "best_optim.pt")



    