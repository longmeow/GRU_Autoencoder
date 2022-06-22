import torch.nn as nn


class GRUAutoencoder(nn.Module):
    def __init__(self, d_model, embed_dim, n_layers, l_win, batch_size, p_drop=0.2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers #Default = 1 only for n_layers=1, with dropout model needs n_layers > 1
        self.hidden_dim_1 = embed_dim * 4
        self.hidden_dim_2 = embed_dim * 2 
        self.embed_dim = embed_dim
        self.l_win = l_win
        self.batch_size = batch_size
        
        #GRU
        self.gru1 = nn.GRU(self.d_model, self.hidden_dim_1, self.n_layers, batch_first=True, dropout=p_drop)
        self.gru2 = nn.GRU(self.hidden_dim_1, self.hidden_dim_2, self.n_layers, batch_first=True, dropout=p_drop)
        self.gru3 = nn.GRU(self.hidden_dim_2, self.embed_dim, self.n_layers, batch_first=True, dropout=p_drop)
        self.gru4 = nn.GRU(self.embed_dim, self.hidden_dim_2, self.n_layers, batch_first=True, dropout=p_drop)
        self.gru5 = nn.GRU(self.hidden_dim_2, self.hidden_dim_1, self.n_layers, batch_first=True, dropout=p_drop)
        
        #Linear
        self.inp_len = self.l_win * self.hidden_dim_1
        self.out_len = self.l_win * self.d_model
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.inp_len, self.out_len),
            nn.Unflatten(1, (self.l_win, self.d_model))
        )
    def forward(self, x, h1, h2, h3, h4, h5):
        out, h1 = self.gru1(x, h1)
        out, h2 = self.gru2(out, h2)
        out, h3 = self.gru3(out, h3)
        out, h4 = self.gru4(out, h4)
        out, h5 = self.gru5(out, h5)
        out = self.linear(out)
        return out
    
    def init_hidden(self, hidden_dim):
        #Init hidden_state 
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, hidden_dim).zero_()
        return hidden


def create_gru_autoencoder(d_model, embed_dim, n_layers, l_win, batch_size, p_drop=0.2):
    model = GRUAutoencoder(d_model, embed_dim, n_layers, l_win, batch_size, p_drop=0.2)
    return model

