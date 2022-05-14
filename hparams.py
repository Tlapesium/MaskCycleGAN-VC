import torch

# learning settings
l_cycle = 10
l_ident = 5
ident_end = 10000
batch_size = 1
learn_length = 64
d_optimizer = torch.optim.Adam
g_optimizer = torch.optim.Adam
disc_lr = 1e-4
gen_lr = 2e-4
beta = (0.5,0.999)
epochs = 50

# fft settings
sr = 22050
fft_size = 1024
hop_size = 256
win_size = 1024
fmin = 0
fmax = None
n_mels = 80
