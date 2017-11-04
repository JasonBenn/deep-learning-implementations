import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor, nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch

actual_dist_mean = 3
actual_dist_stddev = 2
num_samples = 1000

sample_real_dist = lambda: Variable(Tensor(np.random.normal(actual_dist_mean, actual_dist_stddev, num_samples)))
sample_fake_data = lambda: Variable(Tensor(np.random.rand(num_samples)))


class Generator(nn.Module):
    """
    Transforms a weird distribution into one that is likely to fool the Discriminator.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        return self.linear3(x)


class Discriminator(nn.Module):
    """
    Tries to tell the difference between the real distribution, and a fake distribution output by the Generator network.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        return F.sigmoid(self.linear3(x))

hidden_size = 32
output_size = 1   # 0 is fake dist, 1 is real dist

G = Generator(num_samples, hidden_size, num_samples)  # notice that num_samples is the output, because the generator's output is a sample for the discriminator
D = Discriminator(num_samples, hidden_size, 1)


num_epochs = 25000
print_interval = 500
g_steps = 1
d_steps = 1  # apparently you can do more d steps than g steps if you want
criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(G.parameters())
d_optimizer = torch.optim.Adam(D.parameters())

true_label = Variable(torch.ones(1))
false_label = Variable(torch.zeros(1))

for epoch in range(num_epochs):

    for d_step in range(d_steps):
        D.zero_grad()
        # Discriminator gets trained with real data, labeled correctly...
        real_data = sample_real_dist()
        real_outputs = D(real_data)
        true_real_loss = criterion(real_outputs, true_label)
        true_real_loss.backward()

        # and fake data, labeled correctly...
        fake_inputs = G(sample_fake_data())
        fake_outputs = D(fake_inputs)
        false_fake_loss = criterion(fake_outputs, false_label)
        false_fake_loss.backward()
        d_optimizer.step()

    for g_step in range(g_steps):
        G.zero_grad()
        # And then we train the generator by passing its outputs and _incorrect_ labels to the discriminator.
        fake_inputs = G(sample_fake_data())
        fake_outputs = D(fake_inputs)
        true_fake_loss = criterion(fake_outputs, true_label)  # confusing!
        true_fake_loss.backward()
        g_optimizer.step()

    if epoch % print_interval == 0:
        dist = fake_inputs.data.numpy()
        print("{}: D loss: {:.2f}/{:.2f}\tG loss: {:.2f}\tG μ: {:.2f} (real: {})\tG σ: {:.2f} (real: {})".format(
            epoch,
            true_real_loss.data[0],
            false_fake_loss.data[0],
            true_fake_loss.data[0],
            np.mean(dist),
            actual_dist_mean,
            np.std(dist),
            actual_dist_stddev
        ))
