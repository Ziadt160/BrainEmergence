import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 2e-3
EPOCHS = 5
TIME_STEPS = 20  # Number of simulation steps per image
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Surrogate Gradient Function ---
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # FastSigmoid surrogate gradient
        grad = grad_output / (10 * torch.abs(input) + 1.0)**2
        return grad

spike_func = SurrogateSpike.apply

class LearningSNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Trainable Parameters
        self.w_in = nn.Linear(input_size, hidden_size)
        self.w_rec = nn.Linear(hidden_size, hidden_size)
        self.w_out = nn.Linear(hidden_size, output_size)
        
        # Trainable LIF parameters per neuron
        self.tau = nn.Parameter(torch.ones(hidden_size) * 0.9)
        self.v_thresh = nn.Parameter(torch.ones(hidden_size) * 1.0)
        
    def forward(self, x):
        # Initial states
        mem = torch.zeros(x.shape[0], self.hidden_size).to(DEVICE)
        spk = torch.zeros(x.shape[0], self.hidden_size).to(DEVICE)
        output_spikes = torch.zeros(x.shape[0], 10).to(DEVICE)
        
        # Simulation Loop (BPTT)
        for _ in range(TIME_STEPS):
            # Current injection
            current = self.w_in(x) + self.w_rec(spk)
            
            # Leaky Integrate-and-Fire Dynamics
            mem = mem * self.tau + current - spk * self.v_thresh
            
            # Spike generation using Surrogate Gradient
            spk = spike_func(mem - self.v_thresh)
            
            # Accumulate output spikes for classification
            output_spikes += self.w_out(spk)
            
        return output_spikes / TIME_STEPS

# --- Training Loop ---
def train():
    print(f"Training on {DEVICE}...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    model = LearningSNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping to prevent "Spike Explosion"
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            if i % 200 == 0:
                print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f}")
        
        # Evaluation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                correct += (output.argmax(dim=1) == target).sum().item()
        
        accuracy = correct / len(test_ds)
        print(f"--- Epoch {epoch} Complete | Test Accuracy: {accuracy:.4f} ---")

if __name__ == "__main__":
    train()