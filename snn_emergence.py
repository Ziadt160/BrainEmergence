
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

# --- Configuration & Constants ---
GRID_H, GRID_W = 50, 50
N_NEURONS = GRID_H * GRID_W
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neuron Parameters (LIF)
DT = 1.0          # ms (simulation step)
TAU = 20.0        # ms (membrane time constant)
V_REST = 0.0      # mV
V_RESET = 0.0     # mV
V_THRESH = 0.5    # mV (Lower threshold to encourage firing)
REFRACTORY_T = 3.0 # ms (Faster recovery)

# Connectivity: Mexican Hat
W_EXC_MAX = 2.5   # (Stronger excitation to punch through)
W_INH_MAX = 0.4   # (Weaker inhibition to prevent the 'moat')
CONN_RADIUS = 20.0

# Narrower inhibition to prevent broad suppression
SIGMA_EXC = 2.0
SIGMA_INH = 4.0   # (Was 8.0 - tightening this stops the 'wall' effect)

# STDP Parameters
A_PLUS = 0.01
A_MINUS = 0.012
TAU_STDP = 20.0   # ms
LR_SCALE = 0.1    # Global learning rate scaler

# Synaptic Scaling
TARGET_SUM = 1.5  # Target sum of weights per neuron

# Zones (Percent of height)
SENSORY_PCT = 0.2
MOTOR_PCT = 0.2
# Association is the rest

class NeuroGrid:
    def __init__(self, height=GRID_H, width=GRID_W, device=DEVICE,
                 sigma_exc=SIGMA_EXC, sigma_inh=SIGMA_INH, conn_radius=CONN_RADIUS,
                 w_exc=W_EXC_MAX, w_inh=W_INH_MAX, scale_every=1, gain=3.0):
        self.H = height
        self.W = width
        self.N = height * width
        self.device = device

        # Configurable connectivity / homeostasis (default to module constants).
        # Broader excitation + longer radius let signal traverse the sheet; a
        # larger scale_every applies synaptic scaling less often so STDP can
        # actually imprint structure before homeostasis normalizes it away.
        self.sigma_exc = sigma_exc
        self.sigma_inh = sigma_inh
        self.conn_radius = conn_radius
        self.w_exc = w_exc
        self.w_inh = w_inh
        self.scale_every = scale_every
        self.gain = gain
        self._step = 0

        # --- State Vectors ---
        self.v = torch.ones(self.N, device=device) * V_REST
        self.spikes = torch.zeros(self.N, device=device)
        self.refractory_timer = torch.zeros(self.N, device=device)
        
        # STDP Traces
        self.trace_pre = torch.zeros(self.N, device=device)
        self.trace_post = torch.zeros(self.N, device=device)
        
        # --- Zones Definition ---
        # Coordinate grid: (y, x)
        y_coords = torch.arange(self.H, device=device).repeat_interleave(self.W)
        x_coords = torch.arange(self.W, device=device).repeat(self.H)
        self.coords = torch.stack([y_coords, x_coords], dim=1).float() # (N, 2)
        
        # Masks for zones
        # Sensory at Bottom (High Y), Motor at Top (Low Y)
        self.motor_mask = y_coords < (self.H * MOTOR_PCT)
        self.sensory_mask = y_coords >= (self.H * (1.0 - SENSORY_PCT))
        # Association is ~sensory & ~motor
        self.association_mask = ~(self.sensory_mask | self.motor_mask)
        
        print(f"Zones Initialized: Sensory={self.sensory_mask.sum().item()}, "
              f"Assoc={self.association_mask.sum().item()}, Motor={self.motor_mask.sum().item()}")

        # --- Connectivity Initialization (Mexican Hat) ---
        print("Initializing Weights...")
        self.weights, self.mask = self._init_weights_mexican_hat()
        print("Weights Initialized.")

    def _init_weights_mexican_hat(self):
        # Compute pairwise distances (N, N) - memory heavy but okay for N=2500
        # For larger N, this should be done in chunks or implicit convolution.
        
        # Using broadcasting to get differences
        # shape: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        diff = self.coords.unsqueeze(1) - self.coords.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2) # (N, N)
        dist = torch.sqrt(dist_sq)
        
        # Mexican Hat Kernel
        # E.g. A*exp(-d^2/s1) - B*exp(-d^2/s2)
        w_exc = self.w_exc * torch.exp(-dist_sq / (2 * self.sigma_exc**2))
        w_inh = self.w_inh * torch.exp(-dist_sq / (2 * self.sigma_inh**2))
        weights = w_exc - w_inh

        # Remove self-connections
        weights.fill_diagonal_(0.0)

        # Create Topology Mask (Sparse Radius)
        mask = (dist <= self.conn_radius)
        mask.fill_diagonal_(0) # No self connections in mask either
        
        weights = weights * mask.float()
        
        return weights, mask

    def reset_state(self):
        """Reset neural state (membrane, spikes, traces, refractory) but KEEP the
        learned weights. Used between stimulus presentations so one image does not
        bleed into the next, while STDP structure accumulates across images."""
        self.v = torch.ones(self.N, device=self.device) * V_REST
        self.spikes = torch.zeros(self.N, device=self.device)
        self.refractory_timer = torch.zeros(self.N, device=self.device)
        self.trace_pre = torch.zeros(self.N, device=self.device)
        self.trace_post = torch.zeros(self.N, device=self.device)

    def dynamics_step(self, stim_intensity=1.0, external_input=None, learn=True, return_maps=True):
        """
        Run one simulation step.

        external_input: optional (N,) current vector. When provided it drives the
            sensory zone with an externally encoded stimulus (e.g. a digit) instead
            of the default random Poisson noise.
        learn: when False, STDP weight updates and synaptic scaling are skipped, so
            connectivity is frozen (used while recording / decoding scans).
        return_maps: when False, skip the host copy and return None (faster).
        """
        self._step += 1

        # 1. Input Injection (Sensory)
        input_current = torch.zeros(self.N, device=self.device)

        if external_input is None:
            # Default: random Poisson-like input to the sensory zone
            sensory_noise = torch.rand(self.N, device=self.device)
            input_current += (self.sensory_mask.float() * (sensory_noise < (0.1 * stim_intensity)).float() * 5.0)
        else:
            # Externally encoded stimulus, already shaped over the (sensory) neurons
            input_current += external_input

        input_current += torch.rand(self.N, device=self.device) * 0.3 # Moderate noise
        
        # 2. Synaptic Input (Recurrent)
        # Multiply by a gain factor to ensure propagation
        synaptic_current = torch.mv(self.weights, self.spikes) * self.gain
        
        total_current = input_current + synaptic_current
        
        # 3. LIF Update
        # Refractory logic
        is_refractory = self.refractory_timer > 0
        
        # dv = dt/tau * (v_rest - v + current)
        delta_v = (DT / TAU) * (V_REST - self.v + total_current)
        
        # Update v only if not refractory
        self.v = torch.where(is_refractory, self.v, self.v + delta_v)
        
        # Update timers
        self.refractory_timer = torch.clamp(self.refractory_timer - DT, min=0)
        
        # 4. Spike Generation
        # Softmax-like probability or hard threshold?
        # Standard LIF is hard threshold.
        spikes_now = (self.v >= V_THRESH)
        
        # Reset spiked neurons
        self.v[spikes_now] = V_RESET
        self.refractory_timer[spikes_now] = REFRACTORY_T
        
        # 5. STDP & Learning
        # Update traces (cheap vector op; kept in sync every step)
        self.trace_pre = self.trace_pre * np.exp(-DT / TAU_STDP) + spikes_now.float()
        self.trace_post = self.trace_post * np.exp(-DT / TAU_STDP) + spikes_now.float()

        if learn:
            # Weight Update (pair-based STDP via outer products)
            spike_post_col = spikes_now.float().unsqueeze(1) # (N, 1)
            trace_pre_row = self.trace_pre.unsqueeze(0)      # (1, N)
            ltp = torch.mm(spike_post_col, trace_pre_row)    # [i,j] = post_i * trace_j

            spike_pre_row = spikes_now.float().unsqueeze(0)  # (1, N)
            trace_post_col = self.trace_post.unsqueeze(1)    # (N, 1)
            ltd = torch.mm(trace_post_col, spike_pre_row)    # [i,j] = trace_i * spike_j

            dw = LR_SCALE * (A_PLUS * ltp - A_MINUS * ltd)

            # Apply update only to masked connections
            self.weights += dw * self.mask.float()

            # Clamp weights
            self.weights = torch.clamp(self.weights, min=-1.0, max=1.0)

            # 6. Synaptic Scaling (Homeostasis) -- applied every scale_every steps
            if self._step % self.scale_every == 0:
                with torch.no_grad():
                    row_sums = self.weights.abs().sum(dim=1, keepdim=True) + 1e-6
                    scale_factor = TARGET_SUM / row_sums
                    # Soft update to avoid harsh oscillations
                    self.weights = self.weights * (1.0 - 0.01) + (self.weights * scale_factor) * 0.01

        # Update spike record for next step recurrence
        self.spikes = spikes_now.float()

        if not return_maps:
            return None
        return self.v.cpu().numpy().reshape(self.H, self.W), spikes_now.cpu().numpy().reshape(self.H, self.W)


def run_simulation():
    print(f"Starting SNN Simulation on {DEVICE}...")
    grid = NeuroGrid()
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid.v.view(GRID_H, GRID_W).cpu(), cmap='plasma', vmin=V_REST, vmax=V_THRESH, interpolation='nearest')
    plt.colorbar(im)
    ax.set_title("Neuro-Sensory Emergence (V_mem)")
    
    # Text annotations
    ax.text(1, 2, "Motor", color='white', fontsize=8, fontweight='bold')
    ax.text(1, GRID_H//2, "Association", color='white', fontsize=8, fontweight='bold')
    ax.text(1, GRID_H-2, "Sensory", color='white', fontsize=8, fontweight='bold')
    
    # Zone lines
    ax.axhline(y=GRID_H * SENSORY_PCT, color='white', linestyle='--', alpha=0.5)
    ax.axhline(y=GRID_H * (1 - MOTOR_PCT), color='white', linestyle='--', alpha=0.5)

    def update(frame):
        # Sine wave modulation of input for interesting dynamics
        intensity = (np.sin(frame * 0.1) + 1.0) * 0.5 
        
        v_map, spikes_map = grid.dynamics_step(stim_intensity=intensity)
        
        # Customize visualization
        display_data = v_map.copy()
        display_data[spikes_map > 0] = V_THRESH + 0.5 # Super bright for spikes
        
        im.set_data(display_data)
        ax.set_title(f"Step {frame}: Intensity {intensity:.2f}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    run_simulation()
