import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches

# Simulation parameters (adjustable)
params = {
    'num_particles': 50,
    'temperature': 1.0,  # Higher = more particle movement
    'fusion_distance': 0.5,  # Distance at which particles "fuse"
    'energy_release': 1.5,  # Multiplier for speed after fusion
    'pressure': 1.0,  # External pressure pushing particles together
    'simulation_size': 10,
    'particle_size': 50
}

# Create custom colormap (blue to red)
colors = [(0, 0, 1), (1, 0, 0)]  # Blue to Red
cmap = LinearSegmentedColormap.from_list("temp_cmap", colors)

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle("Simplified Thermonuclear Reaction Simulation", fontsize=14)
ax.set_xlim(0, params['simulation_size'])
ax.set_ylim(0, params['simulation_size'])
ax.set_facecolor('black')

# Create particles
particles = np.random.rand(params['num_particles'], 4) * params['simulation_size']
# Columns: x, y, vx, vy
particles[:, 2:4] = (np.random.rand(params['num_particles'], 2) - 0.5) * params['temperature']

# Track which particles have "fused"
fused = np.zeros(params['num_particles'], dtype=bool)
energy_output = 0

# Create scatter plot for particles
scat = ax.scatter(particles[:, 0], particles[:, 1], 
                  s=params['particle_size'], 
                  c=np.zeros(params['num_particles']), 
                  cmap=cmap, vmin=0, vmax=1)

# Add text for parameters and energy output
param_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, color='white')
energy_text = ax.text(0.02, 0.90, "Energy Released: 0", transform=ax.transAxes, color='yellow')

# Add a "compression chamber" visualization
chamber = patches.Rectangle((1, 1), params['simulation_size']-2, params['simulation_size']-2, 
                            linewidth=2, edgecolor='cyan', facecolor='none')
ax.add_patch(chamber)

def update(frame):
    global particles, fused, energy_output
    
    # Apply pressure (push particles toward center)
    center = params['simulation_size'] / 2
    particles[:, 2] += (center - particles[:, 0]) * 0.01 * params['pressure']
    particles[:, 3] += (center - particles[:, 1]) * 0.01 * params['pressure']
    
    # Update positions
    particles[:, 0] += particles[:, 2] * 0.1
    particles[:, 1] += particles[:, 3] * 0.1
    
    # Boundary collisions
    mask = (particles[:, 0] <= 0) | (particles[:, 0] >= params['simulation_size'])
    particles[mask, 2] *= -1
    mask = (particles[:, 1] <= 0) | (particles[:, 1] >= params['simulation_size'])
    particles[mask, 3] *= -1
    
    # Check for fusion
    for i in range(len(particles)):
        if fused[i]:
            continue
        for j in range(i+1, len(particles)):
            if fused[j]:
                continue
            dist = np.sqrt((particles[i, 0] - particles[j, 0])**2 + 
                          (particles[i, 1] - particles[j, 1])**2)
            if dist < params['fusion_distance']:
                # "Fuse" the particles
                particles[i, 2:4] = (particles[i, 2:4] + particles[j, 2:4]) * params['energy_release']
                particles[j, 0:2] = -10  # Move it out of view
                fused[j] = True
                energy_output += 1
                
    # Update visualization
    colors = np.zeros(len(particles))
    speeds = np.sqrt(particles[:, 2]**2 + particles[:, 3]**2)
    colors = speeds / (params['temperature'] * 2)  # Normalize to 0-1 for colormap
    colors = np.clip(colors, 0, 1)
    
    # Update scatter plot
    scat.set_offsets(particles[:, 0:2])
    scat.set_array(colors)
    
    # Update text
    param_text.set_text(
        f"Temperature: {params['temperature']:.1f} | "
        f"Pressure: {params['pressure']:.1f}\n"
        f"Fusion Distance: {params['fusion_distance']:.1f} | "
        f"Particles: {np.sum(~fused)}/{params['num_particles']}"
    )
    energy_text.set_text(f"Energy Released: {energy_output}")
    
    # Randomly adjust some parameters to simulate reaction dynamics
    if frame % 10 == 0:
        params['temperature'] += energy_output * 0.001
        params['pressure'] = max(0.5, params['pressure'] - 0.01)
    
    return scat, param_text, energy_text

# Create animation
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)

plt.tight_layout()
plt.show()
