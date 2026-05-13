import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyefd

def generate_shape():
    t = np.linspace(0, 2 * np.pi, 100)
    r = 1 + 0.3 * np.sin(5 * t) + 0.1 * np.cos(10 * t)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])

def animate_epicycles(points, n_harmonics=15):
    # Calculate descriptors using pyefd
    coeffs = pyefd.elliptic_fourier_descriptors(points, order=n_harmonics)
    locus = pyefd.calculate_dc_coefficients(points)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    padding = 1.0
    ax.set_xlim(np.min(points[:, 0]) - padding, np.max(points[:, 0]) + padding)
    ax.set_ylim(np.min(points[:, 1]) - padding, np.max(points[:, 1]) + padding)
    ax.set_title("EFD Epicycles (using pyefd)")
    
    ax.plot(points[:, 0], points[:, 1], 'k--', alpha=0.2, label='Original')
    
    trace_x, trace_y = [], []
    trace_line, = ax.plot([], [], 'r-', lw=2, label='EFD Path')
    
    # We'll use multiple circles and lines to represent the epicycles
    circles = [ax.plot([], [], 'b-', alpha=0.3)[0] for _ in range(n_harmonics)]
    arm_lines = [ax.plot([], [], 'g-', alpha=0.5)[0] for _ in range(n_harmonics)]
    
    harmonic_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend(loc='upper right')

    def init():
        trace_line.set_data([], [])
        for c in circles: c.set_data([], [])
        for a in arm_lines: a.set_data([], [])
        return [trace_line] + circles + arm_lines

    # Sub-steps for smoother rotation animation
    n_frames = 200
    phi_steps = np.linspace(0, 2 * np.pi, n_frames)

    def update(frame):
        t = phi_steps[frame]
        
        # Starting point is the locus (centroid)
        curr_x, curr_y = locus
        
        for n_idx in range(n_harmonics):
            n = n_idx + 1 # Harmonic order
            an, bn, cn, dn = coeffs[n_idx]
            
            # Draw the harmonic ellipse for this component
            t_circle = np.linspace(0, 2 * np.pi, 50)
            ex = curr_x + an * np.cos(n * t_circle) + bn * np.sin(n * t_circle)
            ey = curr_y + cn * np.cos(n * t_circle) + dn * np.sin(n * t_circle)
            circles[n_idx].set_data(ex, ey)
            
            # Update the tip position for this harmonic at current time t
            prev_x, prev_y = curr_x, curr_y
            curr_x += an * np.cos(n * t) + bn * np.sin(n * t)
            curr_y += cn * np.cos(n * t) + dn * np.sin(n * t)
            
            # Draw the "arm" for this harmonic
            arm_lines[n_idx].set_data([prev_x, curr_x], [prev_y, curr_y])
            
        trace_x.append(curr_x)
        trace_y.append(curr_y)
        trace_line.set_data(trace_x, trace_y)
        
        harmonic_text.set_text(f'Harmonics: {n_harmonics}')
        return [trace_line] + circles + arm_lines

    ani = FuncAnimation(fig, update, frames=n_frames,
                        init_func=init, blit=True, interval=50, repeat=False)
    
    output_file = "efd_epicycles.gif"
    print(f"Saving epicycle animation to {output_file}...")
    ani.save(output_file, writer='pillow')
    print("Done!")
    return ani

if __name__ == "__main__":
    shape = generate_shape()
    animate_epicycles(shape, n_harmonics=10)
