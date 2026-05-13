import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyefd

def generate_shape():
    """Generate an interesting closed shape (a modified star)."""
    t = np.linspace(0, 2 * np.pi, 100)
    r = 1 + 0.3 * np.sin(5 * t) + 0.1 * np.cos(10 * t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return np.column_stack([x, y])

def animate_reconstruction(points, max_harmonics=20):
    coeffs = pyefd.elliptic_fourier_descriptors(points, order=max_harmonics)
    locus = pyefd.calculate_dc_coefficients(points)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(np.min(points[:, 0]) - 0.5, np.max(points[:, 0]) + 0.5)
    ax.set_ylim(np.min(points[:, 1]) - 0.5, np.max(points[:, 1]) + 0.5)
    ax.set_title("EFD Reconstruction (using pyefd)")
    
    # Plot original contour
    ax.plot(points[:, 0], points[:, 1], 'k--', alpha=0.3, label='Original')
    
    line, = ax.plot([], [], 'r-', lw=2, label='Reconstruction')
    harmonic_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend(loc='upper right')

    def init():
        line.set_data([], [])
        harmonic_text.set_text('')
        return line, harmonic_text

    def update(frame):
        n = frame + 1
        # Reconstruct using only the first n harmonics
        reconstruction = pyefd.reconstruct_contour(coeffs[:n], locus=locus, num_points=200)
        line.set_data(reconstruction[:, 0], reconstruction[:, 1])
        harmonic_text.set_text(f'Harmonics: {n}')
        return line, harmonic_text

    ani = FuncAnimation(fig, update, frames=max_harmonics,
                        init_func=init, blit=True, interval=200, repeat=True)
    
    output_file = "efd_reconstruction.gif"
    print(f"Saving animation to {output_file}...")
    ani.save(output_file, writer='pillow')
    print("Done!")
    return ani

if __name__ == "__main__":
    shape = generate_shape()
    animate_reconstruction(shape, max_harmonics=30)
