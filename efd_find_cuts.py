from sim_palm import generate_palm_with_cuts
from icecream import ic
from pyefd import elliptic_fourier_descriptors, reconstruct_contour
import numpy as np
import cv2
import matplotlib.pyplot as plt


def efd_find_cuts(original_contour, original_mask, order=40, ksize=(7,7), iterations=1, return_plot_data=False):
    """ 
    This function was created to detect and locate defects in binary masks 
    of coconut palms, such as v-shaped cuts caused by coconut rhinoceros beetles.
    
    Synthetic contours and masks for testing be provided by the generate_palms_with_cuts function. 
    
    Elliptic Fourier descriptors are calculated and used to reconstruct a "smoothed version" of the original mask.
    Cuts are apparent in the difference between the smoothed mask and the original mask.
    
    In the final step, noise is removed filtered out using a morphological operation called "opening". 
    
    Required arguments:
        original_contour    binary mask of a coconut palm
        original_mask       a binary mask (filled original_contour)
        
    Arguments with defaults:
        order               an EFD parameter which determines the size of the descriptor 
        ksize               a tuple defining the size of the kernel used by the morphological operation
        iterations          number of times the morphological operation is applied to a mask
        return_plot_data    a binary flag which determines if plot_data are calculated and returned
        
    Return values:
        contours            a tuple containing contours (each contour is a numpy array of points; dtype=int32)
        plot_data           a dict containing binary masks for visualization  
    """

    ic.disable()

    # original_contour, original_mask = generate_palm_with_cuts(ncuts)
    # ic(original_contour)
    # ic(original_mask)
    # ic(np.sum(original_mask))

    # calc EFDs
    coeffs = elliptic_fourier_descriptors(original_contour, order, normalize=False)

    # reconstruct contour
    reconstructed_contour = reconstruct_contour(coeffs, num_points=original_contour.shape[0])

    # Calculate the centroid of the original to shift the reconstruction back
    # EFD reconstruction is often centered at (0,0) or uses the DC component (coeffs[0])
    centroid = np.mean(original_contour, axis=0)
    ic(centroid)
    reconstructed_contour += centroid
    reconstructed_contour = reconstructed_contour.astype(np.int32)
    ic(reconstructed_contour)

    reconstructed_mask = cv2.fillPoly(np.zeros_like(original_mask), pts=[reconstructed_contour], color=1)
    ic(reconstructed_mask)
    ic(np.sum(reconstructed_mask))

    # Calculate the difference mask
    diff_mask = reconstructed_mask & ~original_mask
    ic(diff_mask)

    # Create clean_mask
    # Define kernel (size depends on how thick the "thin" features are)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=ksize)
    # Apply Opening
    clean_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
    
    # get vcut contours
    contours, _ = cv2.findContours(image=clean_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    ic.enable()
    
    if return_plot_data:
        plot_data = {
            'original_mask': original_mask, 
            'reconstructed_mask': reconstructed_mask,
            'diff_mask': diff_mask, 
            'clean_mask': clean_mask
        }
        return contours, plot_data
    
    return contours

if __name__ == "__main__":

    ncuts = 4
    order = 40
    ksize = (7,7)
    iterations=1

    contour, mask = generate_palm_with_cuts(ncuts)

    ic('USE CASE 1 - without plots')
    vcut_contours = efd_find_cuts(
        original_contour=contour, 
        original_mask=mask, 
        order=order,
        ksize=ksize,
        iterations=iterations)
    ncuts_detected = len(vcut_contours)
    ic(ncuts_detected);

    ic('USE CASE 2 - with plots (return_plot_data=True)')
    vcut_contours, plot_data = efd_find_cuts(
        original_contour=contour, 
        original_mask=mask, 
        order=order,
        ksize=ksize,
        iterations=iterations,
        return_plot_data=True)
    ncuts_detected = len(vcut_contours)

    # Plot example:
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    masks = [
        plot_data['original_mask'], 
        plot_data['reconstructed_mask'], 
        plot_data['diff_mask'], 
        plot_data['clean_mask']
        ]
    titles = [
        f'Original ({ncuts=})', 
        f'EFD Reconstruction ({order=})', 
        'Diff (Recon & ~Orig)', 
        f'Filtered Diff (k={ksize} i={iterations} n={ncuts_detected})']
    for ax, mask, title in zip(axes, masks, titles):
        ax.imshow(mask, cmap='gray')
        ax.set_title(title)
    plt.tight_layout()
    plt.show()
