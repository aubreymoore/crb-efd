from sim_palm import generate_palm_with_cuts



from efd_animation_simple import animate_reconstruction
from efd_animation import animate_epicycles
from efd_defect_counts import count_defects

from icecream import ic

contour, mask = generate_palm_with_cuts(2)
# animate_reconstruction(contour, max_harmonics=40)
# animate_epicycles(points=contour, n_harmonics=40)
count_defects(contour=contour, order=40)