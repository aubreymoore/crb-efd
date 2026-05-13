# README.md

GitHub repository: [https://github.com/aubreymoore/crb-efd](https://github.com/aubreymoore/crb-efd)

This repo explores use of elliptic Fourier descriptors (EFDs) for for finding and locating "defects" in the form of v-shape to coconut palm fronds, a distinctive symptom of coconut rhinoceros beetle damage.

## Counting defects

### counting_defects.ipynb

Code in this notebook comes from a [Gemini response](https://gemini.google.com/share/082fd52ef82e) to this prompt:
```
How can I estimate the number of defects on a contour using elliptic fourier descriptors? Please provide a Jupyter notebook which shows me how develop a pipeline to do this using sklearn.
```

It was very easy to convert the notebook to a python file using:
```bash
uv add nbconvert
jupyter nbconvert --to python counting_defects.ipynb
```

After converting, I converted the `counting_defects.py` into a Python module and saved it as `efd_defect_counts.py`.

## EFD animation

I used the Gemini CLI to create two EFD animations as [summarized here](SESSION_SUMMARY.md).

[efd_animation_simple.py](efd_animation_simple.py) shows how EFDs can be used to reconstruct a contour (
[view animation](efd_reconstruction.gif)).

[efd_animation.py](efd_animation.py) also shows how EFDs can be used to reconstruct a contour. But also animates epicycles to illustrate the reconstruction method ([view animation](efd_epicycles.gif)).

## Counting and locating defects using EFDs

### [counting_defects_with_efds.ipynb](counting_defects_with_efds.ipynb)

Code in this notebook comes from a [Gemini response](https://gemini.google.com/share/082fd52ef82e) to this prompt:
```
How can I estimate the number of defects on a contour using elliptic fourier descriptors? Please provide a Jupyter notebook which shows me how develop a pipeline to do this using sklearn.
```

The code was improved and resulted a function I exported to a module saved in [efd_find_cuts.py](efd_find_cuts.py).

### [fine_tune_efd_find_cuts.ipynb](fine_tune_efd_find_cuts.ipynb)

Code in this notebook optimizes parameters for `efd_find_cuts.py`, namely `EDF order` and `kernel size`.

## Modules

To streamline, I created some Python modules. Have a look at [efd_sim_palm.py](efd_sim_palm.py) and [efd_find_cuts.py](efd_find_cuts.py).
