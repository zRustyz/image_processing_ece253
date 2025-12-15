# De-Blurring Pipeline
instructions:
1. Create Venv
2. pip install numpy opencv-python matplotlib scipy scikit-image
3. run the degrader.py file (will replace the files in the current output folder, recommended to make a copy before proceeding)
4. run the processor.py file (will also replace the files in the current output folder)

The code:
1. Creates random linear blur kernels for each image and applies the blur to the images
2. Blind deconvolutes the blur and creates the output images

# Super-Resolution Comparison Pipeline
instructions: run the upscale.ipynb in order

or each image in the dataset, the code:

1. Loads the original high-resolution image (ground truth)
2. Loads the corresponding 4× downsampled image
3. Upscales the downsampled image using:
   - Bicubic interpolation (OpenCV)
   - Bicubic interpolation (SciPy)
   - Real-ESRGAN (pretrained ×4 generator)
4. Saves all upscaled images to disk
5. Computes image quality metrics for each method
6. Aggregates metrics across the dataset
7. Generates comparison plots

---
