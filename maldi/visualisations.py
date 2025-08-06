#!/usr/bin/env python3

import numpy as np
import napari
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
from vispy.color import Colormap, Color
from qtpy.QtWidgets import QApplication # Import QApplication to process events
import time


# --- Configuration ---
# Define a dummy volume directory and video directory for demonstration.
# In your actual use, these would point to your data.

# --- Main Processing Loop ---
image_output_dir = Path("output_images")
video_dir = image_output_dir
image_output_dir.mkdir(exist_ok=True, parents=True)
files = ["/Users/Daniel/Downloads/HexCer 42_2;O2_volume255.npy"]

def create_intensity_colormap():
    """
    Creates a custom colormap for napari with intensity-based transparency.
    Less intense voxels (lower values) will be more transparent.
    More intense voxels (higher values) will be more opaque and colored.
    """
    # Define colors and their corresponding alpha (opacity) values
    # Format: (value_position, R, G, B, A)
    # value_position: 0.0 to 1.0, mapping to data range
    # R, G, B, A: 0.0 to 1.0
    colors = [
        (0.0, [0.0, 0.0, 0.0, 0.0]),  # Fully transparent black at min intensity
        (0.1, [0.0, 0.0, 0.0, 0.05]), # Slightly visible at low intensity
        (0.3, [0.1, 0.2, 0.8, 0.2]),  # Faint blue at medium-low intensity
        (0.6, [0.8, 0.4, 0.1, 0.6]),  # Orange at medium-high intensity
        (1.0, [1.0, 1.0, 0.0, 1.0])   # Opaque yellow at max intensity
    ]
    return Colormap(colors)

for cur_file in tqdm(files, desc="Processing volumes"):
    volume = np.load(cur_file)
    cur_volume_name = Path(cur_file).stem

    # Normalize volume data to 0-1 for consistent colormap application
    volume_min = np.nanmin(volume)
    volume_max = np.nanmax(volume)
    if volume_max - volume_min > 0:
        volume = (volume - volume_min) / (volume_max - volume_min)
    else:
        volume = np.zeros_like(volume) # Handle flat volumes

    # Create a custom colormap for intensity-based transparency
    #custom_colormap = create_intensity_colormap()

    # --- 1. Create the napari viewer for the current volume ---
    viewer = napari.Viewer()

    # Add the main volume layer with volume rendering and custom transparency
    # 'attenuation' helps control how light passes through the volume,
    # making the transparency more pronounced.
    main_volume_layer = viewer.add_image(
        volume,
        name=f'{cur_volume_name}_intensity',
        colormap='inferno',
        rendering='mip', # Crucial for true volume rendering with transparency
        blending='translucent', # 'additive' or 'translucent' often work well with transparency
        attenuation=0.05, # Adjust this value (e.g., 0.01 to 0.1) for desired transparency
                          # Lower value = more transparent
    )

    # --- 3. Prepare for Screenshots (2D Cuts and 3D Isometric View) ---

    # Set initial 3D display for the isometric view first
    viewer.dims.ndisplay = 3
    viewer.camera.zoom = 0.7 # Adjust zoom as needed
    # Set camera for an isometric-like view (azimuth, elevation, roll)
    # These angles provide a good starting point for an isometric perspective.
    viewer.camera.angles = (30+180, 45+180, 0+180) # (elevation, azimuth, roll)
    # Give napari a moment to render the view before screenshotting
   # viewer.window.qt_viewer.update_console()
    QApplication.processEvents()
    time.sleep(0.1)
    isometric_screenshot = viewer.screenshot(path=None) # Take screenshot as numpy array

    # Now, switch to 2D display and take screenshots of cuts
    viewer.dims.ndisplay = 2
    # Close the napari viewer after taking all screenshots for this volume

    # --- 4. Combine Screenshots into a Multi-Panel Matplotlib Figure ---
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    fig.suptitle(f"Mice Brain Visualization: {cur_volume_name}", fontsize=16)
    
    # Axial Cut (X-Y plane, changing Z)
    axial_slice_index = volume.shape[0] // 2  # Middle slice along Z axis
    viewer.dims.set_point(0, axial_slice_index)  # Set Z axis slice
    QApplication.processEvents()
    time.sleep(0.9)
    axial_screenshot = viewer.screenshot(path=None)
    axs[0, 0].imshow(axial_screenshot)
    axs[0, 0].set_title("Axial Cut (X-Y plane)")
    axs[0, 0].axis('off')
    
    # Coronal Cut (X-Z plane, changing Y)
    coronal_slice_index = volume.shape[1] // 2  # Middle slice along Y axis
    viewer.dims.set_point(1, coronal_slice_index)  # Set Y axis slice
    QApplication.processEvents()
    time.sleep(0.9)
    coronal_screenshot = viewer.screenshot(path=None)
    axs[0, 1].imshow(coronal_screenshot)
    axs[0, 1].set_title("Coronal Cut (X-Z plane)")
    axs[0, 1].axis('off')

    # Sagittal Cut (Y-Z plane, changing X)
    sagittal_slice_index = volume.shape[2] // 2  # Middle slice along X axis
    viewer.dims.set_point(2, sagittal_slice_index)  # Set X axis slice
    QApplication.processEvents()
    time.sleep(0.9)
    sagittal_screenshot = viewer.screenshot(path=None)
    axs[1, 0].imshow(sagittal_screenshot)
    axs[1, 0].set_title("Sagittal Cut (Y-Z plane)")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(isometric_screenshot)
    axs[1, 1].set_title("Isometric 3D View")
    axs[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.savefig(image_output_dir / f"{cur_volume_name}_multi_panel.png", dpi=300)
    plt.close(fig) # Close the matplotlib figure to free memory
    viewer.close() # Close the napari viewer after taking all screenshots for this volume
def void():

    print(f"Generated multi-panel image for {cur_volume_name}")
    # end script for the moment we do not need to save the images

    # --- Original Animation Code (kept separate as requested) ---
    # Re-open viewer for animation if desired, as the previous one was closed.
    # This part can be commented out if you only need the static images.
    viewer_animation = napari.Viewer()
    layer_animation = viewer_animation.add_image(
        volume,
        name=f'{cur_volume_name}_animation',
        colormap='inferno', # Use the same custom colormap
        rendering='mip',
        blending='translucent',
        attenuation=0.05
    )
    viewer_animation.dims.ndisplay = 3
    viewer_animation.camera.zoom = 0.7

    # Ensure napari-animation is installed: pip install napari-animation
    try:
        from napari_animation import Animation
        animation = Animation(viewer_animation)

        total_frames = 60
        for i in range(total_frames):
            angle = i * (360 / total_frames)
            # Rotate around the Z-axis (azimuth)
            viewer_animation.camera.angles = (30, angle, 30)
            animation.capture_keyframe()

        animation.animate(video_dir / f"{str(cur_volume_name)}_brain_rotation.mp4", canvas_only=True, fps=30)
        print(f"Generated rotation video for {cur_volume_name}")
    except ImportError:
        print("napari-animation not found. Skipping video generation.")
        print("Install with: pip install napari-animation")
    finally:
        viewer_animation.close()

print("\nProcessing complete. Check 'output_images' and 'output_videos' directories.")
