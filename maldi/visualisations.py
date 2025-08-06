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

    # --- 1. Create the napari viewer for the current volume ---
    viewer = napari.Viewer()

    # Add the main volume layer with volume rendering and custom transparency
    main_volume_layer = viewer.add_image(
        volume,
        name=f'{cur_volume_name}_intensity',
        colormap='inferno',
        rendering='mip', # Crucial for true volume rendering with transparency
        blending='translucent', # 'additive' or 'translucent' often work well with transparency
        attenuation=0.05, # Adjust this value (e.g., 0.01 to 0.1) for desired transparency
                          # Lower value = more transparent
    )

    # --- 3. Prepare for Screenshots of 3D Isometric Views from different angles ---
    viewer.dims.ndisplay = 3
    viewer.camera.zoom = 0.7 # Adjust zoom as needed
    
    # Get 3D isometric views from different angles to match each cut direction
    isometric_views = []
    
    # Axial view - looking from top (X-Y plane)
    viewer.camera.angles = (0, 0, 0)  # Top-down view for axial (elevation, azimuth, roll)
    QApplication.processEvents()
    time.sleep(0.5)
    axial_isometric = viewer.screenshot(path=None)
    isometric_views.append(axial_isometric)
    
    # Coronal view - looking from front (X-Z plane)
    viewer.camera.angles = (90, 0, 0)  # Front view for coronal
    QApplication.processEvents()
    time.sleep(0.5)
    coronal_isometric = viewer.screenshot(path=None)
    isometric_views.append(coronal_isometric)
    
    # Sagittal view - looking from side (Y-Z plane)
    viewer.camera.angles = (90, 90, 0)  # Side view for sagittal
    QApplication.processEvents()
    time.sleep(0.5)
    sagittal_isometric = viewer.screenshot(path=None)
    isometric_views.append(sagittal_isometric)

    # --- 4. Combine Screenshots into a Multi-Panel Matplotlib Figure ---
    # Create a larger figure with 3 rows (one for each plane) and 7 columns (6 slices + isometric)
    fig, axs = plt.subplots(nrows=3, ncols=7, figsize=(24, 12))
    fig.suptitle(f"Mice Brain Visualization: {cur_volume_name}", fontsize=16)
    
    # Switch to 2D display for slice views
    viewer.dims.ndisplay = 2
    
    # Calculate slice positions for 6 evenly distributed positions
    # For each axis, at approximately 10%, 25%, 40%, 55%, 70%, and 85%
    z_positions = [
        int(volume.shape[0] * 0.10),  # 10%
        int(volume.shape[0] * 0.25),  # 25%
        int(volume.shape[0] * 0.40),  # 40%
        int(volume.shape[0] * 0.55),  # 55%
        int(volume.shape[0] * 0.70),  # 70%
        int(volume.shape[0] * 0.85),  # 85%
    ]
    
    y_positions = [
        int(volume.shape[1] * 0.10),  # 10%
        int(volume.shape[1] * 0.25),  # 25%
        int(volume.shape[1] * 0.40),  # 40%
        int(volume.shape[1] * 0.55),  # 55%
        int(volume.shape[1] * 0.70),  # 70%
        int(volume.shape[1] * 0.85),  # 85%
    ]
    
    x_positions = [
        int(volume.shape[2] * 0.10),  # 10%
        int(volume.shape[2] * 0.25),  # 25%
        int(volume.shape[2] * 0.40),  # 40%
        int(volume.shape[2] * 0.55),  # 55%
        int(volume.shape[2] * 0.70),  # 70%
        int(volume.shape[2] * 0.85),  # 85%
    ]
    
    # Position labels for titles
    position_labels = ["10%", "25%", "40%", "55%", "70%", "85%"]
    
    # Axial Cuts (X-Y plane, viewing from top, changing Z)
    for i, z_pos in enumerate(z_positions):
        axial_data = volume[z_pos, :, :]
        axial_layer = viewer.add_image(
            axial_data,
            name=f'axial_slice_{i}',
            colormap='inferno',
            visible=True
        )
        main_volume_layer.visible = False  # Hide the volume temporarily
        QApplication.processEvents()
        time.sleep(0.5)
        axial_screenshot = viewer.screenshot(path=None)
        axs[0, i].imshow(axial_screenshot)
        axs[0, i].set_title(f"Axial (X-Y) - {position_labels[i]}")
        axs[0, i].axis('off')
        viewer.layers.remove(axial_layer)  # Remove the layer after screenshot
    
    # Coronal Cuts (X-Z plane, viewing from front, changing Y)
    for i, y_pos in enumerate(y_positions):
        coronal_data = volume[:, y_pos, :]
        coronal_layer = viewer.add_image(
            coronal_data, 
            name=f'coronal_slice_{i}',
            colormap='inferno',
            visible=True
        )
        QApplication.processEvents()
        time.sleep(0.5)
        coronal_screenshot = viewer.screenshot(path=None)
        axs[1, i].imshow(coronal_screenshot)
        axs[1, i].set_title(f"Coronal (X-Z) - {position_labels[i]}")
        axs[1, i].axis('off')
        viewer.layers.remove(coronal_layer)  # Remove the layer after screenshot
    
    # Sagittal Cuts (Y-Z plane, viewing from side, changing X)
    for i, x_pos in enumerate(x_positions):
        sagittal_data = volume[:, :, x_pos]
        sagittal_layer = viewer.add_image(
            sagittal_data,
            name=f'sagittal_slice_{i}',
            colormap='inferno',
            visible=True
        )
        QApplication.processEvents()
        time.sleep(0.5)
        sagittal_screenshot = viewer.screenshot(path=None)
        axs[2, i].imshow(sagittal_screenshot)
        axs[2, i].set_title(f"Sagittal (Y-Z) - {position_labels[i]}")
        axs[2, i].axis('off')
        viewer.layers.remove(sagittal_layer)  # Remove the layer after screenshot
    
    # Restore the main volume visibility
    main_volume_layer.visible = True
    
    # Add the matching isometric view to the last column of each row
    for i in range(3):
        axs[i, 6].imshow(isometric_views[i])
        view_names = ["Axial Orientation", "Coronal Orientation", "Sagittal Orientation"]
        axs[i, 6].set_title(f"3D View - {view_names[i]}")
        axs[i, 6].axis('off')

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
