import tensorflow as tf
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from PIL import Image



def load_images(directory):
    image_paths = glob(os.path.join(directory, "*"))
    print(f"Found {len(image_paths)} images in {directory}")
    return sorted(image_paths)



def tile_image(image, tile_height, tile_width, grid_size_y, grid_size_x):
   
    # Ensure the image is the right size for the grid
    expected_height = tile_height * grid_size_y
    expected_width = tile_width * grid_size_x
    
    if image.shape[0] != expected_height or image.shape[1] != expected_width:
        print(f"Warning: Image size {image.shape} doesn't match expected size ({expected_height}, {expected_width})")
        image = np.array(Image.fromarray(image).resize((expected_width, expected_height)))
    
    tiles = []
    for y in range(grid_size_y):
        for x in range(grid_size_x):
            y_start = y * tile_height
            y_end = (y + 1) * tile_height
            x_start = x * tile_width
            x_end = (x + 1) * tile_width
            
            tile = image[y_start:y_end, x_start:x_end]
            tiles.append(tile)
    
    return tiles

def reconstruct_from_tiles(tiles, grid_size_y, grid_size_x):

    # Get dimensions from the first tile
    tile_height, tile_width = tiles[0].shape[0], tiles[0].shape[1]
    
    # Create empty array for the reconstructed image
    channels = tiles[0].shape[2] if len(tiles[0].shape) > 2 else 1
    reconstructed = np.zeros((tile_height * grid_size_y, tile_width * grid_size_x, channels))
    
    # Place each tile back in its position
    tile_idx = 0
    for y in range(grid_size_y):
        for x in range(grid_size_x):
            y_start = y * tile_height
            y_end = (y + 1) * tile_height
            x_start = x * tile_width
            x_end = (x + 1) * tile_width
            
            reconstructed[y_start:y_end, x_start:x_end] = tiles[tile_idx]
            tile_idx += 1
    
    return reconstructed

def process_images(image_paths, tileParams, ort_session=None, model=None, modeSelected="onnx"):

    # Unpack the tile parameters
    tile_height = tileParams[0]
    tile_width = tileParams[1]
    grid_size_y = tileParams[2]
    grid_size_x = tileParams[3]

    results = list()
    
    for img_path in tqdm(image_paths, desc="Processing images"):

        img = np.array(Image.open(img_path))
        tiles = tile_image(img, tile_height, tile_width, grid_size_y, grid_size_x)
        
        # Extract a batch of tiles
        original_tiles = list()
        segmented_tiles = list()
        
        # Apply model prediction
        if modeSelected == "onnx":
            for tile in tiles:
                tile = tf.image.rgb_to_grayscale(tile)
                tile = tf.cast(tile, tf.float32) / 255.0
                original_tiles.append(tile)
            input_name = ort_session.get_inputs()[0].name
            segmented_tiles = ort_session.run(None, {input_name: original_tiles})[0]

        elif modeSelected == "keras":
            for tile in tiles:
                tile = tf.image.rgb_to_grayscale(tile)
                tile = tf.cast(tile, tf.float32) / 255.0
                tile = np.expand_dims(tile, axis=0)
                segmentation = model.predict(tile, verbose=0)[0]
                segmented_tiles.append(segmentation)

        # Reconstruct the full segmentation
        full_segmentation = reconstruct_from_tiles(segmented_tiles, grid_size_y, grid_size_x)
        
        # Save or collect results
        results.append({"image_path": img_path,
                        "segmentation": full_segmentation})
        
    return results

def plotBwImage(image_matrix):
    
    alpha_channel = np.where(image_matrix == 0, 0, 150)  # 0 for black, 150 for non-black
    alpha_channel = alpha_channel.astype(np.uint8)
    
    # Make sure the alpha channel is in the correct range [0, 1]
    zeros = np.zeros(shape=image_matrix.shape).astype(np.uint8)
    rgba_image = np.dstack((image_matrix, zeros, zeros, alpha_channel))

    plt.imshow(rgba_image)
    plt.axis('off')  # Optional: Hide axis for a cleaner view


