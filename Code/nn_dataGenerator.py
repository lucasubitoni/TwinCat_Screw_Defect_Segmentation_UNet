import tensorflow as tf
import os
from tensorflow.keras.utils import Sequence
import numpy as np
import math


class customDataGenerator(Sequence):

    # BASE, NOT OPTIMIZED (one optimization is to load all images in RAM, but we have to ensure to have sufficient space)

    def __init__(self, imgPath, maskPath, batchSize, tileSize, shuffle: bool, augment: bool, overlap: bool, overlapPercentage=None, thrAvgBlackImg=0.05, epochCurriculumLearning=0):

        super().__init__()

        if overlap:
            assert overlapPercentage<=1 and overlapPercentage>=0

        self.imgPath = imgPath
        self.maskPath = maskPath
        self.batchSize = batchSize
        self.tileSize = tileSize
        self.shuffle = shuffle
        self.augment = augment
        self.overlap = overlap
        self.overlapPercentage = overlapPercentage
        self.thrAvgBlackImg = thrAvgBlackImg
        self.currentEpoch = 0
        self.epochCurriculumLearning = epochCurriculumLearning

    
    def _filterAlmostBlack(self, imgTiles, maskTiles, which="img"):

        # "which" parameter has to be either "img" or "mask"

        if which == "img":
            MeanValue = tf.reduce_mean(imgTiles, axis=[1, 2, 3])
        elif which == "mask":
            MeanValue = tf.reduce_mean(maskTiles, axis=[1, 2, 3])
        nonBlack = MeanValue > self.thrAvgBlackImg
        
        # Use the mask to filter out black tiles
        imgTiles = tf.boolean_mask(imgTiles, nonBlack)
        maskTiles = tf.boolean_mask(maskTiles, nonBlack)
        
        return imgTiles, maskTiles
    

    def _extractTiles(self, img, mask):

        # Assumes that images and masks have shape: [batch=1, height, width, channels] (i.e., one image at a time)

        tileHeight, tileWidth = self.tileSize

        strideHeight = tileHeight if not self.overlap else int(self.overlapPercentage*tileHeight)
        strideWidth = tileWidth if not self.overlap else int(self.overlapPercentage*tileWidth)

        # Expand dims to add fictitious batch size
        img = tf.expand_dims(img, axis=0)
        mask = tf.expand_dims(mask, axis=0)

        imgTiles = tf.image.extract_patches(images = img,
                                            sizes = [1, tileHeight, tileWidth, 1],
                                            strides = [1, strideHeight, strideWidth, 1],
                                            rates = [1, 1, 1, 1],
                                            padding = "VALID")
        
        maskTiles = tf.image.extract_patches(images = mask,
                                             sizes = [1, tileHeight, tileWidth, 1],
                                             strides = [1, strideHeight, strideWidth, 1],
                                             rates = [1, 1, 1, 1],
                                             padding = "VALID")
        
        # Reshape the last dimension to (num_patches, tileHeight, tileWidth, channels)
        num_patches_h = imgTiles.shape[1]
        num_patches_w = imgTiles.shape[2]
        num_channels = img.shape[-1]

        imgTiles = tf.reshape(imgTiles, [num_patches_h * num_patches_w, tileHeight, tileWidth, num_channels])
        maskTiles = tf.reshape(maskTiles, [num_patches_h * num_patches_w, tileHeight, tileWidth, 1])  # Assuming mask has 1 channel

        
        return imgTiles, maskTiles
    

    def _loadImageAndMask(self, imgPath, maskPath):

        # Load and decode image in B/W
        img = tf.io.read_file(imgPath)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.rgb_to_grayscale(img)

        # Load and decode mask
        mask = tf.io.read_file(maskPath)
        mask = tf.image.decode_png(mask, channels=1)        

        return img, mask
    

    def _normalizeImageAndMask(self, img, mask, normalizationFactor=255.0):

        img = tf.cast(img, tf.float32) / normalizationFactor
        mask = tf.cast(mask, tf.float32) / normalizationFactor

        return img, mask
    
    
    def _augmentColor(self, img):
        # Apply random color change with 50% chance
        if tf.random.uniform(shape=()) < 0.5:
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        return img

    def _augmentGeometric(self, img, mask):
        # Flip horizontally with a 50% chance
        if tf.random.uniform(shape=()) < 0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)

        # Flip vertically with a 50% chance
        if tf.random.uniform(shape=()) < 0:
            
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)

        # Apply random rotation with 50% chance
        if tf.random.uniform(shape=()) < 0.5:
            rotation_angle = np.random.uniform(low=-math.pi/6, high=math.pi/6)
            # Compute the affine transformation matrix (rotation and translation)
            transform = self._generate_projective_transform(translation=[0,0], 
                                                            rotation_angle=rotation_angle)
            img = tf.keras.ops.image.affine_transform(img, transform)
            mask = tf.keras.ops.image.affine_transform(mask, transform)

        return img, mask

    def _generate_projective_transform(self, translation, rotation_angle):
        # Unpack the translation
        tx, ty = translation
        # Rotation matrix components
        cos_theta = math.cos(rotation_angle)
        sin_theta = math.sin(rotation_angle)
        # Compute the projective transform matrix parameters
        a0 = cos_theta
        a1 = -sin_theta
        a2 = tx
        b0 = sin_theta
        b1 = cos_theta
        b2 = ty
        # c0 and c1 are only effective with TensorFlow backend so zero them out
        c0 = 0
        c1 = 0
        # The transformation vector (a0, a1, a2, b0, b1, b2, c0, c1)
        transform_vector = np.array([a0, a1, a2, b0, b1, b2, c0, c1])
        return transform_vector
    
   
    def on_epoch_end(self):
        self.currentEpoch += 1
         

    def take(self):
        idx = 0
        return self.__getitem__(idx)
    
    @property # A cosa serve questo decorator?
    def num_batches(self):
        return len(self.imgPath)


    def __getitem__(self, idx):

        # Initializing X and y of shape [batch, tileX, tileY, channels]
        X = np.ones((self.batchSize, *self.tileSize, 1))
        y = np.ones((self.batchSize, *self.tileSize, 1))

        # Converting the paths in which the images are stored into a set
        currentImgPath = set(self.imgPath)
        currentMaskPath = set(self.maskPath)

        # We iteratively generate the batch
        nExtracted = 0
        while nExtracted < self.batchSize:

            # Drawing a random index to extract image and mask from the set
            imgIdx = np.random.randint(low=0, high=len(currentImgPath)-1)
            
            # Load and normalize data
            img, mask = self._loadImageAndMask(np.sort(list(currentImgPath))[imgIdx], np.sort(list(currentMaskPath))[imgIdx])
            img, mask = self._normalizeImageAndMask(img, mask)

            # Geometric data augmentation 
            if self.augment:
                img, mask = self._augmentGeometric(img, mask)
            
            # Extracting and filtering tiles
            imgTiles, maskTiles = self._extractTiles(img, mask)
            imgTiles, maskTiles = self._filterAlmostBlack(imgTiles, maskTiles, which="img")
            if self.currentEpoch < self.epochCurriculumLearning:
                imgTiles, maskTiles = self._filterAlmostBlack(imgTiles, maskTiles, which="mask")

            # Brightness data augmentation
            if self.augment:
                img = self._augmentColor(img)

            # Random shuffle of the indexes (same for images and masks)
            if self.shuffle:
                indices = tf.random.shuffle(tf.range(tf.shape(imgTiles)[0]))
                imgTiles = tf.gather(imgTiles, indices)
                maskTiles = tf.gather(maskTiles, indices)

            # Add the extracted tiles to the X and y tensors
            remainingSlots = self.batchSize - nExtracted
            tilesToAdd = min(len(imgTiles), remainingSlots)
            X[nExtracted:nExtracted+tilesToAdd] = tf.reshape(imgTiles[:tilesToAdd], (tilesToAdd, *self.tileSize, 1))
            y[nExtracted:nExtracted+tilesToAdd] = tf.reshape(maskTiles[:tilesToAdd], (tilesToAdd, *self.tileSize, 1))

            # Safety mechanism (in case we exhaust images when using big batch sizes, repopulate the sets)
            if len(currentImgPath) == 0:
                currentImgPath = set(self.imgPath)
                currentMaskPath = set(self.imgPath)
            else:
                # Normally we remove the used image in order to enrich the diversity of the training set
                currentImgPath.remove(np.sort(list(currentImgPath))[imgIdx])
                currentMaskPath.remove(np.sort(list(currentMaskPath))[imgIdx])
            
            nExtracted += tilesToAdd

        return X, y





class TileDataGenerator:
    """TensorFlow Data Pipeline for Image Segmentation with Tile Extraction using tf.image.extract_patches."""

    def __init__(self, image_paths, mask_paths, batch_size=64, tile_size=(144*2, 108*2), shuffle=True, augment=True, overlap=True, black_threshold=0.05):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.tile_size = tile_size
        self.shuffle = shuffle
        self.augment = augment
        self.overlap = overlap  # If True, allow overlapping tiles (otherwise non-overlapping)
        self.black_threshold = black_threshold  # Threshold to identify almost black tiles

    def _load_image_mask(self, img_path, mask_path):
        """Loads and preprocesses a single image and mask."""
        # Load and decode image
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.rgb_to_grayscale(img) # Convert to grayscale
        img = tf.cast(img, tf.float32) / 255.0

        # Load and decode mask
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.cast(mask, tf.float32) / 255.0

        # Extract tiles from the image and mask
        img_tiles, mask_tiles = self._extract_tiles(img, mask)
        
        # Filter out almost black tiles
        img_tiles, mask_tiles = self._filter_black_tiles(img_tiles, mask_tiles)

        # Filter out almost black mask
        #img_tiles, mask_tiles = self._filter_black_masks(img_tiles, mask_tiles)

        if self.augment:
            img_tiles, mask_tiles = self._augment(img_tiles, mask_tiles)

        return img_tiles, mask_tiles

    def _extract_tiles(self, img, mask):
        """Extracts tiles from the image and mask using tf.image.extract_patches."""
        tile_height, tile_width = self.tile_size

        # Determine strides based on overlap option
        stride_height = tile_height if not self.overlap else tile_height // 4
        stride_width = tile_width if not self.overlap else tile_width // 4

        # Expand dims to add batch dimension: [1, height, width, channels]
        img_exp = tf.expand_dims(img, axis=0)
        mask_exp = tf.expand_dims(mask, axis=0)

        # Use tf.image.extract_patches to get patches.
        patches_img = tf.image.extract_patches(
            images=img_exp,
            sizes=[1, tile_height, tile_width, 1],
            strides=[1, stride_height, stride_width, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches_mask = tf.image.extract_patches(
            images=mask_exp,
            sizes=[1, tile_height, tile_width, 1],
            strides=[1, stride_height, stride_width, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        # Determine the number of patches in each dimension.
        patch_dims = tf.shape(patches_img)

        # Reshape the patches to get individual tiles.
        patches_img = tf.reshape(patches_img, [-1, tile_height, tile_width, tf.shape(img_exp)[-1]])
        patches_mask = tf.reshape(patches_mask, [-1, tile_height, tile_width, 1])

        return patches_img, patches_mask

    def _filter_black_tiles(self, img_tiles, mask_tiles):
        """Filters out almost black tiles by checking the mean pixel value."""
        img_mean = tf.reduce_mean(img_tiles, axis=[1, 2, 3])  # Mean across height, width, and channels
        non_black_mask = img_mean > self.black_threshold
        
        # Use the mask to filter out black tiles
        img_tiles = tf.boolean_mask(img_tiles, non_black_mask)
        mask_tiles = tf.boolean_mask(mask_tiles, non_black_mask)

        return img_tiles, mask_tiles
    

    def _filter_black_masks(self, img_tiles, mask_tiles):
        """Filters out almost black tiles by checking the mean pixel value.USELESS"""
        mask_mean = tf.reduce_mean(mask_tiles, axis=[1, 2, 3])  # Mean across height, width, and channels
        non_black_mask = mask_mean > self.black_threshold
        
        # Use the mask to filter out black tiles
        img_tiles = tf.boolean_mask(img_tiles, non_black_mask)
        mask_tiles = tf.boolean_mask(mask_tiles, non_black_mask)

        return img_tiles, mask_tiles


    def _augment(self, img_tiles, mask_tiles):
        """Applies simple data augmentation."""
        img_tiles = tf.image.random_brightness(img_tiles, max_delta=0.1)
        img_tiles = tf.image.random_contrast(img_tiles, lower=0.8, upper=1.2)

        if tf.random.uniform(()) < 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        
        if tf.random.uniform(()) < 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
        
        if tf.random.uniform(()) < 0.5:
            k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
            image = tf.image.rot90(image, k=k)
            mask = tf.image.rot90(mask, k=k)
        

        return img_tiles, mask_tiles

    def create_dataset(self):
        """Creates a tf.data.Dataset pipeline that randomly samples patches."""
        dataset = tf.data.Dataset.from_tensor_slices((self.image_paths, self.mask_paths))
        dataset = dataset.map(self._load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Flatten the dataset so each element is a (tile, mask) pair.
        dataset = dataset.flat_map(
            lambda img_tiles, mask_tiles: tf.data.Dataset.from_tensor_slices((img_tiles, mask_tiles))
        )
        
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
            
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset