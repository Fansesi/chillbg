import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from sklearn.cluster import KMeans

def apply_painterly_effect(image, brush_detail=5, color_simplification=8, edge_preservation=0.7, 
                           saturation_boost=1.2, contrast_boost=1.1):
    """
    Apply a painterly effect to an image.
    
    Parameters:
    -----------
    image : numpy.ndarray or PIL.Image
        Input image in RGB format.
    brush_detail : int, optional
        Controls the level of detail. Lower values create larger brush strokes. Range: 1-10.
    color_simplification : int, optional
        Number of colors to quantize the image to. Range: 4-32.
    edge_preservation : float, optional
        How much to preserve edges. Range: 0.0-1.0.
    saturation_boost : float, optional
        Saturation multiplier. Range: 0.5-2.0.
    contrast_boost : float, optional
        Contrast multiplier. Range: 0.5-2.0.
        
    Returns:
    --------
    PIL.Image
        Image with painterly effect applied.
    """
    # Convert input to PIL Image if it's a numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))
    elif not isinstance(image, Image.Image):
        raise TypeError("Input must be a numpy array or PIL Image")
    
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Step 1: Edge-preserving smoothing (bilateral filter)
    img_array = np.array(image)
    
    # Convert to float32 for processing
    img_float = img_array.astype(np.float32) / 255.0
    
    # Apply bilateral filter for edge-preserving smoothing
    # Adjust parameters based on brush_detail
    d = int(10 - brush_detail) * 3  # Filter size
    sigma_color = 0.1 * (10 - brush_detail) + 0.05
    sigma_space = 0.1 * (10 - brush_detail) + 0.05
    
    bilateral = cv2.bilateralFilter(img_float, d, sigma_color, sigma_space)
    
    # Step 2: Edge detection for detail preservation
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    edges = cv2.Canny(np.uint8(gray * 255), 50, 150).astype(np.float32) / 255.0
    
    # Dilate edges slightly
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel)
    
    # Step 3: Color quantization for a painterly look
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=color_simplification, random_state=0, n_init=10).fit(pixels)
    labels = kmeans.predict(pixels)
    quantized = kmeans.cluster_centers_[labels].reshape(img_array.shape)
    
    # Step 4: Combine the bilateral filtered image with the quantized image
    # Use edges to control the blend
    edges_3d = np.stack([edges] * 3, axis=2)
    blended = (quantized * 0.6 + bilateral * 255 * 0.4).astype(np.uint8)
    
    # Apply edge preservation
    preserved = (blended * (1 - edges_3d * edge_preservation) + 
                img_array * (edges_3d * edge_preservation)).astype(np.uint8)
    
    # Convert back to PIL
    result = Image.fromarray(preserved)
    
    # Step 5: Enhance colors and contrast
    enhancer = ImageEnhance.Color(result)
    result = enhancer.enhance(saturation_boost)
    
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(contrast_boost)
    
    # Apply a slight sharpening for brush stroke effect
    result = result.filter(ImageFilter.EDGE_ENHANCE_MORE)
    
    # Optional: Apply a subtle texture
    canvas_texture = np.random.normal(1, 0.03, result.size[::-1] + (3,))
    canvas_texture = np.clip(canvas_texture, 0.9, 1.1)
    
    textured = np.array(result).astype(np.float32) * canvas_texture
    textured = np.clip(textured, 0, 255).astype(np.uint8)
    
    return Image.fromarray(textured)

def apply_watercolor_effect(image, simplification=8, blur_strength=2, edge_intensity=0.7):
    """
    Apply a watercolor-like effect to an image.
    
    Parameters:
    -----------
    image : numpy.ndarray or PIL.Image
        Input image in RGB format.
    simplification : int, optional
        Number of colors to quantize the image to. Range: 4-16.
    blur_strength : int, optional
        Controls the level of blurring for watercolor wash effect. Range: 1-5.
    edge_intensity : float, optional
        How prominent the edges should be. Range: 0.0-1.0.
        
    Returns:
    --------
    PIL.Image
        Image with watercolor effect applied.
    """
    # Convert input to PIL Image if it's a numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image))
    elif not isinstance(image, Image.Image):
        raise TypeError("Input must be a numpy array or PIL Image")
    
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    # Step 1: Apply median filter to create flat color regions
    median_filtered = image.filter(ImageFilter.MedianFilter(3))
    
    # Step 2: Apply slight Gaussian blur for softening
    blurred = median_filtered.filter(ImageFilter.GaussianBlur(blur_strength))
    
    # Step 3: Enhance saturation slightly
    enhancer = ImageEnhance.Color(blurred)
    saturated = enhancer.enhance(1.2)
    
    # Step 4: Color quantization
    img_array = np.array(saturated)
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=simplification, random_state=0, n_init=10).fit(pixels)
    labels = kmeans.predict(pixels)
    quantized = kmeans.cluster_centers_[labels].reshape(img_array.shape).astype(np.uint8)
    
    # Step 5: Edge detection for watercolor borders
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    
    # Add slight distortion to edges for a hand-painted look
    edges_distorted = np.copy(edges)
    noise = np.random.normal(0, 2, edges.shape).astype(np.int8)
    edges_distorted = cv2.add(edges_distorted, noise)
    edges_distorted = cv2.dilate(edges_distorted, np.ones((2, 2), np.uint8))
    
    # Step 6: Combine quantized image with edges
    edges_3d = np.expand_dims(edges_distorted, axis=2) / 255.0 * edge_intensity
    edges_3d = np.repeat(edges_3d, 3, axis=2)
    
    result_array = quantized * (1 - edges_3d) + edges_3d * (30, 30, 50)  # Dark blue-gray edge color
    result_array = np.clip(result_array, 0, 255).astype(np.uint8)
    
    # Step 7: Add paper texture
    paper_texture = np.random.normal(1, 0.03, result_array.shape)
    paper_texture = np.clip(paper_texture, 0.95, 1.05)
    
    textured_array = (result_array.astype(np.float32) * paper_texture).astype(np.uint8)
    
    return Image.fromarray(textured_array)

def apply_oil_painting_effect(image, brush_size=5, intensity=10, color_levels=8):
    """
    Apply an oil painting effect to an image.
    
    Parameters:
    -----------
    image : numpy.ndarray or PIL.Image
        Input image in RGB format.
    brush_size : int, optional
        Size of the brush strokes. Range: 2-10.
    intensity : int, optional
        Intensity of the effect. Range: 1-20.
    color_levels : int, optional
        Number of color levels. Range: 4-32.
        
    Returns:
    --------
    PIL.Image
        Image with oil painting effect applied.
    """
    # Convert input to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    elif isinstance(image, np.ndarray):
        img_array = image
    else:
        raise TypeError("Input must be a numpy array or PIL Image")
    
    # Ensure the image is in RGB format
    if len(img_array.shape) == 2 or img_array.shape[2] == 1:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    
    # Apply oil painting effect using OpenCV
    oil_painting = cv2.xphoto.oilPainting(img_array, brush_size, intensity)
    
    # Apply color quantization for more artistic look
    pixels = oil_painting.reshape(-1, 3)
    kmeans = KMeans(n_clusters=color_levels, random_state=0, n_init=10).fit(pixels)
    labels = kmeans.predict(pixels)
    quantized = kmeans.cluster_centers_[labels].reshape(oil_painting.shape).astype(np.uint8)
    
    # Enhance contrast
    result = Image.fromarray(quantized)
    enhancer = ImageEnhance.Contrast(result)
    result = enhancer.enhance(1.2)
    
    # Apply canvas texture
    canvas_array = np.array(result)
    canvas_texture = np.random.normal(1, 0.02, canvas_array.shape)
    canvas_texture = np.clip(canvas_texture, 0.97, 1.03)
    
    textured_array = (canvas_array.astype(np.float32) * canvas_texture).astype(np.uint8)
    
    return Image.fromarray(textured_array)

def apply_vignette(image, intensity=1.0, radius=0.5, falloff=0.75, black=True):
    """
    Apply a vignette effect to an image.
    
    Parameters:
    -----------
    image : numpy.ndarray or PIL.Image
        Input image in RGB format.
    intensity : float, optional
        Strength of the vignette effect. Higher values create a stronger effect.
        Range: 0.0-2.0, default is 1.0.
    radius : float, optional
        Relative size of the unaffected center area. Lower values create a smaller
        bright center. Range: 0.1-1.5, default is 1.0.
    falloff : float, optional
        Controls how quickly the vignette transitions from center to edges.
        Higher values create a more gradual transition. Range: 0.1-1.0, default is 0.75.
    black : bool, optional
        If True, creates a dark vignette (edges darker than center).
        If False, creates a white vignette (edges brighter than center).
        Default is True.
        
    Returns:
    --------
    PIL.Image
        Image with vignette effect applied.
    """
    # Convert input to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        img_array = np.array(image)
        input_is_pil = True
    elif isinstance(image, np.ndarray):
        img_array = image.copy()
        input_is_pil = False
    else:
        raise TypeError("Input must be a numpy array or PIL Image")
    
    # Ensure the image is in RGB format
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Remove alpha channel
    
    # Get image dimensions
    height, width = img_array.shape[:2]
    
    # Create a normalized coordinate grid
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    
    # Calculate distance from center (normalized)
    y_dist = ((y - center_y) / center_y) ** 2
    x_dist = ((x - center_x) / center_x) ** 2
    dist = np.sqrt(y_dist + x_dist)
    
    # Adjust for non-square images to maintain circular vignette
    aspect_ratio = width / height
    if aspect_ratio > 1:
        dist = np.sqrt(y_dist + x_dist / (aspect_ratio ** 2))
    else:
        dist = np.sqrt(y_dist * (aspect_ratio ** 2) + x_dist)
    
    # Create vignette mask
    # Adjust the radius to control where the vignette effect begins
    # and apply falloff to control the transition
    mask = 1 - (np.clip((dist - radius) / falloff, 0, 1) * intensity)
    
    # Apply the mask to the image
    if black:
        # Dark vignette (traditional)
        mask = mask[:, :, np.newaxis]  # Add channel dimension
        vignetted = img_array * mask
        vignetted = np.clip(vignetted, 0, 255).astype(np.uint8)
    else:
        # White vignette (inverted)
        mask = 1 - mask
        mask = mask[:, :, np.newaxis]  # Add channel dimension
        white_value = 255
        vignetted = img_array * (1 - mask) + white_value * mask
        vignetted = np.clip(vignetted, 0, 255).astype(np.uint8)
    
    # Return in the same format as input
    if input_is_pil:
        return Image.fromarray(vignetted)
    else:
        return vignetted

def apply_advanced_vignette(image, intensity=1.0, shape='circle', offset_x=0.0, offset_y=0.0, 
                           color=(0, 0, 0), opacity=0.2, feather=0.5):
    """
    Apply an advanced vignette effect with more customization options.
    
    Parameters:
    -----------
    image : numpy.ndarray or PIL.Image
        Input image in RGB format.
    intensity : float, optional
        Strength of the vignette effect. Range: 0.0-2.0, default is 1.0.
    shape : str, optional
        Shape of the vignette. Options: 'circle', 'rectangle', 'horizontal', 'vertical'.
        Default is 'circle'.
    offset_x : float, optional
        Horizontal offset of the vignette center as a fraction of width.
        Range: -0.5 to 0.5, default is 0.0 (centered).
    offset_y : float, optional
        Vertical offset of the vignette center as a fraction of height.
        Range: -0.5 to 0.5, default is 0.0 (centered).
    color : tuple, optional
        RGB color of the vignette effect. Default is (0, 0, 0) for black.
    opacity : float, optional
        Maximum opacity of the vignette. Range: 0.0-1.0, default is 1.0.
    feather : float, optional
        Controls the softness of the vignette edge. Range: 0.0-1.0, default is 0.2.
        
    Returns:
    --------
    PIL.Image
        Image with advanced vignette effect applied.
    """
    # Convert input to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        img_array = np.array(image)
        input_is_pil = True
    elif isinstance(image, np.ndarray):
        img_array = image.copy()
        input_is_pil = False
    else:
        raise TypeError("Input must be a numpy array or PIL Image")
    
    # Ensure the image is in RGB format
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]  # Remove alpha channel
    
    # Get image dimensions
    height, width = img_array.shape[:2]
    
    # Create a coordinate grid with offset center
    y, x = np.ogrid[:height, :width]
    center_y = height * (0.5 + offset_y)
    center_x = width * (0.5 + offset_x)
    
    # Calculate distance from center based on shape
    if shape == 'circle':
        # For circular vignette
        aspect_ratio = width / height
        y_dist = ((y - center_y) / (height / 2)) ** 2
        x_dist = ((x - center_x) / (width / 2)) ** 2
        
        # Adjust for aspect ratio
        if aspect_ratio > 1:
            dist = np.sqrt(y_dist + x_dist / (aspect_ratio ** 2))
        else:
            dist = np.sqrt(y_dist * (aspect_ratio ** 2) + x_dist)
            
    elif shape == 'rectangle':
        # For rectangular vignette
        y_dist = np.abs(y - center_y) / (height / 2)
        x_dist = np.abs(x - center_x) / (width / 2)
        dist = np.maximum(y_dist, x_dist)
        
    elif shape == 'horizontal':
        # For horizontal vignette (top and bottom edges)
        dist = np.abs(y - center_y) / (height / 2)
        
    elif shape == 'vertical':
        # For vertical vignette (left and right edges)
        dist = np.abs(x - center_x) / (width / 2)
        
    else:
        raise ValueError("Shape must be one of: 'circle', 'rectangle', 'horizontal', 'vertical'")
    
    # Create vignette mask with adjusted feathering
    feather_adjusted = max(0.001, feather)  # Prevent division by zero
    mask = np.clip((dist - (1 - intensity)) / feather_adjusted, 0, 1)
    mask = 1 - (mask * opacity)
    
    # Apply the mask to the image with color
    mask_3d = mask[:, :, np.newaxis]  # Add channel dimension
    color_array = np.array(color, dtype=np.float32)
    
    # Blend original image with the vignette color
    vignetted = img_array * mask_3d + color_array * (1 - mask_3d)
    vignetted = np.clip(vignetted, 0, 255).astype(np.uint8)
    
    # Return in the same format as input
    if input_is_pil:
        return Image.fromarray(vignetted)
    else:
        return vignetted
