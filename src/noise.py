import numpy as np
import random
from numba import njit

@njit
def _perlin_generate_octave(shape, scale):
    noise_map = np.zeros(shape)
    grid = np.random.rand(shape[0] // scale + 2, shape[1] // scale + 2) * 2 - 1

    for y in range(shape[0]):
        for x in range(shape[1]):
            # Get grid cell coordinates
            x0 = x // scale
            y0 = y // scale
            x1 = x0 + 1
            y1 = y0 + 1

            # Get interpolation weights
            sx = (x % scale) / scale
            sy = (y % scale) / scale

            # Interpolate between grid point values
            n0 = grid[y0, x0]
            n1 = grid[y0, x1]
            ix0 = n0 + sx * (n1 - n0)

            n0 = grid[y1, x0]
            n1 = grid[y1, x1]
            ix1 = n0 + sx * (n1 - n0)

            value = ix0 + sy * (ix1 - ix0)
            noise_map[y, x] = value

    return noise_map

@njit
def generate_perlin_noise(
    shape, scale=100, octaves=6, persistence=0.5, lacunarity=2.0, seed=None
):
    """Generate Perlin noise"""
    if seed is not None:
        np.random.seed(seed)

    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1

    for _ in range(octaves):
        noise += amplitude * _perlin_generate_octave(shape, int(scale / frequency))
        frequency *= lacunarity
        amplitude *= persistence

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise


@njit
def _value_generate_octave(shape, scale):
    grid_h = shape[0] // scale + 2
    grid_w = shape[1] // scale + 2

    # Generate random values at grid points
    grid = np.random.rand(grid_h, grid_w)

    noise_map = np.zeros(shape)

    for y in range(shape[0]):
        for x in range(shape[1]):
            # Get grid cell coordinates
            x0 = x // scale
            y0 = y // scale

            # Get interpolation weights
            sx = (x % scale) / scale
            sy = (y % scale) / scale

            # Smoothstep interpolation
            sx = sx * sx * (3 - 2 * sx)
            sy = sy * sy * (3 - 2 * sy)

            # Interpolate
            n0 = grid[y0, x0]
            n1 = grid[y0, x0 + 1]
            n2 = grid[y0 + 1, x0]
            n3 = grid[y0 + 1, x0 + 1]

            ix0 = n0 + sx * (n1 - n0)
            ix1 = n2 + sx * (n3 - n2)
            value = ix0 + sy * (ix1 - ix0)

            noise_map[y, x] = value

    return noise_map

@njit
def generate_value_noise(
    shape, scale=100, octaves=6, persistence=0.5, lacunarity=2.0, seed=None
):
    """Generate Value noise"""
    if seed is not None:
        np.random.seed(seed)

    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1

    for _ in range(octaves):
        noise += amplitude * _value_generate_octave(shape, int(scale / frequency))
        frequency *= lacunarity
        amplitude *= persistence

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

@njit
def generate_white_noise(shape, intensity=0.2, seed=None):
    """Generate white noise"""
    if seed is not None:
        np.random.seed(seed)

    noise = np.random.rand(*shape)
    # Scale down to control intensity
    noise = noise * intensity
    return noise

@njit
def generate_gaussian_noise(shape, intensity=0.2, sigma=1.0, seed=None):
    """Generate Gaussian filtered noise using a custom implementation without scipy"""
    if seed is not None:
        np.random.seed(seed)

    # Generate white noise
    noise = np.random.rand(*shape)

    # Create a Gaussian kernel for filtering
    kernel_size = int(6 * sigma + 1)  # Kernel size based on sigma (6-sigma rule)
    if kernel_size % 2 == 0:  # Make sure kernel size is odd
        kernel_size += 1

    # Create a 1D Gaussian kernel
    k = kernel_size // 2
    x = np.arange(-k, k + 1)
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / np.sum(kernel_1d)  # Normalize

    # Apply separable convolution (much faster than 2D convolution)
    # First convolve rows
    temp = np.zeros_like(noise)
    for i in range(shape[0]):
        temp[i, :] = np.convolve(noise[i, :], kernel_1d, mode="same")

    # Then convolve columns
    filtered = np.zeros_like(noise)
    for j in range(shape[1]):
        filtered[:, j] = np.convolve(temp[:, j], kernel_1d, mode="same")

    # Normalize to [0,1] range and scale by intensity
    filtered = (
        (filtered - filtered.min()) / (filtered.max() - filtered.min()) * intensity
    )

    return filtered

@njit
def generate_simplex_noise(
    shape, scale=100, octaves=6, persistence=0.5, lacunarity=2.0, seed=None
):
    """A simplified version of simplex-like noise using perlin functions"""
    if seed is not None:
        np.random.seed(seed)

    # This is a simplified approximation since true simplex requires more complex implementation
    shape_extended = (shape[0] + 20, shape[1] + 20)  # Add margins to avoid edge effects
    noise = generate_perlin_noise(
        shape_extended, scale, octaves, persistence, lacunarity, seed
    )

    # Cut the margins
    noise = noise[10 : shape[0] + 10, 10 : shape[1] + 10]

    # Apply slight distortion for simplex-like effect
    distortion = np.zeros(shape)
    for y in range(shape[0]):
        for x in range(shape[1]):
            angle = noise[y, x] * 2 * np.pi
            dx = np.cos(angle) * scale / 10
            dy = np.sin(angle) * scale / 10

            y_sample = min(max(int(y + dy), 0), shape[0] - 1)
            x_sample = min(max(int(x + dx), 0), shape[1] - 1)

            distortion[y, x] = noise[y_sample, x_sample]

    # Blend original and distortion
    noise = noise * 0.7 + distortion * 0.3

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    return noise

@njit
def generate_voronoi_noise(shape, num_points=10, metric='euclidean', jitter=0.0, seed=None):
    """
    Generate Voronoi noise using only NumPy (no scipy dependency).
    
    Parameters:
    -----------
    shape : tuple
        The shape of the output array (height, width).
    num_points : int, optional
        Number of random points to generate for Voronoi cells, default is 10.
    metric : str, optional
        Distance metric to use. Options: 'euclidean', 'manhattan', 'chebyshev'.
        Default is 'euclidean'.
    jitter : float, optional
        Amount of random displacement to add to the output, default is 0.0.
    seed : int, optional
        Random seed for reproducibility, default is None.
        
    Returns:
    --------
    numpy.ndarray
        A 2D array containing Voronoi noise values normalized between 0 and 1.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    height, width = shape
    
    # Generate random points for Voronoi cells
    points = np.random.rand(num_points, 2)
    points[:, 0] *= width
    points[:, 1] *= height
    
    # Create coordinate arrays
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    
    # Initialize the distance field with infinity
    min_distances = np.full(shape, np.inf)
    
    # Calculate distances for each point and update the minimum distance field
    for point in points:
        # Create coordinate grids relative to the current point
        x_diff = x_coords - point[0]
        y_diff = y_coords[:, np.newaxis] - point[1]
        
        # Calculate distances based on the selected metric
        if metric == 'euclidean':
            distances = np.sqrt(x_diff[np.newaxis, :]**2 + y_diff**2)
        elif metric == 'manhattan':
            distances = np.abs(x_diff[np.newaxis, :]) + np.abs(y_diff)
        elif metric == 'chebyshev':
            distances = np.maximum(np.abs(x_diff[np.newaxis, :]), np.abs(y_diff))
        else:
            raise ValueError("Metric must be 'euclidean', 'manhattan', or 'chebyshev'")
        
        # Update the minimum distance field
        min_distances = np.minimum(min_distances, distances)
    
    # Normalize to range [0, 1]
    noise = (min_distances - np.min(min_distances)) / (np.max(min_distances) - np.min(min_distances))
    
    # Add jitter if specified
    if jitter > 0:
        noise += np.random.normal(0, jitter, noise.shape)
        noise = np.clip(noise, 0, 1)  # Ensure values stay in [0, 1]
    
    return noise

@njit
def generate_colored_voronoi(shape, num_points=10, seed=None):
    """
    Generate colored Voronoi diagram where each cell has a unique color value.
    Uses only NumPy (no scipy dependency).
    
    Parameters:
    -----------
    shape : tuple
        The shape of the output array (height, width).
    num_points : int, optional
        Number of random points to generate for Voronoi cells, default is 10.
    seed : int, optional
        Random seed for reproducibility, default is None.
        
    Returns:
    --------
    numpy.ndarray
        A 2D array where each Voronoi cell has a unique normalized value.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    height, width = shape
    
    # Generate random points for Voronoi cells
    points = np.random.rand(num_points, 2)
    points[:, 0] *= width
    points[:, 1] *= height
    
    # Generate random color for each cell
    cell_colors = np.random.rand(num_points)
    
    # Create coordinate arrays
    x_coords = np.arange(width)
    y_coords = np.arange(height)
    
    # Initialize arrays to store closest point index and minimum distance
    closest_indices = np.zeros(shape, dtype=int)
    min_distances = np.full(shape, np.inf)
    
    # Calculate distances for each point and find the closest one for each pixel
    for i, point in enumerate(points):
        # Create coordinate grids relative to the current point
        x_diff = x_coords - point[0]
        y_diff = y_coords[:, np.newaxis] - point[1]
        
        # Calculate Euclidean distances
        distances = np.sqrt(x_diff[np.newaxis, :]**2 + y_diff**2)
        
        # Update minimum distances and closest indices
        mask = distances < min_distances
        min_distances[mask] = distances[mask]
        closest_indices[mask] = i
    
    # Assign colors based on the closest point
    noise = cell_colors[closest_indices]
    
    return noise
