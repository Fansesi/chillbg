from PIL import Image
import numpy as np

from .noise import (
    generate_white_noise,
    generate_perlin_noise,
    generate_simplex_noise,
    generate_value_noise,
    generate_white_noise,
    generate_voronoi_noise,
    generate_colored_voronoi
)
try:
    from .post_process import (
        apply_painterly_effect,
        apply_watercolor_effect,
        apply_oil_painting_effect,
        apply_vignette,
        apply_advanced_vignette
    )
except:
    print("[WARNING] Post-processing is not available.")


from .utils import hex_to_rgb, get_dominant_color


def generate_desktop_background(
    image_path,
    bg_color,
    output_path,
    noise_type="perlin",
    intensity=0.1,
    noise_scale=100,
    noise_seed=None,
    border_width=0,
    border_color="#FFFFFF",
    width=1920,
    height=1080,
    image_scale_percent=70,
):
    """Generate a desktop background with a poster image on a colored noisy background"""
    # If bg_color is None, get dominant color from image
    if bg_color is None:
        bg_color = get_dominant_color(image_path)
        print(f"Auto-selected background color: {bg_color}")

    # Convert hex colors to RGB
    bg_rgb = hex_to_rgb(bg_color)
    border_rgb = hex_to_rgb(border_color)

    # Create background image
    background = np.ones((height, width, 3), dtype=np.uint8)

    # Apply base color
    background[:, :, 0] = bg_rgb[0]
    background[:, :, 1] = bg_rgb[1]
    background[:, :, 2] = bg_rgb[2]

    # Generate noise based on selected type
    if noise_type == "perlin":
        noise = generate_perlin_noise(
            (height, width), scale=noise_scale, seed=noise_seed
        )
    elif noise_type == "value":
        noise = generate_value_noise(
            (height, width), scale=noise_scale, seed=noise_seed
        )
    elif noise_type == "simplex":
        noise = generate_simplex_noise(
            (height, width), scale=noise_scale, seed=noise_seed
        )
    elif noise_type == "white":
        noise = generate_white_noise(
            (height, width), intensity=intensity, seed=noise_seed
        )
    elif noise_type == "gaussian":
        noise = generate_gaussian_noise(
            (height, width),
            intensity=intensity,
            sigma=noise_scale / 25,
            seed=noise_seed,
        )
    elif noise_type == "voronoi":
        if not isinstance(intensity, int):
            raise ValueError("intensity for the voronoi is the number of edges. Use integer values.")
        noise = generate_voronoi_noise(
            (height, width), intensity=intensity, seed=noise_seed
        )
    elif noise_type == "None":
        pass
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    if noise_type != "None":
        # Apply noise to each color channel
        for c in range(3):
            # Convert to float for calculations
            channel = background[:, :, c].astype(float)

            # Apply noise with intensity
            modified = channel * (1.0 - intensity + noise * intensity * 2)

            # Clip values to valid range and convert back to uint8
            background[:, :, c] = np.clip(modified, 0, 255).astype(np.uint8)

    # Convert to PIL Image
    background_img = Image.fromarray(background)

    # Open and paste the poster image
    try:
        poster = Image.open(image_path)

        # Resize poster while maintaining aspect ratio
        poster_width, poster_height = poster.size
        aspect_ratio = poster_width / poster_height

        # Account for border in size calculations
        max_height = int(height * 0.8) - (border_width * 2)
        max_width = int(width * 0.8) - (border_width * 2)

        # Apply image_scale_percent if provided
        if image_scale_percent is not None:
            # Scale based on percentage of maximum allowed size
            scale_factor = image_scale_percent / 100.0
            new_height = int(max_height * scale_factor)
            new_width = int(new_height * aspect_ratio)

            # If poster is too wide after scaling, scale by width instead
            if new_width > max_width * scale_factor:
                new_width = int(max_width * scale_factor)
                new_height = int(new_width / aspect_ratio)
        else:
            # Default behavior (fit to 80% of background)
            new_height = max_height
            new_width = int(new_height * aspect_ratio)

            # If poster is too wide, scale by width instead
            if new_width > max_width:
                new_width = max_width
                new_height = int(new_width / aspect_ratio)

        poster = poster.resize((new_width, new_height), Image.LANCZOS)

        # Create a new image with border
        if border_width > 0:
            bordered_width = new_width + (border_width * 2)
            bordered_height = new_height + (border_width * 2)
            bordered_img = Image.new(
                "RGB", (bordered_width, bordered_height), border_color
            )

            # Paste poster onto border
            bordered_img.paste(poster, (border_width, border_width))
            poster = bordered_img
            new_width = bordered_width
            new_height = bordered_height

        # Calculate position to center the poster
        pos_x = (width - new_width) // 2
        pos_y = (height - new_height) // 2

        # Paste the poster onto the background
        # If poster has transparency, use it
        if poster.mode == "RGBA":
            background_img.paste(poster, (pos_x, pos_y), poster)
        else:
            background_img.paste(poster, (pos_x, pos_y))

    except Exception as e:
        print(f"Error processing poster image: {e}")
        return False

    return background_img

def post_process(
    background_img,
    post_effect="None",
    vignette=False,
    vig_intensity=0.5,
    vig_opacity=0.5,
):
    if post_effect == "water":
        background_img=apply_watercolor_effect(background_img)
    elif post_effect == "oil":
        background_img=apply_oil_painting_effect(background_img)
    elif post_effect == "painterly":
        background_img=apply_painterly_effect(background_img)
    elif post_effect == "None":
        pass
    else:
        raise ValueError(f"Post processes effect is set {post_effect}, should be one of the following: water, oily, painterly")

    if vignette:
        background_img=apply_advanced_vignette(background_img, intensity=vig_intensity, opacity=vig_opacity)

    return background_img
