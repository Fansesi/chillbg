import os
import sys
import numpy as np
import argparse
import tomllib

from src import generate_desktop_background, post_process


def load_config(config_path):
    """Load configuration from TOML file"""
    try:
        with open(config_path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Create desktop backgrounds with images and noise effects"
    )
    parser.add_argument(
        "--config", default="config.toml", help="Path to TOML configuration file"
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=False,
        help="Path to the poster image (overrides config file)",
    )

    args = parser.parse_args()

    # Load configuration from TOML
    config = load_config(args.config)
    if not config:
        return 1

    # Use command-line image path if provided, otherwise from config
    image_path = args.image_path or config.get("image", {}).get("path")

    if not image_path:
        print("Error: No image path provided in config or command line")
        return 1

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return 1

    # Get parameters from config
    bg_color = config.get("background", {}).get("color")

    # If bg_color is the string "None", convert to actual None
    if bg_color and bg_color.lower() == "none":
        bg_color = None

    noise_type = config.get("noise", {}).get("type", "perlin")
    intensity = config.get("noise", {}).get("intensity", 0.1)
    noise_scale = config.get("noise", {}).get("scale", 100)
    noise_seed = config.get("noise", {}).get("seed")

    border_width = config.get("border", {}).get("width", 0)
    border_color = config.get("border", {}).get("color", "#FFFFFF")

    width = config.get("output", {}).get("width", 1920)
    height = config.get("output", {}).get("height", 1080)

    image_scale_percent = config.get("image", {}).get("scale_percent")

    post = config.get("post", {}).get("post")
    post_effect = config.get("post", {}).get("effect")
    vignette = config.get("post", {}).get("vignette")
    vig_intensity = config.get("post", {}).get("vig_intensity")
    vig_opacity = config.get("post", {}).get("vig_opacity")

    # Output path
    output_path = config.get("output", {}).get("path")
    if output_path and output_path.lower() == "none":
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"output/{base_name}_{noise_type}_bg.png"

    background_img = generate_desktop_background(
        image_path,
        bg_color,
        output_path,
        noise_type=noise_type,
        intensity=intensity,
        noise_scale=noise_scale,
        noise_seed=noise_seed,
        border_width=border_width,
        border_color=border_color,
        width=width,
        height=height,
        image_scale_percent=image_scale_percent,
    )
    
    if post:
        background_img = post_process(
            background_img,
            post_effect=post_effect,
            vignette=vignette,
            vig_intensity=vig_intensity,
            vig_opacity=vig_opacity,
        )

    # Save the final image
    try:
        background_img.save(output_path)
        print(f"Background saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


    return 1


if __name__ == "__main__":
    sys.exit(main())
