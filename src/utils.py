from PIL import Image


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def get_dominant_color(image_path, num_colors=5):
    """Extract the most dominant color from an image"""
    try:
        # Open the image
        img = Image.open(image_path)

        # Resize image to speed up processing
        img = img.copy()
        img.thumbnail((100, 100))

        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Get pixel data
        pixels = list(img.getdata())

        # Count occurrences of each color
        color_counts = {}
        for pixel in pixels:
            if pixel in color_counts:
                color_counts[pixel] += 1
            else:
                color_counts[pixel] = 1

        # Sort colors by count
        sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)

        # Get the most common colors
        top_colors = sorted_colors[:num_colors]

        # Get the most dominant color
        dominant_rgb = top_colors[0][0]

        # Convert RGB to hex
        dominant_hex = "#{:02x}{:02x}{:02x}".format(
            dominant_rgb[0], dominant_rgb[1], dominant_rgb[2]
        )

        return dominant_hex

    except Exception as e:
        print(f"Error analyzing image colors: {e}")
        return "#242424"  # Default to dark gray on error
