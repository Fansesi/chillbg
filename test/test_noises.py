import unittest
import os
import numpy as np
from PIL import Image
import tempfile
import shutil
from pathlib import Path

from src.generate import (
    generate_desktop_background,
    post_process,
)
from src.utils import (
    hex_to_rgb,
    get_dominant_color,
)
from src.noise import (
    generate_perlin_noise,
    generate_value_noise,
    generate_simplex_noise,
    generate_white_noise,
    generate_gaussian_noise,
    generate_voronoi_noise,
)
from src.post_process import (
    apply_watercolor_effect,
    apply_oil_painting_effect,
    apply_painterly_effect,
    apply_advanced_vignette
)

class TestDesktopBackgroundGenerator(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create a simple test image
        self.test_image_path = os.path.join(self.test_dir, "test_image.png")
        test_img = Image.new('RGB', (300, 200), color='red')
        test_img.save(self.test_image_path)
        
        # Output path for generated backgrounds
        self.output_path = os.path.join(self.test_dir, "output.png")
    
    def tearDown(self):
        # Clean up temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_generate_desktop_background_basic(self):
        """Test basic functionality of generate_desktop_background."""
        bg_img = generate_desktop_background(
            image_path=self.test_image_path,
            bg_color="#0000FF",  # Blue background
            output_path=self.output_path,
            width=800,
            height=600
        )
        
        # Check that an image was returned
        self.assertIsInstance(bg_img, Image.Image)
        
        # Check dimensions
        self.assertEqual(bg_img.width, 800)
        self.assertEqual(bg_img.height, 600)
    
    def test_generate_desktop_background_auto_color(self):
        """Test auto color selection from image."""
        bg_img = generate_desktop_background(
            image_path=self.test_image_path,
            bg_color=None,  # Should auto-select from image
            output_path=self.output_path
        )
        
        self.assertIsInstance(bg_img, Image.Image)
    
    def test_noise_types(self):
        """Test all available noise types."""
        noise_types = ["perlin", "value", "simplex", "white", "gaussian", "voronoi", "None"]
        
        for noise_type in noise_types:
            with self.subTest(noise_type=noise_type):
                intensity = 5 if noise_type == "voronoi" else 0.2
                
                bg_img = generate_desktop_background(
                    image_path=self.test_image_path,
                    bg_color="#333333",
                    output_path=self.output_path,
                    noise_type=noise_type,
                    intensity=intensity,
                    noise_scale=50
                )
                
                self.assertIsInstance(bg_img, Image.Image)
    
    def test_border_options(self):
        """Test border width and color options."""
        bg_img = generate_desktop_background(
            image_path=self.test_image_path,
            bg_color="#000000",
            output_path=self.output_path,
            border_width=20,
            border_color="#FF0000"  # Red border
        )
        
        self.assertIsInstance(bg_img, Image.Image)
    
    def test_image_scaling(self):
        """Test different image scaling options."""
        scaling_percentages = [30, 50, 90]
        
        for scale in scaling_percentages:
            with self.subTest(image_scale_percent=scale):
                bg_img = generate_desktop_background(
                    image_path=self.test_image_path,
                    bg_color="#222222",
                    output_path=self.output_path,
                    image_scale_percent=scale
                )
                
                self.assertIsInstance(bg_img, Image.Image)
    
    def test_noise_parameters(self):
        """Test noise parameters like scale and seed."""
        bg_img = generate_desktop_background(
            image_path=self.test_image_path,
            bg_color="#444444",
            output_path=self.output_path,
            noise_type="perlin",
            noise_scale=150,
            noise_seed=42,
            intensity=0.3
        )
        
        self.assertIsInstance(bg_img, Image.Image)
    
    def test_post_process_effects(self):
        """Test all post-processing effects."""
        bg_img = generate_desktop_background(
            image_path=self.test_image_path,
            bg_color="#555555",
            output_path=self.output_path
        )
        
        effects = ["water", "oil", "painterly", "None"]
        
        for effect in effects:
            with self.subTest(post_effect=effect):
                processed_img = post_process(
                    bg_img,
                    post_effect=effect
                )
                
                self.assertIsInstance(processed_img, Image.Image)
    
    def test_vignette_effect(self):
        """Test vignette effect with different intensities."""
        bg_img = generate_desktop_background(
            image_path=self.test_image_path,
            bg_color="#666666",
            output_path=self.output_path
        )
        
        intensities = [0.2, 0.5, 0.8]
        opacities = [0.3, 0.6, 0.9]
        
        for intensity, opacity in zip(intensities, opacities):
            with self.subTest(intensity=intensity, opacity=opacity):
                processed_img = post_process(
                    bg_img,
                    post_effect="None",
                    vignette=True,
                    vig_intensity=intensity,
                    vig_opacity=opacity
                )
                
                self.assertIsInstance(processed_img, Image.Image)
    
    def test_combined_post_processing(self):
        """Test combining post-processing effects with vignette."""
        bg_img = generate_desktop_background(
            image_path=self.test_image_path,
            bg_color="#777777",
            output_path=self.output_path
        )
        
        processed_img = post_process(
            bg_img,
            post_effect="water",
            vignette=True,
            vig_intensity=0.6,
            vig_opacity=0.4
        )
        
        self.assertIsInstance(processed_img, Image.Image)
    
    def test_integration(self):
        """Test full integration of both functions."""
        bg_img = generate_desktop_background(
            image_path=self.test_image_path,
            bg_color="#888888",
            output_path=self.output_path,
            noise_type="perlin",
            intensity=0.2,
            border_width=10,
            border_color="#00FF00"
        )
        
        processed_img = post_process(
            bg_img,
            post_effect="painterly",
            vignette=True,
            vig_intensity=0.5,
            vig_opacity=0.5
        )
        
        # Save the final image
        final_output_path = os.path.join(self.test_dir, "final_output.png")
        processed_img.save(final_output_path)
        
        # Check that file exists
        self.assertTrue(os.path.exists(final_output_path))
        
        # Open and check the saved image
        saved_img = Image.open(final_output_path)
        self.assertEqual(saved_img.width, 1920)  # Default width
        self.assertEqual(saved_img.height, 1080)  # Default height

    def test_error_handling(self):
        """Test error handling for invalid parameters."""
        # Test invalid noise type
        with self.assertRaises(ValueError):
            generate_desktop_background(
                image_path=self.test_image_path,
                bg_color="#999999",
                output_path=self.output_path,
                noise_type="invalid_noise"
            )
        
        # Test invalid post-processing effect
        bg_img = generate_desktop_background(
            image_path=self.test_image_path,
            bg_color="#AAAAAA",
            output_path=self.output_path
        )
        
        with self.assertRaises(ValueError):
            post_process(
                bg_img,
                post_effect="invalid_effect"
            )
        
        # Test non-integer intensity for voronoi noise
        with self.assertRaises(ValueError):
            generate_desktop_background(
                image_path=self.test_image_path,
                bg_color="#BBBBBB",
                output_path=self.output_path,
                noise_type="voronoi",
                intensity=0.5  # Should be an integer
            )

if __name__ == '__main__':
    unittest.main()
    