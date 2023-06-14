import cairosvg
from PIL import Image, ImageOps

def convert_svg_to_png(svg_path, png_path, dpi, border_thickness):
    # Convert SVG to PNG using cairosvg library
    cairosvg.svg2png(url=svg_path, write_to=png_path, dpi=dpi)

    # Open the generated PNG image
    image = Image.open(png_path)

    # Calculate the border size in pixels based on the DPI
    border_size = int(border_thickness * dpi / 25.4)

    # Add the white border
    image_with_border = ImageOps.expand(image, border_size, fill='white')

    # Save the modified image
    image_with_border.save(png_path)

# Example usage
svg_path = "analysis.svg"
png_path = "output.png"
dpi = 300  # Set the desired DPI here
border_thickness = 50  # Set the desired border thickness in millimeters

convert_svg_to_png(svg_path, png_path, dpi, border_thickness)
