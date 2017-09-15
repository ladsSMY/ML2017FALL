from PIL import Image, ImageDraw
import sys
s = sys.argv[1]

img = Image.open(s)
width, height = img.size
white = (255, 255, 255)
image = Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image)

for y in range(height):
    for x in range(width):
        rgb = img.getpixel((x,y))
        RGB = []
        
        for i in range(len(rgb)):
        		RGB.append(int(rgb[i] / 2) )

        color = (RGB[0],RGB[1],RGB[2])
        draw.point((x,y), color)

image.save("Q2.png")