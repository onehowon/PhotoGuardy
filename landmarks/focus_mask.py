from PIL import Image, ImageDraw

def apply_focus_mask(image, landmarks):
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    for (x, y) in landmarks:
        draw.ellipse((x-10, y-10, x+10, y+10), fill=255)
    return mask
