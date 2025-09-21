from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imageio
import imageio.v3 as iio


def tiff_to_gif(input_tiff: str, 
                output_gif: str, 
                duration: int = 50, 
                loop: int = 1):
    """
    Convert a multi-frame TIFF image to an animated GIF.
    Args:
        input_tiff: Path to the input multi-frame TIFF file.
        output_gif: Path to save the output GIF file.
        duration: Duration of each frame in milliseconds.
        loop: Number of loops (0 for infinite looping).
    """
    with Image.open(input_tiff) as img:
        frames = []
        try:
            while True:
                frames.append(img.convert("RGB").copy()) 
                img.seek(img.tell() + 1)
        except EOFError:
            pass 
        if frames:
            frames[0].save(output_gif, save_all=True, append_images=frames[1:], duration=duration, loop=loop, disposal=2)
    print(f"GIF saved as {output_gif}")


def create_transition_gif(noisy_gif_path, denoised_gif_path, output_path, duration=0.1, width_ratio=0.005, loop: int = 0):
    noisy = Image.open(noisy_gif_path)
    processed = Image.open(denoised_gif_path)
    frames_noisy, frames_processed = [], []
    try:
        while True:
            frames_noisy.append(noisy.convert("RGB").copy())  
            frames_processed.append(processed.convert("RGB").copy())
            noisy.seek(noisy.tell() + 1)
            processed.seek(processed.tell() + 1)
    except EOFError:
        pass  

    num_frames = min(len(frames_noisy), len(frames_processed))
    width, height = frames_noisy[0].size
    bar_width = int(width * width_ratio)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(height * 0.05))
    except IOError:
        font = ImageFont.load_default(int(height * width_ratio * 10))
    new_frames = []
    for i in range(num_frames):
        noisy_frame = frames_noisy[i]
        denoised_frame = frames_processed[i]
        new_frame = Image.new("RGB", (width, height))
        bar_position = int(i / (num_frames - 1) * width)
        new_frame.paste(denoised_frame.crop((0, 0, bar_position, height)), (0, 0))
        new_frame.paste(noisy_frame.crop((bar_position, 0, width, height)), (bar_position, 0))
        draw = ImageDraw.Draw(new_frame)
        draw.rectangle([bar_position - bar_width // 2, 0, bar_position + bar_width // 2, height], fill="white")
        draw.rectangle([0, 0, width - 1, height - 1], outline="white", width=bar_width)
        draw.text((20, 20), "Denoised image", font=font, fill="white")
        right_label_x = width - 20 - draw.textlength("Noisy image", font=font)
        if bar_position < right_label_x:
            draw.text((right_label_x, 10), "Noisy image", font=font, fill="white")  
        new_frames.append(np.array(new_frame))
    imageio.mimsave(output_path, new_frames, duration=duration, loop=loop)