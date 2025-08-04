from PIL import Image, ImageDraw, ImageFont
import re


class FragmentCreator:
    def __init__(
        self,
        fragment_width=1654,
        fragment_height=2339,
        font_path="arial.ttf",
        font_size_text=14,
        font_size_caption=12,
        line_spacing=1.3,
        horizontal_padding_ratio=0.00,
        n_images_per_row=2,
    ):
        """Create a fragment image from a text and a list of images with captions.

        Args:
            fragment_width (int, optional): Maximum fragment width. Defaults to 1654.
            fragment_height (int, optional): Maximum fragment height. Defaults to 2339.
            font_path (str, optional): Path to the font file. Defaults to "arial.ttf".
            font_size_text (int, optional): Font size for paragraph. Defaults to 14.
            font_size_caption (int, optional): Font size for captions. Defaults to 12.
            line_spacing (float, optional): Vertical space between lines. Defaults to 1.3.
            horizontal_padding_ratio (float, optional): Horizontal padding for images and fragment. Defaults to 0.00.
            n_images_per_row (int, optional): No. of image-caption pairs to put in a row of the grid. Defaults to 2.
        """
        self.fragment_width = fragment_width
        self.fragment_height = fragment_height

        self.font_path = font_path
        self.font_size_text = font_size_text
        self.font_size_caption = font_size_caption
        self.font_text = ImageFont.truetype(self.font_path, self.font_size_text)
        self.font_caption = ImageFont.truetype(self.font_path, self.font_size_caption)

        self.line_spacing = line_spacing
        self.horizontal_padding_ratio = horizontal_padding_ratio
        self.n_images_per_row = n_images_per_row

        self.real_fragment_width = int(
            self.fragment_width * (1 - 2 * self.horizontal_padding_ratio)
        )

        self.image_width = int(self.real_fragment_width / (self.n_images_per_row * 2))
        self.real_image_width = int(
            self.image_width * (1 - 2 * self.horizontal_padding_ratio)
        )

    def wrap_text(self, text, is_caption=False):
        lines = []
        words = text.split(" ")

        font = self.font_caption if is_caption else self.font_text
        max_width = self.real_image_width if is_caption else self.real_fragment_width

        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            line_width = font.getlength(test_line)
            if line_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line[:-1])
                current_line = word + " "

        lines.append(current_line[:-1])
        return lines

    def text_to_image(self, text, is_caption=False):
        font = self.font_caption if is_caption else self.font_text
        font_size = self.font_size_caption if is_caption else self.font_size_text
        width = self.image_width if is_caption else self.fragment_width

        image = Image.new("RGB", (width, self.fragment_height), (255, 255, 255))

        # Find all link positions in the original text
        split_text = re.split(r'<a href="[^"]*">(.*?)</a>', text)

        # Wrap the text after processing
        wrapped_text = self.wrap_text("".join(split_text), is_caption)

        draw = ImageDraw.Draw(image)

        x = int(width * self.horizontal_padding_ratio)
        y = int(font_size * (self.line_spacing - 1))

        # Current position in processed text
        split_index = 0
        split_char_index = 0

        # Draw each line of wrapped text
        for line in wrapped_text:
            line_x = x
            line_chars = []

            for i, char in enumerate(line):
                color = (0, 0, 0) if split_index % 2 == 0 else (0, 0, 255)

                if split_index < len(split_text) and not split_text[split_index]:
                    color = (0, 0, 0) if color == (0, 0, 255) else (0, 0, 255)
                    split_index += 1

                line_chars.append((char, color))

                # Move to the next split text segment
                split_char_index += 1
                if split_index < len(split_text) and split_char_index >= len(split_text[split_index]):
                    split_index += 1
                    split_char_index = 0

            # Draw each segment
            for part, color in line_chars:
                draw.text((line_x, y), part, color, font=font)
                line_x += draw.textlength(part, font=font)
                
            y += int(font_size * self.line_spacing)
            # Move to the next split text segment
            split_char_index += 1
            if split_index < len(split_text) and split_char_index >= len(split_text[split_index]):
                split_index += 1
                split_char_index = 0

        # Cut the image height to the text height
        image = image.crop((0, 0, width, y))
        return image

    def image_caption_to_image(self, image, caption):
        caption_image = self.text_to_image(caption, is_caption=True)

        max_height = max(image.height, caption_image.height)

        composite_image = Image.new(
            "RGB", (self.image_width * 2, max_height), (255, 255, 255)
        )

        composite_image.paste(image, (0, 0))
        composite_image.paste(caption_image, (self.image_width, 0))

        return composite_image

    def example_to_image(self, example):
        text_image = self.text_to_image(example["text"])
        image_caption_images = [
            self.image_caption_to_image(img, caption)
            for img, caption in zip(
                example["images"]["image"], example["images"]["caption"]
            )
        ]

        current_height = text_image.height
        image_caption_images_to_keep = []
        img_cap_heights = [img_cap.height for img_cap in image_caption_images]
        for row in range(0, len(image_caption_images), self.n_images_per_row):
            max_row_height = max(img_cap_heights[row : row + self.n_images_per_row])
            if current_height + max_row_height > self.fragment_height:
                break
            image_caption_images_to_keep.extend(
                image_caption_images[row : row + self.n_images_per_row]
            )
            current_height += max_row_height

        composite_image = Image.new(
            "RGB", (self.fragment_width, current_height), (255, 255, 255)
        )

        # paste the image caption images
        x_offset = int(self.image_width * self.horizontal_padding_ratio)
        y_offset = 0
        for row in range(0, len(image_caption_images_to_keep), self.n_images_per_row):
            max_row_height = 0
            for i, img_cap_image in enumerate(
                image_caption_images_to_keep[row : row + self.n_images_per_row]
            ):
                composite_image.paste(img_cap_image, (x_offset, y_offset))
                x_offset += img_cap_image.width
                max_row_height = max(max_row_height, img_cap_image.height)
            x_offset = 0
            y_offset += max_row_height

        # paste the text image
        composite_image.paste(text_image, (0, y_offset))

        return composite_image
