from IPython.display import HTML
from base64 import b64encode

from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import numpy as np
import os

# manage and display colored bounding boxes on images, 
# typically for tracked objects in tasks such as object detection or tracking

class VisTrack:
    def __init__(self, unique_colors=400):
        """
        unique_colors (int): The number of unique colors (the number of unique colors dos not need to be greater than the max id)
        """
        self._unique_colors = unique_colors
        self._id_dict = {}
        self.p = np.zeros(unique_colors)
        self._colors = (np.array(sns.color_palette("hls", unique_colors))*255).astype(np.uint8)

    def _get_color(self, i):
        return tuple(self._colors[i])

    def _color(self, i):
        if i not in self._id_dict:
            inp = (self.p.max() - self.p ) + 1 
            if any(self.p == 0):
                nzidx = np.where(self.p != 0)[0]
                inp[nzidx] = 0
            soft_inp = inp / inp.sum()

            ic = np.random.choice(np.arange(self._unique_colors, dtype=int), p=soft_inp)
            self._id_dict[i] = ic

            self.p[ic] += 1

        ic = self._id_dict[i]
        return self._get_color(ic)

    def draw_bounding_boxes(self, im: np.ndarray, bboxes: np.ndarray, ids: np.ndarray,
                        names: dict, scores: np.ndarray) -> np.ndarray:
        """
        im (PIL.Image): The image 
        bboxes (np.ndarray): The bounding boxes. [[x1,y1,x2,y2],...]
        ids (np.ndarray): The id's for the bounding boxes
        scores (np.ndarray): The scores's for the bounding boxes
        """
        if len(ids)==0:
            return im

        im = im.copy()

        # convert im to PIL.Image to use draw
        im=Image.fromarray(im) 
        draw = ImageDraw.Draw(im)

        # Define a font with a larger size
        font_size = 20  # You can adjust this value
        font = ImageFont.truetype("Helvetica.ttc", font_size)  # make sure this font is available in /System/Library/Fonts/

        for bbox, id_, score in zip(bboxes, ids, scores):
            color = self._color(id_)
            draw.rectangle((*bbox.astype(np.int64),), outline=color)

            text = f'{id_}: {names[str(id_)]} - {score:.2f}'


            # get text width and height 
            box = draw.textbbox((0, 0), text, font=font) # get the region for text
            text_w, text_h = box[2] - box[0], box[3] - box[1]

            # draw rectangular boxes
            draw.rectangle((bbox[0], bbox[1], bbox[0] + text_w, bbox[1] + text_h), fill=color, outline=color)
            draw.text((bbox[0], bbox[1]), text, fill=(0, 0, 0), font=font)

        return np.array(im)