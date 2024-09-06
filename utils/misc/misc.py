import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def to_jpg(img, quality=100):
    # # opencv
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # img = cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # PIL
    buffered = io.BytesIO()
    Image.fromarray(img).save(buffered, format="JPEG", quality=quality)
    img = Image.open(buffered)

    return img

    
def upsample(img, size=224):
    # img_rsz = Image.fromarray((img*255).astype(np.uint8))
    # img_rsz = img_rsz.resize((size, size), Image.LANCZOS)
    # img_rsz = np.array(img_rsz).astype(np.float32) / 255
    img_bgr = cv2.COLOR_RGB2BGR(img)
    img_rsz = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    img_rsz = cv2.COLOR_BGR2RGB(img_rsz)

    return img_rsz
