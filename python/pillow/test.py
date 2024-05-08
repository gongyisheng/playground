# Pillow decompression bomb protection
from PIL import Image
from io import BytesIO

Image.MAX_IMAGE_PIXELS = 466550000

import warnings
warnings.simplefilter('error', Image.DecompressionBombWarning)

with open('test.jpg', 'rb') as f:
    im = Image.open(BytesIO(f.read()))


# To protect against potential DOS attacks caused by “decompression bombs” (i.e. malicious files which decompress into a huge amount of data and are designed to crash or cause disruption by using up a lot of memory), Pillow will issue a DecompressionBombWarning if the number of pixels in an image is over a certain limit, MAX_IMAGE_PIXELS.
# This threshold can be changed by setting MAX_IMAGE_PIXELS. It can be disabled by setting Image.MAX_IMAGE_PIXELS = None.
# If desired, the warning can be turned into an error with warnings.simplefilter('error', Image.DecompressionBombWarning) or suppressed entirely with warnings.simplefilter('ignore', Image.DecompressionBombWarning). See also the logging documentation to have warnings output to the logging facility instead of stderr.
# If the number of pixels is greater than twice MAX_IMAGE_PIXELS, then a DecompressionBombError will be raised instead.
# ref: https://pillow.readthedocs.io/en/latest/reference/Image.html#PIL.Image.open