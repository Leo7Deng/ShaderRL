import io

import cairosvg
from PIL import Image


def render_svg(svg_text: str, size: int = 512) -> Image.Image | None:
    start = svg_text.find("<svg")
    end = svg_text.rfind("</svg>")
    if start == -1 or end == -1:
        return None
    svg_text = svg_text[start : end + len("</svg>")]

    try:
        png_bytes = cairosvg.svg2png(
            bytestring=svg_text.encode("utf-8"),
            output_width=size,
            output_height=size,
        )
    except Exception:
        return None

    if png_bytes is None:
        return None

    return Image.open(io.BytesIO(png_bytes))