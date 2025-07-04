import cv2
import numpy as np
from pathlib import Path
import sys
from typing import Tuple, Union


def draw_bbox_only(
    img: np.ndarray,
    *,
    shirt_bbox_px: Tuple[int, int, int, int] | None = None,
    shirt_bbox_rel: Tuple[float, float, float, float] | None = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 4,
) -> np.ndarray:
    """
    Return a copy of *img* with a green rectangle drawn around the shirt bbox.

    Parameters
    ----------
    img : BGR image (np.ndarray)
    shirt_bbox_px  : (x0, y0, x1, y1) in pixels          — use either…
    shirt_bbox_rel : (x0, y0, x1, y1) in relative [0-1]  — …but not both
    color          : rectangle BGR colour
    thickness      : rectangle line thickness (px)
    """
    if (shirt_bbox_px is None) == (shirt_bbox_rel is None):
        raise ValueError("Specify exactly one of shirt_bbox_px or shirt_bbox_rel")

    h, w = img.shape[:2]

    # convert relative → absolute pixels
    if shirt_bbox_px is None:
        x0_rel, y0_rel, x1_rel, y1_rel = shirt_bbox_rel
        shirt_bbox_px = (
            int(x0_rel * w),
            int(y0_rel * h),
            int(x1_rel * w),
            int(y1_rel * h),
        )

    x0, y0, x1, y1 = shirt_bbox_px
    img_out = img.copy()
    cv2.rectangle(img_out, (x0, y0), (x1, y1), color, thickness)
    return img_out


# ------------------------------------------------------------------
# demo  (run `python draw_bbox_only.py sample.jpg`)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Get the directory where *this script* is located.
    # This is the key to robustness.
    script_directory = Path(__file__).parent

    # Define the default image filename, assumed to be in the same directory as the script
    default_image_filename = "test.jpg"

    # Construct the full path to the default image, relative to the script's location
    default_image_path = script_directory / default_image_filename

    # Determine the final path to use:
    # 1. If an argument is provided, use that path (assuming it's either absolute
    #    or relative to the CWD, but it's safer to provide absolute paths via args)
    # 2. Otherwise, use the calculated default_image_path (relative to the script)
    if len(sys.argv) > 1:
        # If the argument is a relative path, it will be relative to the CWD.
        # If it's an absolute path, Path() handles it.
        # For maximum robustness, you might want to resolve sys.argv[1] against script_directory
        # if you expect command-line args to be relative to the script.
        # For now, let's assume sys.argv[1] is either absolute or relative to CWD.
        path = Path(sys.argv[1])
    else:
        path = default_image_path

    # --- Crucial Debugging Prints ---
    import os
    print(f"Current working directory (CWD): {os.getcwd()}")
    print(f"Directory where this script is located: {script_directory}")
    print(f"Attempting to load image from absolute path: {path.resolve()}")
    # ----------------------------------

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Could not find or open the image at: {path.resolve()}. Please check the path, filename, and file integrity.")

    # Use relative coordinates
    vis = draw_bbox_only(bgr, shirt_bbox_rel=(0, 0.5, 1, 1))

    # Or use pixel coordinates
    # vis = draw_bbox_only(bgr, shirt_bbox_px=(120, 350, 880, 1260))

    cv2.imshow("bbox preview", vis)
    cv2.imwrite("bbox_preview.png", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()