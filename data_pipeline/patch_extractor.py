"""
Patch extraction and spatial tiling engine for high-resolution satellite scenes.
Splits large multi-band rasters into uniform patches (e.g., 256x256) and reconstructs full mosaics.
"""

from typing import List, Tuple, Optional
import numpy as np


class PatchExtractor:
    """
    Extracts fixed-dimension patches (e.g., 256x256) from 2D and 3D geospatial arrays,
    and supports seamless re-assembly.
    """

    def __init__(self, patch_size: int = 256, stride: Optional[int] = None):
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size

    def extract_patches(
        self,
        array: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], Tuple[int, int]]:
        """
        Extracts patches from an array of shape (C, H, W) or (H, W).

        Returns:
            patches: np.ndarray of shape (N, C, patch_size, patch_size) or (N, patch_size, patch_size)
            coords: List of (top, bottom, left, right) bounding boxes
            original_shape: (H, W)
        """
        is_3d = (array.ndim == 3)
        if is_3d:
            c, h, w = array.shape
        else:
            h, w = array.shape
            c = 1

        # Pad array if dimensions are smaller than patch_size or not multiple
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)

        if pad_h > 0 or pad_w > 0:
            if is_3d:
                pad_width = ((0, 0), (0, pad_h), (0, pad_w))
            else:
                pad_width = ((0, pad_h), (0, pad_w))
            padded_array = np.pad(array, pad_width, mode="reflect")
        else:
            padded_array = array

        if is_3d:
            _, cur_h, cur_w = padded_array.shape
        else:
            cur_h, cur_w = padded_array.shape

        patches_list = []
        coords_list = []

        for y in range(0, cur_h - self.patch_size + 1, self.stride):
            for x in range(0, cur_w - self.patch_size + 1, self.stride):
                top, bottom = y, y + self.patch_size
                left, right = x, x + self.patch_size

                if is_3d:
                    patch = padded_array[:, top:bottom, left:right]
                else:
                    patch = padded_array[top:bottom, left:right]

                patches_list.append(patch)
                coords_list.append((top, bottom, left, right))

        patches_arr = np.array(patches_list)
        return patches_arr, coords_list, (h, w)

    def reconstruct_from_patches(
        self,
        patches: np.ndarray,
        coords: List[Tuple[int, int, int, int]],
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Reassembles extracted patches into the original array shape.
        """
        orig_h, orig_w = original_shape
        is_3d = (patches.ndim == 4)

        if is_3d:
            num_patches, c, ph, pw = patches.shape
            canvas = np.zeros((c, orig_h, orig_w), dtype=patches.dtype)
            count_matrix = np.zeros((orig_h, orig_w), dtype=np.float32)

            for i, (top, bottom, left, right) in enumerate(coords):
                # Clip bounds to original shape
                clip_b = min(bottom, orig_h)
                clip_r = min(right, orig_w)
                h_len = clip_b - top
                w_len = clip_r - left

                if h_len > 0 and w_len > 0:
                    canvas[:, top:clip_b, left:clip_r] += patches[i, :, :h_len, :w_len]
                    count_matrix[top:clip_b, left:clip_r] += 1.0

            # Normalize overlapping regions
            count_matrix[count_matrix == 0] = 1.0
            return canvas / count_matrix[np.newaxis, :, :]

        else:
            num_patches, ph, pw = patches.shape
            canvas = np.zeros((orig_h, orig_w), dtype=np.float32)
            count_matrix = np.zeros((orig_h, orig_w), dtype=np.float32)

            for i, (top, bottom, left, right) in enumerate(coords):
                clip_b = min(bottom, orig_h)
                clip_r = min(right, orig_w)
                h_len = clip_b - top
                w_len = clip_r - left

                if h_len > 0 and w_len > 0:
                    canvas[top:clip_b, left:clip_r] += patches[i, :h_len, :w_len]
                    count_matrix[top:clip_b, left:clip_r] += 1.0

            count_matrix[count_matrix == 0] = 1.0
            return canvas / count_matrix
