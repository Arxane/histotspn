from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class PatchRecord:
	patch: np.ndarray
	label: int
	source_id: str
	x: int
	y: int
	tissue_fraction: float


class QuPathPatchExtractor:
	def __init__(self, patch_size, stride, min_tissue_fraction, blur_kernel, morph_kernel, min_patch_size: Optional[int] = None):
		if patch_size <= 0:
			raise ValueError("patch_size must be positive")
		if stride is not None and stride <= 0:
			raise ValueError("stride must be positive when provided")
		if not 0.0 <= min_tissue_fraction <= 1.0:
			raise ValueError("min_tissue_fraction must be between 0 and 1")

		self.patch_size = patch_size
		self.stride = stride or patch_size
		self.min_tissue_fraction = min_tissue_fraction
		self.blur_kernel = self._ensure_odd(blur_kernel)
		self.morph_kernel = self._ensure_odd(morph_kernel)
		self.min_patch_size = min_patch_size or patch_size

	@staticmethod
	def _ensure_odd(value):
		if value <= 0:
			raise ValueError("kernel sizes must be positive")
		return value if value % 2 == 1 else value + 1

	@staticmethod
	def _normalize_image(image):
		if image.ndim == 2:
			return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
		if image.ndim == 3 and image.shape[-1] == 4:
			return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
		if image.ndim == 3 and image.shape[-1] == 3:
			return image
		raise ValueError(f"Expected 2D grayscale or 3-channel image, got shape {image.shape}")

	def tissue_mask(self, image):
		rgb_image = self._normalize_image(image)

		hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
		saturation = hsv[:, :, 1]   # 0–255
		value      = hsv[:, :, 2]   # 0–255; background ≈ 240+

		# Tissue = bright white excluded by brightness OR kept by having colour
		background = (value > 230) & (saturation < 15)
		mask = (~background).astype(np.uint8) * 255

		# Blur before morphology to smooth noisy boundaries
		blurred_mask = cv2.GaussianBlur(mask, (self.blur_kernel, self.blur_kernel), 0)
		_, mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

		# Remove isolated speckle (OPEN) then fill holes inside tissue (CLOSE)
		kernel = np.ones((self.morph_kernel, self.morph_kernel), dtype=np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		return mask.astype(bool)

	def _iter_tiles(self, image):
		height, width = image.shape[:2]

		if height < self.min_patch_size or width < self.min_patch_size:
			yield 0, 0, image
			return

		for y in range(0, height - self.patch_size + 1, self.stride):
			for x in range(0, width - self.patch_size + 1, self.stride):
				patch = image[y : y + self.patch_size, x : x + self.patch_size]
				yield x, y, patch

	def extract_patches_from_image(self, image, label, source_id):
		rgb_image = self._normalize_image(image)
		mask = self.tissue_mask(rgb_image)
		patches: List[PatchRecord] = []

		for x, y, patch in self._iter_tiles(rgb_image):
			if patch.size == 0:
				continue

			patch_mask = mask[y : y + patch.shape[0], x : x + patch.shape[1]]
			tissue_fraction = float(patch_mask.mean()) if patch_mask.size else 0.0

			if tissue_fraction >= self.min_tissue_fraction:
				patches.append(
					PatchRecord(
						patch=patch.copy(),
						label=int(label),
						source_id=str(source_id),
						x=int(x),
						y=int(y),
						tissue_fraction=tissue_fraction,
					)
				)

		if not patches:
			patches.append(
				PatchRecord(
					patch=rgb_image.copy(),
					label=int(label),
					source_id=str(source_id),
					x=0,
					y=0,
					tissue_fraction=float(mask.mean()),
				)
			)

		return patches

	def extract_from_dataset(self,data: Union[Tuple[Sequence[np.ndarray], Sequence[int]],Tuple[Sequence[np.ndarray], Sequence[int], Sequence[str]],Dict[str, object],], source_ids):
		images, labels, image_ids = self._coerce_dataset_input(data, source_ids)

		all_patches: List[PatchRecord] = []
		# Tissue-aware patch extraction: Otsu threshold → morphological open/close → tile grid
		for index, (image, label, source_id) in enumerate(tqdm(
			zip(images, labels, image_ids),
			total=len(images),
			desc="[Patches] Extracting tissue patches",
			unit="WSI",
		)):
			source_name = source_id if source_id is not None else f"wsi_{index}"
			all_patches.extend(self.extract_patches_from_image(image, int(label), source_name))

		return all_patches

	def extract_arrays(self,data: Union[Tuple[Sequence[np.ndarray], Sequence[int]],Tuple[Sequence[np.ndarray], Sequence[int], Sequence[str]],Dict[str, object],],source_ids):
		patches = self.extract_from_dataset(data, source_ids=source_ids)
		patch_arrays = np.stack([record.patch for record in patches], axis=0)
		labels = np.array([record.label for record in patches], dtype=np.int64)
		source_ids_out = np.array([record.source_id for record in patches], dtype=object)
		coordinates = np.array([[record.x, record.y] for record in patches], dtype=np.int64)
		tissue_fraction = np.array([record.tissue_fraction for record in patches], dtype=np.float32)

		return {
			"patches": patch_arrays,
			"labels": labels,
			"source_ids": source_ids_out,
			"coordinates": coordinates,
			"tissue_fraction": tissue_fraction,
		}

	def _coerce_dataset_input(self,data: Union[Tuple[Sequence[np.ndarray], Sequence[int]], Tuple[Sequence[np.ndarray], Sequence[int], Sequence[str]], Dict[str, object]], source_ids):
		if isinstance(data, dict):
			if "images" in data and "labels" in data:
				images = data["images"]
				labels = data["labels"]
				image_ids = data.get("image_ids")
				return self._coerce_sequences(images, labels, image_ids, source_ids)

			if len(data) == 1:
				first_value = next(iter(data.values()))
				if isinstance(first_value, tuple) and len(first_value) in {2, 3}:
					return self._coerce_dataset_input(first_value, source_ids)

			raise ValueError(
				"Unsupported dictionary format. Expected keys like 'images' and 'labels', "
				"or a modality mapping such as {'HE': (images, labels)}."
			)

		if isinstance(data, tuple) and len(data) in {2, 3}:
			images = data[0]
			labels = data[1]
			image_ids = data[2] if len(data) == 3 else None
			return self._coerce_sequences(images, labels, image_ids, source_ids)

		raise ValueError(
			"Unsupported data format. Expected (images, labels), (images, labels, image_ids), or a compatible dict."
		)

	@staticmethod
	def _coerce_sequences(images: Sequence[np.ndarray], labels: Sequence[int], image_ids: Optional[Sequence[str]], source_ids: Optional[Sequence[str]]) -> Tuple[Sequence[np.ndarray], Sequence[int], List[Optional[str]]]:
		if len(images) != len(labels):
			raise ValueError("images and labels must have the same length")

		if source_ids is not None:
			if len(source_ids) != len(images):
				raise ValueError("source_ids must match the number of images")
			return images, labels, list(source_ids)

		if image_ids is not None:
			if len(image_ids) != len(images):
				raise ValueError("image_ids must match the number of images")
			return images, labels, list(image_ids)

		return images, labels, [None] * len(images)


def get_patches_from_get_data_output(data: Union[Tuple[Sequence[np.ndarray], Sequence[int]], Tuple[Sequence[np.ndarray], Sequence[int], Sequence[str]], Dict[str, object] ], patch_size, stride, min_tissue_fraction, blur_kernel: int = 5, morph_kernel: int = 5) -> Dict[str, np.ndarray]:
	extractor = QuPathPatchExtractor(
		patch_size=patch_size,
		stride=stride,
		min_tissue_fraction=min_tissue_fraction,
		blur_kernel=blur_kernel,
		morph_kernel=morph_kernel,
	)
	return extractor.extract_arrays(data)