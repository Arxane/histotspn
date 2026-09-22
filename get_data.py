import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, List, Dict, Set, Union

VALIDATION_FRACTION = 0.10
VALIDATION_SEED = 42


def get_validation_image_ids(data_root: Union[str, Path], fraction: float = VALIDATION_FRACTION, seed: int = VALIDATION_SEED) -> Set[str]:
    """Return deterministic, stratified image IDs held out from the train split."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"Validation fraction must be between 0 and 1, got {fraction}")

    root = Path(data_root)
    he_files = sorted((root / "HE" / "train").glob("*.png"))
    ihc_files = sorted((root / "IHC" / "train").glob("*.png"))
    he_by_id = {
        BaseModalityDataLoader.extract_image_id_from_filename(path.name): BaseModalityDataLoader.extract_label_from_filename(path.name) for path in he_files
    }
    ihc_by_id = {
        BaseModalityDataLoader.extract_image_id_from_filename(path.name): BaseModalityDataLoader.extract_label_from_filename(path.name) for path in ihc_files
    }

    paired_by_label: Dict[int, List[str]] = {label: [] for label in range(4)}
    for image_id in sorted(set(he_by_id) & set(ihc_by_id)):
        label = he_by_id[image_id]
        ihc_label = ihc_by_id[image_id]
        if label == ihc_label:
            paired_by_label[label].append(image_id)

    rng = np.random.default_rng(seed)
    validation_ids: Set[str] = set()
    for label, image_ids in paired_by_label.items():
        if not image_ids:
            continue
        shuffled = np.array(sorted(set(image_ids)), dtype=object)
        rng.shuffle(shuffled)
        count = int(round(len(shuffled) * fraction))
        if count == 0 and len(shuffled) >= 2:
            count = 1
        validation_ids.update(str(image_id) for image_id in shuffled[:count])

    return validation_ids


class BaseModalityDataLoader:
    def __init__(self, modality, data_root):
        if modality not in ["HE", "IHC"]:
            raise ValueError(f"Modality must be 'HE' or 'IHC', got {modality}")

        self.modality = modality
        self.data_root = Path(data_root)
        self.splits = ["train", "validation", "test"]

    @staticmethod
    def extract_label_from_filename(filename):
        parts = filename.rsplit("_", 1)
        if len(parts) < 2:
            raise ValueError(f"Cannot parse filename: {filename}")

        label_str = parts[-1].replace(".png", "").replace("+", "")
        try:
            label = int(label_str)
            if label not in [0, 1, 2, 3]:
                raise ValueError(f"Label {label} not in valid range [0, 1, 2, 3]")
            return label
        except ValueError as e:
            raise ValueError(f"Cannot extract valid label from {filename}: {e}")

    @staticmethod
    def extract_image_id_from_filename(filename):
        parts = filename.rsplit("_", 2)
        if len(parts) < 2:
            raise ValueError(f"Cannot parse filename: {filename}")
        return parts[0]

    def load_images_and_labels(self, split):
        if split not in self.splits:
            raise ValueError(f"Split must be one of {self.splits}")

        source_split = "train" if split == "validation" else split
        data_path = self.data_root / self.modality / source_split

        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")

        images = []
        labels = []
        image_ids = []

        image_files = sorted(data_path.glob("*.png"))
        validation_ids = get_validation_image_ids(self.data_root) if split in {"train", "validation"} else set()

        for img_file in image_files:
            try:
                img = Image.open(img_file)
                img_array = np.array(img)

                label = self.extract_label_from_filename(img_file.name)
                image_id = self.extract_image_id_from_filename(img_file.name)
                if split == "validation" and image_id not in validation_ids:
                    continue
                if split == "train" and image_id in validation_ids:
                    continue

                images.append(img_array)
                labels.append(label)
                image_ids.append(image_id)
            except Exception as e:
                print(f"[{self.modality}] Error loading {img_file.name}: {e}")
                continue

        return images, labels, image_ids

    def get_training_data(self):
        images, labels, _ = self.load_images_and_labels("train")
        return np.array(images), np.array(labels)

    def get_test_data(self):
        images, labels, _ = self.load_images_and_labels("test")
        return np.array(images), np.array(labels)

    def get_all_data(self):
        return {
            "train": self.get_training_data(),
            "validation": tuple(np.array(values) for values in self.load_images_and_labels("validation")[:2]),
            "test": self.get_test_data(),
        }

    def get_data_info(self):
        train_imgs, train_labels = self.get_training_data()
        validation_imgs, validation_labels, _ = self.load_images_and_labels("validation")
        test_imgs, test_labels = self.get_test_data()

        train_unique, train_counts = np.unique(train_labels, return_counts=True)
        validation_unique, validation_counts = np.unique(validation_labels, return_counts=True)
        test_unique, test_counts = np.unique(test_labels, return_counts=True)

        return {
            "modality": self.modality,
            "train": {
                "num_samples": len(train_labels),
                "image_shape": train_imgs[0].shape if len(train_imgs) > 0 else None,
                "label_distribution": dict(zip(train_unique, train_counts)),
            },
            "validation": {
                "num_samples": len(validation_labels),
                "image_shape": (validation_imgs[0].shape if len(validation_imgs) > 0 else None),
                "label_distribution": dict(zip(validation_unique, validation_counts)),
            },
            "test": {
                "num_samples": len(test_labels),
                "image_shape": test_imgs[0].shape if len(test_imgs) > 0 else None,
                "label_distribution": dict(zip(test_unique, test_counts)),
            },
        }


class HEDataLoader(BaseModalityDataLoader):

    def __init__(self, data_root: str = "BCI_dataset"):
        super().__init__(modality="HE", data_root=data_root)


class IHCDataLoader(BaseModalityDataLoader):

    def __init__(self, data_root: str = "BCI_dataset"):
        super().__init__(modality="IHC", data_root=data_root)


class HistoTSPNDataLoader:
    def __init__(self, data_root: str = "BCI_dataset"):
        self.data_root = Path(data_root)
        self.he_loader = HEDataLoader(data_root)
        self.ihc_loader = IHCDataLoader(data_root)

    def get_he_data(self, split: str = "train"):
        if split == "train":
            return self.he_loader.get_training_data()
        elif split == "validation":
            images, labels, _ = self.he_loader.load_images_and_labels("validation")
            return np.array(images), np.array(labels)
        elif split == "test":
            return self.he_loader.get_test_data()
        else:
            raise ValueError(f"Split must be 'train', 'validation', or 'test', got {split}")

    def get_ihc_data(self, split: str = "train"):
        if split == "train":
            return self.ihc_loader.get_training_data()
        elif split == "validation":
            images, labels, _ = self.ihc_loader.load_images_and_labels("validation")
            return np.array(images), np.array(labels)
        elif split == "test":
            return self.ihc_loader.get_test_data()
        else:
            raise ValueError(f"Split must be 'train', 'validation', or 'test', got {split}")

    def get_modality_data(self, modality: str, split: str = "train"):
        if modality == "HE":
            return self.get_he_data(split)
        elif modality == "IHC":
            return self.get_ihc_data(split)
        else:
            raise ValueError(f"Modality must be 'HE' or 'IHC', got {modality}")

    def get_all_data(self, modality: str):
        if modality == "HE":
            return self.he_loader.get_all_data()
        elif modality == "IHC":
            return self.ihc_loader.get_all_data()
        else:
            raise ValueError(f"Modality must be 'HE' or 'IHC', got {modality}")

    def get_multimodal_data(self, split: str = "train"):
        return {"HE": self.get_he_data(split), "IHC": self.get_ihc_data(split)}

    def get_paired_data(self, split: str = "train"):
        he_imgs, he_labels, he_ids = self.he_loader.load_images_and_labels(split)
        ihc_imgs, ihc_labels, ihc_ids = self.ihc_loader.load_images_and_labels(split)

        he_dict = {img_id: (img, label) for img, label, img_id in zip(he_imgs, he_labels, he_ids)}
        ihc_dict = {img_id: (img, label) for img, label, img_id in zip(ihc_imgs, ihc_labels, ihc_ids)}

        he_ids_set = set(he_ids)
        ihc_ids_set = set(ihc_ids)
        common_ids = sorted(he_ids_set & ihc_ids_set)

        he_only = he_ids_set - ihc_ids_set
        ihc_only = ihc_ids_set - he_ids_set

        if he_only or ihc_only:
            print(f"\n[Warning] {split.upper()} split has mismatched images:")
            if he_only:
                print(f"  HE only (not in IHC): {len(he_only)} images")
            if ihc_only:
                print(f"  IHC only (not in HE): {len(ihc_only)} images")
            print(f"  Using {len(common_ids)} common images for paired training")

        paired_he = []
        paired_ihc = []
        paired_labels = []

        for img_id in common_ids:
            he_img, he_label = he_dict[img_id]
            ihc_img, ihc_label = ihc_dict[img_id]

            if he_label != ihc_label:
                print(f"[Warning] Image {img_id} has mismatched labels: HE={he_label}, IHC={ihc_label}")
                continue

            paired_he.append(he_img)
            paired_ihc.append(ihc_img)
            paired_labels.append(he_label)

        return {"HE": np.array(paired_he), "IHC": np.array(paired_ihc), "labels": np.array(paired_labels)}

    def get_data_info(self, modality: str = "HE"):
        if modality == "HE":
            return self.he_loader.get_data_info()
        elif modality == "IHC":
            return self.ihc_loader.get_data_info()
        else:
            raise ValueError(f"Modality must be 'HE' or 'IHC', got {modality}")

    def get_paired_data_info(self, split: str = "train"):
        paired = self.get_paired_data(split)
        labels = paired["labels"]
        unique_labels, label_counts = np.unique(labels, return_counts=True)

        return {
            "split": split,
            "num_paired_samples": len(labels),
            "he_shape": paired["HE"][0].shape if len(paired["HE"]) > 0 else None,
            "ihc_shape": paired["IHC"][0].shape if len(paired["IHC"]) > 0 else None,
            "label_distribution": dict(zip(unique_labels, label_counts)),
        }
