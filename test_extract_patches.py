from pathlib import Path
import json

import numpy as np
from PIL import Image

from get_patches import QuPathPatchExtractor


def main() -> None:
    image_path = sorted(Path("BCI_dataset/HE/train").glob("*.png"))[0]
    label = int(image_path.stem.rsplit("_", 1)[-1].replace("+", ""))
    image = np.array(Image.open(image_path))

    extractor = QuPathPatchExtractor(patch_size=256, stride=256, min_tissue_fraction=0.5)
    records = extractor.extract_patches_from_image(image, label, image_path.stem)

    out_dir = Path("patch_test_output") / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, record in enumerate(records):
        patch_name = (
            f"{image_path.stem}_label{record.label}_x{record.x}_y{record.y}"
            f"_t{record.tissue_fraction:.3f}_{index}.png"
        )
        Image.fromarray(record.patch).save(out_dir / patch_name)

    summary = {
        "source_image": image_path.name,
        "source_shape": list(image.shape),
        "label": label,
        "patch_count": len(records),
        "output_dir": str(out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
