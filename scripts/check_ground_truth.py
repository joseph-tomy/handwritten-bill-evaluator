from pathlib import Path
import json

def main():
    root = Path(__file__).resolve().parent.parent
    gt_path = root / "dataset" / "ground_truth.json"
    imgs_dir = root / "dataset" / "images"
    payload = json.load(gt_path.open("r", encoding="utf-8"))
    # debug: show payload type and a truncated repr to help diagnose parsing issues
    print(f"DEBUG payload type: {type(payload)}")
    try:
        preview = repr(payload)[:1000]
    except Exception:
        preview = "<unrepresentable>"
    print("DEBUG payload preview:\n", preview)
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if samples is None:
        # handle dict mapping
        indexed = {k: v for k, v in payload.items()}
        gt_names = [k for k in indexed.keys()]
    else:
        gt_names = [str(s.get("image_name") or "").strip() for s in samples]

    print("Ground-truth entries:")
    for name in gt_names:
        print(repr(name))

    imgs = [p.name for p in sorted(imgs_dir.iterdir()) if p.is_file()]
    print("\nImage files:")
    for n in imgs:
        print(n)

    print("\nMatches:)")
    missing = []
    for img in imgs:
        normalized = img.lower()
        found = any((n or "").strip().lower() == normalized for n in gt_names)
        print(f"{img} -> {'FOUND' if found else 'MISSING'}")
        if not found:
            missing.append(img)

    if missing:
        print("\nMissing ground truth for:")
        for m in missing:
            print(m)
    else:
        print("\nAll images have ground-truth entries.")

if __name__ == '__main__':
    main()
