# Handwritten Bill Evaluator

Python 3.11+ project scaffold for evaluating multimodal LLMs on handwritten Indian bills.

## Folder purpose

- `dataset/` stores the sample set and labels used for scoring.
- `dataset/images/` stores original bill images.
- `dataset/redacted/` stores privacy-safe bill copies.
- `dataset/ground_truth.json` stores the reference annotations for evaluation.
- `models/` is where model wrappers for Gemini, Claude, and future providers belong.
- `evaluation/` is where accuracy metrics, comparison logic, and report generation live.
- `zoho/` is where Zoho Books integration code belongs.
- `outputs/` stores generated reports, predictions, and exported artifacts.
- `ui/` is the optional Streamlit interface folder.

## Structure

```text
handwritten-bill-evaluator/
├── dataset/
│   ├── images/
│   ├── redacted/
│   └── ground_truth.json
├── evaluation/
├── models/
├── outputs/
├── ui/
├── zoho/
├── .env.example
├── README.md
├── main.py
└── requirements.txt
```

## Usage

1. Place bill images in `dataset/images/`.
2. Place redacted copies in `dataset/redacted/`.
3. Add evaluation labels to `dataset/ground_truth.json`.
4. Copy `.env.example` to `.env` and fill in your provider keys.
5. Install dependencies with `pip install -r requirements.txt`.
6. Run `python main.py` when you are ready to execute the project entry point.

## Notes

- The structure is intentionally modular so you can add new models without changing the evaluation or Zoho layers.
- The optional UI folder is separate so the core pipeline stays usable without Streamlit.
