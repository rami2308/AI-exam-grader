# AI Exam Grader

A prototype that grades a student's exam answer from images. Given the exam question, the professor's correct answer and the student's answer, it produces a grade (0-100) and a one-sentence explanation.

Final-year project (team of two), SCE Ashdod.

## How it works

For each exam, three images are used: the question, the professor's answer and the student's answer.

1. **Weighted rubric from the professor's answer.** A pre-trained **Qwen2.5-VL-7B-Instruct** (4-bit quantized) is prompted to split the correct answer into weighted components (core concepts 30-50 points, supporting points 15-25, minor details 5-10, weights summing to 100).
2. **Partial credit.** The model scores the student against each component (100%, 50-75%, 25%, 0%), deducts points for incorrect statements, and returns a **coverage score** (0-100) plus a one-sentence summary. Decoding is deterministic (`do_sample=False`).
3. **Calibration.** **GraderNet**, a small PyTorch network trained in this repo, maps the coverage score to the final grade, so the output follows the professor's grading scale.

```
question + professor's answer + student's answer (3 images)
              |
Qwen2.5-VL-7B-Instruct (4-bit, inference only)
   weighted components -> partial credit -> deductions
              |
        coverage score (0-100) + summary
              |
GraderNet (MLP 1 -> 16 -> 8 -> 1, trained here)
              |
        final grade (0-100)
```

The vision-language model is used as is. **It is not fine-tuned**; the only model trained in this project is GraderNet.

## How it was developed

The approach changed several times, and each change came from looking at where grades disagreed with the professor's:

- Splitting the grade into percentages for clarity, accuracy and logic failed, because the model's logic rarely matched the professor's.
- Comparing the two answers directly was unreliable.
- Extracting key points from the professor's answer worked better, and giving each point a **weight** (some points matter more than others) worked best.

## Results

Average grading error (MAE, points on a 0-100 scale, lower is better):

| Version | Change | Avg. error |
|---|---|---|
| V1 | Binary prompt (correct / incorrect) | 20.4 |
| V2 | Weighted partial credit per component (79 samples) | 13.5 |
| V3 | 150 samples across many subjects, noise filter | 11.0 |

GraderNet: train MAE 12.6, test MAE 11.0 (103 samples were left after filtering, split 80/20 into train and test).

### Limitations (please read)
- **All data is synthetic**: exam images were generated with code (rendered text), not scanned from real students. The model can read handwriting in general, but this project has **not** been evaluated on real handwritten exams.
- The test set is small (about 20 samples), so the numbers are indicative, not a benchmark.
- Outlier filtering (47 of 150 samples dropped where the model's score was far from the real grade) is applied **before** the train/test split, so the test set is filtered too. Results on unfiltered data would likely be worse.
- GraderNet currently takes a single input, the coverage score.
- Results come from a single run.

## Project structure

| File | Purpose |
|---|---|
| `generate_synthetic_data.py`, `generate_more_data.py` | Generate the synthetic exam images and grades (the first set and the extended set) |
| `extract_features.py` | Runs Qwen2.5-VL on every exam, saves the coverage score and summary to `training_features.csv` |
| `train_model.py` | Filters outliers, trains GraderNet, saves `grader_model.pth`, `scaler.pkl` and `loss_curve.png` |
| `demo.py` | Pick an exam from a menu and see the predicted grade |
| `data/questions`, `data/solutions`, `data/answers` | The three images per exam (same file name in each folder) |
| `train.csv` | File name and real grade for each exam |

## Run it

Requires Python and a CUDA GPU (the 7B model runs in 4-bit). The first run downloads the model from Hugging Face.

```
pip install -r requirements.txt
python extract_features.py    # score every exam with Qwen2.5-VL
python train_model.py         # train GraderNet
python demo.py                # grade an exam interactively
```

To grade your own exam, put three images with the same file name in `data/questions/`, `data/solutions/` and `data/answers/`, then run `demo.py`.

## Background

The idea was inspired by VEHME (Vision-Language Model for Evaluating Handwritten Mathematics Expressions, EMNLP 2025). An earlier version of this repository included that project's training code; it was never used by this pipeline and has been removed.

## Possible next steps
Evaluate on real handwritten exams, a web interface for upload and grading, and calibration to an individual professor's grading style.

**Author:** Rami Tal · [LinkedIn](https://linkedin.com/in/rami-tal) · [GitHub](https://github.com/rami2308)
