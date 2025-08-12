# Archive File Structure

This document describes the organization and purpose of the files in this archive.

## Usage

Compress this folder in root, not to compress the whole folder file. Upload to [Codabench](https://www.codabench.org/competitions/) -> Benchmarks > Management -> Upload. 

## scoring_program
**Files:**
- `metrics.py`
- `requirements.txt`
- `metadata`
- `score.py`

Example of running `score.py` locally:
```bash
python score.py \
  --solution_dir=/path/to/your/reference_data \
  --prediction_dir=/path/to/your/input_data \
  --score_dir=/path/to/save/your/score \
  --skip_ingestion
```
### Important Notes

- The current implementation of `score.py` is **hard-coded** to read from the submitted `input_data` directory and the local `reference_data` directory using a predefined folder structure.  
  This structure may change before the final version of the challenge is released.
- The script will terminate if the participant’s submission (`input_data`) contains:
  - Incorrectly named files  
  - Missing required files  
  - Additional files that are not part of the expected submission format
- To test locally, you must first update the challenge start date in the `competition.yaml` file of each phase to a date earlier than your current time.  Otherwise, submissions will be blocked until the predefined start date is reached.

---

## Components

- **Root (`/`)**
  - `competition.yaml`: Metadata and configuration for the Codabench competition.
  - `overview.html`, `terms.html`, `data.html`, `submission.html`, `evaluation.html`: HTML content pages shown on the competition website.
  - `logo.png`: Competition logo image.

- **`ingestion_program/`**
  - Contains the ingestion logic (`ingestion.py`) that processes participants' submissions.
  - `metadata`: Metadata describing the ingestion program.

- **`scoring_program/`**
  - `metrics.py`: Defines metrics (including `FAD_CLAP`) used to evaluate submissions.
  - `score.py`: Script to get evaluation metrics.
  - `requirements.txt`: Python dependencies for the scoring program.
  - `metadata`: Metadata describing the scoring program.

---
