# Radiology Prior Relevance API

Predicts whether prior radiology examinations should be shown to a radiologist reading a current exam.

## Quick Start

```bash
pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uvicorn src.app:app --host 0.0.0.0 --port 8000
```

## Test It

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "challenge_id": "relevant-priors-v1",
    "schema_version": 1,
    "cases": [{
      "case_id": "1001016",
      "patient_id": "606707",
      "current_study": {
        "study_id": "3100042",
        "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
        "study_date": "2026-03-08"
      },
      "prior_studies": [
        {
          "study_id": "2453245",
          "study_description": "MRI BRAIN STROKE LIMITED WITHOUT CONTRAST",
          "study_date": "2020-03-08"
        },
        {
          "study_id": "992654",
          "study_description": "CT HEAD WITHOUT CNTRST",
          "study_date": "2021-03-08"
        }
      ]
    }]
  }'
```

## Local Evaluation

```bash
# Download public_eval.json from the challenge page, then:
python eval.py --data public_eval.json --config config/<config file with tunable params>


## Run Tests

```bash
pip install pytest
pytest tests/ -v
```

## Docker

```bash
docker build -t radiology-priors .
docker run -p 8000:8000 radiology-priors
```

## Architecture



See `WRITEUP.md` for full experiment details and next steps.
