# Flight Delay Prediction — Backend Service

An end-to-end ML pipeline that predicts whether a flight will arrive **≥ 15 minutes late** (`ArrDel15`), built on the [BTS Carrier On-Time Performance dataset](https://www.kaggle.com/datasets/mexwell/carrier-on-time-performance-dataset).

---

## Architecture at a Glance

```
archive.zip ─► ingestion ─► validation ─► transformation ─► training
                                                               │
                                          DagsHub/MLflow  ◄────┘
                                               │
                                          best_model.joblib
                                               │
                                          Flask API  ──► Railway
```

## Quick Start

### 1. Clone & install

```bash
git clone <your-backend-repo-url>
cd Python
pip install -r requirements.txt
pip install -e .
```

### 2. Set environment variables

Create a `.env` file in the project root (it is gitignored):

```env
DAGSHUB_USERNAME=your_dagshub_username
DAGSHUB_TOKEN=your_dagshub_token
```

### 3. Prepare the data

Download the dataset from Kaggle and place `archive.zip` in `D:\Project\data\`.

### 4. Run the pipeline

```bash
python -m src.pipeline.run_pipeline
```

This will:
- unzip and load the CSV,
- validate & clean the schema,
- engineer features,
- train + evaluate models,
- log everything to DagsHub MLflow.

### 5. Start the API server

```bash
python app.py
```

The API is now live at `http://localhost:5000`.

| Endpoint       | Method | Description                        |
|----------------|--------|------------------------------------|
| `/health`      | GET    | Returns `{"status": "healthy"}`    |
| `/predict`     | POST   | Accepts flight features, returns delay prediction |

### 6. Deploy to Railway

The repo ships with a `Procfile`:

```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

Push to GitHub, link the repo in Railway, add your `DAGSHUB_USERNAME` and `DAGSHUB_TOKEN` env vars, and deploy.

---

## Project Structure

```
Python/
├── app.py                  # Flask entry point
├── Procfile                # Railway process definition
├── requirements.txt        # pinned dependencies
├── setup.py                # editable install support
├── config/
│   └── config.yaml         # all tunables in one place
├── src/
│   ├── data/               # ingestion, validation, transformation
│   ├── model/              # trainer, evaluator
│   ├── experiment/         # DagsHub + MLflow helpers
│   ├── pipeline/           # end-to-end orchestrator
│   ├── api/                # Flask routes & request schemas
│   └── utils/              # logging, yaml loader, path helpers
├── models/                 # (gitignored) saved model artefacts
└── logs/                   # (gitignored) runtime logs
```

---

## Tech Stack

| Layer               | Tool                   |
|---------------------|------------------------|
| Language            | Python 3.10+           |
| Web framework       | Flask + Gunicorn       |
| ML                  | scikit-learn           |
| Experiment tracking | MLflow via DagsHub     |
| Deployment          | Railway                |

---

## License

This project is for educational purposes. The dataset is provided by the U.S. Bureau of Transportation Statistics and hosted on Kaggle.
