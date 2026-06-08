# Among Us MLOps Predictor

A fun Flask + scikit-learn project that predicts survival chance and estimated sabotages fixed for an Among Us style match. The app includes a modern space-themed frontend, a machine learning backend, and reusable training code for rebuilding the models.

## What this project does

- Takes match inputs from the UI or API.
- Sends them to a Flask prediction service.
- Uses trained scikit-learn pipelines to produce:
  - survival percentage
  - predicted sabotages fixed
  - confidence score and confidence band
- Shows the result in a premium glassmorphism-style frontend.
- Displays animated character images from `assets/characters/` as a flying background.
- Uses a custom logo from `assets/among-us-logo.jpg` at the top of the page.

## Project structure

```text
among-us-mlops/
  app.py                # Flask API for prediction and model info
  index.html            # Frontend UI and API client
  train.py              # Model training and serialization script
  requirements.txt      # Python dependencies
  assets/               # Logo and animated background images
  data/                 # Dataset files used for experimentation/training
  models/               # Saved trained model artifacts
```

## Step-by-step implementation

### 1. Prepare the training data

The training script expects a CSV dataset such as `User1.csv` with fields similar to the following:

- `Team`
- `Task Completed`
- `Imposter Kills`
- `Game Length`
- `Murdered`
- `Ejected`
- `Sabotages Fixed`

The script cleans and converts the raw values into model-ready features.

### 2. Engineer features in `train.py`

The training script creates the features used by the models:

- `Team`
- `Task Completed`
- `Imposter Kills`
- `Game Length Sec`

It also derives targets:

- `Survived` from `Murdered` and `Ejected`
- `Sabotages Fixed` as a numeric regression target

This keeps the model input simple while still making the app feel like a game-based predictor.

### 3. Build preprocessing pipelines

`train.py` uses a `ColumnTransformer` so the model can handle different feature types:

- numeric columns are standardized with `StandardScaler`
- `Team` is one-hot encoded with `OneHotEncoder`

This is wrapped inside two full pipelines:

- `survive_model.pkl` for survival classification
- `sabotage_model.pkl` for sabotage regression

### 4. Train and save the models

Run the training script to fit the models and save them into the `models/` folder:

```bash
python train.py
```

After this finishes, the following artifacts are created:

- `models/survive_model.pkl`
- `models/sabotage_model.pkl`

### 5. Start the Flask API and frontend

`app.py` loads the trained models into memory and hosts both the API and the frontend UI.

Run it with:

```bash
python app.py
```

Then open the app in your browser at:

```text
http://127.0.0.1:5000
```

### 6. Use the frontend

The UI is now served directly from the Flask app, so you can use the same URL above without opening `index.html` manually.

If you still prefer a static preview, you can also open `index.html` directly, but the recommended workflow is to use the server-hosted version so API calls work consistently.

## Docker support

A Docker container is included so the app can run with the same environment everywhere.

Build the container with:

```bash
docker build -t among-us-mlops .
```

Run it with:

```bash
docker run --rm -p 5000:5000 among-us-mlops
```

Then visit:

```text
http://127.0.0.1:5000
```

## Tests

A lightweight test suite is included for API sanity checks. Install the development requirements and run:

```bash
pip install -r requirements-dev.txt
pytest
```

The frontend:

- uses a deep space visual theme
- shows flying character images from `assets/characters/`
- includes a logo from `assets/among-us-logo.jpg`
- calls the Flask prediction API automatically
- supports live what-if predictions while you edit the inputs

### 7. Test the API directly

Send a `POST` request to:

```text
http://localhost:5000/predict
```

Sample JSON body:

```json
{
  "Team": "Crewmate",
  "Task Completed": 5,
  "Imposter Kills": 0,
  "Game Length Sec": 600
}
```

The API returns:

- `survival_percentage`
- `predicted_sabotages_fixed`
- `confidence_score`
- `confidence_band`
- `model_version`

### 8. Check model metadata

The API also includes a model metadata endpoint:

```text
GET /model-info
```

This returns information such as:

- model version
- trained timestamp
- expected features
- artifact paths

## API reference

### `POST /predict`

Request body:

- `Team`: `Crewmate` or `Imposter`
- `Task Completed`: non-negative number
- `Imposter Kills`: non-negative number
- `Game Length Sec`: non-negative number

Response example:

```json
{
  "status": "success",
  "model_version": "1.0.0-beta",
  "survival_percentage": 72.41,
  "predicted_sabotages_fixed": 3.14,
  "confidence_score": 44.82,
  "confidence_band": "Medium"
}
```

### `GET /model-info`

Response example:

```json
{
  "status": "success",
  "model_version": "1.0.0-beta",
  "trained_at_utc": "2026-06-07T12:00:00+00:00",
  "features": [
    "Team",
    "Task Completed",
    "Imposter Kills",
    "Game Length Sec"
  ],
  "targets": [
    "survival_percentage",
    "predicted_sabotages_fixed"
  ]
}
```

## Frontend features

The UI in `index.html` includes:

- glassmorphism card styling
- neon input and button effects
- animated starfield and floating background elements
- randomized flying character images from `assets/characters/`
- top logo using `assets/among-us-logo.jpg`
- live what-if predictions as you change inputs
- animated results panel
- confidence color bands:
  - green for high confidence
  - amber for medium confidence
  - red for low confidence

## Backend features

The Flask app in `app.py` includes:

- JSON input validation
- clear error responses
- trained model loading via `joblib`
- survival classification prediction
- sabotage regression prediction
- confidence band logic
- model info endpoint for version visibility

## Training features

The training script in `train.py` includes:

- cleaning raw CSV fields
- parsing game length into seconds
- building preprocessing pipelines
- training a classifier and regressor
- saving serialized model files into `models/`

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

If you are using a virtual environment, activate it first.

## Typical run order

1. Install packages.
2. Train the models.
3. Start the Flask API.
4. Open the frontend.
5. Test predictions.

## Example workflow

```bash
python train.py
python app.py
```

Then open `index.html` and try values like:

- Crewmate, 5 tasks, 0 kills, 600 seconds
- Imposter, 1 task, 3 kills, 420 seconds

## Notes

- If the frontend says it cannot connect, make sure Flask is running.
- If the models are missing, run `python train.py` again.
- If you change the training data, retrain the model so the API uses the latest artifacts.

## Future improvements

Possible upgrades for the next version:

- confidence meter calibration
- feature importance explanation panel
- additional Among Us gameplay inputs
- downloadable result card
- leaderboard or history tracking
- model metadata saved automatically during training

## License

This project is for learning and demo purposes.

---

## 🤝 Let's Connect!

If you enjoy this course, consider:
- Follow me on **[LinkedIn](https://www.linkedin.com/in/thanvir-assif-1b3435203/)**
- Subscribe to my YouTube channels:
        
    * [Thanvir Assif](https://www.youtube.com/@thanvirassif731) 
    * [Learn With Ash - Tamil](https://www.youtube.com/@learnwithashtamil7)

- Book 1:1 guidance via **[Topmate](https://topmate.io/thanvir_assif/)**

Stay tuned for more courses and tutorials!

---