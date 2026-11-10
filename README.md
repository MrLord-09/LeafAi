# LeafAI Plant Disease Scanner — Multi-Crop Upgrade

This is the upgraded version of the supplied project. It keeps the existing FastAPI + Streamlit + web frontend and adds a training/data pipeline for Mango, Rice, Wheat, Banana, Cotton, Papaya, Guava, Onion, Cabbage, Chilli, Brinjal, Groundnut, Sunflower, Tea, Strawberry, Peach and the original PlantVillage classes.

## What changed

- General plant-image preprocessing instead of assuming every image is a leaf.
- Dynamic model labels loaded from `backend/class_names.json`.
- `/api/v1/classes` endpoint for the website disease/crop grid.
- Disease/treatment catalog and SQLite seeding for the new crops.
- Dataset normalizer: `training/prepare_dataset.py`.
- PlantVillage downloader/exporter: `training/download_open_datasets.py`.
- MobileNetV3 transfer-learning trainer: `training/train.py`.
- Keras → TFLite conversion: `training/convert_tflite.py`.
- No fake prediction fallback: the API now tells you when the trained model is missing.
- Scans with less than 60% model confidence are rejected with supported-crop guidance; unsupported inputs cannot be reliably detected when a model is overconfident.
- Optional `Other___Not_a_Plant` training class lets the model reject non-plant/material examples rather than forcing every image into a crop disease label.
- Chemical treatment text is intentionally label/registration based; verify current Indian registration, dose, PHI and local extension guidance before publishing a pesticide recommendation.

## Dataset folder format

Put licensed images under `data/raw/` using either of these layouts:

```text
data/raw/
  my_source/
    Mango___Anthracnose/
      image1.jpg
      image2.jpg
    Mango___healthy/
      image3.jpg
    Rice___Blast/
      image4.jpg
```

or:

```text
data/raw/my_source/Mango/Anthracnose/*.jpg
data/raw/my_source/Mango/healthy/*.jpg
```

To teach the model to reject non-plant objects and materials, add varied example
photos under `data/raw/not_a_plant/`, grouped into folders by type. These folder
names are used only to organize examples; all images are trained as the single
`Other___Not_a_Plant` class:

```text
data/raw/not_a_plant/
  household_objects/*.jpg
  tools/*.jpg
  packaging/*.jpg
  soil_or_empty_scenes/*.jpg
```

Include diverse examples captured in conditions similar to the plant photos.
The model can only learn to reject objects represented by its training examples;
the negative class improves rejection but cannot guarantee recognition of every
possible unseen material.

Then run:

```powershell
python training/prepare_dataset.py --clean
```

The preparation script supports the added Banana, Cotton, Guava, Papaya, Rice, and
Wheat datasets already under `data/raw/`. It uses the original Banana photos (not
their augmented copies), keeps provided train/validation splits, and converts the
Rice YOLO boxes into image crops for classification. By default it hard-links
images where possible and copies them when hard-linking is unavailable; pass
`--copy` to always copy. The Mango YOLO dataset is skipped until its numeric class
IDs have a verified class-name mapping.

## Open datasets

PlantVillage is documented as an open dataset with 38 categories across 14 species; its standardized repository documents CC BY 3.0. It includes Peach and Strawberry but not all requested crops.

PlantDoc is documented as CC BY 4.0 in its standardized repository and is useful for field-style images.

Use `data/source_manifest.csv` to record the exact source and license for every dataset you add. Do not scrape random search-engine images with unknown rights.

## Train

Recommended Python environment: Python 3.10 or 3.11.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For PlantVillage:

```powershell
python training/download_open_datasets.py
python training/prepare_dataset.py --raw data/raw --out data/organized --copy
```

For your additional crop datasets, copy them into `data/raw/` first, then run the prepare command above.

Train (the reusable training code is `training/train.py`):

```powershell
python training/train.py --data data/organized --epochs 10 --batch-size 32
```

The existing MobileNetV3 base weights are reused when the current Keras model is
present. The class names are saved to `backend/class_names.json`.

To continue training an existing classifier without resetting its classification
head, pass `--resume-model` and the matching labels file. Use a separate output
path to validate a candidate before replacing the current model:

```powershell
python training/train.py --data data/organized --epochs 5 --batch-size 32 `
  --resume-model backend/plant_disease_model.keras `
  --resume-labels backend/class_names.json `
  --output backend/plant_disease_model.candidate.keras `
  --labels backend/class_names.candidate.json
```

### Train the plant/non-plant detector

Leafybud2 is used as a separate binary gate before disease diagnosis. It must
contain `1 training`, `2 validating`, and `3 testing` folders, each with
`plant` and `not plant` subfolders. Images can be nested under category folders;
the top-level label folder determines the binary label. The test split is held
out and evaluated after training.

```powershell
python training/train_plant_detector.py --data data/raw/Leafybud2 --epochs 10 --batch-size 32
python training/convert_tflite.py --model backend/plant_detector.keras --output backend/plant_detector.tflite
```

The detector artifacts are saved as `backend/plant_detector.keras`,
`backend/plant_detector.tflite`, and
`backend/plant_detector_class_names.json`. The existing disease classifier stays
separate and continues to use `backend/class_names.json`.

Convert to TFLite:

```powershell
python training/convert_tflite.py
```

This creates:

```text
backend/plant_disease_model.keras
backend/model_quantized.tflite
backend/class_names.json
```

## Seed the treatment database

```powershell
python scripts/init_db.py
```

## Run the website

```powershell
python run_app.py
```

Open `http://127.0.0.1:8000`.

## Deploy on Render

The repository includes a Render Blueprint (`render.yaml`) and a Docker image
for the FastAPI app. The service uses a persistent disk at `/var/data` for the
remedy database and private scan history. Keep one service instance while using
SQLite and this disk; do not scale this configuration to multiple instances.
The deployment uses a paid Render instance because it needs both a persistent
disk and memory for TensorFlow.

1. Push the project to the GitHub repository connected to your Render account.
   The provided `MrLord-09/LeafAi` repository was empty when checked.
   From the project folder, you can initialize and push the first deployment
   commit with:

   ```powershell
   git init -b main
   git add .
   git status --short
   git commit -m "Prepare LeafAI for Render deployment"
   git remote add origin https://github.com/MrLord-09/LeafAi.git
   git push -u origin main
   ```

   Review the staged files before committing. `.gitignore` excludes local
   databases, scan images, training datasets, and Keras checkpoints.
2. In Render, choose **New → Blueprint**, select that repository, and apply the
   `render.yaml` configuration. Render will build the Docker image and create
   the web service and persistent disk.
3. During setup, enter a unique admin username and a strong admin password
   when Render asks for `ADMIN_USERNAME` and `ADMIN_PASSWORD`. Keep the
   generated `ADMIN_SESSION_SECRET`; do not commit secrets to the repository.
4. Wait for the `/health` check to pass, then open the HTTPS `onrender.com`
   URL shown by Render. Add that URL to the phone's browser to install the PWA.

The deployment configuration does not include local scan history, local
database contents, training datasets, or Keras training/candidate models. It
includes the TFLite models required for scans. Back up the Render disk before
replacing or deleting the service; destroying the disk permanently deletes
saved scan history. Set up a custom domain in Render later if needed.

## Install LeafAI on a phone

LeafAI can be installed from a supported mobile browser as a Progressive Web App
(PWA). Serve the app over HTTPS on a phone-accessible host, open it in the
browser, and choose **Install app** (or the browser's **Add to Home Screen**
menu). On iPhone or iPad, open the site in Safari, tap **Share**, then **Add to
Home Screen**. The local `http://127.0.0.1:8000` development address only works
on the computer running the server; a phone needs a deployed HTTPS address.

The app shell and static assets are cached after the first successful load, so
the interface can reopen offline. Plant diagnosis, crop news, and other API
features still need a working connection to the LeafAI backend. The browser
camera and PWA installation also require HTTPS outside localhost.

### Diagnose with an IP Webcam

Install and start the IP Webcam app on an Android phone connected to the same
local network as the machine running LeafAI. In LeafAI, choose **IP Webcam** and
enter the server address shown by the app, for example
`http://192.168.1.25:8080` (you can also enter `192.168.1.25:8080`; LeafAI adds
the `/shot.jpg` snapshot path automatically). Use the IP address and port shown
in the app, which may differ from this example.
Keep the crop in view and choose **Capture 5 & Diagnose**. LeafAI analyzes five
frames, selects the most consistent disease prediction, and stores one scan in
history. All accepted frames and their individual predictions are also stored
as image data in the local `ip_camera_snapshots` table in
`backend/official_cures.db`, linked to that scan. Deleting a scan from history
also deletes its associated snapshots. The IP camera must be reachable by the
LeafAI backend over the local network; the browser-only camera option remains
available as before.

After each scan, the IP Webcam stays connected so you can move to another leaf
and run **Capture 5 & Diagnose** again without reconnecting. For the built-in
browser camera, use **Retake Photo** to capture another image without restarting
the camera; **Stop Camera** turns off the camera while keeping the current
preview available. Uploaded and captured images can also be diagnosed repeatedly
with the scan button. Errors appear in a small foreground alert.

Single-image and IP Webcam scans below 60% model confidence are rejected with a
message listing the trained crops and asking for a clear crop/plant photo. If
the optional `Other___Not_a_Plant` class is included during training, predictions
of that class are rejected instead of showing a disease diagnosis. A classifier
can still misclassify materials that are unlike its negative training examples.
The diagnosis displays the image used in a compact preview frame; IP Webcam shows
the exact saved snapshot selected for the consensus.

## Admin dashboard and scan history

The site admin page (`/admin`) accepts these default local login credentials:

- Username: `Alam`
- Password: `Alam@2304`

Override either default with `ADMIN_USERNAME` or `ADMIN_PASSWORD` environment
variables before starting the app:

```powershell
$env:ADMIN_USERNAME = "your-admin-name"
$env:ADMIN_PASSWORD = "your-strong-password"
$env:ADMIN_SESSION_SECRET = python -c "import secrets; print(secrets.token_urlsafe(48))"
python run_app.py
```

Open `http://127.0.0.1:8000/admin` to review scan totals, search/filter prior
diagnoses, view their uploaded images, delete individual or all scan records, and
edit the symptoms, organic/IPM guidance, chemical guidance and source note for
each trained model class. Scan history and image files are retained until an
administrator deletes them; images are stored outside the public static directory
and are only served to an authenticated admin. IP Webcam frame images are stored
as BLOBs in `backend/official_cures.db`. Back up the database and
`backend/history_images/` together if retaining history is important.

The default password and automatically generated session signing key are for
local development only. The development launcher binds to `127.0.0.1`; do not
expose the default credentials on a network-facing deployment. Use a unique
strong password and a persistent secret of at least 32 characters, HTTPS,
`ADMIN_COOKIE_SECURE=1`, and an appropriately configured production server.
Chemical guidance is label-based: verify current Indian CIB&RC registration and
local agricultural-extension recommendations before use.

## Important model limitation

The project can contain disease names in the catalog before images exist, but the AI cannot reliably recognize a class until that class has enough correctly labeled training images. The actual model's `class_names.json` is therefore generated from the classes present in `data/organized` at training time.

For production, keep separate train/validation/test images by plant/field/source where possible and evaluate the model on real phone photographs, not only clean laboratory images.


## Trainning
& 'C:\Users\ASUS\anaconda3\python.exe' .\training\prepare_dataset.py --clean
& 'C:\Users\ASUS\anaconda3\python.exe' .\training\train.py --data .\data\organized --epochs 10 --batch-size 32
& 'C:\Users\ASUS\anaconda3\python.exe' .\training\convert_tflite.py
& 'C:\Users\ASUS\anaconda3\python.exe' .\run_app.py
