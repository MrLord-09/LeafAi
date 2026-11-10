import json
import html
import ipaddress
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import requests
from fastapi import Depends, FastAPI, UploadFile, File, HTTPException, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.admin_auth import (
    SESSION_COOKIE,
    SESSION_LIFETIME_SECONDS,
    authenticate,
    create_session,
    require_admin,
    require_csrf,
    validate_admin_configuration,
)
from backend.preprocess import preprocess_leaf_image
from backend.model_inference import PlantModel
from backend.database import (
    delete_all_scans,
    delete_scan,
    get_remedy_data,
    get_scan_history,
    get_scan_image,
    history_summary,
    initialize_database,
    list_remedies,
    save_scan,
    update_remedy_data,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")
DATA_DIR = os.environ.get("LEAF_AI_DATA_DIR", BASE_DIR)
MODEL_PATH = os.path.join(BASE_DIR, "model_quantized.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "class_names.json")
PLANT_DETECTOR_MODEL_PATH = os.path.join(BASE_DIR, "plant_detector.tflite")
PLANT_DETECTOR_LABELS_PATH = os.path.join(BASE_DIR, "plant_detector_class_names.json")
DB_PATH = os.path.join(DATA_DIR, "official_cures.db")
HISTORY_IMAGES_DIR = os.path.join(DATA_DIR, "history_images")
MIN_DIAGNOSTIC_CONFIDENCE = 60.0
MIN_PLANT_DETECTOR_CONFIDENCE = 60.0
NON_PLANT_CLASS_NAME = "Other___Not_a_Plant"
NON_PLANT_DETECTOR_CLASS_NAME = "not_plant"
NEWS_FEED_URL = "https://www.bing.com/news/search"
NEWS_CACHE_SECONDS = 15 * 60
news_cache = {"expires_at": 0.0, "data": None}

# This catalog is the target label space. Training creates model weights for whichever
# classes actually have images; unsupported/untrained classes are not claimed as detectable.
CATALOG_PATH = os.path.join(PROJECT_DIR, "data", "disease_catalog.json")
with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    DISEASE_CATALOG = json.load(f)

CLASS_NAMES = []
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        CLASS_NAMES = json.load(f)

SUPPORTED_PLANT_CLASSES = [
    class_name for class_name in CLASS_NAMES
    if class_name != NON_PLANT_CLASS_NAME and "___" in class_name
]

model_instance = None
plant_detector_instance = None

@asynccontextmanager
async def lifespan(_: FastAPI):
    global model_instance, plant_detector_instance
    if os.environ.get("APP_ENV", "").casefold() == "production":
        validate_admin_configuration()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(HISTORY_IMAGES_DIR, exist_ok=True)
    initialize_database(DB_PATH, SUPPORTED_PLANT_CLASSES, DISEASE_CATALOG)
    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        try:
            model_instance = PlantModel(MODEL_PATH, LABELS_PATH)
        except Exception as exc:
            print(f"Model load warning: {exc}")
            model_instance = None
    if os.path.exists(PLANT_DETECTOR_MODEL_PATH) and os.path.exists(PLANT_DETECTOR_LABELS_PATH):
        try:
            plant_detector_instance = PlantModel(
                PLANT_DETECTOR_MODEL_PATH,
                PLANT_DETECTOR_LABELS_PATH,
            )
        except Exception as exc:
            print(f"Plant/non-plant model load warning: {exc}")
            plant_detector_instance = None
    if os.environ.get("APP_ENV", "").casefold() == "production":
        if model_instance is None or plant_detector_instance is None:
            raise RuntimeError(
                "Both TFLite models must load successfully before production can start."
            )
    yield


app = FastAPI(
    title="Plant Disease & Health Diagnostic API",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    return FileResponse(index_path, media_type="text/html") if os.path.exists(index_path) else {"status": "online"}


@app.get("/manifest.webmanifest", include_in_schema=False)
def app_manifest():
    return FileResponse(
        os.path.join(FRONTEND_DIR, "manifest.webmanifest"),
        media_type="application/manifest+json",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/service-worker.js", include_in_schema=False)
def app_service_worker():
    return FileResponse(
        os.path.join(FRONTEND_DIR, "service-worker.js"),
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"},
    )


@app.get("/admin")
def admin_page():
    return FileResponse(os.path.join(FRONTEND_DIR, "admin.html"), media_type="text/html")

@app.get("/health")
def health():
    return {
        "status": "online",
        "model_loaded": model_instance is not None,
        "plant_detector_loaded": plant_detector_instance is not None,
        "classes": len(SUPPORTED_PLANT_CLASSES),
    }

@app.get("/api/v1/classes")
def classes():
    catalog = {
        guide["class_name"]: guide
        for guide in list_remedies(DB_PATH, SUPPORTED_PLANT_CLASSES)
    }
    return {
        "classes": SUPPORTED_PLANT_CLASSES,
        "catalog": catalog,
        "model_loaded": model_instance is not None,
        "plant_detector_loaded": plant_detector_instance is not None,
    }


@app.get("/api/v1/news")
def crop_news():
    if news_cache["data"] is not None and time.monotonic() < news_cache["expires_at"]:
        return news_cache["data"]

    try:
        response = requests.get(
            NEWS_FEED_URL,
            params={
                "q": f"India agriculture farmers crops {datetime.now().year}",
                "format": "rss",
            },
            headers={"User-Agent": "LeafAI/1.0 (agriculture news reader)"},
            timeout=10,
        )
        response.raise_for_status()
        if len(response.content) > 1024 * 1024:
            raise HTTPException(status_code=502, detail="The crop news feed response was too large.")
        feed = ET.fromstring(response.content)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Crop news is temporarily unavailable.") from exc
    except ET.ParseError as exc:
        raise HTTPException(status_code=502, detail="The crop news feed returned invalid data.") from exc

    articles = []
    for item in feed.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        feed_url = (item.findtext("link") or "").strip()
        query = parse_qs(urlsplit(feed_url).query)
        url = (query.get("url") or [feed_url])[0].strip()
        parsed_url = urlsplit(url)
        if not title or parsed_url.scheme != "https" or not parsed_url.hostname:
            continue

        description = item.findtext("description") or ""
        summary = html.unescape(re.sub(r"<[^>]*>", " ", description))
        summary = re.sub(r"\s+", " ", summary).strip()[:360]
        image_url = ""
        image_node = next(
            (node for node in item if node.tag.endswith("}Image")),
            None,
        )
        if image_node is not None and image_node.text:
            candidate = image_node.text.strip()
            try:
                image_parts = urlsplit(candidate)
                if (
                    image_parts.scheme in {"http", "https"}
                    and image_parts.hostname in {"bing.com", "www.bing.com"}
                ):
                    image_url = image_parts._replace(scheme="https").geturl()
            except ValueError:
                pass

        published = item.findtext("pubDate") or ""
        try:
            published = parsedate_to_datetime(published).isoformat()
        except (TypeError, ValueError, OverflowError):
            published = ""
        source_node = next(
            (node for node in item if node.tag.endswith("}Source")),
            None,
        )
        articles.append(
            {
                "title": title,
                "url": url,
                "image_url": image_url,
                "source": (source_node.text or "").strip() if source_node is not None else "",
                "published_at": published,
                "summary": summary,
            }
        )
        if len(articles) == 6:
            break

    data = {
        "articles": articles,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    news_cache["data"] = data
    news_cache["expires_at"] = time.monotonic() + NEWS_CACHE_SECONDS
    return data


@app.get("/api/v1/news/image")
def crop_news_image(url: str):
    try:
        image_url = urlsplit(url)
        if (
            image_url.scheme != "https"
            or image_url.hostname != "www.bing.com"
            or image_url.username is not None
            or image_url.password is not None
            or image_url.port not in (None, 443)
            or image_url.path != "/th"
            or not image_url.query
            or image_url.fragment
        ):
            raise HTTPException(status_code=400, detail="Invalid news thumbnail URL.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid news thumbnail URL.") from exc

    try:
        with requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 LeafAI/1.0"},
            timeout=8,
            allow_redirects=False,
            stream=True,
        ) as image_response:
            image_response.raise_for_status()
            media_type = image_response.headers.get("content-type", "").split(";", 1)[0].lower()
            if media_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise HTTPException(status_code=502, detail="News provider returned an unsupported image.")
            image_data = bytearray()
            for chunk in image_response.iter_content(chunk_size=64 * 1024):
                image_data.extend(chunk)
                if len(image_data) > 4 * 1024 * 1024:
                    raise HTTPException(status_code=502, detail="News thumbnail exceeds the 4MB limit.")
            if not image_data:
                raise HTTPException(status_code=502, detail="News provider returned an empty image.")
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="News thumbnail is temporarily unavailable.") from exc

    return Response(
        content=bytes(image_data),
        media_type=media_type,
        headers={
            "Cache-Control": "public, max-age=900",
            "X-Content-Type-Options": "nosniff",
        },
    )


@app.post("/api/v1/scan")
async def scan_plant(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Use JPEG, PNG, JPG, or WebP.")
    bytes_data = await file.read()
    if len(bytes_data) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds the 8MB limit.")
    if model_instance is None:
        raise HTTPException(status_code=503, detail="AI model is not installed/trained yet. Run training/train.py and training/convert_tflite.py first.")
    try:
        detected_class, confidence, report = predict_plant(bytes_data)
        if confidence < MIN_DIAGNOSTIC_CONFIDENCE:
            raise HTTPException(status_code=400, detail=unsupported_image_detail())
        history_id = save_scan(
            DB_PATH,
            HISTORY_IMAGES_DIR,
            bytes_data,
            file.content_type,
            file.filename or "plant-image",
            detected_class,
            report["disease_name"],
            confidence,
        )
        return {
            "confidence": confidence,
            "class_name": detected_class,
            "report": report,
            "history_id": history_id,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {exc}")


def predict_plant(bytes_data: bytes):
    if model_instance is None:
        raise HTTPException(status_code=503, detail="AI model is not installed/trained yet. Run training/train.py and training/convert_tflite.py first.")
    if plant_detector_instance is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Plant/non-plant model is not installed yet. Run "
                "training/train_plant_detector.py and convert its model to TFLite."
            ),
        )
    try:
        tensor, _ = preprocess_leaf_image(bytes_data)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="This file could not be read as a valid image. Upload a clear JPEG or PNG photo of a crop, vegetable, fruit, or plant.",
        ) from exc
    detector_class_idx, detector_confidence = plant_detector_instance.predict(tensor)
    if detector_class_idx >= len(plant_detector_instance.labels):
        raise HTTPException(status_code=500, detail="Plant/non-plant model labels do not match model output size.")
    detector_class = plant_detector_instance.labels[detector_class_idx]
    if detector_class not in {NON_PLANT_DETECTOR_CLASS_NAME, "plant"}:
        raise HTTPException(status_code=500, detail="Plant/non-plant model contains an unknown class label.")
    if detector_confidence < MIN_PLANT_DETECTOR_CONFIDENCE:
        raise HTTPException(
            status_code=400,
            detail="The image could not be confidently identified as a plant. Upload a clear plant photo.",
        )
    if detector_class == NON_PLANT_DETECTOR_CLASS_NAME:
        raise HTTPException(
            status_code=400,
            detail="This does not appear to be a plant. Upload a clear photo of a supported crop, fruit, vegetable, or plant.",
        )
    class_idx, confidence = model_instance.predict(tensor)
    if class_idx >= len(CLASS_NAMES):
        raise HTTPException(status_code=500, detail="Model label file does not match model output size.")
    detected_class = CLASS_NAMES[class_idx]
    if detected_class == NON_PLANT_CLASS_NAME:
        raise HTTPException(
            status_code=400,
            detail="This does not appear to be a supported crop, fruit, vegetable, or plant. Please submit a clear image of a crop included in the trained dataset.",
        )
    report = get_remedy_data(DB_PATH, detected_class)
    if not report:
        crop_name = detected_class.split("___", 1)[0].replace("_", " ")
        raise HTTPException(
            status_code=400,
            detail=f"{crop_name} is not available in the local disease database. Please use an image of a supported crop.",
        )
    return detected_class, confidence, report


def unsupported_image_detail() -> str:
    supported_crops = sorted({
        class_name.split("___", 1)[0].replace("_", " ")
        for class_name in SUPPORTED_PLANT_CLASSES
        if "___" in class_name
    })
    crop_list = ", ".join(supported_crops) if supported_crops else "none (the model has no trained crop classes)"
    return (
        "The image could not be confidently matched to a supported crop. "
        "The crop may not be available, or the image may not clearly show a crop, "
        "vegetable, fruit, or plant. Upload a clear, well-lit plant image. "
        f"Currently supported crops: {crop_list}."
    )


@app.post("/api/v1/scan/batch")
async def scan_plant_batch(files: list[UploadFile] = File(...)):
    if not 4 <= len(files) <= 5:
        raise HTTPException(status_code=400, detail="Submit four or five images for a batch diagnosis.")
    predictions = []
    try:
        for file in files:
            if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
                raise HTTPException(status_code=400, detail="Invalid file type. Use JPEG, PNG, JPG, or WebP.")
            bytes_data = await file.read()
            if len(bytes_data) > 8 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="File size exceeds the 8MB limit.")
            detected_class, confidence, report = predict_plant(bytes_data)
            predictions.append({
                "class_name": detected_class,
                "confidence": confidence,
                "report": report,
                "bytes": bytes_data,
                "content_type": file.content_type,
                "filename": file.filename or "ip-webcam-image.jpg",
            })

        grouped = defaultdict(list)
        for prediction in predictions:
            grouped[prediction["class_name"]].append(prediction)
        detected_class = max(
            grouped,
            key=lambda name: (len(grouped[name]), sum(item["confidence"] for item in grouped[name]) / len(grouped[name])),
        )
        winners = grouped[detected_class]
        selected = max(winners, key=lambda item: item["confidence"])
        confidence = round(sum(item["confidence"] for item in winners) / len(winners), 2)
        if confidence < MIN_DIAGNOSTIC_CONFIDENCE:
            raise HTTPException(status_code=400, detail=unsupported_image_detail())
        selected_snapshot_index = next(
            index for index, prediction in enumerate(predictions) if prediction is selected
        )
        history_id = save_scan(
            DB_PATH,
            HISTORY_IMAGES_DIR,
            selected["bytes"],
            selected["content_type"],
            selected["filename"],
            detected_class,
            selected["report"]["disease_name"],
            confidence,
            snapshots=predictions,
        )
        return {
            "confidence": confidence,
            "class_name": detected_class,
            "report": selected["report"],
            "history_id": history_id,
            "sample_count": len(predictions),
            "vote_count": len(winners),
            "selected_snapshot_index": selected_snapshot_index,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Diagnostic error: {exc}")


@app.get("/api/v1/ip-camera/snapshot")
def ip_camera_snapshot(url: str):
    try:
        if "://" not in url:
            url = f"http://{url}"
        parsed_url = urlsplit(url)
    except ValueError:
        raise HTTPException(status_code=400, detail="The IP Webcam URL is invalid.")
    if parsed_url.path in {"", "/"}:
        url = f"{url.rstrip('/')}/shot.jpg"
        parsed_url = urlsplit(url)
    if (
        parsed_url.scheme != "http"
        or parsed_url.username is not None
        or parsed_url.password is not None
        or parsed_url.path != "/shot.jpg"
        or parsed_url.query
        or parsed_url.fragment
    ):
        raise HTTPException(status_code=400, detail="Use an IP Webcam URL in the form http://PRIVATE_IP:8080/shot.jpg.")
    try:
        address = ipaddress.ip_address(parsed_url.hostname or "")
        port = parsed_url.port or 8080
    except ValueError:
        raise HTTPException(status_code=400, detail="The camera URL must use a private IPv4 address.")
    if (
        not isinstance(address, ipaddress.IPv4Address)
        or not address.is_private
        or address.is_loopback
        or address.is_link_local
        or not 1024 <= port <= 65535
    ):
        raise HTTPException(status_code=400, detail="Use the IP Webcam's private IPv4 address and its configured port (1024–65535).")
    if parsed_url.port is None:
        url = parsed_url._replace(netloc=f"{parsed_url.netloc}:8080").geturl()

    try:
        with requests.get(url, timeout=5, allow_redirects=False, stream=True) as camera_response:
            camera_response.raise_for_status()
            content_type = camera_response.headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise HTTPException(status_code=502, detail="The camera did not return a supported image.")
            image_data = bytearray()
            for chunk in camera_response.iter_content(chunk_size=64 * 1024):
                image_data.extend(chunk)
                if len(image_data) > 8 * 1024 * 1024:
                    raise HTTPException(status_code=502, detail="The camera image exceeds the 8MB limit.")
            if not image_data:
                raise HTTPException(status_code=502, detail="The camera returned an empty image.")
            return Response(content=bytes(image_data), media_type=content_type, headers={"Cache-Control": "no-store"})
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach {parsed_url.hostname}:{port}. Check that the phone and computer are on the same Wi-Fi, IP Webcam is started, and the port is correct. ({exc})",
        )


class AdminCredentials(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


class DiseaseGuideUpdate(BaseModel):
    disease_name: str = Field(min_length=1, max_length=200)
    symptoms: str = Field(min_length=1, max_length=5000)
    organic_control: str = Field(min_length=1, max_length=5000)
    chemical_control: str = Field(min_length=1, max_length=5000)
    source: str = Field(min_length=1, max_length=1000)


@app.post("/api/admin/login")
def admin_login(credentials: AdminCredentials, response: Response):
    if not authenticate(credentials.username, credentials.password):
        raise HTTPException(status_code=401, detail="Invalid admin username or password.")
    token, csrf_token = create_session(credentials.username)
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_LIFETIME_SECONDS,
        httponly=True,
        secure=os.environ.get("ADMIN_COOKIE_SECURE", "").casefold() in {"1", "true", "yes"},
        samesite="strict",
        path="/api/admin",
    )
    return {"username": credentials.username, "csrf_token": csrf_token}


@app.get("/api/admin/session")
def admin_session(session: dict = Depends(require_admin)):
    return {"username": session["username"], "csrf_token": session["csrf"]}


@app.post("/api/admin/logout")
def admin_logout(response: Response):
    response.delete_cookie(SESSION_COOKIE, path="/api/admin", httponly=True, samesite="strict")
    return {"status": "logged_out"}


@app.get("/api/admin/overview", dependencies=[Depends(require_admin)])
def admin_overview():
    summary = history_summary(DB_PATH)
    summary["guide_count"] = len(SUPPORTED_PLANT_CLASSES)
    return summary


@app.get("/api/admin/history", dependencies=[Depends(require_admin)])
def admin_history(
    q: str = "",
    class_name: str | None = None,
    limit: int = 30,
    offset: int = 0,
):
    if class_name and class_name not in SUPPORTED_PLANT_CLASSES:
        raise HTTPException(status_code=400, detail="Unknown disease class.")
    if not 1 <= limit <= 100 or offset < 0:
        raise HTTPException(status_code=400, detail="Invalid history pagination.")
    items, total = get_scan_history(DB_PATH, q[:200], class_name, limit, offset)
    for item in items:
        item["image_url"] = f"/api/admin/history/{item['id']}/image"
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@app.get("/api/admin/history/{scan_id}/image", dependencies=[Depends(require_admin)])
def admin_history_image(scan_id: int):
    image = get_scan_image(DB_PATH, scan_id)
    if image is None:
        raise HTTPException(status_code=404, detail="Scan history item not found.")
    image_root = Path(HISTORY_IMAGES_DIR).resolve()
    image_path = (image_root / image[0]).resolve()
    if image_path.parent != image_root or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Stored scan image is unavailable.")
    return FileResponse(
        image_path,
        media_type=image[1],
        headers={"Cache-Control": "private, no-store"},
    )


@app.delete("/api/admin/history/{scan_id}", dependencies=[Depends(require_csrf)])
def admin_delete_scan(scan_id: int):
    if not delete_scan(DB_PATH, HISTORY_IMAGES_DIR, scan_id):
        raise HTTPException(status_code=404, detail="Scan history item not found.")
    return {"deleted": scan_id}


@app.delete("/api/admin/history", dependencies=[Depends(require_csrf)])
def admin_delete_all_scans():
    return {"deleted": delete_all_scans(DB_PATH, HISTORY_IMAGES_DIR)}


@app.get("/api/admin/guides", dependencies=[Depends(require_admin)])
def admin_guides():
    return {"guides": list_remedies(DB_PATH, SUPPORTED_PLANT_CLASSES)}


@app.put("/api/admin/guides/{class_name}", dependencies=[Depends(require_csrf)])
def admin_update_guide(
    class_name: str,
    update: DiseaseGuideUpdate,
):
    if class_name not in SUPPORTED_PLANT_CLASSES:
        raise HTTPException(status_code=404, detail="Disease class not found.")
    try:
        update_remedy_data(DB_PATH, class_name, update.model_dump())
    except KeyError:
        raise HTTPException(status_code=404, detail="Disease guidance record not found.")
    return {"guide": get_remedy_data(DB_PATH, class_name)}
