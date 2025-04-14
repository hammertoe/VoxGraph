# -*- coding: utf-8 -*-
import requests
import threading
import os
from logger_setup import logger, get_log_prefix
import config

# --- Import from local CAR library ---
try:
    from car_library import generate_cid, cid_to_string, generate_car
    logger.info("Successfully imported functions from car_library.py")
except ImportError as e:
    logger.critical(f"Failed to import from car_library.py: {e}. Local CID/CAR generation will fail.")
    # Define dummy functions if import fails
    def generate_cid(data): return b"DUMMY_CID_BYTES_LIB_MISSING"
    def cid_to_string(cid_bytes): return "bDummyCIDStringLibMissing"
    def generate_car(text): return b"DUMMY_CAR_DATA_LIB_MISSING"

def generate_car_and_cid(text_data: str, session_id: str, sid: str) -> tuple[str | None, bytes | None]:
    """Generates an IPFS CID (string) and CAR file data locally using car_library."""
    log_prefix = get_log_prefix(session_id, sid, "CARGen")
    try:
        logger.debug(f"{log_prefix} Generating CAR/CID locally...")
        # Ensure text_data is string for generate_car, encode for generate_cid
        if not isinstance(text_data, str):
             raise TypeError("Input text_data must be a string")

        car_bytes = generate_car(text_data)
        if not car_bytes:
             raise ValueError("generate_car returned empty data")

        cid_bytes = generate_cid(text_data.encode('utf-8'))
        cid_string = cid_to_string(cid_bytes)

        if not cid_string or not car_bytes:
             raise ValueError("Failed to generate valid CID string or CAR bytes")

        logger.debug(f"{log_prefix} Generated CID: {cid_string}, CAR size: {len(car_bytes)} bytes")
        return cid_string, car_bytes
    except Exception as e:
        logger.error(f"{log_prefix} Error generating CAR/CID locally: {e}", exc_info=True)
        return None, None

def _upload_task(car_data: bytes, cid_str: str, session_id: str, sid: str):
    """Internal function to run the upload in a separate thread."""
    log_prefix = get_log_prefix(session_id, sid, "UploadThread")
    threading.current_thread().name = f"Upload-{session_id[:6]}-{sid[:6]}"

    if not config.LIGHTHOUSE_API_KEY:
        logger.error(f"{log_prefix} Lighthouse API Key missing. Cannot upload CAR {cid_str}.")
        return
    if not car_data:
        logger.error(f"{log_prefix} No CAR data provided for {cid_str}. Cannot upload.")
        return

    upload_url = "https://node.lighthouse.storage/api/v0/add"
    headers = {"Authorization": f"Bearer {config.LIGHTHOUSE_API_KEY}"}
    files = {'file': (f'{cid_str}.car', car_data, 'application/octet-stream')}

    try:
        logger.info(f"{log_prefix} Uploading CAR {cid_str} ({len(car_data)} bytes) to Lighthouse...")
        response = requests.post(upload_url, headers=headers, files=files, timeout=180) # 3 min timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        uploaded_cid = response_data.get('Hash')
        uploaded_size = response_data.get('Size')
        logger.info(f"{log_prefix} Upload successful. Lighthouse CID: {uploaded_cid}, Size: {uploaded_size} (Local CID: {cid_str})")
        # Optionally emit success to client? Needs socketio instance. Maybe return status?

    except requests.exceptions.Timeout:
         logger.error(f"{log_prefix} Lighthouse CAR upload timed out for {cid_str}.")
    except requests.exceptions.RequestException as e:
        logger.error(f"{log_prefix} Lighthouse CAR upload failed for {cid_str}: {e}", exc_info=True)
        # Optionally emit failure to client? Needs socketio instance.
    except json.JSONDecodeError:
        logger.error(f"{log_prefix} Failed to decode Lighthouse API response for {cid_str}. Response: {response.text}")
    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error during Lighthouse CAR upload for {cid_str}: {e}", exc_info=True)

def upload_car_to_lighthouse_async(car_data: bytes, cid_str: str, session_id: str, sid: str):
    """Starts the Lighthouse upload in a background thread."""
    log_prefix = get_log_prefix(session_id, sid, "UploadMgr")
    if not config.LIGHTHOUSE_API_KEY:
        logger.warning(f"{log_prefix} Skipping CAR upload for {cid_str} - API key missing.")
        return
    if not car_data or not cid_str:
        logger.warning(f"{log_prefix} Skipping CAR upload - invalid data or CID.")
        return

    logger.info(f"{log_prefix} Starting async CAR upload thread for CID: {cid_str}")
    upload_thread = threading.Thread(
        target=_upload_task,
        args=(car_data, cid_str, session_id, sid),
        daemon=True # Allows main thread to exit even if upload is running
    )
    upload_thread.start()
