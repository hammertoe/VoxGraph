# -*- coding: utf-8 -*-
import eventlet # For sleep
from flask import request
from flask_socketio import emit, join_room, leave_room, disconnect

# Import necessary components from other modules
from logger_setup import logger, get_log_prefix
import config
from session_manager import (
    get_session_id_for_sid, get_or_create_session, get_session,
    map_sid_to_session, unmap_sid, remove_session_if_empty,
    get_client_state, ClientState # Import ClientState for type hint
)
from rdf_utils import graph_to_visjs, update_graph_visualization
from ipfs_utils import generate_car_and_cid, upload_car_to_lighthouse_async
from audio_handler import run_async_session_manager, terminate_audio_session, put_status_update
# Need the llm_handler_instance to check availability
from llm_handler import llm_handler_instance
# NLTK for sentence tokenization (if needed, currently bypassed by utterance approach)
import nltk
from nltk.tokenize import sent_tokenize

# NLTK Download Check (moved here as it's used by a handler potentially)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("NLTK 'punkt' tokenizer not found. Downloading...")
    try:
        nltk.download('punkt')
        logger.info("NLTK 'punkt' tokenizer downloaded.")
    except Exception as e:
        logger.error(f"Failed to download NLTK 'punkt': {e}")


# --- Status Queue Poller ---
def status_queue_poller(session_id: str, sid: str, socketio):
    """Eventlet background task polling status queue for a specific client."""
    log_prefix = get_log_prefix(session_id, sid, "Poller")
    logger.info(f"{log_prefix} Starting.")
    should_run = True

    while should_run:
        client_state = get_client_state(sid) # Check if client still exists each loop

        if not client_state:
            logger.info(f"{log_prefix} Client state no longer found. Stopping poller.")
            should_run = False
            break # Exit loop

        status_queue = client_state.status_queue
        if not status_queue:
             logger.error(f"{log_prefix} Status queue missing! Stopping poller.")
             should_run = False
             break

        try:
            # Get update with a small timeout to prevent busy-waiting and allow checking exit condition
            update = status_queue.get(block=True, timeout=0.1)
            event = update.get('event')
            data = update.get('data', {})

            # Process based on event type
            if event == 'new_transcription':
                 # Handle the transcribed text (CID gen, upload, Quick LLM)
                 handle_transcription_result(data.get('text',''), session_id, sid, socketio)
            elif event in ['audio_started', 'audio_stopped', 'error', 'reconnecting', 'reconnected', 'connection_lost']:
                # Emit status directly to the specific client SID
                # logger.debug(f"{log_prefix} Emitting status '{event}' to client.")
                socketio.emit(event, data, room=sid) # Use room=sid for direct message
            else:
                logger.warning(f"{log_prefix} Received unknown status event type: {event}")

            status_queue.task_done() # Mark task as done

        except QueueEmpty:
            # Timeout occurred, loop continues and checks client validity again
            eventlet.sleep(0.05) # Small sleep after timeout
            continue
        except Exception as e:
             logger.error(f"{log_prefix} Error processing status queue: {e}", exc_info=True)
             eventlet.sleep(0.5) # Longer sleep on error

    logger.info(f"{log_prefix} Stopped.")


# --- Transcription Result Handler ---
def handle_transcription_result(text: str, session_id: str, sid: str, socketio):
    """Processes transcribed text: generates CAR/CID, starts upload, triggers LLM."""
    log_prefix = get_log_prefix(session_id, sid, "TranscriptHdlr")
    session = get_session(session_id)
    if not session:
        logger.warning(f"{log_prefix} Session not found for transcription result.")
        return
    text = text.strip()
    if not text:
        return

    logger.info(f"{log_prefix} Received Transcription: '{text[:100]}...'")

    # 1. Generate CAR and CID locally
    cid_string, car_bytes = generate_car_and_cid(text, session_id, sid)

    # 2. Start async upload if successful
    if cid_string and car_bytes:
        upload_car_to_lighthouse_async(car_bytes, cid_string, session_id, sid)
    elif not cid_string or not car_bytes:
        logger.warning(f"{log_prefix} Failed to generate CAR/CID. Proceeding without archiving.")
        cid_string = None # Ensure CID is None if generation failed
    # Upload function handles missing API key warning internally

    # 3. Trigger Quick LLM processing via the Session object in a background task
    logger.info(f"{log_prefix} Starting background task for Quick LLM (CID: {cid_string})")
    socketio.start_background_task(
        session.process_with_quick_llm,
        text,
        sid,
        transcription_cid=cid_string
    )


# --- SocketIO Event Handlers ---
def register_handlers(socketio):
    logger.info("Registering SocketIO event handlers...")

    @socketio.on('connect')
    def handle_connect():
        sid = request.sid
        logger.info(f"[Connect|SID:{sid[:6]}] Client connected. Waiting for 'join_session'.")
        # Wait for client to send session_id via 'join_session'

    @socketio.on('join_session')
    def handle_join_session(data):
        """Handles client joining a specific session room."""
        sid = request.sid
        session_id = data.get('session_id') if isinstance(data, dict) else None

        if not session_id or not isinstance(session_id, str):
            logger.error(f"[Join|SID:{sid[:6]}] Invalid or missing session_id. Data: {data}. Disconnecting.")
            # Emit error before disconnecting if possible
            try: emit('error', {'message': 'Invalid session ID provided.'}, room=sid)
            except: pass # Ignore if emit fails during disconnect
            disconnect(sid)
            return

        log_prefix = get_log_prefix(session_id, sid, "Join")
        logger.info(f"{log_prefix} Received join request.")

        try:
            # --- Session/Client Initialization ---
            session = get_or_create_session(session_id, socketio)
            client_state = session.add_client(sid) # Creates/replaces client state

            # --- SocketIO Room and SID Mapping ---
            join_room(session_id, sid=sid)
            map_sid_to_session(sid, session_id)
            logger.info(f"{log_prefix} Client added to session room and SID mapped.")

            # --- Start Status Poller for this Client ---
            poller_task = socketio.start_background_task(status_queue_poller, session_id, sid, socketio)
            client_state.status_poller_task = poller_task # Store reference if needed
            logger.info(f"{log_prefix} Started status queue poller task.")

            # --- Send Initial Graph State to Newly Joined Client ---
            initial_graph = session.get_graph()
            vis_data = graph_to_visjs(initial_graph)
            emit('update_graph', vis_data, room=sid) # Send only to the new client
            logger.info(f"{log_prefix} Initial graph state ({len(initial_graph)} triples) sent to client.")

            # Optionally, send confirmation of successful join
            emit('session_joined', {'session_id': session_id, 'message': 'Successfully joined session.'}, room=sid)

        except Exception as e:
            logger.error(f"{log_prefix} Error during join_session: {e}", exc_info=True)
            emit('error', {'message': f'Server error joining session: {e}'}, room=sid)
            # Clean up potentially partial setup
            unmap_sid(sid)
            session = get_session(session_id)
            if session:
                 session.remove_client(sid) # Will trigger cleanup in ClientState
                 remove_session_if_empty(session_id)
            try: leave_room(session_id, sid=sid)
            except: pass
            disconnect(sid)


    @socketio.on('disconnect')
    def handle_disconnect():
        sid = request.sid
        session_id = get_session_id_for_sid(sid) # Find session before unmapping

        if not session_id:
            logger.warning(f"[Disconnect|SID:{sid[:6]}] Client disconnected, but no session mapping found.")
            return # Already unmapped or never joined properly

        log_prefix = get_log_prefix(session_id, sid, "Disconnect")
        logger.info(f"{log_prefix} Client disconnected.")

        # --- Unmap SID First ---
        unmap_sid(sid) # Remove mapping

        # --- Clean up Client and Session Resources ---
        session = get_session(session_id)
        if session:
            session.remove_client(sid) # Removes client state and triggers ClientState.cleanup()
            remove_session_if_empty(session_id) # Checks if session needs removal
        else:
             logger.warning(f"{log_prefix} Session object not found during disconnect for session ID: {session_id}")

        # --- Leave SocketIO Room ---
        # This might fail if client already disconnected abruptly, hence try/except
        try:
            leave_room(session_id, sid=sid)
            logger.info(f"{log_prefix} Client removed from SocketIO session room.")
        except Exception as e:
            logger.warning(f"{log_prefix} Error removing client from room (may already be gone): {e}")

        logger.info(f"{log_prefix} Disconnect handling complete.")


    @socketio.on('start_audio')
    def handle_start_audio():
        sid = request.sid
        client_state = get_client_state(sid) # Use helper to get ClientState

        if not client_state:
            logger.error(f"[StartAudio|SID:{sid[:6]}] Received 'start_audio' from unknown/invalid client.")
            emit('error', {'message': 'Invalid session context. Please rejoin.'}, room=sid)
            return

        log_prefix = get_log_prefix(client_state.session_id, sid, "StartAudio")

        # Check if Live API is available globally
        if not llm_handler_instance.is_live_client_available():
            logger.error(f"{log_prefix} Google Live API client is unavailable.")
            emit('error', {'message': 'Live transcription service is currently unavailable.'}, room=sid)
            return

        # Ensure clean state: Terminate any potentially lingering previous session
        logger.info(f"{log_prefix} Ensuring clean state before starting...")
        # Pass client_state directly to termination function
        terminate_audio_session(client_state.session_id, sid, client_state)

        # Start new session
        logger.info(f"{log_prefix} Received 'start_audio'. Starting fresh audio session.")
        client_state.is_receiving_audio = True # Set flag **before** starting thread

        # Start the async manager in its own thread, passing the client_state
        thread = threading.Thread(
            target=run_async_session_manager,
            args=(client_state.session_id, sid, client_state), # Pass client_state
            daemon=True
        )
        client_state.live_session_thread = thread # Store thread reference
        thread.start()
        logger.info(f"{log_prefix} Audio manager thread started (ID: {thread.ident}).")
        # Note: 'audio_started' event is sent by the manager thread via status_queue


    @socketio.on('stop_audio')
    def handle_stop_audio():
        sid = request.sid
        client_state = get_client_state(sid)

        if not client_state:
            logger.warning(f"[StopAudio|SID:{sid[:6]}] Received 'stop_audio' from unknown/invalid client.")
            # Emit stopped just in case client UI is stuck waiting
            emit('audio_stopped', {'message': 'Audio recording stop requested, but session context invalid.'}, room=sid)
            return

        log_prefix = get_log_prefix(client_state.session_id, sid, "StopAudio")

        if not client_state.is_receiving_audio:
            logger.warning(f"{log_prefix} Received 'stop_audio' but client was not marked as receiving.")
            # Still attempt termination for robustness, might clean up lingering thread/refs
            terminate_audio_session(client_state.session_id, sid, client_state)
            emit('audio_stopped', {'message': 'Audio recording was already stopped.'}, room=sid)
            return

        logger.info(f"{log_prefix} Received 'stop_audio'. Initiating termination.")
        terminate_audio_session(client_state.session_id, sid, client_state)

        # Explicitly send confirmation immediately (don't wait for manager thread's final status)
        emit('audio_stopped', {'message': 'Audio recording stopped by user request.'}, room=sid)


    @socketio.on('audio_chunk')
    def handle_audio_chunk(data):
        # High frequency event - minimize logging unless debugging
        sid = request.sid
        client_state = get_client_state(sid)

        # Fail silently if client/session is invalid or not receiving audio
        if not client_state or not client_state.is_receiving_audio:
            # logger.warning(f"[AudioChunk|SID:{sid[:6]}] Dropping audio - invalid client or not receiving.")
            return

        # Basic data validation
        if not isinstance(data, bytes):
            logger.error(f"{get_log_prefix(client_state.session_id, sid, 'AudioChunk')} Received non-bytes audio data.")
            return

        audio_queue = client_state.audio_queue
        if not audio_queue:
            logger.error(f"{get_log_prefix(client_state.session_id, sid, 'AudioChunk')} Audio queue missing!")
            return

        try:
            # Construct message expected by Google API (assuming PCM audio)
            msg = {"data": data, "mime_type": "audio/pcm"}
            audio_queue.put_nowait(msg)
            # logger.debug(f"{get_log_prefix(client_state.session_id, sid, 'AudioChunk')} Queued audio ({len(data)} bytes). QSize: {audio_queue.qsize()}")
        except QueueFull:
            logger.warning(f"{get_log_prefix(client_state.session_id, sid, 'AudioChunk')} Audio queue full. Dropping chunk.")
        except Exception as e:
             logger.error(f"{get_log_prefix(client_state.session_id, sid, 'AudioChunk')} Error queuing audio: {e}", exc_info=True)


    @socketio.on('query_graph')
    def handle_query_graph(data):
        sid = request.sid
        session_id = get_session_id_for_sid(sid)
        session = get_session(session_id) if session_id else None

        if not session:
            logger.error(f"[Query|SID:{sid[:6]}] Query from unknown/invalid client/session.")
            emit('query_result', {'answer': "Error: Invalid session. Please refresh.", 'error': True}, room=sid)
            return

        log_prefix = get_log_prefix(session_id, sid, "Query")
        query_text = data.get('query', '').strip() if isinstance(data, dict) else ''

        if not query_text:
            logger.warning(f"{log_prefix} Received empty query.")
            emit('query_result', {'answer': "Please enter a query.", 'error': True}, room=sid)
            return

        # Delegate query processing to the Session object, which handles LLM checks
        # This runs in a background task initiated by the session method
        session.process_query(query_text, sid)


    logger.info("SocketIO event handlers registered.")