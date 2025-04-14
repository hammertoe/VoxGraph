# -*- coding: utf-8 -*-
# Step 1: Import and enable eventlet monkey patching FIRST
import eventlet
# Tell eventlet NOT to patch the standard threading module
# This allows standard threading for asyncio isolation within audio_handler
eventlet.monkey_patch(thread=False)

# Step 2: Now import other essential modules
import os
import sys
import uuid
import threading # Keep standard threading import

# Step 3: Import Flask and SocketIO
from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, emit

# Step 4: Import application components AFTER patching
from logger_setup import logger, get_log_prefix # Initialize logger early
import config # Load configuration
from llm_handler import llm_handler_instance # Check LLM clients
from session_manager import get_session # Import session functions if needed by routes
from socketio_handlers import register_handlers # Import handler registration function
from rdf_utils import update_graph_visualization # Potentially needed if routes update graph

# --- Flask App and SocketIO Initialization ---
logger.info("Initializing Flask app and SocketIO...")
app = Flask(__name__)
# Use a more persistent secret key in production if sessions are used server-side
app.config['SECRET_KEY'] = os.urandom(24)
# Initialize SocketIO with eventlet async mode
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")
logger.info("Flask app and SocketIO initialized.")

# --- Initial Service Status Checks ---
live_status = "Available" if llm_handler_instance.is_live_client_available() else "Unavailable"
llm_status = "Available" if llm_handler_instance.is_base_client_available() else "Unavailable (RDF/Query Disabled)"
archive_status = "Enabled" if config.LIGHTHOUSE_API_KEY else "Disabled (No API Key)"

logger.info(f"--- Service Status ---")
logger.info(f"Live Transcription API Client: {live_status}")
logger.info(f"Base LLM API Client: {llm_status}")
logger.info(f"Lighthouse Archiving: {archive_status}")
logger.info(f"----------------------")

if not llm_handler_instance.is_live_client_available():
    logger.critical("CRITICAL: Live API Client is UNAVAILABLE. Live transcription will not function.")
if not llm_handler_instance.is_base_client_available():
    logger.critical("CRITICAL: Base LLM Client is UNAVAILABLE. RDF generation and Query features will not function.")


# --- Register SocketIO Handlers ---
register_handlers(socketio)


# --- Flask Routes ---
@app.route('/')
def index():
    """Generates a new session ID and redirects to the session-specific viewer page."""
    new_session_id = str(uuid.uuid4())
    log_prefix = get_log_prefix(component="Route:/")
    logger.info(f"{log_prefix} New session requested. Redirecting to /session/{new_session_id}")
    return redirect(url_for('session_viewer', session_id=new_session_id))

@app.route('/session/<string:session_id>')
def session_viewer(session_id):
    """Serves the main GRAPH VIEWER page for a specific session."""
    log_prefix = get_log_prefix(session_id, component="Route:/session")
    # Basic validation of session_id format (UUID)
    try:
        uuid.UUID(session_id, version=4)
        logger.info(f"{log_prefix} Serving viewer page.")
        # Pass session_id to the template for client-side connection
        return render_template('viewer.html', session_id=session_id)
    except ValueError:
        logger.error(f"{log_prefix} Invalid session ID format requested: {session_id}")
        return "Invalid session ID format.", 400
    except Exception as e:
        logger.error(f"{log_prefix} Error serving viewer page: {e}", exc_info=True)
        return "Server error.", 500


@app.route('/mobile/<string:session_id>')
def mobile_client(session_id):
    """Serves the MOBILE INTERFACE page for a specific session."""
    log_prefix = get_log_prefix(session_id, component="Route:/mobile")
    # Basic validation of session_id format (UUID)
    try:
        uuid.UUID(session_id, version=4)
        logger.info(f"{log_prefix} Serving mobile page.")
        # Pass session_id to the template
        return render_template('mobile.html', session_id=session_id)
    except ValueError:
        logger.error(f"{log_prefix} Invalid session ID format requested: {session_id}")
        return "Invalid session ID format.", 400
    except Exception as e:
        logger.error(f"{log_prefix} Error serving mobile page: {e}", exc_info=True)
        return "Server error.", 500


# --- Main Execution ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    host = '0.0.0.0' # Listen on all interfaces for container/network deployment
    logger.info(f"--- Starting VoxGraph Server (Session-Based) ---")
    logger.info(f"Attempting to start server on http://{host}:{port}")
    logger.info(f"Using async mode: {socketio.async_mode}")

    # Ensure Eventlet is used if specified
    if socketio.async_mode != 'eventlet':
         logger.warning(f"SocketIO async mode is '{socketio.async_mode}', but 'eventlet' was expected. Monkey patching might not be effective.")

    try:
        # Disable Flask/SocketIO default logging if using custom configured logger
        # use_reloader=False is important for stability, especially with background threads/tasks
        socketio.run(app, debug=False, host=host, port=port, use_reloader=False, log_output=False)
    except OSError as e:
         if "Address already in use" in str(e):
              logger.error(f"FATAL: Port {port} is already in use. Cannot start server.")
         else:
              logger.error(f"FATAL: Failed to start server due to OS error: {e}", exc_info=True)
         sys.exit(1) # Exit with error code
    except Exception as e:
        logger.error(f"FATAL: Unexpected error during server startup: {e}", exc_info=True)
        sys.exit(1) # Exit with error code

    logger.info("--- VoxGraph Server Shutdown ---")