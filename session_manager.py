# -*- coding: utf-8 -*-
import uuid
from rdflib import Graph
from queue import Queue as ThreadSafeQueue, Full as QueueFull
from threading import Timer, Lock
import time

from logger_setup import logger, get_log_prefix
import config
from llm_handler import llm_handler_instance # Use the global instance
from rdf_utils import process_turtle_data, update_graph_visualization # Import necessary functions
# Import audio_handler functions later to avoid circular dependency if needed,
# or pass necessary functions/objects during initialization/method calls.

# --- Global Session Storage ---
# Use dictionaries protected by locks for basic thread safety if mixing threads
sessions: dict[str, 'Session'] = {}
sid_to_session: dict[str, str] = {}
_sessions_lock = Lock()
_sid_map_lock = Lock()

class ClientState:
    """Holds the state and resources for a single connected client within a session."""
    def __init__(self, sid: str, session_id: str, socketio):
        self.sid = sid
        self.session_id = session_id
        self.socketio = socketio # Needed for background tasks and emitting errors/updates

        self.log_prefix = get_log_prefix(session_id, sid, "ClientState")

        # Buffers
        self.sentence_buffer: list[str] = [] # Less used now
        self.quick_llm_results: list[str] = []

        # Timers
        self.fast_llm_timer: Timer | None = None
        self.slow_llm_timer: Timer | None = None

        # Audio Processing State
        self.audio_queue = ThreadSafeQueue(maxsize=config.MAX_AUDIO_QUEUE_SIZE)
        self.status_queue = ThreadSafeQueue(maxsize=config.MAX_STATUS_QUEUE_SIZE)
        self.live_session_thread: threading.Thread | None = None
        self.live_session_object = None # Holds the actual Google API session object
        self.is_receiving_audio: bool = False
        self.status_poller_task = None # Holds the eventlet task for polling status queue

        logger.info(f"{self.log_prefix} Initialized.")

    # --- Timer Management ---
    def _clear_timer(self, timer_attr: str):
        """Safely cancels and clears a timer attribute."""
        timer = getattr(self, timer_attr, None)
        if timer:
            try:
                timer.cancel()
                # logger.debug(f"{self.log_prefix} Cancelled timer '{timer_attr}'.")
            except Exception as e:
                logger.warning(f"{self.log_prefix} Error cancelling timer '{timer_attr}': {e}", exc_info=False)
            finally:
                setattr(self, timer_attr, None)

    def schedule_fast_llm_timeout(self):
        """Schedules or reschedules the fast LLM timeout."""
        self._clear_timer('fast_llm_timer')
        timer = Timer(config.FAST_LLM_TIMEOUT, self.flush_sentence_buffer) # Call method on self
        timer.daemon = True
        timer.start()
        self.fast_llm_timer = timer
        # logger.debug(f"{self.log_prefix} Scheduled fast timeout ({config.FAST_LLM_TIMEOUT}s).")

    def schedule_slow_llm_timeout(self):
        """Schedules or reschedules the slow LLM timeout."""
        self._clear_timer('slow_llm_timer')
        timer = Timer(config.SLOW_LLM_TIMEOUT, self.flush_quick_llm_results) # Call method on self
        timer.daemon = True
        timer.start()
        self.slow_llm_timer = timer
        logger.info(f"{self.log_prefix} Scheduled slow analysis timeout ({config.SLOW_LLM_TIMEOUT}s).")

    # --- Buffer Flushing ---
    def flush_sentence_buffer(self):
        """Forces processing of the sentence buffer due to timeout."""
        # This method is less relevant now but kept for potential future use
        self.fast_llm_timer = None # Clear timer flag first
        if not self.sentence_buffer:
            return

        count = len(self.sentence_buffer)
        logger.info(f"{self.log_prefix} Fast timeout flushing {count} sentences (less common path).")
        sentences_to_process = list(self.sentence_buffer)
        self.sentence_buffer.clear()
        text = " ".join(sentences_to_process)

        # Need access to the Session's process_with_quick_llm method
        session = get_session(self.session_id)
        if session:
            # Start background task via session or socketio (socketio is easier here)
            self.socketio.start_background_task(
                session.process_with_quick_llm,
                text,
                self.sid,
                transcription_cid="timeout_flush_sentence" # Indicate source
            )
        else:
             logger.error(f"{self.log_prefix} Session not found during sentence buffer flush.")

    def flush_quick_llm_results(self):
        """Forces processing of the quick LLM results buffer due to timeout."""
        self._clear_timer('slow_llm_timer') # Clear timer flag first
        if not self.quick_llm_results:
            return

        count = len(self.quick_llm_results)
        logger.info(f"{self.log_prefix} Slow timeout flushing {count} quick LLM result chunks.")
        results_to_process = list(self.quick_llm_results)
        self.quick_llm_results.clear()
        combined_turtle = "\n\n".join(results_to_process)

        # Need access to the Session's process_with_slow_llm method
        session = get_session(self.session_id)
        if session:
             # Start background task via session or socketio
            self.socketio.start_background_task(
                session.process_with_slow_llm,
                combined_turtle,
                self.sid
            )
        else:
            logger.error(f"{self.log_prefix} Session not found during quick results buffer flush.")


    # --- Chunk Checking ---
    def check_slow_llm_chunk(self):
        """Checks if the quick LLM results buffer is full and processes it."""
        count = len(self.quick_llm_results)

        if count >= config.SLOW_LLM_CHUNK_SIZE:
            logger.info(f"{self.log_prefix} Slow analysis chunk size ({count}/{config.SLOW_LLM_CHUNK_SIZE}) reached.")
            self._clear_timer('slow_llm_timer') # Cancel pending timeout

            results_to_process = list(self.quick_llm_results)
            self.quick_llm_results.clear()
            combined_turtle = "\n\n".join(results_to_process)

            # Need access to the Session's process_with_slow_llm method
            session = get_session(self.session_id)
            if session:
                self.socketio.start_background_task(
                    session.process_with_slow_llm,
                    combined_turtle,
                    self.sid
                )
            else:
                 logger.error(f"{self.log_prefix} Session not found during slow chunk check.")

        # If buffer has items but not full, ensure timeout is scheduled (or rescheduled)
        elif count > 0:
             if not self.slow_llm_timer:
                 self.schedule_slow_llm_timeout()
        # If buffer is empty, clear any existing timer
        elif count == 0:
             self._clear_timer('slow_llm_timer')


    def cleanup(self):
        """Cleans up resources associated with this client."""
        logger.info(f"{self.log_prefix} Cleaning up client resources.")
        # Cancel timers
        self._clear_timer('fast_llm_timer')
        self._clear_timer('slow_llm_timer')

        # Stop audio processing (sets flag, signals queue, joins thread)
        # Need the audio_handler.terminate_audio_session function here
        from audio_handler import terminate_audio_session # Local import okay here
        terminate_audio_session(self.session_id, self.sid, self) # Pass self (ClientState)

        # Clear buffers and queues (termination should handle queues, but clear for safety)
        self.sentence_buffer.clear()
        self.quick_llm_results.clear()
        try:
            while not self.audio_queue.empty(): self.audio_queue.get_nowait()
        except: pass
        try:
             while not self.status_queue.empty(): self.status_queue.get_nowait()
        except: pass

        # Status poller task should stop itself when it sees the client is gone from session

        logger.info(f"{self.log_prefix} Client resource cleanup complete.")


class Session:
    """Represents a single user session with its graph, LLM chats, and clients."""
    def __init__(self, session_id: str, socketio):
        self.session_id = session_id
        self.socketio = socketio # Needed for background tasks, graph updates
        self.graph = Graph()
        self.clients: dict[str, ClientState] = {}
        self.log_prefix = get_log_prefix(session_id, component="Session")

        # LLM Chat objects (managed by LLMHandler)
        self.quick_chat = None
        self.slow_chat = None
        self.query_chat = None
        self._llms_initialized = False

        self._bind_namespaces()
        self._initialize_llms() # Attempt initialization on creation

        logger.info(f"{self.log_prefix} Created.")

    def _bind_namespaces(self):
        """Binds standard prefixes to the session graph."""
        for prefix, namespace in config.NAMESPACES.items():
            self.graph.bind(prefix, namespace)
        logger.debug(f"{self.log_prefix} Namespaces bound to graph.")

    def _initialize_llms(self):
        """Uses the LLMHandler to initialize chat models for this session."""
        if not self._llms_initialized:
            # Pass the session's state dictionary (self.__dict__) for modification
            success = llm_handler_instance.initialize_session_chats(self.__dict__, self.session_id)
            self._llms_initialized = True # Mark as attempted even if failed
            if not success:
                 logger.warning(f"{self.log_prefix} One or more LLM chat models failed to initialize.")
            else:
                 logger.info(f"{self.log_prefix} LLM chat models initialized (or already existed).")


    def add_client(self, sid: str) -> ClientState:
        """Adds a new client to the session."""
        log_prefix = get_log_prefix(self.session_id, sid, "Session")
        if sid in self.clients:
            logger.warning(f"{log_prefix} Client already exists in session. Re-initializing client state.")
            self.clients[sid].cleanup() # Clean up old state first

        client_state = ClientState(sid, self.session_id, self.socketio)
        self.clients[sid] = client_state
        logger.info(f"{log_prefix} Client added. Total clients: {len(self.clients)}")
        return client_state

    def remove_client(self, sid: str):
        """Removes a client from the session and cleans up its resources."""
        log_prefix = get_log_prefix(self.session_id, sid, "Session")
        if sid in self.clients:
            logger.info(f"{log_prefix} Removing client...")
            client_state = self.clients.pop(sid)
            client_state.cleanup() # Trigger resource cleanup
            logger.info(f"{log_prefix} Client removed. Remaining clients: {len(self.clients)}")
        else:
            logger.warning(f"{log_prefix} Attempted to remove non-existent client.")

    def get_client(self, sid: str) -> ClientState | None:
        """Gets the state object for a specific client SID."""
        return self.clients.get(sid)

    def get_graph(self) -> Graph:
        """Returns the RDF graph for this session."""
        return self.graph

    def is_empty(self) -> bool:
        """Checks if the session has any active clients."""
        return not self.clients

    # --- LLM Processing Methods ---
    def process_with_quick_llm(self, text_chunk: str, sid: str, transcription_cid: str | None = None):
        """Processes a text chunk with the Quick LLM."""
        client_log_prefix = get_log_prefix(self.session_id, sid, "QuickLLM")
        client_state = self.get_client(sid)

        if not client_state:
            logger.error(f"{client_log_prefix} Client state not found for processing.")
            return
        if not self.quick_chat:
            logger.error(f"{client_log_prefix} Quick LLM chat is not available for this session.")
            self.socketio.emit('error', {'message': 'RDF generation service unavailable.'}, room=sid)
            return

        # Format message using CID if available
        user_message = f"[{transcription_cid if transcription_cid else 'NO_CID'}] {text_chunk}"
        logger.info(f"{client_log_prefix} Processing (CID: {transcription_cid}): '{text_chunk[:100]}...'")

        try:
            # Ensure LLMs are initialized (should be, but double-check)
            self._initialize_llms()
            if not self.quick_chat: raise RuntimeError("Quick LLM chat failed initialization.")

            response = self.quick_chat.send_message(user_message)
            turtle_response = response.text
            logger.info(f"{client_log_prefix} Response received (CID: {transcription_cid}).")

            # Process the Turtle data and update the session graph
            triples_added = process_turtle_data(turtle_response, self.graph, self.session_id, sid)

            if triples_added:
                # Add result to client buffer for potential slow LLM processing
                client_state.quick_llm_results.append(turtle_response)
                logger.info(f"{client_log_prefix} Added Quick result. Buffer size: {len(client_state.quick_llm_results)}")
                # Update graph visualization for all clients in the session
                update_graph_visualization(self.socketio, self.session_id, self.graph)
                # Check if slow LLM chunk is ready
                client_state.check_slow_llm_chunk()
            else:
                 logger.info(f"{client_log_prefix} No new triples added from Quick LLM response (CID: {transcription_cid}).")

        except Exception as e:
            logger.error(f"{client_log_prefix} Error processing with Quick LLM (CID: {transcription_cid}): {e}", exc_info=True)
            self.socketio.emit('error', {'message': f'Error generating RDF: {e}'}, room=sid)


    def process_with_slow_llm(self, combined_quick_results_turtle: str, sid: str):
        """Processes combined Turtle results with the Slow LLM."""
        client_log_prefix = get_log_prefix(self.session_id, sid, "SlowLLM")
        client_state = self.get_client(sid)

        if not client_state:
            logger.error(f"{client_log_prefix} Client state not found for processing.")
            return
        if not self.slow_chat:
            logger.error(f"{client_log_prefix} Slow LLM chat is not available for this session.")
            # Don't emit error to client for background analysis failure
            return

        logger.info(f"{client_log_prefix} Processing {len(combined_quick_results_turtle.splitlines())} lines from quick results.")

        try:
             # Ensure LLMs are initialized
            self._initialize_llms()
            if not self.slow_chat: raise RuntimeError("Slow LLM chat failed initialization.")

            # Serialize current graph state (with truncation)
            current_graph_turtle = self.graph.serialize(format='turtle')
            if len(current_graph_turtle) > config.MAX_GRAPH_CONTEXT_SLOW:
                logger.warning(f"{client_log_prefix} Truncating Slow LLM graph context ({len(current_graph_turtle)} > {config.MAX_GRAPH_CONTEXT_SLOW}).")
                current_graph_turtle = current_graph_turtle[-config.MAX_GRAPH_CONTEXT_SLOW:] # Keep the end

            # Construct input prompt
            slow_llm_input = (
                f"Existing Knowledge Graph (Session: {self.session_id[:6]}):\n```turtle\n{current_graph_turtle}\n```\n\n"
                f"New Information/Triples:\n```turtle\n{combined_quick_results_turtle}\n```\n\n"
                "Analyze the new information in the context of the existing graph..."
            )

            response = self.slow_chat.send_message(slow_llm_input)
            turtle_response = response.text
            logger.info(f"{client_log_prefix} Response received.")

            # Process the Turtle data and update the session graph
            triples_added = process_turtle_data(turtle_response, self.graph, self.session_id, sid)
            if triples_added:
                update_graph_visualization(self.socketio, self.session_id, self.graph)
            else:
                 logger.info(f"{client_log_prefix} No new triples added from Slow LLM analysis.")

        except Exception as e:
            logger.error(f"{client_log_prefix} Error processing with Slow LLM: {e}", exc_info=True)
            # Maybe emit a subtle warning? For now, just log.


    def process_query(self, query_text: str, sid: str):
        """Processes a user query against the session graph using the Query LLM."""
        client_log_prefix = get_log_prefix(self.session_id, sid, "QueryLLM")
        client_state = self.get_client(sid)

        if not client_state:
            logger.error(f"{client_log_prefix} Client state not found for query.")
            self.socketio.emit('query_result', {'answer': "Error: Client session state lost.", 'error': True}, room=sid)
            return
        if not self.query_chat:
            logger.error(f"{client_log_prefix} Query LLM chat is not available for this session.")
            self.socketio.emit('query_result', {'answer': "Error: Query service unavailable.", 'error': True}, room=sid)
            return

        logger.info(f"{client_log_prefix} Processing query: '{query_text}'")
        self.socketio.emit('query_result', {'answer': "Processing query against session graph...", 'processing': True}, room=sid)

        try:
            # Ensure LLMs are initialized
            self._initialize_llms()
            if not self.query_chat: raise RuntimeError("Query LLM chat failed initialization.")

            # Serialize current graph state (with truncation)
            current_graph_turtle = self.graph.serialize(format='turtle')
            if len(current_graph_turtle) > config.MAX_GRAPH_CONTEXT_QUERY:
                logger.warning(f"{client_log_prefix} Truncating Query LLM graph context ({len(current_graph_turtle)} > {config.MAX_GRAPH_CONTEXT_QUERY}).")
                current_graph_turtle = current_graph_turtle[-config.MAX_GRAPH_CONTEXT_QUERY:]

            # Construct input prompt
            query_prompt = (
                f"Knowledge Graph (Session: {self.session_id[:6]}):\n```turtle\n{current_graph_turtle}\n```\n\n"
                f"User Query: \"{query_text}\"\n\n"
                "---\nBased *only* on the graph..."
            )

            # --- Define background task ---
            def run_query_task():
                task_log_prefix = get_log_prefix(self.session_id, sid, "QueryTask")
                try:
                    logger.info(f"{task_log_prefix} Sending query to LLM...")
                    response = self.query_chat.send_message(query_prompt)
                    answer = response.text
                    logger.info(f"{task_log_prefix} Query response received.")
                    self.socketio.emit('query_result', {'answer': answer, 'error': False}, room=sid)
                except Exception as task_e:
                    logger.error(f"{task_log_prefix} Query LLM error: {task_e}", exc_info=True)
                    self.socketio.emit('query_result', {'answer': f"Error processing query: {task_e}", 'error': True}, room=sid)
            # --- Start background task ---
            self.socketio.start_background_task(run_query_task)

        except Exception as e:
            logger.error(f"{client_log_prefix} Error preparing query or starting task: {e}", exc_info=True)
            self.socketio.emit('query_result', {'answer': f"Server error preparing query: {e}", 'error': True}, room=sid)


# --- Session Management Functions ---

def get_session(session_id: str) -> Session | None:
    """Retrieves a session object by its ID."""
    with _sessions_lock:
        return sessions.get(session_id)

def get_or_create_session(session_id: str, socketio) -> Session:
    """Gets an existing session or creates a new one."""
    with _sessions_lock:
        if session_id not in sessions:
            logger.info(f"[SessionMgr] Creating new session: {session_id}")
            sessions[session_id] = Session(session_id, socketio)
        else:
             logger.debug(f"[SessionMgr] Rejoining existing session: {session_id}")
        return sessions[session_id]

def remove_session_if_empty(session_id: str):
    """Removes a session if it no longer has any clients."""
    with _sessions_lock:
        session = sessions.get(session_id)
        if session and session.is_empty():
            logger.info(f"[SessionMgr] Session {session_id} is empty. Removing.")
            del sessions[session_id]
            # Optional: Add more cleanup here if needed (e.g., explicitly close LLM chats?)

def get_session_id_for_sid(sid: str) -> str | None:
    """Finds the session ID associated with a client SID."""
    with _sid_map_lock:
        return sid_to_session.get(sid)

def map_sid_to_session(sid: str, session_id: str):
    """Maps a client SID to a session ID."""
    with _sid_map_lock:
        sid_to_session[sid] = session_id

def unmap_sid(sid: str) -> str | None:
    """Removes the SID mapping and returns the session ID it was mapped to."""
    with _sid_map_lock:
        return sid_to_session.pop(sid, None)

def get_client_state(sid: str) -> ClientState | None:
    """Convenience function to get the ClientState object directly from an SID."""
    session_id = get_session_id_for_sid(sid)
    if session_id:
        session = get_session(session_id)
        if session:
            return session.get_client(sid)
    return None
