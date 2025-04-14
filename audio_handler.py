# -*- coding: utf-8 -*-
import asyncio
import threading
import time
from queue import Queue as ThreadSafeQueue, Empty as QueueEmpty, Full as QueueFull
import websockets.exceptions
from google.api_core import exceptions as google_exceptions

import config
from logger_setup import logger, get_log_prefix
from llm_handler import llm_handler_instance # Need the live client
# Avoid importing session_manager here if possible to prevent cycles.
# Pass necessary state (client_state) or specific objects (queues, flags)

# --- Helper Function ---
def put_status_update(status_queue: ThreadSafeQueue | None, update_dict: dict, log_ctx: str = "[StatusQueue]"):
    """Safely puts status update messages onto the thread-safe queue."""
    if not status_queue:
        logger.warning(f"{log_ctx} Status queue is None, cannot put update: {update_dict.get('event')}")
        return
    try:
        status_queue.put_nowait(update_dict)
    except QueueFull:
        logger.warning(f"{log_ctx} Queue full, dropping status update: {update_dict.get('event')}")
    except Exception as e:
        logger.error(f"{log_ctx} Error putting status update: {e}")

# --- Core Async Tasks (Run within the manager thread's loop) ---

async def live_api_sender(session_id: str, sid: str, client_state: 'ClientState', session_object):
    """Async task sending audio from the queue to Google, checking termination."""
    log_prefix = get_log_prefix(session_id, sid, "Sender")
    audio_queue = client_state.audio_queue
    status_queue = client_state.status_queue
    logger.info(f"{log_prefix} Starting...")
    is_active = True

    while is_active:
        try:
            # Check termination flag *before* blocking queue get
            if not client_state.is_receiving_audio:
                logger.info(f"{log_prefix} Termination flag detected (before get). Stopping.")
                is_active = False
                break

            # Get audio chunk with timeout to allow checking flag periodically
            try:
                msg = audio_queue.get(block=True, timeout=0.5) # Block with timeout
            except QueueEmpty:
                continue # Timeout, loop back and check flag

            # Check termination flag *after* getting from queue
            if not client_state.is_receiving_audio:
                logger.info(f"{log_prefix} Termination flag detected (after get). Stopping.")
                if msg is not None: audio_queue.task_done() # Mark task done if we got data
                is_active = False
                break

            if msg is None:
                logger.info(f"{log_prefix} Received termination signal (None) from queue.")
                is_active = False
                audio_queue.task_done() # Mark None task done
                break

            # Process audio message if session is valid
            if session_object and getattr(session_object, 'send', None):
                # logger.debug(f"{log_prefix} Sending audio chunk ({len(msg.get('data',b''))} bytes)")
                await session_object.send(input=msg)
                # Minimal sleep to yield control, prevent hogging loop
                await asyncio.sleep(0.001)
            else:
                logger.warning(f"{log_prefix} Google API session object is invalid or closed. Cannot send.")
                # If session is bad, wait a bit longer before trying again
                await asyncio.sleep(0.1)

            audio_queue.task_done() # Mark audio data task done

        except asyncio.CancelledError:
            logger.info(f"{log_prefix} Cancelled.")
            is_active = False
        except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError) as e:
             logger.warning(f"{log_prefix} WebSocket connection closed by Google during send: {e}. Signalling connection lost.")
             put_status_update(status_queue, {'event': 'connection_lost', 'data': {'message': f'Google send connection lost: {e}'}}, log_prefix)
             is_active = False # Let manager handle reconnect
        except Exception as e:
            logger.error(f"{log_prefix} Unexpected error: {e}", exc_info=True)
            put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Audio Send Error: {e}'}}, log_prefix)
            is_active = False # Stop sender on unexpected error

    logger.info(f"{log_prefix} Stopped.")


async def live_api_receiver(session_id: str, sid: str, client_state: 'ClientState', session_object):
    """Async task receiving transcriptions from Google, checking termination."""
    log_prefix = get_log_prefix(session_id, sid, "Receiver")
    status_queue = client_state.status_queue
    logger.info(f"{log_prefix} Starting...")
    is_active = True
    current_segment = ""

    while is_active:
        try:
            # Check termination flag
            if not client_state.is_receiving_audio:
                 logger.info(f"{log_prefix} Termination flag detected. Stopping.")
                 is_active = False
                 break

            if not session_object or not getattr(session_object, 'receive', None):
                 logger.warning(f"{log_prefix} Google API session object invalid or closed. Cannot receive.")
                 await asyncio.sleep(0.5) # Wait before checking again
                 continue

            # Receive data from Google asynchronously
            turn = session_object.receive()
            async for response in turn:
                 # Check flag again inside the loop (important!)
                 if not client_state.is_receiving_audio:
                    logger.info(f"{log_prefix} Termination flag detected (inside receive loop). Stopping.")
                    is_active = False
                    break # Break inner async for loop

                 # Process response text if available
                 if text := response.text:
                    # logger.debug(f"{log_prefix} Received text fragment: '{text}'")
                    current_segment += text
                    # Simple segmentation: end of sentence punctuation or length limit
                    # Consider adding newline as well if Google uses it
                    if text.endswith(('.', '?', '!')) or len(current_segment) > 120: # Increased length slightly
                        segment = current_segment.strip()
                        current_segment = ""
                        if segment:
                            # logger.debug(f"{log_prefix} Putting segment to status queue: '{segment}'")
                            put_status_update(status_queue, {'event': 'new_transcription', 'data': {'text': segment}}, log_prefix)

            # --- After turn finishes (or breaks) ---
            if not is_active: break # Exit outer loop if inner loop broke due to flag

            # Send any remaining partial segment *if* client is still active
            if client_state.is_receiving_audio and current_segment.strip():
                 segment = current_segment.strip()
                 logger.debug(f"{log_prefix} Putting final segment from turn: '{segment}'")
                 put_status_update(status_queue, {'event': 'new_transcription', 'data': {'text': segment}}, log_prefix)
            elif not client_state.is_receiving_audio:
                 logger.info(f"{log_prefix} Client stopped before sending final segment from turn.")

            current_segment = "" # Reset segment for next turn
            await asyncio.sleep(0.01) # Small yield between turns

        except asyncio.CancelledError:
            logger.info(f"{log_prefix} Cancelled.")
            is_active = False
        except (websockets.exceptions.ConnectionClosedOK, websockets.exceptions.ConnectionClosedError) as e:
            logger.warning(f"{log_prefix} WebSocket connection closed during receive: {e}")
            put_status_update(status_queue, {'event': 'connection_lost', 'data': {'message': f'Google receive connection lost: {e}'}}, log_prefix)
            is_active = False # Let manager handle reconnect
        except google_exceptions.DeadlineExceeded:
             logger.warning(f"{log_prefix} Google API Deadline Exceeded during receive. Ending turn.")
             # Often indicates end of speech or network issue, let manager decide on reconnect
             put_status_update(status_queue, {'event': 'connection_lost', 'data': {'message': 'Google service timed out. Reconnecting...'}}, log_prefix)
             is_active = False
        except Exception as e:
            logger.error(f"{log_prefix} Unexpected error: {e}", exc_info=True)
            put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Receive Error: {e}'}}, log_prefix)
            is_active = False # Stop receiving on error

    # Final check for any remaining segment if loop exited unexpectedly
    if current_segment.strip():
        logger.info(f"{log_prefix} Putting final remaining segment after loop exit: '{current_segment.strip()}'")
        put_status_update(status_queue, {'event': 'new_transcription', 'data': {'text': current_segment.strip()}}, log_prefix)

    logger.info(f"{log_prefix} Stopped.")


async def manage_live_session(session_id: str, sid: str, client_state: 'ClientState'):
    """Manages the Google Live API connection lifecycle (reconnects, runs tasks)."""
    log_prefix = get_log_prefix(session_id, sid, "Manager")
    thread_id = threading.get_ident() # For checking if thread is superseded
    logger.info(f"{log_prefix} Async manager starting (Thread ID: {thread_id}).")

    status_queue = client_state.status_queue
    audio_queue = client_state.audio_queue
    max_retries = config.AUDIO_MANAGER_MAX_RETRIES
    retry_count = 0
    base_delay = config.AUDIO_MANAGER_RETRY_BASE_DELAY

    # Initial checks
    if not client_state.is_receiving_audio:
        logger.warning(f"{log_prefix} Client no longer active at manager start.")
        return
    if not llm_handler_instance.is_live_client_available():
        logger.error(f"{log_prefix} Google Live API client is not available.")
        put_status_update(status_queue, {'event': 'error', 'data': {'message': 'Server config error: Live transcription unavailable.'}}, log_prefix)
        client_state.is_receiving_audio = False # Ensure flag is off
        return

    # Main connection and task loop
    while retry_count <= max_retries:
        # --- Check Flags at Start of Loop ---
        if not client_state.is_receiving_audio:
            logger.info(f"{log_prefix} Client stopped (flag check). Exiting manager loop.")
            break
        # Check if this thread is still the designated one
        current_thread_ref = client_state.live_session_thread
        if not current_thread_ref or current_thread_ref.ident != thread_id:
            logger.warning(f"{log_prefix} Thread superseded (Expected: {current_thread_ref.ident if current_thread_ref else 'None'}, Actual: {thread_id}). Exiting.")
            break

        session_object = None # Reset session object for this attempt
        try:
            # --- Retry Delay Logic ---
            if retry_count > 0:
                delay = min(base_delay * (2 ** (retry_count - 1)), 15) # Exponential backoff capped
                logger.info(f"{log_prefix} Retry {retry_count}/{max_retries}. Waiting {delay:.1f}s...")
                put_status_update(status_queue, {'event': 'reconnecting', 'data': {'attempt': retry_count, 'max': max_retries, 'delay': delay}}, log_prefix)
                # Wait asynchronously, checking termination flag periodically
                start_time = time.time()
                while (time.time() - start_time < delay):
                    if not client_state.is_receiving_audio: break # Break wait loop if stopped
                    await asyncio.sleep(0.1)
                if not client_state.is_receiving_audio:
                    logger.info(f"{log_prefix} Client stopped during retry delay.")
                    break # Break outer manager loop

            # --- Establish Connection ---
            logger.info(f"{log_prefix} Attempting connection (Attempt {retry_count + 1})...")
            live_api_config = llm_handler_instance.get_live_api_config()
            # Use the live client from the handler instance
            async with llm_handler_instance.live_client.aio.live.connect(
                model=config.GOOGLE_LIVE_API_MODEL, config=live_api_config
            ) as session:
                session_object = session
                client_state.live_session_object = session_object # Store active session
                logger.info(f"{log_prefix} Connection established successfully.")

                # Check state *after* connection, before starting tasks
                if not client_state.is_receiving_audio:
                    logger.info(f"{log_prefix} Client stopped immediately after connection. Closing.")
                    break # Exit manager loop

                # Notify client of success
                event_type = 'reconnected' if retry_count > 0 else 'audio_started'
                put_status_update(status_queue, {'event': event_type, 'data': {}}, log_prefix)
                if retry_count > 0: logger.info(f"{log_prefix} Successfully reconnected.")
                retry_count = 0 # Reset retries on success

                # --- Run Sender/Receiver Tasks Concurrently ---
                try:
                    # Use TaskGroup for structured concurrency
                    async with asyncio.TaskGroup() as tg:
                        logger.info(f"{log_prefix} Creating sender and receiver tasks.")
                        receiver_task = tg.create_task(live_api_receiver(session_id, sid, client_state, session_object))
                        sender_task = tg.create_task(live_api_sender(session_id, sid, client_state, session_object))
                        logger.info(f"{log_prefix} Sender/Receiver tasks running.")

                        # TaskGroup implicitly waits for all tasks. We just need to monitor the flag.
                        while client_state.is_receiving_audio:
                            await asyncio.sleep(0.2) # Check flag periodically

                        # If loop exits, client stopped. TaskGroup will handle cancellation on exit.
                        logger.info(f"{log_prefix} Termination flag detected. Exiting TaskGroup (will cancel tasks).")

                except* asyncio.CancelledError: # Handle TaskGroup cancellation (expected on stop)
                    logger.info(f"{log_prefix} Task group cancelled (likely due to client stop).")
                    # Propagate cancellation to exit manager loop cleanly
                    raise asyncio.CancelledError("Client stopped")
                except* Exception as group_e: # Handle errors within tasks
                    # TaskGroup gathers exceptions. Log them.
                    logger.error(f"{log_prefix} Error within task group: {group_e.exceptions}", exc_info=False) # Log the collected exceptions
                    # Increment retry count and continue to the outer loop
                    retry_count += 1
                    # Clear the potentially invalid session object ref before retry
                    client_state.live_session_object = None

            # --- Post-Session Handling (Clean Exit from `async with session`) ---
            logger.warning(f"{log_prefix} Google session ended cleanly or tasks completed unexpectedly. Incrementing retry count.")
            client_state.live_session_object = None # Clear session ref
            retry_count += 1

        # --- Exception Handling for Connection/Setup ---
        except (websockets.exceptions.ConnectionClosedError, google_exceptions.GoogleAPIError) as e:
            retry_count += 1
            logger.warning(f"{log_prefix} Connection failed or closed during setup: {e}. Retry {retry_count}/{max_retries}")
            put_status_update(status_queue, {'event': 'connection_lost', 'data': {'message': f'Connection failed: {e}'}}, log_prefix)
            client_state.live_session_object = None # Clear potentially bad ref
            continue # To next retry iteration
        except asyncio.CancelledError:
            logger.info(f"{log_prefix} Manager task cancelled (likely client stop).")
            break # Exit manager loop cleanly
        except Exception as e:
            retry_count += 1
            logger.error(f"{log_prefix} Unexpected error during connection/management: {e}", exc_info=True)
            put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Live session error: {e}'}}, log_prefix)
            client_state.live_session_object = None # Clear potentially bad ref
            if retry_count > max_retries:
                logger.error(f"{log_prefix} Max retries ({max_retries}) reached. Giving up.")
                put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Connection failed after {max_retries} attempts.'}}, log_prefix)
                break # Exit manager loop

    # --- Final Cleanup Actions ---
    logger.info(f"{log_prefix} Exiting manager loop (Thread ID: {thread_id}). Final cleanup.")

    # Ensure client state reflects stoppage
    client_state.is_receiving_audio = False
    client_state.live_session_object = None # Clear final reference

    # Signal audio queue to unblock sender if it's waiting on queue.get()
    try:
        audio_queue.put_nowait(None)
    except QueueFull: logger.warning(f"{log_prefix} Audio queue full during final signal.")
    except Exception: pass # Ignore other errors here

    # Send a final 'audio_stopped' event unless client explicitly requested stop elsewhere
    # Check flag one last time
    put_status_update(status_queue, {'event': 'audio_stopped', 'data': {'message': 'Session terminated or failed.'}}, log_prefix)

    logger.info(f"{log_prefix} Async manager finished (Thread ID: {thread_id}).")


def run_async_session_manager(session_id: str, sid: str, client_state: 'ClientState'):
    """Wrapper function to run the asyncio manager in a separate thread."""
    log_prefix = get_log_prefix(session_id, sid, "AsyncRunner")
    thread_id = threading.get_ident()
    threading.current_thread().name = f"AudioMgr-{session_id[:6]}-{sid[:6]}"
    logger.info(f"{log_prefix} Thread started (ID: {thread_id}).")

    # Double-check if still active before starting loop
    if not client_state or not client_state.is_receiving_audio:
        logger.warning(f"{log_prefix} Client stopped or invalid before async loop started.")
        # Clean up thread reference if it points to this dying thread
        if client_state and client_state.live_session_thread and client_state.live_session_thread.ident == thread_id:
             client_state.live_session_thread = None
        return

    status_queue = client_state.status_queue # For error reporting from this wrapper

    # Create and manage asyncio loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    logger.info(f"{log_prefix} Created new asyncio loop for thread.")

    try:
        # Run the core session management logic until it completes
        loop.run_until_complete(manage_live_session(session_id, sid, client_state))
    except Exception as e:
        logger.error(f"{log_prefix} Unhandled error in async manager's main loop execution: {e}", exc_info=True)
        client_state.is_receiving_audio = False # Ensure stopped state
        put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Critical session error: {e}'}}, log_prefix)
    finally:
        # Graceful asyncio loop cleanup
        logger.info(f"{log_prefix} Shutting down async tasks and closing loop.")
        try:
            # Shutdown async generators first
            loop.run_until_complete(loop.shutdown_asyncgens())

            # Cancel any remaining tasks in the loop
            pending = asyncio.all_tasks(loop=loop)
            if pending:
                 logger.warning(f"{log_prefix} Cancelling {len(pending)} outstanding tasks.")
                 for task in pending: task.cancel()
                 # Wait for tasks to finish cancellation
                 loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            # Close the loop
            loop.close()
            logger.info(f"{log_prefix} Loop closed.")
        except Exception as close_err:
            logger.error(f"{log_prefix} Error during loop cleanup: {close_err}", exc_info=True)

        # Final state cleanup: Remove thread reference *only* if it's still pointing to this thread
        if client_state: # Check if client_state object still exists
            current_thread_ref = client_state.live_session_thread
            if current_thread_ref and current_thread_ref.ident == thread_id:
                logger.info(f"{log_prefix} Clearing own thread reference from ClientState.")
                client_state.live_session_thread = None
                client_state.live_session_object = None # Ensure session object is cleared
                client_state.is_receiving_audio = False # Explicitly set flag off
            else:
                # This might happen if stop_audio was called concurrently
                logger.info(f"{log_prefix} Not clearing thread reference - likely superseded or already cleaned up by terminate call.")
        else:
             logger.info(f"{log_prefix} Client state object disappeared during cleanup.")

        logger.info(f"{log_prefix} Thread finishing (ID: {thread_id}).")


def terminate_audio_session(session_id: str, sid: str, client_state: 'ClientState', wait_time: float = config.AUDIO_TERMINATION_WAIT):
    """Forcibly terminate the audio session for a client and clean up resources."""
    log_prefix = get_log_prefix(session_id, sid, "Terminator")

    if not client_state:
        logger.warning(f"{log_prefix} No client state found to terminate.")
        return

    # Step 1: Signal termination via flag
    was_receiving = client_state.is_receiving_audio
    client_state.is_receiving_audio = False
    logger.info(f"{log_prefix} Setting is_receiving_audio = False (was {was_receiving})")

    # Step 2: Signal termination via audio queue (unblocks sender)
    audio_queue = client_state.audio_queue
    if audio_queue:
        try:
             # Clear existing items first? Optional, None signal should suffice.
             # while not audio_queue.empty(): audio_queue.get_nowait(); audio_queue.task_done()
             audio_queue.put_nowait(None) # Signal termination
             logger.debug(f"{log_prefix} Put None signal onto audio queue.")
        except QueueFull:
             logger.warning(f"{log_prefix} Audio queue full, couldn't put None signal.")
        except Exception as e: logger.warning(f"{log_prefix} Error signalling audio queue: {e}")

    # Step 3: Wait for the manager thread to finish
    thread = client_state.live_session_thread
    if thread and thread.is_alive():
        logger.info(f"{log_prefix} Waiting up to {wait_time}s for audio manager thread (ID: {thread.ident}) to terminate...")
        thread.join(timeout=wait_time)
        if thread.is_alive():
            logger.warning(f"{log_prefix} Audio manager thread (ID: {thread.ident}) did NOT terminate in time!")
            # What to do here? Thread might be stuck. OS will kill eventually if daemon.
        else:
            logger.info(f"{log_prefix} Audio manager thread (ID: {thread.ident}) terminated successfully.")
    elif thread:
        logger.info(f"{log_prefix} Audio manager thread (ID: {thread.ident}) was already finished.")

    # Step 4: Clean up references (thread should clean itself, but do it here too for safety)
    client_state.live_session_thread = None
    client_state.live_session_object = None

    # Step 5: Clear queues again (manager might have put status updates after flag check)
    # Note: The poller might miss the very last status updates if cleared here.
    # Maybe rely on the poller stopping naturally when client state is removed.
    # For robustness, clear them.
    # try:
    #     while not audio_queue.empty(): audio_queue.get_nowait(); audio_queue.task_done()
    # except: pass
    # try:
    #      while not status_queue.empty(): status_queue.get_nowait(); status_queue.task_done()
    # except: pass

    logger.info(f"{log_prefix} Audio session termination sequence complete for client.")