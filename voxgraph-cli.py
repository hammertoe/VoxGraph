# -*- coding: utf-8 -*-
"""
VoxGraph CLI Client - Real-time Microphone Transcription to Knowledge Graph

This script listens to the default microphone input, transcribes it using
Google Generative AI Live API, and sends the transcriptions to the VoxGraph
server via Socket.IO for knowledge graph processing.

Based on patterns from Google Generative AI examples (cli-transcribe.py).
"""

import asyncio
import os
import sys
import traceback
import json
import argparse
import time

import pyaudio
import socketio  # Use python-socketio client
from google import genai # Keep the import
from google.genai import types
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file

# Command line arguments
parser = argparse.ArgumentParser(description='VoxGraph CLI Client')
parser.add_argument('--server', type=str, default='http://localhost:5001',
                    help='Socket.IO server URL (default: http://localhost:5001)')
parser.add_argument('--model', type=str, default='models/gemini-2.0-flash-exp',
                    help='Gemini model to use for transcription (e.g., models/gemini-2.0-flash-exp)')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug output for Socket.IO')
args = parser.parse_args()

# Check for API key - Environment variable is implicitly used by Client()
if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    print("The google.genai Client will likely fail to authenticate.")
    # Allow proceeding, but it will probably fail later
    # sys.exit(1) # Uncomment to force exit if key is missing


# Audio settings (from cli-transcribe)
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024

# Gemini API settings (from cli-transcribe)
MODEL = args.model # Use model from command line args
CONFIG = {
    "response_modalities": ["TEXT"],
    "system_instruction": types.Content(
        parts=[
            types.Part(
                text=
                    """
                    You are a transcription assistant.
                    Transcribe the audio input accurately, preserving meaning.
                    Format transcription as complete sentences when possible.
                    Return nothing else apart from the transcription text.
                    """
            )
        ]
    ),
}

# Initialize PyAudio
pya = pyaudio.PyAudio()

# Initialize Google Generative AI Client directly for live.connect (Alpha API)
# This implicitly uses the GOOGLE_API_KEY environment variable.
try:
    client = genai.Client(http_options={"api_version": "v1alpha"})
    print("Google GenAI Client initialized (using v1alpha for live connect).")
except Exception as e:
    print(f"Error initializing Google GenAI Client: {e}")
    print("Ensure the GOOGLE_API_KEY environment variable is set correctly.")
    sys.exit(1)


# --- Socket.IO Setup ---
sio = socketio.AsyncClient(logger=args.debug, engineio_logger=args.debug)
is_connected = False # Simple flag to track connection

def debug_print(*messages, **kwargs): # Renamed `args` to `messages` for clarity
    global args  # Declare that we are using the global 'args' object
    if args.debug:
        print(*messages, **kwargs) # Use 'messages' to print the actual messages passed in

@sio.event
async def connect():
    global is_connected
    print(f"Connected to VoxGraph server at {args.server} (sid: {sio.sid})")
    is_connected = True

@sio.event
async def connect_error(data):
    global is_connected
    print(f"Connection to VoxGraph server failed: {data}")
    is_connected = False

@sio.event
async def disconnect():
    global is_connected
    print("Disconnected from VoxGraph server")
    is_connected = False

@sio.event
async def message(data): # Example handler for messages *from* server
    debug_print(f"Received message from server: {data}")

async def send_transcription(text):
    """Send a transcription chunk to the VoxGraph server via Socket.IO"""
    global is_connected
    if is_connected and text.strip():
        try:
            await sio.emit('transcription_chunk', {'text': text})
            debug_print(f"Sent to server: {text}")
        except socketio.exceptions.BadNamespaceError:
             print("Error: Not connected to server namespace.")
             is_connected = False # Update status
        except Exception as e:
            print(f"Error sending transcription via Socket.IO: {e}")
    elif not is_connected:
         debug_print("Skipping send, not connected.")


# --- Main Transcription Class (adapted from cli-transcribe) ---

class MicrophoneTranscriber:
    def __init__(self):
        self.audio_out_queue = None
        self.session = None
        self.audio_stream = None
        self._running = True # Flag to control loops

    async def listen_audio(self):
        """Captures audio from the microphone and puts it into the queue."""
        try:
            mic_info = pya.get_default_input_device_info()
            print(f"Using microphone: {mic_info['name']}")

            self.audio_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=self._audio_callback # Efficient callback method
            )

            print("Microphone stream opened. Listening...")
            self.audio_stream.start_stream()

            while self._running and self.audio_stream.is_active():
                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Error in listen_audio: {e}")
            traceback.print_exc()
            self._running = False
        finally:
            print("Stopping microphone listener...")
            if self.audio_stream:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
            if self.audio_out_queue:
                 self.audio_out_queue.put_nowait(None)


    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function. Puts audio data into the async queue."""
        try:
            if self.audio_out_queue:
                self.audio_out_queue.put_nowait({"data": in_data, "mime_type": "audio/pcm"})
        except asyncio.QueueFull:
             print("Warning: Audio queue is full, dropping frame.")
        except Exception as e:
            print(f"Error in audio callback: {e}")
            self._running = False
        return (in_data, pyaudio.paContinue if self._running else pyaudio.paComplete)


    async def send_audio(self):
        """Sends audio chunks from the queue to the Gemini live.connect session."""
        print("Starting audio sender...")
        while self._running:
            try:
                msg = await asyncio.wait_for(self.audio_out_queue.get(), timeout=1.0)

                if msg is None:
                    print("Termination signal received in send_audio.")
                    break

                if self.session:
                    await self.session.send(input=msg)
                else:
                    print("Warning: Session not ready, skipping audio chunk.")
                    await asyncio.sleep(0.1)

                self.audio_out_queue.task_done()

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                print("send_audio task cancelled.")
                break
            except Exception as e:
                print(f"Error in send_audio: {e}")
                traceback.print_exc()
                self._running = False
                break
        print("Audio sender stopped.")


    async def receive_transcription(self):
        """Receives text transcription from Gemini, prints it, and sends to Socket.IO server."""
        print("Starting transcription receiver...")
        current_segment = ""
        while self._running:
            try:
                if not self.session:
                    await asyncio.sleep(0.5)
                    continue

                turn = self.session.receive()
                async for response in turn:
                    if not self._running: break
                    if text := response.text:
                        print(text, end="", flush=True)
                        current_segment += text
                        if text.endswith(('.', '?', '!')) or len(current_segment) > 80:
                            await send_transcription(current_segment.strip())
                            current_segment = ""

                if current_segment.strip() and self._running:
                     print()
                     await send_transcription(current_segment.strip())
                     current_segment = ""

                if not self._running: break

            except asyncio.CancelledError:
                print("\nreceive_transcription task cancelled.")
                break
            except Exception as e:
                print(f"\nError receiving transcription: {e}")
                traceback.print_exc()
                self._running = False
                break

        if current_segment.strip():
            await send_transcription(current_segment.strip())
        print("\nTranscription receiver stopped.")


    async def run(self):
        """Main execution function setting up tasks for audio and transcription."""
        self._running = True
        self.audio_out_queue = asyncio.Queue(maxsize=30)

        try:
            print(f"Connecting to Google Generative AI Live API (Model: {MODEL})...")
            # Use the global 'client' initialized earlier
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                print("Connected to Google API! Press Ctrl+C to stop.")

                async with asyncio.TaskGroup() as tg:
                    print("Creating asyncio tasks...")
                    listen_task = tg.create_task(self.listen_audio())
                    send_task = tg.create_task(self.send_audio())
                    receive_task = tg.create_task(self.receive_transcription())
                    print("Tasks created and running.")

                print("TaskGroup finished.")

        except asyncio.CancelledError:
            print("\nCancellation requested by TaskGroup or external signal. Shutting down...")
        except Exception as e:
             print(f"\nAn error occurred during execution: {e}")
             traceback.print_exc()
        finally:
            print("Initiating cleanup...")
            self._running = False
            await asyncio.sleep(0.5) # Allow tasks to react

            # Cleanup PyAudio
            if self.audio_stream:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                if not self.audio_stream.is_stopped():
                     self.audio_stream.close()
            # Call terminate *after* ensuring streams are closed
            pya.terminate()
            print("PyAudio terminated.")
            print("Transcription process stopped.")


# --- Script Execution ---
if __name__ == "__main__":

    async def connect_socketio():
        global is_connected
        print(f"Attempting to connect to VoxGraph server at {args.server}...")
        try:
            # Removed waits=True argument
            await sio.connect(args.server, wait_timeout=10)
        except socketio.exceptions.ConnectionError as e:
            print(f"Failed to connect to VoxGraph server: {e}")
            is_connected = False
        # Let the main function check is_connected

    async def main():
        await connect_socketio()

        if not is_connected:
            print("Warning: Could not connect to VoxGraph server. Transcription will proceed but data won't be sent.")
            # Decide on behavior: exit or continue without sending
            # return # Uncomment to exit if server connection is essential

        transcriber = MicrophoneTranscriber()
        try:
            await transcriber.run()
        finally:
            if sio.connected:
                print("Disconnecting from VoxGraph server...")
                await sio.disconnect()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Exiting.")
    except Exception as e:
        print(f"\nUnhandled exception in main execution: {e}")
        traceback.print_exc()
    finally:
         # Removed the problematic check: if pya.get_stream_count() > 0 :
         print("Final PyAudio termination check (calling terminate regardless).")
         # Call terminate as a final safeguard if it wasn't called before
         # Note: Calling terminate on an already terminated PyAudio object is safe.
         pya.terminate()
         print("Client shutdown complete.")