# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
# (License header from original example retained)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Real-time Microphone Transcription using Google Generative AI Live API.

This script listens to the default microphone input, sends the audio
stream to the Google Generative AI API, and prints the real-time
transcription to the console.
"""

import asyncio
import os
import sys
import traceback

import pyaudio
from google import genai

from dotenv import load_dotenv

# --- Configuration ---
# Ensure you have the necessary libraries installed:
# pip install google-genai pyaudio

# Before running, set the GOOGLE_API_KEY environment variable:
# export GOOGLE_API_KEY="YOUR_API_KEY"

load_dotenv()  # Load environment variables from .env file

if "GOOGLE_API_KEY" not in os.environ:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    sys.exit(1)

# Audio settings
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
SEND_SAMPLE_RATE = 16000  # Sample rate for audio sent to the API
CHUNK_SIZE = 1024  # Size of audio chunks to process

# Gemini API settings
MODEL = "models/gemini-2.0-flash-exp"  # Or another suitable model
# Configure the API to expect audio input and respond with text transcription
CONFIG = {
    "response_modalities": ["TEXT"], 
    "system_instruction": "You are a transcription assistant. Transcribe the audio input. Return nothing else apart from the transcription."
    }

# --- Main Transcription Class ---

# Handle compatibility for older Python versions if needed
if sys.version_info < (3, 11, 0):
    try:
        import taskgroup
        import exceptiongroup
        asyncio.TaskGroup = taskgroup.TaskGroup
        asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup
    except ImportError:
        print("Warning: taskgroup/exceptiongroup libraries not found. "
              "Required for Python < 3.11 for robust error handling.")
        # Basic fallback, might not handle multiple exceptions as gracefully
        asyncio.TaskGroup = asyncio.gather
        asyncio.ExceptionGroup = Exception # Simple fallback

# Initialize PyAudio
pya = pyaudio.PyAudio()

# Initialize Google Generative AI Client
# Using alpha version for live.connect
client = genai.Client(http_options={"api_version": "v1alpha"})


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
                stream_callback=self._audio_callback # More efficient callback method
            )

            print("Microphone stream opened. Listening...")
            self.audio_stream.start_stream()

            # Keep the task alive while the stream is active
            while self._running and self.audio_stream.is_active():
                await asyncio.sleep(0.1)

        except Exception as e:
            print(f"Error in listen_audio: {e}")
            traceback.print_exc()
            self._running = False # Signal other tasks to stop
        finally:
            print("Stopping microphone listener...")
            if self.audio_stream:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                self.audio_stream.close()
            # Signal termination by putting None in the queue
            if self.audio_out_queue:
                 await self.audio_out_queue.put(None)


    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback function."""
        # This runs in a separate thread managed by PyAudio
        try:
            # Use put_nowait for non-blocking behavior in the callback thread
            self.audio_out_queue.put_nowait({"data": in_data, "mime_type": "audio/pcm"})
        except asyncio.QueueFull:
             print("Warning: Audio queue is full, dropping frame.")
        except Exception as e:
            print(f"Error in audio callback: {e}")
            self._running = False # Signal stop on error
        return (in_data, pyaudio.paContinue if self._running else pyaudio.paComplete)


    async def send_audio(self):
        """Sends audio chunks from the queue to the Gemini API."""
        print("Starting audio sender...")
        while self._running:
            try:
                # Wait briefly for an item, avoids busy-waiting if queue is empty
                msg = await asyncio.wait_for(self.audio_out_queue.get(), timeout=1.0)

                if msg is None: # Termination signal
                    print("Termination signal received in send_audio.")
                    break

                if self.session:
                     # Use await with session.send as it's an async operation
                    await self.session.send(input=msg)
                else:
                    print("Warning: Session not ready, skipping audio chunk.")
                    await asyncio.sleep(0.1) # Wait briefly if session isn't ready yet

                self.audio_out_queue.task_done() # Mark task as done

            except asyncio.TimeoutError:
                # No audio received in a while, just continue waiting
                continue
            except asyncio.CancelledError:
                print("send_audio task cancelled.")
                break # Exit loop if task is cancelled
            except Exception as e:
                print(f"Error in send_audio: {e}")
                traceback.print_exc()
                self._running = False # Signal stop on error
                break # Exit loop on error
        print("Audio sender stopped.")


    async def receive_transcription(self):
        """Receives text transcription from the Gemini API and prints it."""
        print("Starting transcription receiver...")
        while self._running:
            try:
                if not self.session:
                    print("Waiting for session...")
                    await asyncio.sleep(0.5)
                    continue

                # Use async for loop to iterate through turns
                turn = self.session.receive()
                async for response in turn:
                    if not self._running: break # Check running flag frequently
                    if text := response.text:
                        print(text, end="", flush=True) # Print incrementally

                # If the loop finishes naturally (e.g., model ends turn), check running flag
                if not self._running: break

            except asyncio.CancelledError:
                print("receive_transcription task cancelled.")
                break # Exit loop if task is cancelled
            except Exception as e:
                # Catch potential API errors or other issues
                print(f"\nError receiving transcription: {e}")
                traceback.print_exc()
                self._running = False # Signal stop on error
                break # Exit loop on error
        print("\nTranscription receiver stopped.")


    async def run(self):
        """Main execution function setting up tasks."""
        self._running = True
        self.audio_out_queue = asyncio.Queue(maxsize=20) # Increased buffer size slightly

        try:
            print("Connecting to Google Generative AI...")
            # Use TaskGroup for managing concurrent tasks
            async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
                self.session = session
                print("Connected! Press Ctrl+C to stop.")

                async with asyncio.TaskGroup() as tg:
                    print("Creating tasks...")
                    # Start tasks for listening, sending, and receiving
                    listen_task = tg.create_task(self.listen_audio())
                    send_task = tg.create_task(self.send_audio())
                    receive_task = tg.create_task(self.receive_transcription())
                    print("Tasks created.")

                # TaskGroup finishes when all tasks are done or one raises an unhandled exception
                print("TaskGroup finished.")

        except asyncio.CancelledError:
            print("\nCancellation requested. Shutting down...")
        except Exception as e: # Catch potential connection errors or other top-level issues
             print(f"\nAn error occurred during execution: {e}")
             traceback.print_exc()
        finally:
            print("Initiating cleanup...")
            self._running = False # Ensure all loops know to stop

            # Give tasks a moment to react to the _running flag change
            await asyncio.sleep(0.5)

            # Explicitly cancel tasks if they haven't finished (important for graceful shutdown)
            all_tasks = asyncio.all_tasks()
            current_task = asyncio.current_task()
            for task in all_tasks:
                 if task is not current_task and not task.done():
                     task.cancel()
                     try:
                         # Allow tasks to handle cancellation gracefully
                         await asyncio.wait_for(task, timeout=1.0)
                     except (asyncio.CancelledError, asyncio.TimeoutError):
                         pass # Ignore errors during cancellation cleanup
                     except Exception as task_exc:
                         print(f"Error during task cancellation: {task_exc}")


            # Ensure PyAudio resources are released
            if self.audio_stream:
                if self.audio_stream.is_active():
                    self.audio_stream.stop_stream()
                if not self.audio_stream.is_stopped(): # Double check
                     self.audio_stream.close()
            pya.terminate()
            print("PyAudio terminated.")
            print("Transcription stopped.")


# --- Script Execution ---
if __name__ == "__main__":
    transcriber = MicrophoneTranscriber()
    try:
        asyncio.run(transcriber.run())
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Exiting.")
    except Exception as e:
        print(f"\nUnhandled exception in main: {e}")
        traceback.print_exc()