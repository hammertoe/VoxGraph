# VoxGraph: Real-time Audio to Knowledge Graph

VoxGraph is a real-time audio processing system that captures microphone input, transcribes it using the Google Gemini Live API, extracts structured information as RDF triples using LLMs, builds a knowledge graph, and provides web interfaces for visualization and querying.

## Features

*   **Real-time Audio Capture:** Uses the Web Audio API (AudioWorklet) in the browser for efficient microphone input handling.
*   **Live Transcription:** Leverages the experimental Google Gemini Live API (via the Python SDK's v1alpha client) for low-latency speech-to-text.
*   **Server-Side Processing:** A Python backend built with Flask and Flask-SocketIO manages connections, API interactions, and graph processing.
*   **RDF Knowledge Graph:**
    *   Uses LLMs (e.g., Gemini Flash) to extract RDF triples (Turtle syntax) from transcriptions.
    *   Uses more advanced LLMs (e.g., Gemini Pro) for graph refinement and analysis (though this might need further development).
    *   Stores the combined knowledge graph using `rdflib`.
*   **Real-time Visualization:** A dedicated web interface (`/`) uses Vis.js to display the evolving knowledge graph.
*   **Natural Language Querying:** A separate mobile-oriented interface (`/mobile`) allows users to ask natural language questions about the graph, answered by an LLM (e.g., Gemini 1.5 Pro) configured as a query assistant.


## Architecture Overview

1.  **Mobile Interface (`/mobile`):**
    *   Captures microphone audio using the Web Audio API (`AudioWorklet`).
    *   Sends raw audio chunks (Int16 PCM) via SocketIO (`audio_chunk` event) to the server.
    *   Provides controls to start/stop listening.
    *   Includes a panel to send natural language queries (`query_graph` event) and display results (`query_result` event).
2.  **Python Server (`app.py`):**
    *   Listens for SocketIO connections.
    *   On `start_audio`, spawns a dedicated OS thread for the specific client (SID).
    *   **Worker Thread (Asyncio):**
        *   Initializes an `asyncio` event loop.
        *   Connects to the Google Gemini Live API using `live_client.aio.live.connect`.
        *   Runs two concurrent `asyncio` tasks:
            *   **Sender:** Gets audio chunks from a thread-safe `audio_queue` (filled by the main thread's `handle_audio_chunk`) and sends them to the Google API.
            *   **Receiver:** Receives transcription text from the Google API. Puts status updates (transcriptions, errors, start/stop signals) onto a thread-safe `status_queue`.
    *   **Main Thread (Eventlet):**
        *   `handle_audio_chunk`: Receives bytes from the client and puts them onto the client's `audio_queue`.
        *   `status_queue_poller` (one per client, runs as eventlet background task): Reads messages from the client's `status_queue`.
            *   If it's a transcription, calls `handle_transcription_result`.
            *   If it's a status update (`audio_started`, `audio_stopped`, `error`), emits it to the specific client using `socketio.server.emit(..., to=sid)`.
        *   `handle_transcription_result`: Takes transcribed text, uses NLTK to tokenize sentences, adds them to a buffer.
        *   Chunking/Timeout Logic: Uses `Timer` objects to trigger LLM processing (`process_with_quick_llm`, `process_with_slow_llm`) on sentence buffers or collected RDF results after certain thresholds or timeouts.
        *   `process_with_quick/slow_llm`: Sends text/RDF to configured Gemini chat models to generate/refine Turtle triples.
        *   `process_turtle_data`: Parses Turtle syntax and merges new triples into the shared `rdflib.Graph`.
        *   `update_graph_visualization`: Converts the graph to Vis.js format and broadcasts (`socketio.emit`) the `update_graph` event to *all* connected clients (viewers and mobile).
        *   `handle_query_graph`: Receives query text, formats a prompt including the current graph state, sends it to the query LLM, and emits the result back via `query_result`.
    *   **Viewer Interface (`/`):**
        *   Connects via SocketIO.
        *   Listens for `update_graph` events and renders the data using Vis.js.

## Prerequisites

*   **Python:** 3.10 or higher recommended.
*   **pip:** Python package installer.
*   **Google Cloud Project:** With the Gemini API enabled.
*   **Google API Key:** An API key authorized to use the Gemini API. **Crucial!**
*   **Modern Web Browser:** Supporting Web Audio API, AudioWorklet, and WebSockets (Chrome, Firefox, Safari, Edge).
*   **(Optional but Recommended for Mobile Testing):** `ngrok` ([https://ngrok.com/](https://ngrok.com/)) to create a secure tunnel to your local server, necessary for microphone access on mobile browsers when accessing via IP address.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/hammertoe/VoxGraph
    cd VoxGraph
    ```

2.  **Create and Activate Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK Data (if needed):** The script attempts to download the 'punkt' tokenizer data automatically on the first run if it's missing. Ensure you have an internet connection during the first startup. You can also download it manually:
    ```bash
    python -m nltk.downloader punkt
    ```

## Configuration

1.  **API Key:** The application requires your Google API Key. The recommended way is using a `.env` file:
    *   Create a file named `.env` in the project's root directory.
    *   Add your API key to this file:
        ```dotenv
        GOOGLE_API_KEY=YOUR_ACTUAL_API_KEY_HERE
        ```
    *   **Important:** Add `.env` to your `.gitignore` file to avoid committing your key.

2.  **(Optional) Model Names:** You can override the default LLM models used by setting environment variables (e.g., in your `.env` file or system environment):
    *   `GOOGLE_LIVE_MODEL` (Default: `models/gemini-2.0-flash-exp`)
    *   `QUICK_LLM_MODEL` (Default: `gemini-2.0-flash`)
    *   `SLOW_LLM_MODEL` (Default: `gemini-2.5-pro-exp-03-25`)
    *   `QUERY_LLM_MODEL` (Default: `gemini-1.5-pro`)

## Running the Application

1.  **Activate Virtual Environment:**
    ```bash
    source venv/bin/activate 
    ```
2.  **Run the Server:**
    ```bash
    python app.py
    ```
3.  The server will start, typically on `http://0.0.0.0:5001`. It will log status information, including the availability of the Live API client and other LLMs.

## Usage

There are two web interfaces:

1.  **Graph Viewer:**
    *   Access: Open `http://localhost:5001/` in your browser.
    *   Functionality: Displays the RDF knowledge graph as it's built in real-time. It listens for `update_graph` events from the server.

2.  **Mobile Input/Query Interface:**
    *   Access: Open `http://localhost:5001/mobile` in your browser.
    *   **Important for Mobile:** If accessing from a different device (like your phone) using your computer's IP address (e.g., `http://192.168.1.100:5001/mobile`), microphone access will likely fail due to browser security restrictions (Web Audio API requires HTTPS or localhost).
    *   **Solution for Mobile:** Use `ngrok` to create an HTTPS tunnel. Run `ngrok http 5001` and access the provided `https://....ngrok-free.app/mobile` URL on your phone.
    *   **Functionality:**
        *   **Start/Stop Listening:** Click the button to start capturing audio from your microphone. The audio is streamed to the server for transcription. Click again to stop.
        *   **Query:** Type a natural language question about the information likely captured in the graph into the input box and click "Ask". The server will query the graph using the Query LLM and display the answer.


