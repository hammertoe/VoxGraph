# VoxGraph - Speech-to-Knowledge Graph

VoxGraph captures and transcribes live audio using Google's Gemini API, processes the transcriptions into an RDF knowledge graph, and visualizes this data in real-time.

## Architecture

VoxGraph now consists of two main components:

1. **VoxGraph Server** (`app.py`): A Flask/SocketIO-based web application that:
   - Processes transcriptions into RDF triples using Gemini LLMs
   - Builds and maintains a knowledge graph
   - Provides a real-time visualization interface
   - Allows users to query the knowledge graph

2. **VoxGraph CLI Client** (`voxgraph_cli.py`): A command-line client that:
   - Captures audio from the microphone
   - Transcribes it in real-time using Google Gemini's Audio API
   - Sends transcriptions to the server via WebSockets
   - Provides a seamless integration between audio capture and knowledge graph generation

## Features

- **Real-time audio transcription** using Google Gemini's Audio API
- **Sentence-based processing** for natural language boundaries
- **Two-tier LLM processing**:
  - Fast model for immediate RDF triple generation
  - Slow model for deeper, higher-level concept identification
- **Timeout-based processing** to ensure responsiveness
- **Dynamic graph visualization** with real-time updates
- **Knowledge graph query interface**

## Prerequisites

- Python 3.9+
- Google API key with access to Gemini models
- Modern web browser
- Microphone for audio capture

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/voxgraph.git
   cd voxgraph
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

### Running the Server

1. Start the VoxGraph server:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5001
   ```

### Running the CLI Client

1. In a separate terminal (with the virtual environment activated), start the CLI client:
   ```
   python voxgraph_cli.py
   ```

2. The client will connect to the server and begin transcribing audio from your microphone.

3. Speak clearly into your microphone, and you'll see:
   - Transcription displayed in the CLI
   - Transcription sent to the server
   - Knowledge graph updated in the web interface

### Command-line Options

The CLI client supports several options:

- `--server URL`: Specify the WebSocket server URL (default: ws://localhost:5001)
- `--model MODEL`: Specify the Gemini model to use (default: models/gemini-2.0-flash-exp)
- `--debug`: Enable debug output for detailed logs

Example:
```
python voxgraph_cli.py --server ws://example.com:5001 --debug
```

## Configuration

You can adjust the following parameters in the web interface:

- **Sentence Chunk Size**: Number of sentences to collect before processing with the fast LLM
- **Slow LLM Chunk Size**: Number of fast LLM results to collect before processing with the slow LLM
- **Fast LLM Timeout**: Maximum time to wait before processing sentences (seconds)
- **Slow LLM Timeout**: Maximum time to wait before processing with slow LLM (seconds)

## How It Works

1. **Audio Capture**: The CLI client captures audio from your microphone in real-time.

2. **Transcription**: Google's Gemini API transcribes the audio into text.

3. **Sentence Processing**: 
   - Transcribed text is broken into sentences
   - When enough sentences accumulate (or timeout occurs), they're sent to the fast LLM

4. **Knowledge Graph Construction**:
   - The fast LLM converts text to RDF triples
   - The slow LLM identifies higher-level concepts and relationships
   - The graph is continuously updated and visualized

5. **Query Processing**:
   - Users can ask questions about the knowledge graph
   - The LLM analyzes the graph and provides answers based on its contents

## Requirements

See `requirements.txt` for the complete list of dependencies. Key libraries include:

- Flask and Flask-SocketIO for the web server
- google-generativeai for accessing Gemini models
- pyaudio for microphone access
- websocket-client for CLI-to-server communication
- rdflib for knowledge graph management
- nltk for natural language processing
