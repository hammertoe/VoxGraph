# Speech-to-Knowledge Graph

An application that captures and transcribes live audio from a web browser, processes the transcriptions into an RDF knowledge graph, and visualizes this data in real-time.

## Features

- Real-time audio capture and transcription using Faster Whisper
- Automatic conversion of transcriptions to RDF triples using Google Gemini LLMs
- Two-tier LLM processing (fast model for immediate processing, slow model for deeper insights)
- Dynamic knowledge graph visualization
- Interactive query interface to the knowledge graph
- Configurable processing parameters

## Architecture

The application consists of four main components:

1. **Audio Capture Frontend**: Records and streams audio from the user's browser
2. **Transcription Backend**: Processes audio data and converts it to text
3. **Knowledge Graph Processing**: Converts transcriptions to RDF triples and builds a knowledge graph
4. **Visualization & Query Interface**: Displays the knowledge graph and allows users to query it

## Prerequisites

- Python 3.9+ 
- FFmpeg installed on your system
- Google API key for Gemini models
- Modern web browser with microphone access

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/speech-to-knowledge-graph.git
   cd speech-to-knowledge-graph
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

4. Create a `.env` file from the template:
   ```
   cp .env.template .env
   ```

5. Edit the `.env` file and add your Google API key:
   ```
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5001
   ```

3. Allow microphone access when prompted

4. Click "Start Recording" to begin capturing audio

5. Speak clearly into your microphone. The application will:
   - Transcribe your speech in real-time
   - Process transcriptions into knowledge graph triples
   - Display the growing knowledge graph visually
   - Periodically enhance the graph with higher-level concepts

6. Use the query box to ask questions about the information in the knowledge graph

## Configuration

You can adjust the following parameters in the web interface:

- **Quick LLM Chunk Size**: Number of transcription chunks to collect before processing with the fast LLM
- **Slow LLM Chunk Size**: Number of fast LLM results to collect before processing with the slow LLM

## Project Structure

```
.
├── app.py                 # Main application logic
├── templates/
│   └── index.html         # Web interface
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create from .env.template)
└── .env.template          # Template for environment variables
```

## Technical Details

- **Backend**: Flask with Flask-SocketIO for real-time communication
- **Speech Recognition**: Faster Whisper for efficient transcription
- **LLM Processing**: Google Gemini models (2.0-flash for quick processing, 2.5-pro for deeper analysis)
- **Knowledge Graph**: RDFLib for RDF triple storage and management
- **Visualization**: Vis.js Network for interactive graph display
- **Frontend Communication**: Socket.IO for real-time updates

## Troubleshooting

- **Microphone not working**: Ensure your browser has permission to access the microphone
- **LLM errors**: Verify your Google API key is correct and has access to the Gemini models
- **Transcription issues**: Ensure FFmpeg is properly installed and accessible
- **Performance problems**: Try reducing the chunk sizes for faster processing
