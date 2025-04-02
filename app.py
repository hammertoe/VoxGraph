# Step 1: Import and enable eventlet monkey patching FIRST
import eventlet
eventlet.monkey_patch()

# Step 2: Now import all other modules
import os
import logging
import json
import time
from threading import Thread
from queue import Queue
import numpy as np
from tempfile import NamedTemporaryFile

# Step 3: Import Flask and related libraries
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

# Step 4: Import the rest of the dependencies
# Google Gemini API imports
import google.generativeai as genai
from google.generativeai import types
# RDF handling imports
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD
# Environment and config
from dotenv import load_dotenv

# --- Configuration & Setup ---
load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Flask app and socketio
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # Secret key for session management
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Whisper Model Setup (Optional) ---
whisper_model = None
try:
    # Only attempt to import and load Whisper if needed
    USE_WHISPER = True  # Set to True to enable Whisper

    if USE_WHISPER:
        from faster_whisper import WhisperModel
        import ffmpeg
        import sys
        
        # Increase recursion limit to prevent issues
        sys.setrecursionlimit(5000)
        
        # Use a small model for quicker transcription
        whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        logger.info("Whisper model loaded successfully")
    else:
        logger.info("Whisper model disabled (USE_WHISPER=False)")
        
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")
    whisper_model = None

# --- Google Gemini LLM Setup ---
# System prompts for the LLMs
quick_llm_system_prompt = """
You will convert transcribed speech into an RDF knowledge graph using Turtle syntax.
Return only the new RDF Turtle triples representing entities and relationships mentioned in the text.
Use the 'ex:' prefix for examples.

Follow these steps:
1. Identify entities (people, places, concepts, etc.)
2. Create URIs for entities using ex: prefix and CamelCase (e.g., ex:EntityName)
3. Identify relationships between entities
4. Format as valid Turtle triples

Example output format:
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .

ex:EntityA a ex:Type ;
   ex:relatesTo ex:EntityB ;
   ex:hasProperty "literal value" .

Note: Keep your response ONLY to the Turtle syntax - no explanations or other text.
"""

slow_llm_system_prompt = """
You are analyzing an existing RDF knowledge graph and a new piece of text to identify higher-level concepts and connections.

1. Review the existing graph structure provided
2. Analyze the new text segment
3. Identify:
   - Higher-level concepts that connect existing entities
   - Implicit relationships not explicitly stated
   - Categories or classifications that organize entities
   - Potential inconsistencies to resolve

Return ONLY new Turtle triples that enhance the knowledge graph with these insights.
Do not repeat existing triples. Focus on adding value through abstraction and connection.
"""

query_llm_system_prompt = """
You are analyzing a knowledge graph to answer a user query.
Use the provided RDF graph to:
1. Identify relevant entities and relationships that address the query
2. Analyze connections between concepts
3. Synthesize information from across the graph
4. Provide a clear, concise answer based on the knowledge graph contents ONLY

Respond in natural language, explaining how you arrived at your conclusion by referencing specific entities and relationships from the graph.
"""

# Instantiate Gemini clients
client_quick = None
client_slow = None
client_query = None
quick_chat = None
slow_chat = None
client_chat = None

# --- Fixed Gemini LLM Setup ---
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        api_key = "dummy_api_key"
        logger.critical("CRITICAL: GOOGLE_API_KEY environment variable not set. Using a dummy key. LLM calls will fail.")
    else:
        logger.info("Using GOOGLE_API_KEY from environment variable.")

    # Configure the API client
    genai.configure(api_key=api_key)
    
    # Set up generation config for quick model (without safety_settings)
    quick_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
    
    # Set up generation config for slow model (without safety_settings)
    slow_config = {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 4096,
    }

    # Set up generation config for query model (without safety_settings)
    query_config = {
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }
    
    # Initialize quick model
    quick_model_name = "gemini-1.5-flash"
    quick_model = genai.GenerativeModel(
        model_name=quick_model_name,
        generation_config=quick_config,
        system_instruction=quick_llm_system_prompt
    )
    
    # Initialize slow model
    slow_model_name = "gemini-2.5-pro-exp-03-25"
    slow_model = genai.GenerativeModel(
        model_name=slow_model_name,
        generation_config=slow_config,
        system_instruction=slow_llm_system_prompt
    )

    # Initialize query model
    query_model_name = "gemini-1.5-pro"
    query_model = genai.GenerativeModel(
        model_name=query_model_name,
        generation_config=query_config,
        system_instruction=query_llm_system_prompt
    )

    
    # Create chat sessions
    quick_chat = quick_model.start_chat(history=[])
    slow_chat = slow_model.start_chat(history=[])
    query_chat = query_model.start_chat(history=[])

    logger.info(f"LLM models initialized - Quick: {quick_model_name}, Slow: {slow_model_name}, Query: {query_model_name}")

except Exception as e:
    logger.error(f"Error initializing Google Gemini clients: {e}")
    client_quick = None
    client_slow = None
    client_query = None
    quick_chat = None
    slow_chat = None
    client_chat = None

# --- Global State Management ---
# Queue for transcribed chunks
transcription_queue = Queue()
# Buffer for collecting transcriptions before processing
transcription_buffer = []
# Config for chunk sizes
TRANSCRIPTION_CHUNK_SIZE = 3  # Number of transcriptions to collect before quick LLM processing
SLOW_LLM_CHUNK_SIZE = 5  # Number of quick LLM results to collect before slow LLM processing
# RDF graph for storing the knowledge graph
accumulated_graph = Graph()
# Add common prefixes to the main graph
EX = URIRef("http://example.org/")
accumulated_graph.bind("rdf", RDF)
accumulated_graph.bind("rdfs", RDFS)
accumulated_graph.bind("owl", OWL)
accumulated_graph.bind("xsd", XSD)
accumulated_graph.bind("ex", EX)
# List to collect quick LLM results for slow LLM processing
quick_llm_results = []

# --- Helper Functions ---
def extract_label(uri_or_literal):
    """Helper to get a readable label from URI or Literal"""
    if isinstance(uri_or_literal, URIRef):
        try:
            prefix, namespace, name = accumulated_graph.compute_qname(uri_or_literal, generate=False)
            return name
        except:
            if '#' in uri_or_literal:
                return uri_or_literal.split('#')[-1]
            return uri_or_literal.split('/')[-1]
    elif isinstance(uri_or_literal, Literal):
        return str(uri_or_literal)
    else:
        return str(uri_or_literal)

def graph_to_visjs(graph):
    """Converts an rdflib Graph to Vis.js nodes and edges format."""
    nodes = []
    edges = []
    node_ids = set()  # Keep track of added node URIs/IDs

    # Define schema URIs to potentially filter out or style differently
    schema_node_uris = {
        RDF.type, RDFS.Class, OWL.Class, OWL.ObjectProperty,
        OWL.DatatypeProperty, RDFS.subClassOf, OWL.Thing,
        RDFS.Resource, RDFS.Literal, RDFS.domain, RDFS.range
    }
    schema_prefixes = (str(RDF), str(RDFS), str(OWL), str(XSD))

    # Iterate through all triples in the graph
    for s, p, o in graph:
        subject_str = str(s)
        predicate_str = str(p)
        object_str = str(o)

        # Add subject node if not already added and not a schema definition URI
        if subject_str not in node_ids and s not in schema_node_uris and not subject_str.startswith(schema_prefixes):
            nodes.append({"id": subject_str, "label": extract_label(s), "title": f"URI: {subject_str}"})
            node_ids.add(subject_str)

        # Handle the object
        if isinstance(o, URIRef):
            # Add object node if it's a resource, not already added, and not a schema definition URI
            if object_str not in node_ids and o not in schema_node_uris and not object_str.startswith(schema_prefixes):
                nodes.append({"id": object_str, "label": extract_label(o), "title": f"URI: {object_str}"})
                node_ids.add(object_str)

            # Add edge if it's a relationship between two resources (nodes)
            if p not in {RDF.type, RDFS.subClassOf, RDFS.domain, RDFS.range} and \
                s not in schema_node_uris and o not in schema_node_uris and \
                not subject_str.startswith(schema_prefixes) and not object_str.startswith(schema_prefixes) and \
                not predicate_str.startswith(schema_prefixes):
                edges.append({
                    "from": subject_str,
                    "to": object_str,
                    "label": extract_label(p),
                    "title": f"Predicate: {predicate_str}",
                    "arrows": "to"
                })

        elif isinstance(o, Literal):
            # Add literal value as property to the subject node's title (tooltip)
            for node in nodes:
                if node["id"] == subject_str:
                    prop_label = extract_label(p)
                    lit_label = extract_label(o)
                    # Append to existing title or create new one
                    node['title'] = node.get('title', f"URI: {subject_str}") + f"\n{prop_label}: {lit_label}"
                    # Optionally, add primary literals like name/label directly to node label
                    if p == RDFS.label or predicate_str.endswith("Name") or predicate_str.endswith("name"):
                        if node['label'] == extract_label(s):  # Avoid overwriting if already complex
                            node['label'] = lit_label  # Use literal as main label
                        else:
                            node['label'] += f"\n({lit_label})"  # Append in parenthesis
                    break

    # Add rdfs:subClassOf as hierarchical edges
    for s, p, o in graph.triples((None, RDFS.subClassOf, None)):
        subject_str = str(s)
        object_str = str(o)
        if subject_str in node_ids and object_str in node_ids:
            edges.append({
                "from": subject_str,
                "to": object_str,
                "label": "subClassOf",
                "arrows": "to",
                "color": {"color": "#888888", "highlight": "#555555"},
                "dashes": True,
                "title": "rdfs:subClassOf"
            })

    # Add rdf:type information to node titles/labels
    for s, p, o in graph.triples((None, RDF.type, None)):
        subject_str = str(s)
        object_str = str(o)
        if subject_str in node_ids and isinstance(o, URIRef):
            type_label = extract_label(o)
            if o not in {OWL.Class, RDFS.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.Thing, RDFS.Resource}:
                for node in nodes:
                    if node["id"] == subject_str:
                        node['title'] = node.get('title', f"URI: {subject_str}") + f"\nType: {type_label}"
                        if f"({type_label})" not in node['label'] and type_label not in node['label']:
                            node['label'] += f"\n({type_label})"
                        break

    # Remove duplicate edges
    unique_edges_set = set()
    unique_edges = []
    
    for edge in edges:
        edge_tuple = (edge['from'], edge['to'], edge.get('label', ''))
        if edge_tuple not in unique_edges_set:
            unique_edges.append(edge)
            unique_edges_set.add(edge_tuple)
    
    return {"nodes": nodes, "edges": unique_edges}

def process_turtle_data(turtle_data):
    """Process Turtle data and add it to the accumulated graph"""
    if not turtle_data:
        logger.warning("Empty Turtle data received, nothing to process")
        return False
    
    try:
        # Strip potential markdown code fences and leading/trailing whitespace
        turtle_data = turtle_data.strip()
        if turtle_data.startswith("```turtle"):
            turtle_data = turtle_data[len("```turtle"):].strip()
        elif turtle_data.startswith("```"):
            turtle_data = turtle_data[len("```"):].strip()
        if turtle_data.endswith("```"):
            turtle_data = turtle_data[:-len("```")].strip()
                
        # Define prefixes string to provide context for parsing fragment
        prefixes = """
            @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
            @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
            @prefix owl: <http://www.w3.org/2002/07/owl#> .
            @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
            @prefix ex: <http://example.org/> .
        """
        # Combine prefixes and new data for parsing
        full_turtle_for_parsing = prefixes + "\n" + turtle_data

        # Create a temporary graph to parse into, avoiding polluting main graph on error
        temp_graph = Graph()
        temp_graph.parse(data=full_turtle_for_parsing, format="turtle")

        # Count new triples before adding
        original_count = len(accumulated_graph)
        
        # If parsing succeeded, add triples from temp_graph to accumulated_graph
        for triple in temp_graph:
            accumulated_graph.add(triple)

        # Log information about the update
        new_count = len(accumulated_graph) - original_count
        logger.info(f"Added {new_count} new triples to the graph. Total: {len(accumulated_graph)}")
        
        return new_count > 0  # Return True if new triples were added
        
    except Exception as e:
        logger.error(f"Error parsing Turtle data: {e}")
        logger.error(f"Problematic Turtle data: {turtle_data}")
        return False

def update_graph_visualization():
    """Update the graph visualization for all clients"""
    try:
        # Convert the graph to Vis.js format
        vis_data = graph_to_visjs(accumulated_graph)
        # Broadcast the updated graph to all connected clients
        socketio.emit('update_graph', vis_data)
        logger.info("Graph visualization updated and broadcast to clients")
    except Exception as e:
        logger.error(f"Error updating graph visualization: {e}")
        socketio.emit('error', {'message': f'Error generating graph visualization: {e}'})

def process_transcription_queue():
    """Process the transcription queue in a separate thread"""
    logger.info("Starting transcription queue processing thread")
    while True:
        try:
            # Get the next item from the queue (will block until an item is available)
            item = transcription_queue.get()
            
            # Process the item (in this case, it's a transcript)
            if item:
                process_transcription(item)
            
            # Mark the task as done
            transcription_queue.task_done()
        
        except Exception as e:
            logger.error(f"Error in transcription queue processing: {e}")
        
        # Small sleep to prevent CPU spinning
        time.sleep(0.1)

def process_transcription(transcript):
    """Process a single transcription and update the buffer"""
    global transcription_buffer
    
    if not transcript:
        return
    
    # Add the transcript to the buffer
    transcription_buffer.append(transcript)
    
    # Update clients with the new transcript
    socketio.emit('transcription_update', {'transcript': transcript})
    
    # Check if we have enough transcriptions to process
    if len(transcription_buffer) >= TRANSCRIPTION_CHUNK_SIZE:
        # Process the buffer with the quick LLM
        combined_text = " ".join(transcription_buffer)
        transcription_buffer = []  # Clear the buffer
        
        # Process in a separate thread to avoid blocking
        Thread(target=process_with_quick_llm, args=(combined_text,)).start()

def process_with_quick_llm(text):
    """Process text with the quick LLM model"""
    global quick_llm_results
    
    if not quick_chat:
        logger.error("Quick LLM chat not available")
        socketio.emit('error', {'message': 'Quick LLM service unavailable'})
        return
    
    try:
        logger.info(f"Processing with quick LLM: {text[:100]}...")
        
        # Send the text to the quick LLM
        response = quick_chat.send_message(text)
        
        # Get the turtle data from the response
        turtle_data = response.text
        
        # Process the turtle data
        if process_turtle_data(turtle_data):
            # Add to the results for slow LLM processing
            quick_llm_results.append(text)
            
            # Update the graph visualization
            update_graph_visualization()
            
            # Check if we have enough quick results to process with the slow LLM
            if len(quick_llm_results) >= SLOW_LLM_CHUNK_SIZE:
                combined_text = " ".join(quick_llm_results)
                quick_llm_results = []  # Clear the buffer
                
                # Process in a separate thread to avoid blocking
                Thread(target=process_with_slow_llm, args=(combined_text,)).start()
    
    except Exception as e:
        logger.error(f"Error processing with quick LLM: {e}")
        socketio.emit('error', {'message': f'Error processing with quick LLM: {e}'})

def process_with_slow_llm(text):
    """Process text with the slow LLM model"""
    if not slow_chat:
        logger.error("Slow LLM chat not available")
        socketio.emit('error', {'message': 'Slow LLM service unavailable'})
        return
    
    try:
        logger.info(f"Processing with slow LLM: {text[:100]}...")
        
        # Get the current graph as Turtle
        current_graph_turtle = accumulated_graph.serialize(format="turtle")
        
        # Prepare the message with the current graph and the new text
        message = f"""
Current Knowledge Graph:
```turtle
{current_graph_turtle}
```

New text to analyze:
```
{text}
```

Please identify any higher-level concepts, connections, or patterns in this data that would enhance the knowledge graph.
Return ONLY the new Turtle triples to add to the graph.
"""
        
        # Send the message to the slow LLM
        response = slow_chat.send_message(message)
        
        # Get the turtle data from the response
        turtle_data = response.text
        
        # Process the turtle data
        if process_turtle_data(turtle_data):
            # Update the graph visualization
            update_graph_visualization()
            
            # Log success
            logger.info("Slow LLM processing completed successfully")
            socketio.emit('info', {'message': 'Enhanced knowledge graph with higher-level concepts'})
    
    except Exception as e:
        logger.error(f"Error processing with slow LLM: {e}")
        socketio.emit('error', {'message': f'Error processing with slow LLM: {e}'})

def process_query(query_text):
    """Process a user query against the knowledge graph using an LLM"""
    if not query_chat:
        logger.error("Query LLM chat not available")
        return "Query processing service is unavailable."
    
    try:
        logger.info(f"Processing query: {query_text}")
        
        # Get the current graph as Turtle
        current_graph_turtle = accumulated_graph.serialize(format="turtle")
        
        # Prepare the message with the current graph and the query
        message = f"""
Current Knowledge Graph:
```turtle
{current_graph_turtle}
```

User Query: {query_text}

Please analyze the knowledge graph to answer this query. Only use information present in the graph.
"""
        
        # Send the message to the LLM
        logger.info(f"Sending query: {message}")
        response = query_chat.send_message(message)
        
        # Return the response text
        logger.info(f"Got response: {response}")

        return response.text
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return f"Error processing query: {str(e)}"

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

# --- SocketIO Events ---
@socketio.on('connect')
def handle_connect():
    """Handles a new client connection."""
    logger.info(f"Client connected: {request.sid}")
    # Send the current graph state to the newly connected client
    try:
        vis_data = graph_to_visjs(accumulated_graph)
        emit('update_graph', vis_data)
    except Exception as e:
        logger.error(f"Error sending initial graph data: {e}")
        emit('error', {'message': f'Error generating graph visualization: {e}'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handles a client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")

# --- Updated Audio Processing Function ---
@socketio.on('audio_data')
def handle_audio_data(data):
    """Handles audio data from the client."""
    try:
        # Get audio bytes from data
        audio_bytes = data.get('audio')
        if not audio_bytes:
            logger.warning("Received empty audio data")
            return
        
        # Log receipt of audio data
        logger.info(f"Received audio chunk: {len(audio_bytes)} bytes")
        
        # Process the audio data
        transcript = process_audio_data(audio_bytes)
        if transcript:
            # Send the transcript back to the client immediately
            emit('transcription', {'text': transcript})
            
            # Add to the transcription queue for processing
            transcription_queue.put(transcript)
            logger.info(f"Transcribed: '{transcript[:50]}...' (truncated)")
        else:
            logger.warning("Transcription produced empty result")
            emit('error', {'message': 'Failed to transcribe audio segment'})
    
    except Exception as e:
        logger.error(f"Error handling audio data: {e}")
        emit('error', {'message': f'Error processing audio: {str(e)}'})

def process_audio_data(audio_bytes):
    """Process audio data for transcription"""
    if not whisper_model:
        # Use a fallback for testing if Whisper is not available
        logger.warning("Whisper model not available, using fallback transcription")
        return "This is a fallback transcription since Whisper is not available."
    
    try:
        with NamedTemporaryFile(suffix=".webm", delete=False) as audio_webm, \
             NamedTemporaryFile(suffix=".wav", delete=False) as audio_wav:
            
            # Store filenames for processing
            webm_path = audio_webm.name
            wav_path = audio_wav.name
            
            # Write audio data to temporary file
            audio_webm.write(audio_bytes)
            audio_webm.flush()
        
        try:
            # Convert WebM to WAV using ffmpeg
            logger.info(f"Converting audio: {webm_path} → {wav_path}")
            ffmpeg.input(webm_path).output(
                wav_path,
                format='wav',
                acodec='pcm_s16le',
                ac=1,
                ar='16000'
            ).overwrite_output().run(quiet=True)
            
            # Transcribe the WAV file
            logger.info("Transcribing audio with Whisper model")
            segments, _ = whisper_model.transcribe(wav_path, language="en")
            transcript = " ".join(segment.text for segment in segments).strip()
            
            if not transcript:
                logger.warning("Empty transcript from non-empty audio")
                return "..."  # Return placeholder to indicate silent audio
            
            return transcript
            
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if hasattr(e, 'stderr') else str(e)}")
            return None
        
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None
            
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(webm_path):
                    os.unlink(webm_path)
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
            except Exception as e:
                logger.error(f"Error removing temporary files: {e}")
                
    except Exception as e:
        logger.error(f"Error processing audio data: {e}")
        return None

@socketio.on('text_input')
def handle_text_input(data):
    """Handles direct text input from the client."""
    text = data.get('text')
    if not text:
        return
    
    logger.info(f"Received text input: {text}")
    
    # Add to the transcription queue
    transcription_queue.put(text)
    # Echo back to the client
    emit('transcription', {'text': text})

@socketio.on('query')
def handle_query(data):
    """Handles a query from the client."""
    query_text = data.get('query')
    if not query_text:
        return
    
    logger.info(f"Received query: {query_text}")
    
    # Process the query in a separate thread
    def process_and_emit():
        response = process_query(query_text)
        socketio.emit('query_response', {'response': response})
    
    Thread(target=process_and_emit).start()

@socketio.on('set_chunk_size')
def handle_set_chunk_size(data):
    """Handles setting the chunk size configuration."""
    global TRANSCRIPTION_CHUNK_SIZE, SLOW_LLM_CHUNK_SIZE
    
    quick_size = data.get('quick_size')
    slow_size = data.get('slow_size')
    
    if quick_size:
        TRANSCRIPTION_CHUNK_SIZE = int(quick_size)
        logger.info(f"Updated quick LLM chunk size to {TRANSCRIPTION_CHUNK_SIZE}")
    
    if slow_size:
        SLOW_LLM_CHUNK_SIZE = int(slow_size)
        logger.info(f"Updated slow LLM chunk size to {SLOW_LLM_CHUNK_SIZE}")
    
    emit('config_updated', {
        'quick_size': TRANSCRIPTION_CHUNK_SIZE,
        'slow_size': SLOW_LLM_CHUNK_SIZE
    })

# --- Main Execution ---
if __name__ == '__main__':
    # Start the transcription queue processing thread
    transcription_thread = Thread(target=process_transcription_queue)
    transcription_thread.daemon = True  # Thread will exit when main program exits
    transcription_thread.start()
    
    # Log configuration information
    logger.info(f"Starting application with configuration:")
    logger.info(f"  - Quick LLM chunk size: {TRANSCRIPTION_CHUNK_SIZE}")
    logger.info(f"  - Slow LLM chunk size: {SLOW_LLM_CHUNK_SIZE}")
    logger.info(f"  - Whisper model: {'Available' if whisper_model else 'Unavailable'}")
    logger.info(f"  - Quick LLM: {'Available' if quick_chat else 'Unavailable'}")
    logger.info(f"  - Slow LLM: {'Available' if slow_chat else 'Unavailable'}")
    
    # Check if critical components are missing
    missing_components = []
    if not whisper_model:
        missing_components.append("Whisper model (transcription will use fallback)")
    if not quick_chat and not slow_chat:
        missing_components.append("LLM models (knowledge graph generation will not work)")
    
    if missing_components:
        logger.warning(f"WARNING: Running with missing components: {', '.join(missing_components)}")
        logger.warning("Some functionality will be limited or unavailable")
    
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Starting server on http://0.0.0.0:{port}")
    socketio.run(app, debug=True, host='0.0.0.0', port=port)
