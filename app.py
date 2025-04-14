# -*- coding: utf-8 -*-
# Step 1: Import and enable eventlet monkey patching FIRST
import eventlet
# *** MODIFICATION: Tell eventlet NOT to patch the standard threading module ***
eventlet.monkey_patch(thread=False) # Allows standard threading for asyncio isolation

# Step 2: Now import all other modules
import os
import sys
import logging # Import logging early
import json
import time
import re
import uuid # <-- Import UUID for session IDs
# *** Import standard threading and queue ***
import threading
from queue import Queue as ThreadSafeQueue, Empty as QueueEmpty, Full as QueueFull
from threading import Timer # Keep for LLM timeouts
import numpy as np # Keep if used elsewhere
import io
import asyncio
import requests # For direct Lighthouse API calls
import websockets.exceptions
import threading


# Step 3: Import Flask and related libraries
from flask import Flask, render_template, request, redirect, url_for # <-- Added redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room

# Step 4: Import the rest of the dependencies
# Google Gemini API imports
from google import genai as genai
from google.genai import types as genai_types
from google.api_core import exceptions as google_exceptions

# RDF handling imports
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL, XSD
# Environment and config
from dotenv import load_dotenv

# NLTK for sentence tokenization
import nltk

# --- Explicit Logging Setup (Moved Earlier) ---
# (No changes needed here)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set the desired level for your app's logger
stream_handler = logging.StreamHandler(sys.stdout) # Use stdout
formatter = logging.Formatter('%(asctime)s - %(name)s [%(threadName)s] %(levelname)s - %(message)s') # Added threadName
stream_handler.setFormatter(formatter)
if not logger.handlers: # Avoid duplicate handlers
    logger.addHandler(stream_handler)

# --- Now safe to import requests ---
import requests # For direct Lighthouse API calls

# --- NLTK Download (Use configured logger now) ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("NLTK 'punkt' tokenizer not found. Downloading...") # Use configured logger
    nltk.download('punkt')
    logger.info("NLTK 'punkt' tokenizer downloaded.") # Use configured logger
from nltk.tokenize import sent_tokenize

# *** Import from your local CAR library ***
# (No changes needed here)
try:
    from car_library import generate_cid, cid_to_string, generate_car
    logger.info("Successfully imported functions from car_library.py") # Use configured logger
except ImportError as e:
    logger.critical(f"Failed to import from car_library.py: {e}. Local CID/CAR generation will fail.") # Use configured logger
    def generate_cid(data): return b"DUMMY_CID_BYTES"
    def cid_to_string(cid_bytes): return "bDummyCIDStringLibsMissing"
    def generate_car(text): return b"DUMMY_CAR_DATA_LIBS_MISSING"

# --- Configuration & Setup ---
load_dotenv()
LIGHTHOUSE_API_KEY = os.getenv("LIGHTHOUSE_API_KEY") # Needed for upload

# --- Google AI Live API Specific Config ---
GOOGLE_LIVE_API_MODEL = os.getenv("GOOGLE_LIVE_MODEL", "models/gemini-2.0-flash-exp")
LIVE_API_SAMPLE_RATE = 16000
LIVE_API_CHANNELS = 1
LIVE_API_CONFIG = genai_types.GenerateContentConfig(
    response_modalities=[genai_types.Modality.TEXT],
    system_instruction=genai_types.Content(parts=[genai_types.Part(text="You are a transcription assistant...")]) # Truncated
)

# Create Flask app and socketio
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Google Gemini Base Client Setup (API Key Check) ---
# Initialize the base client once to reuse the API key
base_client = None # For standard Chat models
live_client = None # For Live API using v1alpha
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.critical("GOOGLE_API_KEY is missing in the environment variables. LLM features will be disabled.")
    else:
        logger.info("GOOGLE_API_KEY found. Initializing base clients.")
        try:
            # Base client for creating chat sessions
            base_client = genai.Client(api_key=api_key)
            logger.info("Standard base Client initialized.")
        except Exception as e:
            logger.error(f"Standard base Client initialization failed: {e}", exc_info=True)
            base_client = None
        try:
            # Live client for streaming transcription
            live_client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
            # Check if live connection is available (optional but good practice)
            _ = live_client.aio.live.connect
            logger.info("v1alpha live Client initialized.")
        except AttributeError:
             logger.error("Failed to access live API functionality. Ensure 'v1alpha' API version is supported by the library and key.")
             live_client = None
        except Exception as e:
             logger.error(f"v1alpha live Client initialization failed: {e}", exc_info=True)
             live_client = None

except Exception as e:
     logger.error(f"General client initialization error: {e}", exc_info=True)
     base_client = None
     live_client = None
     logger.warning("Proceeding with potentially limited LLM/Live functionality.")


# --- System Prompts (No changes needed) ---
quick_llm_system_prompt = """
You will convert transcribed speech into an RDF knowledge graph using Turtle syntax.
IMPORTANT: The user's message will include a CID at the start
You MUST extract the <CID_STRING> from the beginning of the user's message.

Return only the new RDF Turtle triples representing entities and relationships mentioned in the Transcription Text part. Use the 'ex:' prefix for examples (e.g., http://example.org/).

Follow these steps:

1. Identify the Source CID provided at the beginning of the user's message.
2. Identify the full Transcription Text after the Source CID.
3. Create a new transcription node using a URI that incorporates the extracted CID (e.g., ex:transcription_<CID_STRING>).
   This transcription node MUST include:
   - The property ex:sourceTranscriptionCID with the extracted <CID_STRING> (typed as an xsd:string).
   - The property ex:transcriptionText containing the full transcription text.

4. Identify entities (people, places, concepts, times, organizations, etc.) within the transcription.
5. Create URIs for entities using the ex: prefix and CamelCase (e.g., ex:EntityName). Use existing URIs if the same entities are mentioned again.
6. Identify relationships between entities (e.g., ex:worksAt, ex:locatedIn, ex:discussedConcept).
7. Identify properties of entities (e.g., rdfs:label, ex:hasValue, ex:occurredOnDate). Use appropriate datatypes for literals (e.g., "value"^^xsd:string, "123"^^xsd:integer, "2024-01-01"^^xsd:date).
8. For significant entities or statements derived directly from the Transcription Text, do not attach the CID directly. Instead, add a triple linking them to the transcription node using a relation such as ex:derivedFromTranscript.

Example:
ex:AliceJohnson ex:derivedFromTranscript ex:transcription_<CID_STRING> .
ex:ProjectPhoenix ex:derivedFromTranscript ex:transcription_<CID_STRING> .

Format your output as valid Turtle triples. Output ONLY Turtle syntax and do not repeat triples.

Example Input User Message: [bafy...xyz] Acme Corporation announced Project Phoenix. Alice Johnson leads it.

Example Output Format:
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:transcription_bafy_xyz a ex:Transcription ;
    ex:sourceTranscriptionCID "bafy...xyz"^^xsd:string ;
    ex:transcriptionText "Acme Corporation announced Project Phoenix. Alice Johnson leads it." .

ex:AcmeCorporation a ex:Organization ;
    ex:announcedProject ex:ProjectPhoenix ;
    ex:derivedFromTranscript ex:transcription_bafy_xyz .

ex:ProjectPhoenix a ex:Project ;
    rdfs:label "Phoenix"^^xsd:string ;
    ex:ledBy ex:AliceJohnson ;
    ex:derivedFromTranscript ex:transcription_bafy_xyz .

ex:AliceJohnson a ex:Person ;
    ex:derivedFromTranscript ex:transcription_bafy_xyz .
"""

slow_llm_system_prompt = """
You are an advanced knowledge graph analyst. Your task is to analyze an existing RDF knowledge graph (provided in Turtle) and a set of newly added Turtle triples (derived from recent text) to identify higher-level concepts, implicit connections, and potential refinements.

1. Review the 'Existing Knowledge Graph' structure provided below (it may be partial).
2. Analyze the 'New Information/Triples' provided below.
3. Identify and generate ONLY NEW Turtle triples that represent:
   - Higher-level concepts or categories that connect multiple existing entities (e.g., identifying a 'Meeting' concept linking several discussed topics and people).
   - Implicit relationships not explicitly stated but inferable from the context (e.g., if Person A discusses Topic X and Person B discusses Topic X, infer ex:collaboratedOnTopic between Person A and Person B).
   - Potential classifications or typing that organize entities more effectively (e.g., classifying a project as ex:HighPriority based on discussion context).
   - Possible links between new entities and older entities in the existing graph.
   - DO NOT repeat triples already present in the 'Existing Knowledge Graph' or the 'New Information/Triples' sections.
   - DO NOT generate triples that are simple restatements of the new information. Focus on abstraction, inference, and connection.
   - Use the 'ex:' prefix (<http://example.org/>) for new URIs and relationships.

Return ONLY the newly inferred Turtle triples. Output strictly Turtle syntax, without explanations or markdown fences. If no significant new insights are found, return nothing or only essential schema links (like subClassOf).

Example Scenario:
Existing Graph shows: ex:TopicA, ex:TopicB discussed.
New Info adds: ex:PersonX discussed ex:TopicA. ex:PersonY discussed ex:TopicB. ex:Meeting1 mentioned ex:TopicA, ex:TopicB.
Possible Slow LLM Output:
ex:Meeting1 a ex:DiscussionForum .
ex:PersonX ex:participatedIn ex:Meeting1 .
ex:PersonY ex:participatedIn ex:Meeting1 .
ex:TopicA ex:relatedTopic ex:TopicB . # Inferred connection

Format: Use standard Turtle syntax.
"""

query_llm_system_prompt = """
You are a knowledge graph query assistant. Answer user queries based *strictly* on the provided RDF knowledge graph (in Turtle format).

Follow these steps:
1. Analyze the user's query.
2. Examine the provided 'Knowledge Graph'.
3. Identify relevant entities and relationships.
4. Synthesize the information found *only* within the graph into a clear, concise answer.
5. Explain *how* you arrived at the answer by referencing specific entities and relationships.
6. **If an entity or statement has an `ex:sourceTranscriptionCID` property, mention this CID as the source evidence for that piece of information.** Example: "The graph states Project Phoenix is led by Alice Johnson (source: bafy...xyz)."
7. If the information needed is *not present*, state that clearly. Do not invent information.
"""

# --- Session-Based Global State ---
sessions = {} # Dictionary to hold state for each session_id
sid_to_session = {} # Dictionary mapping client SID to session_id

# --- Constants ---
EX = URIRef("http://example.org/")
SENTENCE_CHUNK_SIZE = 1 # Not actively used with current CID-per-utterance approach
SLOW_LLM_CHUNK_SIZE = 5
FAST_LLM_TIMEOUT = 20 # Timeout for sentence buffering (less relevant now)
SLOW_LLM_TIMEOUT = 60 # Timeout for slow LLM buffer flushing

# --- LLM Initialization Helper (Lazy Loading per Session) ---
def initialize_llm_chats(session_id):
    """Initializes LLM chat sessions for a given session_id if not already done."""
    log_prefix = f"[Session:{session_id[:6]}]"
    if session_id not in sessions:
        logger.error(f"{log_prefix} Attempted to initialize LLMs for non-existent session.")
        return False

    session_state = sessions[session_id]

    # Check if base client is available
    if not base_client:
        logger.error(f"{log_prefix} Base Google AI Client not available. Cannot initialize chat models.")
        session_state['quick_chat'] = None
        session_state['slow_chat'] = None
        session_state['query_chat'] = None
        return False

    # Initialize Quick Chat if needed
    if session_state.get('quick_chat') is None:
        try:
            quick_config = genai_types.GenerateContentConfig(
                temperature=0.1, top_p=0.95, top_k=40, max_output_tokens=2048,
                system_instruction=quick_llm_system_prompt
            )
            quick_model_name = os.getenv("QUICK_LLM_MODEL", "gemini-2.0-flash")
            session_state['quick_chat'] = base_client.chats.create(model=quick_model_name, config=quick_config)
            logger.info(f"{log_prefix} Initialized Quick LLM Chat: {quick_model_name}")
        except Exception as e:
            logger.error(f"{log_prefix} Failed to initialize Quick LLM: {e}", exc_info=True)
            session_state['quick_chat'] = None

    # Initialize Slow Chat if needed
    if session_state.get('slow_chat') is None:
        try:
            slow_config = genai_types.GenerateContentConfig(
                temperature=0.3, top_p=0.95, top_k=40, max_output_tokens=4096,
                system_instruction=slow_llm_system_prompt
            )
            slow_model_name = os.getenv("SLOW_LLM_MODEL", "gemini-2.5-pro-exp-03-25")
            session_state['slow_chat'] = base_client.chats.create(model=slow_model_name, config=slow_config)
            logger.info(f"{log_prefix} Initialized Slow LLM Chat: {slow_model_name}")
        except Exception as e:
            logger.error(f"{log_prefix} Failed to initialize Slow LLM: {e}", exc_info=True)
            session_state['slow_chat'] = None

    # Initialize Query Chat if needed
    if session_state.get('query_chat') is None:
        try:
            query_config = genai_types.GenerateContentConfig(
                temperature=0.3, top_p=0.95, top_k=40, max_output_tokens=2048,
                system_instruction=query_llm_system_prompt
            )
            query_model_name = os.getenv("QUERY_LLM_MODEL", "gemini-1.5-pro")
            session_state['query_chat'] = base_client.chats.create(model=query_model_name, config=query_config)
            logger.info(f"{log_prefix} Initialized Query LLM Chat: {query_model_name}")
        except Exception as e:
            logger.error(f"{log_prefix} Failed to initialize Query LLM: {e}", exc_info=True)
            session_state['query_chat'] = None

    # Return True if at least one chat model was potentially initialized (or already existed)
    return session_state['quick_chat'] or session_state['slow_chat'] or session_state['query_chat']


# --- Helper Functions (Modified for Session Context) ---
def extract_label(graph, uri_or_literal): # <-- Added graph argument
    """Helper to get a readable label from URI or Literal for display."""
    if isinstance(uri_or_literal, URIRef):
        try:
            # Use the provided graph instance for qname computation
            prefix, namespace, name = graph.compute_qname(uri_or_literal, generate=False)
            return f"{prefix}:{name}" if prefix else name
        except:
            # Fallback logic remains the same
            if '#' in uri_or_literal:
                return uri_or_literal.split('#')[-1]
            return uri_or_literal.split('/')[-1]
    elif isinstance(uri_or_literal, Literal):
        return str(uri_or_literal)
    else:
        return str(uri_or_literal)

def graph_to_visjs(graph): # <-- Takes the specific graph instance
    """Converts an rdflib Graph to Vis.js nodes and edges format, focusing on instances
       and adding specific styling for provenance elements."""
    nodes_data = {}
    edges = []
    instance_uris = set()
    schema_properties_to_ignore = {RDF.type, RDFS.subClassOf, RDFS.domain, RDFS.range, OWL.inverseOf, OWL.equivalentClass, OWL.equivalentProperty}
    schema_classes_to_ignore = {OWL.Class, RDFS.Class, RDF.Property, OWL.ObjectProperty, OWL.DatatypeProperty, RDFS.Resource, OWL.Thing}
    schema_prefixes = (str(RDF), str(RDFS), str(OWL), str(XSD))

    TRANSCRIPTION_TYPE = URIRef(str(EX) + "Transcription") # Construct full URI
    PROVENANCE_PREDICATE = URIRef(str(EX) + "derivedFromTranscript") # Construct full URI

    # Pass 1: Identify instance URIs (using the provided graph)
    for s, p, o in graph:
        s_str, p_str, o_str = str(s), str(p), str(o)
        # (Logic remains the same, but operates on the passed 'graph')
        if p == RDF.type and isinstance(s, URIRef) and isinstance(o, URIRef) and \
           o not in schema_classes_to_ignore and not s_str.startswith(schema_prefixes) and \
           not o_str.startswith(schema_prefixes): instance_uris.add(s_str)
        elif isinstance(s, URIRef) and isinstance(o, URIRef) and \
             p not in schema_properties_to_ignore and \
             not s_str.startswith(schema_prefixes) and \
             not o_str.startswith(schema_prefixes) and \
             not p_str.startswith(schema_prefixes): instance_uris.add(s_str); instance_uris.add(o_str)
        elif isinstance(s, URIRef) and isinstance(o, Literal) and \
             not s_str.startswith(schema_prefixes) and \
             not p_str.startswith(schema_prefixes): instance_uris.add(s_str)

    # Pass 2: Create nodes for identified instances
    for uri in instance_uris:
        if URIRef(uri) not in schema_classes_to_ignore and not uri.startswith(schema_prefixes):
             # Pass the graph instance to extract_label
             nodes_data[uri] = {"id": uri, "label": extract_label(graph, URIRef(uri)), "title": f"URI: {uri}\n", "group": "Instance"}

    # Pass 3: Add edges and properties, apply provenance styling (using the provided graph)
    for s, p, o in graph:
        s_str, p_str, o_str = str(s), str(p), str(o)
        if s_str in nodes_data:
            node = nodes_data[s_str]
            if o_str in nodes_data and isinstance(o, URIRef) and \
               p not in schema_properties_to_ignore and \
               not p_str.startswith(schema_prefixes):
                 edge_label = extract_label(graph, p) # Pass graph
                 edge_id = f"{s_str}_{p_str}_{o_str}"
                 edge_data = { "id": edge_id, "from": s_str, "to": o_str, "label": edge_label, "title": f"Predicate: {edge_label}", "arrows": "to" }
                 # Apply provenance styling based on the predicate
                 if p == PROVENANCE_PREDICATE:
                     edge_data["dashes"] = True
                     edge_data["color"] = 'lightgray'
                     # edge_data["label"] = "source" # Keep full label for clarity
                 edges.append(edge_data)
            elif p == RDF.type and isinstance(o, URIRef):
                if o == TRANSCRIPTION_TYPE:
                     node['group'] = "Transcription"
                     node['label'] = "Txn: " + extract_label(graph, s) # Pass graph
                elif o not in schema_classes_to_ignore and not o_str.startswith(schema_prefixes):
                     type_label = extract_label(graph, o); node['title'] += f"Type: {type_label}\n"; type_suffix = f" ({type_label})"; # Pass graph
                     if type_suffix not in node['label'] and node['label'] != type_label: node['label'] += type_suffix
                     if node.get('group') != "Transcription": node['group'] = type_label # Assign group based on type URI string
            elif isinstance(o, Literal):
                prop_label = extract_label(graph, p); lit_label = extract_label(graph, o); # Pass graph
                node['title'] += f"{prop_label}: {lit_label}\n"
                if p == RDFS.label:
                    if node.get('group') != "Transcription": node['label'] = lit_label

    # Pass 4: Create final node list and deduplicate edges
    final_nodes = []
    for node in nodes_data.values(): node['title'] = node['title'].strip(); final_nodes.append(node)
    unique_edges_set = set(); unique_edges = []
    for edge in edges:
        if 'from' in edge and 'to' in edge:
            edge_key = (edge['from'], edge['to'], edge.get('label'))
            if edge_key not in unique_edges_set: unique_edges.append(edge); unique_edges_set.add(edge_key)
        else: logger.warning(f"[System] Skipping malformed edge in graph_to_visjs: {edge}")

    return {"nodes": final_nodes, "edges": unique_edges}

def process_turtle_data(turtle_data, session_id, sid): # <-- Added session_id
    """Process Turtle data string, add new triples to the session's graph."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}]" if sid else f"[Session:{session_id[:6]}]"
    if not turtle_data or session_id not in sessions:
        if session_id not in sessions: logger.error(f"{log_prefix} Session not found for processing turtle.")
        return False

    session_graph = sessions[session_id]['graph'] # Get the session-specific graph

    try:
        turtle_data = turtle_data.strip()
        # Remove markdown fences (same logic)
        if turtle_data.startswith("```turtle"):
             turtle_data = turtle_data[len("```turtle"):].strip()
        elif turtle_data.startswith("```"):
             turtle_data = turtle_data[len("```"):].strip()
        if turtle_data.endswith("```"):
             turtle_data = turtle_data[:-len("```")].strip()
        if not turtle_data:
            return False

        # Use the session graph's namespaces
        prefixes = "\n".join(f"@prefix {p}: <{n}> ." for p, n in session_graph.namespaces())
        full_turtle_for_parsing = prefixes + "\n" + turtle_data

        temp_graph = Graph()
        temp_graph.parse(data=full_turtle_for_parsing, format="turtle")

        new_triples_count = 0
        for triple in temp_graph:
            # Add to the session graph if not already present
            if triple not in session_graph:
                 session_graph.add(triple)
                 new_triples_count += 1

        if new_triples_count > 0:
            logger.info(f"{log_prefix} Added {new_triples_count} triples. Session total: {len(session_graph)}")
            return True
        else:
             logger.info(f"{log_prefix} No new triples added to session graph.")
             return False
    except Exception as e:
         logger.error(f"{log_prefix} Turtle parse error: {e}", exc_info=False)
         logger.error(f"{log_prefix} Problematic Turtle data: {turtle_data}", exc_info=False)
         return False

def update_graph_visualization(session_id): # <-- Added session_id
    """Generates Vis.js data from the session's graph and broadcasts it to the session room."""
    log_prefix = f"[Session:{session_id[:6]}]"
    if session_id not in sessions:
        logger.warning(f"{log_prefix} Attempted to update graph for non-existent session.")
        return

    session_graph = sessions[session_id]['graph']
    session_client_sids = list(sessions[session_id]['client_buffers'].keys())

    try:
        vis_data = graph_to_visjs(session_graph) # Use the session graph
        # Emit only to the specific session room
        socketio.emit('update_graph', vis_data, room=session_id)
        logger.info(f"{log_prefix} Graph update sent to session room ({len(session_client_sids)} clients).")
    except Exception as e:
         logger.error(f"{log_prefix} Graph update error for session: {e}", exc_info=True)

# --- Local CAR/CID Generation Helper (No changes needed) ---
def generate_car_and_cid(text_data: str) -> tuple[str | None, bytes | None]:
    """Generates an IPFS CID (raw) and CAR file data locally using car_library."""
    try:
        logger.debug("Generating CAR/CID locally...")
        car_bytes = generate_car(text_data)
        if not car_bytes:
             raise ValueError("generate_car returned empty data")
        cid_bytes = generate_cid(text_data.encode('utf-8'))
        cid_string = cid_to_string(cid_bytes)
        if not cid_string or not car_bytes:
             raise ValueError("Failed to generate valid CID string or CAR bytes")
        logger.debug(f"Generated CID: {cid_string}, CAR size: {len(car_bytes)} bytes")
        return cid_string, car_bytes
    except Exception as e:
        logger.error(f"Error generating CAR/CID locally: {e}", exc_info=True)
        return None, None

# --- Lighthouse CAR Upload Helper (Modified for Session Context Logging) ---
def upload_car_to_lighthouse(car_data: bytes, cid_str: str, session_id: str, sid: str): # <-- Added session_id
    """Uploads CAR data to Lighthouse storage."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}|UploadThread]" # <-- Include session_id
    if not LIGHTHOUSE_API_KEY:
        logger.error(f"{log_prefix} Lighthouse API Key missing. Cannot upload CAR {cid_str}.")
        return
    if not car_data:
        logger.error(f"{log_prefix} No CAR data for {cid_str}. Cannot upload.")
        return

    upload_url = "https://node.lighthouse.storage/api/v0/add"
    headers = {"Authorization": f"Bearer {LIGHTHOUSE_API_KEY}"}
    files = {'file': (f'{cid_str}.car', car_data, 'application/octet-stream')}

    try:
        logger.info(f"{log_prefix} Uploading CAR {cid_str} ({len(car_data)} bytes) to Lighthouse...")
        response = requests.post(upload_url, headers=headers, files=files, timeout=180)
        response.raise_for_status()

        response_data = response.json()
        uploaded_cid = response_data.get('Hash')
        logger.info(f"{log_prefix} Uploaded CAR. Lighthouse CID: {uploaded_cid} (Local CID: {cid_str})")

    except requests.exceptions.RequestException as e:
        logger.error(f"{log_prefix} Lighthouse CAR upload failed: {e}", exc_info=True)
        # Optionally notify the client in the session room about the upload failure
        # socketio.emit('error', {'message': f'Failed to archive transcription {cid_str} to Lighthouse.'}, room=session_id)
    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error during Lighthouse CAR upload for {cid_str}: {e}", exc_info=True)


# --- LLM Processing Functions (Modified for Session Context) ---
def process_with_quick_llm(text_chunk, session_id, sid, transcription_cid=None): # <-- Added session_id
    """Processes text chunk with the session's Quick LLM."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}]"
    if session_id not in sessions:
        logger.error(f"{log_prefix} Session not found for Quick LLM processing.")
        return

    # Lazily initialize LLMs for the session if needed
    if not initialize_llm_chats(session_id):
        logger.error(f"{log_prefix} Failed to initialize LLMs for session.")
        socketio.emit('error', {'message': 'RDF generation service unavailable for this session.'}, room=sid) # Notify specific client
        return

    quick_chat = sessions[session_id].get('quick_chat')
    if not quick_chat:
        logger.error(f"{log_prefix} Quick LLM Chat Session unavailable for this session.")
        socketio.emit('error', {'message': 'RDF generation service unavailable for this session.'}, room=sid) # Notify specific client
        return

    user_message = f"[{transcription_cid}] {text_chunk}"
    logger.info(f"{log_prefix} Processing with Quick LLM (CID: {transcription_cid}): '{text_chunk[:100]}...'")

    try:
        response = quick_chat.send_message(user_message)
        turtle_response = response.text
        logger.info(f"{log_prefix} Quick LLM response received (for CID: {transcription_cid}).")
        # logger.debug(f"{log_prefix} Quick LLM response: '{turtle_response}'") # Optional: Debug log full response

        # Process data using the session's graph
        triples_added = process_turtle_data(turtle_response, session_id, sid)
        if triples_added:
            # Access the correct client buffer within the session
            client_buffers = sessions[session_id]['client_buffers']
            if sid in client_buffers:
                 client_buffers[sid]['quick_llm_results'].append(turtle_response)
                 logger.info(f"{log_prefix} Added Quick result. Buffer size: {len(client_buffers[sid]['quick_llm_results'])}")
                 check_slow_llm_chunk(session_id, sid) # Pass session_id and sid
            else:
                 logger.warning(f"{log_prefix} Client buffer missing in session.")
            update_graph_visualization(session_id) # Update graph for the session room
        else:
             logger.info(f"{log_prefix} Quick LLM done, no new triples added to session graph (for CID: {transcription_cid}).")

    except Exception as e:
        logger.error(f"{log_prefix} Error in Quick LLM (CID: {transcription_cid}): {e}", exc_info=True)
        socketio.emit('error', {'message': f'Error processing text: {e}'}, room=sid) # Notify specific client

def process_with_slow_llm(combined_quick_results_turtle, session_id, sid): # <-- Added session_id, sid
    """Processes combined Turtle results with the session's Slow LLM."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}]"
    if session_id not in sessions:
        logger.error(f"{log_prefix} Session not found for Slow LLM processing.")
        return

    # Ensure LLMs are initialized (should be if quick ran)
    if not initialize_llm_chats(session_id):
        logger.error(f"{log_prefix} Failed to initialize LLMs for session.")
        return # Error already logged by initializer

    slow_chat = sessions[session_id].get('slow_chat')
    if not slow_chat:
        logger.error(f"{log_prefix} Slow LLM Chat Session unavailable for this session.")
        # Don't emit error here, just log, as it's a background process
        return

    session_graph = sessions[session_id]['graph'] # Get session graph

    logger.info(f"{log_prefix} Processing {len(combined_quick_results_turtle.splitlines())} lines with Slow LLM.")
    try:
        # Serialize the current session graph
        current_graph_turtle = session_graph.serialize(format='turtle')
        MAX_GRAPH_CONTEXT_SLOW = 10000 # Keep context limit
        if len(current_graph_turtle) > MAX_GRAPH_CONTEXT_SLOW:
            logger.warning(f"{log_prefix} Truncating Slow LLM graph context ({len(current_graph_turtle)} > {MAX_GRAPH_CONTEXT_SLOW}).")
            current_graph_turtle = current_graph_turtle[-MAX_GRAPH_CONTEXT_SLOW:]

        # Construct input with session's graph context
        slow_llm_input = f"Existing Knowledge Graph (Session: {session_id[:6]}):\n```turtle\n{current_graph_turtle}\n```\n\nNew Information/Triples:\n```turtle\n{combined_quick_results_turtle}\n```\n\nAnalyze the new information in the context of the existing graph..."

        response = slow_chat.send_message(slow_llm_input)
        turtle_response = response.text
        logger.info(f"{log_prefix} Slow LLM response received.")
        # logger.debug(f"{log_prefix} Slow LLM response: '{turtle_response}'") # Optional: Debug log

        # Process data using the session's graph
        triples_added = process_turtle_data(turtle_response, session_id, sid)
        if triples_added:
            update_graph_visualization(session_id) # Update graph for the session room
        else:
            logger.info(f"{log_prefix} Slow LLM done, no new triples added to session graph.")
    except Exception as e:
        logger.error(f"{log_prefix} Error in Slow LLM: {e}", exc_info=True)
        # Optionally notify client? Maybe too noisy for background analysis errors.
        # socketio.emit('error', {'message': f'Background analysis error: {e}'}, room=sid)

# --- Timeout and Chunking Logic (Modified for Session Context) ---

# Note: flush_sentence_buffer is less relevant with CID-per-utterance, but updated for consistency.
def flush_sentence_buffer(session_id, sid): # <-- Added session_id
    """Forces processing of the sentence buffer due to timeout (session-specific)."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}]"
    if session_id not in sessions or sid not in sessions[session_id]['client_buffers']:
        logger.warning(f"{log_prefix} Cannot flush sentence buffer, session or client buffer missing.")
        return

    state = sessions[session_id]['client_buffers'][sid]
    state['fast_llm_timer'] = None # Clear timer flag
    if not state['sentence_buffer']:
        return # Nothing to flush

    count = len(state['sentence_buffer'])
    logger.info(f"{log_prefix} Fast timeout flushing {count} sentences from buffer.")
    sentences = list(state['sentence_buffer'])
    state['sentence_buffer'].clear()
    text = " ".join(sentences)

    # Start background task with session context
    # NOTE: CID context is lost here, which is a limitation of timeout flushing vs utterance-based processing
    socketio.start_background_task(process_with_quick_llm, text, session_id, sid, transcription_cid="timeout_flush")

def flush_quick_llm_results(session_id, sid): # <-- Added session_id
    """Forces processing of the quick LLM results buffer due to timeout (session-specific)."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}]"
    if session_id not in sessions or sid not in sessions[session_id]['client_buffers']:
        logger.warning(f"{log_prefix} Cannot flush quick results buffer, session or client buffer missing.")
        return

    state = sessions[session_id]['client_buffers'][sid]
    state['slow_llm_timer'] = None # Clear timer flag
    if not state['quick_llm_results']:
        return # Nothing to flush

    count = len(state['quick_llm_results'])
    logger.info(f"{log_prefix} Slow timeout flushing {count} quick LLM results.")
    results = list(state['quick_llm_results'])
    state['quick_llm_results'].clear()
    combined_turtle = "\n\n".join(results)

    # Start background task with session context
    socketio.start_background_task(process_with_slow_llm, combined_turtle, session_id, sid)

def schedule_fast_llm_timeout(session_id, sid): # <-- Added session_id
    """Schedules or reschedules the fast LLM timeout for a specific client in a session."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}]"
    if session_id not in sessions or sid not in sessions[session_id]['client_buffers']:
        logger.warning(f"{log_prefix} Cannot schedule fast timeout, session or client buffer missing.")
        return

    state = sessions[session_id]['client_buffers'][sid]
    # Cancel existing timer if any
    if state.get('fast_llm_timer'):
        try: state['fast_llm_timer'].cancel()
        except: pass
    # Create and start new timer
    timer = Timer(FAST_LLM_TIMEOUT, flush_sentence_buffer, args=[session_id, sid])
    timer.daemon = True # Ensure thread doesn't block exit
    timer.start()
    state['fast_llm_timer'] = timer
    # logger.debug(f"{log_prefix} Scheduled fast timeout ({FAST_LLM_TIMEOUT}s).") # Less noisy logging

def schedule_slow_llm_timeout(session_id, sid): # <-- Added session_id
    """Schedules or reschedules the slow LLM timeout for a specific client in a session."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}]"
    if session_id not in sessions or sid not in sessions[session_id]['client_buffers']:
        logger.warning(f"{log_prefix} Cannot schedule slow timeout, session or client buffer missing.")
        return

    state = sessions[session_id]['client_buffers'][sid]
    # Cancel existing timer if any
    if state.get('slow_llm_timer'):
        try: state['slow_llm_timer'].cancel()
        except: pass
    # Create and start new timer
    timer = Timer(SLOW_LLM_TIMEOUT, flush_quick_llm_results, args=[session_id, sid])
    timer.daemon = True # Ensure thread doesn't block exit
    timer.start()
    state['slow_llm_timer'] = timer
    logger.info(f"{log_prefix} Scheduled slow analysis timeout ({SLOW_LLM_TIMEOUT}s).")

# check_fast_llm_chunk remains bypassed due to CID-per-utterance approach
def check_fast_llm_chunk(session_id, sid):
     pass

def check_slow_llm_chunk(session_id, sid): # <-- Added session_id
    """Checks if the quick LLM results buffer is full for a client and processes it."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}]"
    if session_id not in sessions or sid not in sessions[session_id]['client_buffers']:
        logger.warning(f"{log_prefix} Cannot check slow chunk, session or client buffer missing.")
        return

    state = sessions[session_id]['client_buffers'][sid]
    count = len(state['quick_llm_results'])

    if count >= SLOW_LLM_CHUNK_SIZE:
        logger.info(f"{log_prefix} Slow analysis chunk size ({count}/{SLOW_LLM_CHUNK_SIZE}) reached.")
        # Cancel timeout if it exists
        if state.get('slow_llm_timer'):
            try:
                state['slow_llm_timer'].cancel()
                state['slow_llm_timer'] = None
            except: pass
        # Process the chunk
        results = list(state['quick_llm_results'])
        state['quick_llm_results'].clear()
        combined_turtle = "\n\n".join(results)
        socketio.start_background_task(process_with_slow_llm, combined_turtle, session_id, sid)
    # If buffer has items but not full, ensure timeout is scheduled
    elif count > 0 and not state.get('slow_llm_timer'):
        schedule_slow_llm_timeout(session_id, sid)


# --- Live API Interaction (Modified for Session Context) ---

def handle_transcription_result(text, session_id, sid): # <-- Added session_id
    """Processes text, generates CAR/CID, starts upload, triggers LLM processing for a session."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}]"
    if session_id not in sessions or sid not in sessions[session_id]['client_buffers']:
        logger.warning(f"{log_prefix} Transcription result for unknown session/client.")
        return
    text = text.strip()
    if not text:
        return

    logger.info(f"{log_prefix} Received Transcription: '{text[:100]}...'")

    # 1. Generate CAR and CID locally
    cid_string, car_bytes = generate_car_and_cid(text) # Use helper

    # 2. Start async upload if successful and API key exists
    if cid_string and car_bytes and LIGHTHOUSE_API_KEY:
        logger.info(f"{log_prefix} Starting async CAR upload thread for CID: {cid_string}")
        # Pass session_id to upload thread for logging context
        upload_thread = threading.Thread(
            target=upload_car_to_lighthouse,
            args=(car_bytes, cid_string, session_id, sid),
            daemon=True
        )
        upload_thread.start()
    elif not cid_string or not car_bytes:
        logger.warning(f"{log_prefix} Failed to generate CAR/CID. Proceeding without archiving.")
        cid_string = None # Ensure CID is None if generation failed
    elif not LIGHTHOUSE_API_KEY:
         logger.warning(f"{log_prefix} Lighthouse API key missing. Skipping CAR upload for CID: {cid_string}")
         # Keep cid_string as it was generated, just don't upload

    # 3. Trigger Quick LLM processing IMMEDIATELY with the text and CID (if available)
    logger.info(f"{log_prefix} Starting background task for Quick LLM (CID: {cid_string})")
    # Pass session_id to the LLM task
    socketio.start_background_task(process_with_quick_llm, text, session_id, sid, transcription_cid=cid_string)

# Helper to safely put status on the queue (No changes needed, queue is client-specific)
def put_status_update(status_queue, update_dict):
    """Safely puts status update messages onto the thread-safe queue."""
    try:
        if status_queue:
            status_queue.put_nowait(update_dict)
    except QueueFull:
        logger.warning(f"[StatusQueue] Full, dropping: {update_dict.get('event')}")
    except Exception as e:
        logger.error(f"[StatusQueue] Error putting status: {e}")

def terminate_audio_session(session_id, sid, wait_time=3.0): # <-- Added session_id
    """Forcibly terminate an audio session for a client and ensure resources are cleaned up."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}|Terminator]"
    if session_id not in sessions or sid not in sessions[session_id]['client_buffers']:
        logger.warning(f"{log_prefix} No client buffer to terminate in session.")
        return True # Nothing to do

    state = sessions[session_id]['client_buffers'][sid]

    # Step 1: Signal termination via flag
    prev_state = state.get('is_receiving_audio', False)
    state['is_receiving_audio'] = False
    logger.info(f"{log_prefix} Setting is_receiving_audio = False (was {prev_state})")

    # Step 2 & 3: Clear audio queue and wait for thread (logic remains same)
    audio_queue = state.get('audio_queue')
    if audio_queue:
        try:
            while not audio_queue.empty():
                try: audio_queue.get_nowait(); audio_queue.task_done()
                except: pass
            try: audio_queue.put_nowait(None) # Signal termination
            except: pass
        except Exception as e: logger.warning(f"{log_prefix} Error clearing audio queue: {e}")

    thread = state.get('live_session_thread')
    if thread and thread.is_alive():
        logger.info(f"{log_prefix} Waiting up to {wait_time}s for thread (ID: {thread.ident}) to terminate...")
        thread.join(timeout=wait_time)
        if thread.is_alive():
            logger.warning(f"{log_prefix} Thread (ID: {thread.ident}) failed to terminate in {wait_time}s")
        else:
            logger.info(f"{log_prefix} Thread (ID: {thread.ident}) terminated successfully.")

    # Step 4 & 5: Reset state and create fresh queues (logic remains same)
    state['live_session_thread'] = None
    state['live_session_object'] = None
    # Create fresh queues to avoid race conditions
    state['audio_queue'] = ThreadSafeQueue(maxsize=50)
    state['status_queue'] = ThreadSafeQueue(maxsize=50)

    logger.info(f"{log_prefix} Session terminated and resources reset for client.")
    return True


async def live_api_sender(session_id, sid, session, audio_queue, status_queue): # <-- Added session_id
    """Async task sending audio and handling termination (session-aware)."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}|Sender]"
    logger.info(f"{log_prefix} Starting...")
    is_active = True
    while is_active:
        try:
            # Check termination flag *before* blocking queue get
            if session_id not in sessions or sid not in sessions[session_id]['client_buffers'] or \
               not sessions[session_id]['client_buffers'][sid].get('is_receiving_audio'):
                logger.info(f"{log_prefix} Client stopped (flag check before get).")
                is_active = False
                break

            msg = audio_queue.get(block=True, timeout=1.0) # Block with timeout

            # Check termination flag again *after* getting from queue
            if session_id not in sessions or sid not in sessions[session_id]['client_buffers'] or \
               not sessions[session_id]['client_buffers'][sid].get('is_receiving_audio'):
                logger.info(f"{log_prefix} Client stopped (flag check after get).")
                is_active = False
                if msg is not None: audio_queue.task_done() # Mark task done if we got data
                break

            if msg is None:
                logger.info(f"{log_prefix} Received termination signal (None).")
                is_active = False
                audio_queue.task_done() # Mark None task done
                break

            # Process audio message
            if session:
                # logger.debug(f"{log_prefix} Sending audio chunk ({len(msg.get('data',b''))} bytes)") # Verbose
                await session.send(input=msg)
                await asyncio.sleep(0.001) # Yield control slightly
            else:
                logger.warning(f"{log_prefix} Google API session object is invalid. Cannot send.")
                await asyncio.sleep(0.1) # Wait longer if session is bad

            audio_queue.task_done() # Mark audio data task done

        except QueueEmpty:
            # Timeout occurred, loop continues and checks flag again
            continue
        except asyncio.CancelledError:
            logger.info(f"{log_prefix} Cancelled.")
            is_active = False
        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"{log_prefix} WebSocket connection closed normally by Google.")
            is_active = False
            put_status_update(status_queue, {'event': 'connection_lost', 'data': {'message': 'Google service connection closed normally.'}})
        except websockets.exceptions.ConnectionClosedError as e:
             logger.warning(f"{log_prefix} WebSocket connection closed unexpectedly by Google: {e}. Signalling connection lost.")
             is_active = False
             put_status_update(status_queue, {'event': 'connection_lost', 'data': {'message': f'Google service connection lost unexpectedly: {e}'}})
        except Exception as e:
            logger.error(f"{log_prefix} Error: {e}", exc_info=True)
            is_active = False
            put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Audio Send Error: {e}'}})

    logger.info(f"{log_prefix} Stopped.")


async def live_api_receiver(session_id, sid, session, status_queue): # <-- Added session_id
    """Async task receiving transcriptions (session-aware)."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}|Receiver]"
    logger.info(f"{log_prefix} Starting...")
    is_active = True
    current_segment = ""
    while is_active:
        try:
            # Check termination flag
            if session_id not in sessions or sid not in sessions[session_id]['client_buffers'] or \
               not sessions[session_id]['client_buffers'][sid].get('is_receiving_audio'):
                 logger.info(f"{log_prefix} Client stopped (flag check).")
                 is_active = False
                 break

            if not session:
                 logger.warning(f"{log_prefix} Google API session object is invalid. Cannot receive.")
                 await asyncio.sleep(0.5)
                 continue

            # Receive data from Google
            turn = session.receive()
            async for response in turn:
                 # Check flag again inside the loop
                 if session_id not in sessions or sid not in sessions[session_id]['client_buffers'] or \
                    not sessions[session_id]['client_buffers'][sid].get('is_receiving_audio'):
                    logger.info(f"{log_prefix} Client stopped (flag check inside receive loop).")
                    is_active = False
                    break # Break inner loop

                 # Process response text
                 if text := response.text:
                    # logger.debug(f"{log_prefix} Received text fragment: '{text}'") # Verbose
                    current_segment += text
                    # Send complete segments based on punctuation or length
                    # (Consider adding \n as a segment terminator as well)
                    if text.endswith(('.', '?', '!')) or len(current_segment) > 100:
                        segment = current_segment.strip()
                        current_segment = ""
                        if segment:
                            # logger.debug(f"{log_prefix} Sending segment: '{segment}'") # Verbose
                            put_status_update(status_queue, {'event': 'new_transcription', 'data': {'text': segment}})

            if not is_active: break # Exit outer loop if inner loop broke due to flag

            # After turn finishes, send any remaining partial segment if client still active
            if session_id in sessions and sid in sessions[session_id]['client_buffers'] and \
               sessions[session_id]['client_buffers'][sid].get('is_receiving_audio'):
                if current_segment.strip():
                     segment = current_segment.strip()
                     current_segment = ""
                     logger.debug(f"{log_prefix} Sending final segment from turn: '{segment}'")
                     put_status_update(status_queue, {'event': 'new_transcription', 'data': {'text': segment}})
            else:
                 logger.info(f"{log_prefix} Client stopped before sending final segment from turn.")
                 is_active = False # Ensure outer loop terminates

            await asyncio.sleep(0.01) # Small yield

        except asyncio.CancelledError:
            logger.info(f"{log_prefix} Cancelled.")
            is_active = False
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"{log_prefix} WebSocket connection closed: {e}")
            put_status_update(status_queue, {'event': 'connection_lost', 'data': {'message': 'Connection to Google service lost, attempting to reconnect...'}})
            is_active = False # End this receiver, let manager handle reconnect
        except google_exceptions.DeadlineExceeded:
             logger.warning(f"{log_prefix} Google API Deadline Exceeded during receive. Ending turn.")
             # Often indicates end of speech or network issue, let manager decide on reconnect
             is_active = False
             put_status_update(status_queue, {'event': 'connection_lost', 'data': {'message': 'Google service timed out. Reconnecting...'}})
        except Exception as e:
            logger.error(f"{log_prefix} Error: {e}", exc_info=True)
            put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Receive Error: {e}'}})
            is_active = False # Stop receiving on error

    # Send any final remaining segment if receiver loop exits unexpectedly but segment has data
    if current_segment.strip():
        logger.info(f"{log_prefix} Putting final remaining segment after loop exit: '{current_segment.strip()}'")
        put_status_update(status_queue, {'event': 'new_transcription', 'data': {'text': current_segment.strip()}})

    logger.info(f"{log_prefix} Stopped.")


def run_async_session_manager(session_id, sid): # <-- Added session_id
    """Wrapper function to run the asyncio manager in a separate thread (session-aware)."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}|AsyncRunner]"
    thread_id = threading.get_ident()
    threading.current_thread().name = f"AsyncMgr-{session_id[:6]}-{sid[:6]}" # Set thread name
    logger.info(f"{log_prefix} Thread started (ID: {thread_id}).")

    # Verify the session and client still exist and should be running
    if session_id not in sessions or sid not in sessions[session_id]['client_buffers'] or \
       not sessions[session_id]['client_buffers'][sid].get('is_receiving_audio'):
        logger.warning(f"{log_prefix} Client/Session stopped or invalid before thread fully started.")
        # Clean up thread reference if it points to this dying thread
        if session_id in sessions and sid in sessions[session_id]['client_buffers']:
             state = sessions[session_id]['client_buffers'][sid]
             if state.get('live_session_thread') and state['live_session_thread'].ident == thread_id:
                 state['live_session_thread'] = None
        return

    # Get state and queues for this specific client
    state = sessions[session_id]['client_buffers'][sid]
    audio_queue = state.get('audio_queue')
    status_queue = state.get('status_queue')

    if not audio_queue or not status_queue:
        logger.error(f"{log_prefix} State or Queues missing!")
        if session_id in sessions and sid in sessions[session_id]['client_buffers']:
             sessions[session_id]['client_buffers'][sid]['is_receiving_audio'] = False # Mark as stopped
             if state.get('live_session_thread') and state['live_session_thread'].ident == thread_id:
                 state['live_session_thread'] = None
        return

    # Create and manage asyncio loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    logger.info(f"{log_prefix} Created new asyncio loop for thread.")

    try:
        # Run the core session management logic
        loop.run_until_complete(manage_live_session(session_id, sid, audio_queue, status_queue))
    except Exception as e:
        logger.error(f"{log_prefix} Unhandled error in async manager loop: {e}", exc_info=True)
        # Ensure state reflects the failure
        if session_id in sessions and sid in sessions[session_id]['client_buffers']:
            sessions[session_id]['client_buffers'][sid]['is_receiving_audio'] = False
        put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Critical session error: {e}'}})
    finally:
        # Graceful asyncio loop cleanup
        try:
            logger.info(f"{log_prefix} Shutting down async generators and closing loop.")
            loop.run_until_complete(loop.shutdown_asyncgens())
            pending = asyncio.all_tasks(loop=loop)
            if pending:
                 logger.warning(f"{log_prefix} Cancelling {len(pending)} outstanding tasks.")
                 for task in pending:
                     task.cancel()
                 loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

            loop.close()
            logger.info(f"{log_prefix} Loop closed.")
        except Exception as close_err:
            logger.error(f"{log_prefix} Error during loop cleanup: {close_err}", exc_info=True)

        # Final state cleanup: Remove thread reference only if it's still pointing to this thread
        if session_id in sessions and sid in sessions[session_id]['client_buffers']:
            current_thread_ref = sessions[session_id]['client_buffers'][sid].get('live_session_thread')
            if current_thread_ref and current_thread_ref.ident == thread_id:
                logger.info(f"{log_prefix} Clearing own thread reference.")
                sessions[session_id]['client_buffers'][sid]['live_session_thread'] = None
                sessions[session_id]['client_buffers'][sid]['live_session_object'] = None
                sessions[session_id]['client_buffers'][sid]['is_receiving_audio'] = False # Ensure flag is off
            else:
                logger.info(f"{log_prefix} Not clearing thread reference - likely superseded or already cleaned up.")
        else:
             logger.info(f"{log_prefix} Session or client buffer disappeared during cleanup.")

        logger.info(f"{log_prefix} Thread finishing (ID: {thread_id}).")


async def manage_live_session(session_id, sid, audio_queue, status_queue): # <-- Added session_id
    """Manages the Google Live API session for a client within a session."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}|Manager]"
    thread_id = threading.get_ident()
    logger.info(f"{log_prefix} Async manager starting (Thread ID: {thread_id}).")

    max_retries = 5
    retry_count = 0
    base_delay = 1.0

    # Initial check: Should we even be running?
    if session_id not in sessions or sid not in sessions[session_id]['client_buffers'] or \
       not sessions[session_id]['client_buffers'][sid].get('is_receiving_audio', False):
        logger.warning(f"{log_prefix} Client/Session no longer active at manager start.")
        return

    # Check prerequisite: Live Client availability
    if not live_client:
        logger.error(f"{log_prefix} Google Live API client is not available.")
        put_status_update(status_queue, {'event': 'error', 'data': {'message': 'Server configuration error: Live transcription service unavailable.'}})
        if session_id in sessions and sid in sessions[session_id]['client_buffers']:
             sessions[session_id]['client_buffers'][sid]['is_receiving_audio'] = False # Stop flag
        return

    # Main connection and task loop
    while retry_count <= max_retries:
        # Check termination flag at the beginning of each loop iteration
        if session_id not in sessions or sid not in sessions[session_id]['client_buffers'] or \
           not sessions[session_id]['client_buffers'][sid].get('is_receiving_audio', False):
            logger.info(f"{log_prefix} Client/Session stopped. Exiting manager loop.")
            break

        # Check if this thread is still the designated one for this client
        current_thread_ref = sessions[session_id]['client_buffers'][sid].get('live_session_thread')
        if not current_thread_ref or current_thread_ref.ident != thread_id:
            logger.warning(f"{log_prefix} Thread superseded (expected: {current_thread_ref.ident if current_thread_ref else 'None'}, actual: {thread_id}). Exiting manager loop.")
            break

        session_object = None # Reset session object for this attempt
        try:
            # --- Retry Logic with Backoff ---
            if retry_count > 0:
                delay = min(base_delay * (2 ** (retry_count - 1)), 15) # Exponential backoff capped
                logger.info(f"{log_prefix} Retry {retry_count}/{max_retries}. Waiting {delay:.1f}s before reconnecting...")
                put_status_update(status_queue, {'event': 'reconnecting', 'data': {'attempt': retry_count, 'max': max_retries, 'delay': delay}})
                # Wait asynchronously, checking termination flag periodically
                start_time = time.time()
                while (time.time() - start_time < delay):
                    if session_id not in sessions or sid not in sessions[session_id]['client_buffers'] or \
                       not sessions[session_id]['client_buffers'][sid].get('is_receiving_audio', False):
                        logger.info(f"{log_prefix} Client stopped during retry delay.")
                        break # Break inner wait loop
                    await asyncio.sleep(0.1)
                if not (session_id in sessions and sid in sessions[session_id]['client_buffers'] and \
                        sessions[session_id]['client_buffers'][sid].get('is_receiving_audio', False)):
                    break # Break outer manager loop if stopped during delay

            # --- Establish Connection ---
            logger.info(f"{log_prefix} Attempting connection to Google Live API (Attempt {retry_count + 1})...")
            async with live_client.aio.live.connect(model=GOOGLE_LIVE_API_MODEL, config=LIVE_API_CONFIG) as session:
                session_object = session
                logger.info(f"{log_prefix} Connection established successfully.")

                # Check state *after* successful connection
                if session_id not in sessions or sid not in sessions[session_id]['client_buffers'] or \
                   not sessions[session_id]['client_buffers'][sid].get('is_receiving_audio', False):
                    logger.info(f"{log_prefix} Client stopped immediately after connection. Closing.")
                    break # Exit manager loop

                # Store the active session object
                sessions[session_id]['client_buffers'][sid]['live_session_object'] = session_object

                # Notify client of success (first connect or reconnect)
                if retry_count > 0:
                    put_status_update(status_queue, {'event': 'reconnected', 'data': {}})
                    logger.info(f"{log_prefix} Successfully reconnected.")
                else:
                    put_status_update(status_queue, {'event': 'audio_started', 'data': {}})
                retry_count = 0 # Reset retries on successful connection

                # --- Run Sender/Receiver Tasks ---
                try:
                    async with asyncio.TaskGroup() as tg:
                        logger.info(f"{log_prefix} Creating sender and receiver tasks.")
                        receiver_task = tg.create_task(live_api_receiver(session_id, sid, session_object, status_queue))
                        sender_task = tg.create_task(live_api_sender(session_id, sid, session_object, audio_queue, status_queue))
                        logger.info(f"{log_prefix} Sender/Receiver tasks running.")

                        # Monitor tasks and termination flag
                        while True:
                            if session_id not in sessions or sid not in sessions[session_id]['client_buffers'] or \
                               not sessions[session_id]['client_buffers'][sid].get('is_receiving_audio', False):
                                logger.info(f"{log_prefix} Termination flag detected. Cancelling tasks.")
                                # TaskGroup cancellation happens automatically on exit
                                raise asyncio.CancelledError("Client stopped")

                            # Check if tasks are still running
                            done, pending = await asyncio.wait([sender_task, receiver_task], timeout=0.5, return_when=asyncio.FIRST_COMPLETED)

                            if done:
                                logger.info(f"{log_prefix} One or more tasks completed ({len(done)} tasks).")
                                # Check results for errors
                                for task in done:
                                    if task.exception():
                                        logger.warning(f"{log_prefix} Task finished with exception: {task.exception()}")
                                # Let TaskGroup handle propagation or break to outer loop for retry
                                break # Exit monitoring loop, TaskGroup will handle cleanup

                            # If timeout occurred and no tasks finished, loop continues checking flag
                except asyncio.CancelledError as e:
                     logger.info(f"{log_prefix} Task group cancelled: {e}")
                     # Propagate cancellation if needed, or just break
                     break # Exit manager loop as client requested stop
                except Exception as group_e:
                     logger.error(f"{log_prefix} Error within task group: {group_e}", exc_info=True)
                     # Increment retry count and continue to the outer loop
                     retry_count += 1
                     continue

            # --- Post-Session Handling ---
            # If we exited the 'async with session:' block cleanly (e.g., Google closed connection), increment retry
            logger.warning(f"{log_prefix} Google session ended or tasks completed. Incrementing retry count.")
            retry_count += 1

        # --- Exception Handling for Connection/Setup ---
        except websockets.exceptions.ConnectionClosedError as e:
            retry_count += 1
            logger.warning(f"{log_prefix} WebSocket connection closed during setup/connection: {e}. Retry {retry_count}/{max_retries}")
            put_status_update(status_queue, {'event': 'connection_lost', 'data': {'message': f'Connection failed: {e}'}})
            continue # Continue to next retry iteration
        except asyncio.CancelledError:
            logger.info(f"{log_prefix} Manager cancelled during connection/wait.")
            break # Exit manager loop
        except Exception as e:
            retry_count += 1
            logger.error(f"{log_prefix} Error establishing connection or during session management: {e}", exc_info=True)
            put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Live session error: {e}'}})
            if retry_count > max_retries:
                logger.error(f"{log_prefix} Max retries ({max_retries}) reached. Giving up.")
                put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Failed to maintain connection after {max_retries} attempts.'}})
                break # Exit manager loop

    # --- Final Cleanup Actions ---
    logger.info(f"{log_prefix} Exiting manager loop (Thread ID: {thread_id}). Performing final cleanup.")

    # Ensure the client state reflects that audio is stopped
    if session_id in sessions and sid in sessions[session_id]['client_buffers']:
        sessions[session_id]['client_buffers'][sid]['is_receiving_audio'] = False
        sessions[session_id]['client_buffers'][sid]['live_session_object'] = None # Clear session object reference

    # Always signal the audio queue to unblock the sender if it's waiting
    if audio_queue:
        try: audio_queue.put_nowait(None)
        except QueueFull: logger.warning(f"{log_prefix} Audio queue full during final signal.")
        except Exception as e: logger.warning(f"{log_prefix} Error putting final None on audio queue: {e}")

    # Send a final stopped event if it wasn't sent already by an error condition
    # Check flag one last time before sending stopped event
    if session_id in sessions and sid in sessions[session_id]['client_buffers']:
        put_status_update(status_queue, {'event': 'audio_stopped', 'data': {'message': 'Session terminated or failed.'}})
    else:
        logger.info(f"{log_prefix} Client/Session gone, skipping final 'audio_stopped' event.")

    logger.info(f"{log_prefix} Async manager finished (Thread ID: {thread_id}).")


# --- SocketIO Event Handlers (Modified for Session Context) ---

def status_queue_poller(session_id, sid): # <-- Added session_id
    """Eventlet background task polling status queue for a specific client."""
    log_prefix = f"[Session:{session_id[:6]}|SID:{sid[:6]}|Poller]"
    logger.info(f"{log_prefix} Starting.")
    should_run = True
    while should_run:
        # Check if session and client buffer still exist
        if session_id not in sessions or sid not in sessions[session_id]['client_buffers']:
            logger.info(f"{log_prefix} Client or session disconnected/removed. Stopping poller.")
            should_run = False
            break

        state = sessions[session_id]['client_buffers'][sid]
        status_queue = state.get('status_queue')

        if not status_queue:
             logger.error(f"{log_prefix} Status queue missing! Stopping poller.")
             should_run = False
             break
        try:
            # Get update with a small timeout to prevent busy-waiting
            update = status_queue.get(block=True, timeout=0.1)
            event = update.get('event')
            data = update.get('data', {})

            # Process based on event type
            if event == 'new_transcription':
                 # Pass session_id and sid to handler
                 handle_transcription_result(data.get('text',''), session_id, sid)
            elif event in ['audio_started', 'audio_stopped', 'error', 'reconnecting', 'reconnected', 'connection_lost']:
                # Emit status directly to the specific client SID
                logger.debug(f"{log_prefix} Emitting status '{event}' to client.")
                socketio.emit(event, data, room=sid) # Use room=sid for direct message
            else:
                logger.warning(f"{log_prefix} Received unknown status event type: {event}")

            status_queue.task_done() # Mark task as done

        except QueueEmpty:
            # Timeout occurred, loop continues and checks session/client validity
            eventlet.sleep(0.05) # Small sleep after timeout
            continue
        except Exception as e:
             logger.error(f"{log_prefix} Error processing status queue: {e}", exc_info=True)
             eventlet.sleep(0.5) # Longer sleep on error

    logger.info(f"{log_prefix} Stopped.")

@socketio.on('connect')
def handle_connect():
    # Connection is established, but we don't know the session_id yet.
    # Client MUST send 'join_session' immediately after connecting.
    sid = request.sid
    logger.info(f"[Connect|SID:{sid[:6]}] Client connected. Waiting for 'join_session'...")
    # DO NOT initialize buffers here. Wait for join_session.

@socketio.on('join_session')
def handle_join_session(data):
    """Handles client joining a specific session room."""
    sid = request.sid
    session_id = data.get('session_id')

    if not session_id or not isinstance(session_id, str):
        logger.error(f"[Join|SID:{sid[:6]}] Invalid or missing session_id in join request. Disconnecting.")
        emit('error', {'message': 'Invalid session ID provided.'}, room=sid)
        socketio.disconnect(sid)
        return

    log_prefix = f"[Join|Session:{session_id[:6]}|SID:{sid[:6]}]"
    logger.info(f"{log_prefix} Received join request.")

    # --- Session Initialization ---
    if session_id not in sessions:
        logger.info(f"{log_prefix} Creating new session state.")
        new_graph = Graph()
        # Bind prefixes to the new session graph
        new_graph.bind("rdf", RDF)
        new_graph.bind("rdfs", RDFS)
        new_graph.bind("owl", OWL)
        new_graph.bind("xsd", XSD)
        new_graph.bind("ex", EX)
        sessions[session_id] = {
            'graph': new_graph,
            'quick_chat': None, # Lazy init
            'slow_chat': None,  # Lazy init
            'query_chat': None, # Lazy init
            'client_buffers': {} # Store client buffers here
        }
    else:
        logger.info(f"{log_prefix} Joining existing session.")

    # --- Client Buffer Initialization ---
    if sid in sessions[session_id]['client_buffers']:
        logger.warning(f"{log_prefix} Client already has buffer in this session. Re-initializing.")
        # Terminate any previous activity for this SID cleanly
        terminate_audio_session(session_id, sid) # Ensure clean state

    sessions[session_id]['client_buffers'][sid] = {
        'sentence_buffer': [], 'quick_llm_results': [],
        'fast_llm_timer': None, 'slow_llm_timer': None,
        'audio_queue': ThreadSafeQueue(maxsize=50),
        'status_queue': ThreadSafeQueue(maxsize=50),
        'live_session_thread': None,
        'live_session_object': None,
        'is_receiving_audio': False,
        'status_poller_task': None # Will be started below
    }
    logger.info(f"{log_prefix} Client buffer initialized.")

    # --- SocketIO Room and SID Mapping ---
    join_room(session_id, sid=sid)
    sid_to_session[sid] = session_id
    logger.info(f"{log_prefix} Client added to session room and SID mapped.")

    # --- Start Status Poller for this Client ---
    try:
        poller_task = socketio.start_background_task(status_queue_poller, session_id, sid)
        sessions[session_id]['client_buffers'][sid]['status_poller_task'] = poller_task
        logger.info(f"{log_prefix} Started status queue poller task.")
    except Exception as e:
        logger.error(f"{log_prefix} Failed to start status poller: {e}", exc_info=True)
        emit('error', {'message': f'Server error during session setup: {e}'}, room=sid)
        # Consider disconnecting if poller fails?

    # --- Send Initial Graph State to Newly Joined Client ---
    try:
        session_graph = sessions[session_id]['graph']
        vis_data = graph_to_visjs(session_graph)
        emit('update_graph', vis_data, room=sid) # Send only to the new client
        logger.info(f"{log_prefix} Initial graph state sent to client.")
    except Exception as e:
         logger.error(f"{log_prefix} Error sending initial graph: {e}", exc_info=True)
         emit('error', {'message': f'Error loading initial graph state: {e}'}, room=sid)


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    log_prefix_base = f"[Disconnect|SID:{sid[:6]}]"

    # Find the session this client belonged to
    session_id = sid_to_session.pop(sid, None) # Remove mapping and get session_id

    if not session_id or session_id not in sessions:
        logger.warning(f"{log_prefix_base} Client disconnected, but session_id '{session_id}' not found or already cleaned up.")
        return

    log_prefix = f"[Disconnect|Session:{session_id[:6]}|SID:{sid[:6]}]"
    logger.info(f"{log_prefix} Client disconnected.")

    # --- Clean up Client-Specific Resources ---
    if sid in sessions[session_id]['client_buffers']:
        client_state = sessions[session_id]['client_buffers'][sid]

        # Terminate audio session cleanly
        logger.info(f"{log_prefix} Initiating audio termination for disconnecting client.")
        terminate_audio_session(session_id, sid) # Handles flag, queue signal, thread join

        # Cancel LLM timers for this client
        if client_state.get('fast_llm_timer'):
            try: client_state['fast_llm_timer'].cancel()
            except: pass
        if client_state.get('slow_llm_timer'):
            try: client_state['slow_llm_timer'].cancel()
            except: pass

        # The status poller task will stop itself when it detects the buffer is gone
        logger.info(f"{log_prefix} Status poller will stop automatically.")

        # Remove the client's buffer from the session
        del sessions[session_id]['client_buffers'][sid]
        logger.info(f"{log_prefix} Client buffer removed from session.")
    else:
        logger.warning(f"{log_prefix} Client buffer was already missing from session state.")

    # --- Leave SocketIO Room ---
    try:
        leave_room(session_id, sid=sid)
        logger.info(f"{log_prefix} Client removed from session room.")
    except Exception as e:
        logger.error(f"{log_prefix} Error removing client from room: {e}")

    # --- Session Cleanup Check ---
    # Check if this was the last client in the session
    if not sessions[session_id]['client_buffers']:
        logger.info(f"{log_prefix} Last client disconnected from session. Cleaning up session resources.")
        # Add cleanup logic here (e.g., close LLM sessions if applicable, etc.)
        # For now, just remove the session entry
        del sessions[session_id]
        logger.info(f"[Session:{session_id[:6]}] Session removed.")
    else:
        remaining_clients = len(sessions[session_id]['client_buffers'])
        logger.info(f"{log_prefix} Session still active with {remaining_clients} client(s).")


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    sid = request.sid
    session_id = sid_to_session.get(sid)
    if not session_id or session_id not in sessions or sid not in sessions[session_id]['client_buffers']:
        # logger.warning(f"[AudioChunk|SID:{sid[:6]}] Received audio from unknown/invalid client/session.")
        return # Fail silently for audio chunks

    log_prefix = f"[AudioChunk|Session:{session_id[:6]}|SID:{sid[:6]}]"
    state = sessions[session_id]['client_buffers'][sid]

    if not state.get('is_receiving_audio'):
        # logger.warning(f"{log_prefix} Received audio chunk while not in receiving state.")
        return # Fail silently

    if not isinstance(data, bytes):
        logger.error(f"{log_prefix} Received audio data is not bytes.")
        return

    audio_queue = state.get('audio_queue')
    if not audio_queue:
        logger.error(f"{log_prefix} Audio queue missing!")
        return

    try:
        # Construct message expected by Google API
        msg = {"data": data, "mime_type": "audio/pcm"} # Assuming PCM audio
        audio_queue.put_nowait(msg)
        # logger.debug(f"{log_prefix} Queued audio chunk ({len(data)} bytes). Queue size: {audio_queue.qsize()}") # Verbose
    except QueueFull:
        logger.warning(f"{log_prefix} Audio queue full. Dropping chunk.")
    except Exception as e:
         logger.error(f"{log_prefix} Error queuing audio: {e}", exc_info=True)


@socketio.on('start_audio')
def handle_start_audio():
    sid = request.sid
    session_id = sid_to_session.get(sid)
    if not session_id or session_id not in sessions or sid not in sessions[session_id]['client_buffers']:
        logger.error(f"[StartAudio|SID:{sid[:6]}] 'start_audio' from unknown/invalid client/session.")
        emit('error', {'message': 'Invalid session context for starting audio.'}, room=sid)
        return

    log_prefix = f"[StartAudio|Session:{session_id[:6]}|SID:{sid[:6]}]"

    # Check if Live API is available globally
    if not live_client:
        logger.error(f"{log_prefix} Google Live API client is unavailable globally.")
        emit('error', {'message': 'Live transcription service is currently unavailable.'}, room=sid)
        return

    state = sessions[session_id]['client_buffers'][sid]

    # Ensure clean state: Terminate any potentially lingering previous session for this specific client
    logger.info(f"{log_prefix} Ensuring clean state before starting new audio session...")
    terminate_audio_session(session_id, sid) # Use the session-aware terminator

    # Start new session
    logger.info(f"{log_prefix} Received 'start_audio'. Starting fresh audio session for client.")
    state['is_receiving_audio'] = True

    # Start the async manager in its own thread
    thread = threading.Thread(target=run_async_session_manager, args=(session_id, sid,), daemon=True)
    state['live_session_thread'] = thread
    thread.start()
    logger.info(f"{log_prefix} New audio session manager thread started (ID: {thread.ident}).")
    # Note: 'audio_started' event is sent by the manager thread via status_queue on successful connection


@socketio.on('stop_audio')
def handle_stop_audio():
    sid = request.sid
    session_id = sid_to_session.get(sid)
    if not session_id or session_id not in sessions or sid not in sessions[session_id]['client_buffers']:
        logger.warning(f"[StopAudio|SID:{sid[:6]}] 'stop_audio' from unknown/invalid client/session.")
        return

    log_prefix = f"[StopAudio|Session:{session_id[:6]}|SID:{sid[:6]}]"
    state = sessions[session_id]['client_buffers'][sid]

    if not state.get('is_receiving_audio', False):
        logger.warning(f"{log_prefix} Received 'stop_audio' but client was not marked as receiving.")
        # Force cleanup anyway for robustness
        terminate_audio_session(session_id, sid)
        # Send stopped confirmation just in case client UI is stuck
        socketio.emit('audio_stopped', {'message': 'Audio recording was already stopped.'}, room=sid)
        return

    logger.info(f"{log_prefix} Received 'stop_audio'. Terminating audio session for client.")
    terminate_audio_session(session_id, sid) # This handles setting the flag and cleanup

    # Explicitly send stopped confirmation to the client immediately
    socketio.emit('audio_stopped', {'message': 'Audio recording stopped by user request.'}, room=sid)


@socketio.on('query_graph')
def handle_query_graph(data):
    sid = request.sid
    session_id = sid_to_session.get(sid)
    if not session_id or session_id not in sessions:
        logger.error(f"[Query|SID:{sid[:6]}] Query from unknown/invalid client/session.")
        emit('query_result', {'answer': "Error: Invalid session.", 'error': True}, room=sid)
        return

    log_prefix = f"[Query|Session:{session_id[:6]}|SID:{sid[:6]}]"
    query_text = data.get('query', '').strip()
    if not query_text:
        logger.warning(f"{log_prefix} Received empty query.")
        emit('query_result', {'answer': "Please enter a query.", 'error': True}, room=sid)
        return

    # Lazily initialize LLMs for the session if needed
    if not initialize_llm_chats(session_id):
        logger.error(f"{log_prefix} Failed to initialize LLMs for session query.")
        emit('query_result', {'answer': "Error: Query service unavailable for this session.", 'error': True}, room=sid)
        return

    query_chat = sessions[session_id].get('query_chat')
    if not query_chat:
        logger.error(f"{log_prefix} Query LLM Chat Session unavailable for this session.")
        emit('query_result', {'answer': "Error: Query service unavailable for this session.", 'error': True}, room=sid)
        return

    session_graph = sessions[session_id]['graph'] # Get the session's graph

    logger.info(f"{log_prefix} Received query: '{query_text}'")
    try:
        # Serialize the session graph
        current_graph_turtle = session_graph.serialize(format='turtle')
        MAX_GRAPH_CONTEXT_QUERY = 15000 # Keep limit
        if len(current_graph_turtle) > MAX_GRAPH_CONTEXT_QUERY:
            logger.warning(f"{log_prefix} Truncating Query LLM graph context ({len(current_graph_turtle)} > {MAX_GRAPH_CONTEXT_QUERY}).")
            current_graph_turtle = current_graph_turtle[-MAX_GRAPH_CONTEXT_QUERY:]

        # Construct the prompt with session context
        query_prompt = f"Knowledge Graph (Session: {session_id[:6]}):\n```turtle\n{current_graph_turtle}\n```\n\nUser Query: \"{query_text}\"\n\n---\nBased *only* on the graph..."

        # Define the background task function
        def run_query_task():
            task_log_prefix = f"[QueryTask|Session:{session_id[:6]}|SID:{sid[:6]}]"
            try:
                logger.info(f"{task_log_prefix} Sending query to LLM...")
                # Use the session-specific query_chat
                response = query_chat.send_message(query_prompt)
                answer = response.text
                logger.info(f"{task_log_prefix} Query response received from LLM.")
                # Emit result directly to the querying client
                socketio.emit('query_result', {'answer': answer, 'error': False}, room=sid)
            except Exception as e:
                 logger.error(f"{task_log_prefix} Query LLM error: {e}", exc_info=True)
                 socketio.emit('query_result', {'answer': f"Error processing query: {e}", 'error': True}, room=sid)

        # Notify client that processing has started
        emit('query_result', {'answer': "Processing query against session graph...", 'processing': True}, room=sid)
        # Start the background task
        socketio.start_background_task(run_query_task)

    except Exception as e:
        logger.error(f"{log_prefix} Error preparing query or starting task: {e}", exc_info=True)
        emit('query_result', {'answer': f"Server error preparing query: {e}", 'error': True}, room=sid)


# --- Flask Routes (Modified for Sessions) ---
@app.route('/')
def index():
    """Generates a new session ID and redirects to the session-specific viewer page."""
    new_session_id = str(uuid.uuid4())
    logger.info(f"[Index:/] New session requested. Redirecting to /session/{new_session_id}")
    return redirect(url_for('session_viewer', session_id=new_session_id))

@app.route('/session/<string:session_id>')
def session_viewer(session_id):
    """Serves the main GRAPH VIEWER page for a specific session."""
    # Basic validation of session_id format (UUID)
    try:
        uuid.UUID(session_id, version=4)
        logger.info(f"[Viewer:/session/{session_id[:6]}] Serving viewer.html")
        # Pass session_id to the template
        return render_template('viewer.html', session_id=session_id)
    except ValueError:
        logger.error(f"[Viewer] Invalid session ID format requested: {session_id}")
        return "Invalid session ID format.", 400

@app.route('/mobile/<string:session_id>')
def mobile_client(session_id):
    """Serves the MOBILE INTERFACE page for a specific session."""
    # Basic validation of session_id format (UUID)
    try:
        uuid.UUID(session_id, version=4)
        logger.info(f"[Mobile:/mobile/{session_id[:6]}] Serving mobile.html")
        # Pass session_id to the template
        return render_template('mobile.html', session_id=session_id)
    except ValueError:
        logger.error(f"[Mobile] Invalid session ID format requested: {session_id}")
        return "Invalid session ID format.", 400


# --- Main Execution ---
if __name__ == '__main__':
    logger.info(f"--- Starting VoxGraph Server (Session-Based) ---")
    logger.info(f"Configuration: SlowLLMChunk={SLOW_LLM_CHUNK_SIZE}, SlowTimeout={SLOW_LLM_TIMEOUT}s")

    # Log initial status of base clients
    live_status = "Available" if live_client else "Unavailable"
    llm_status = "Available" if base_client else "Unavailable (LLM features disabled)"
    logger.info(f"Service Status: Live API Client={live_status}, Base LLM Client={llm_status}")
    if not live_client: logger.critical("Live API Client is UNAVAILABLE. Live transcription will not function.")
    if not base_client: logger.critical("Base LLM Client is UNAVAILABLE. RDF generation and Query features will not function.")

    port = int(os.environ.get('PORT', 5001))
    host = '0.0.0.0' # Listen on all interfaces
    logger.info(f"Attempting to start server on http://{host}:{port}")
    try:
        socketio.run(app, debug=False, host=host, port=port, use_reloader=False, log_output=False) # Disable Flask/SocketIO default logging if using custom
    except OSError as e:
         if "Address already in use" in str(e):
              logger.error(f"FATAL: Port {port} is already in use. Cannot start server.")
         else:
              logger.error(f"FATAL: Failed to start server due to OS error: {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
        logger.error(f"FATAL: Unexpected error during server startup: {e}", exc_info=True)
        sys.exit(1)