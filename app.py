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
# *** Import standard threading and queue ***
import threading
from queue import Queue as ThreadSafeQueue, Empty as QueueEmpty, Full as QueueFull
from threading import Timer # Keep for LLM timeouts
import numpy as np # Keep if used elsewhere
import io
import asyncio
import requests # For direct Lighthouse API calls

# Step 3: Import Flask and related libraries
from flask import Flask, render_template, request
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
# 1. Get the logger for this application module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set the desired level for your app's logger
# 2. Create a handler (e.g., StreamHandler to output to console)
stream_handler = logging.StreamHandler(sys.stdout) # Use stdout
# 3. Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# 4. Set the formatter for the handler
stream_handler.setFormatter(formatter)
# 5. Add the handler to *your* logger
if not logger.handlers: # Avoid duplicate handlers
    logger.addHandler(stream_handler)
# logger.propagate = False # Optional: Uncomment later if you see duplicate logs

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
# (Assuming car_library.py with the provided code exists in the same directory)
try:
    from car_library import generate_cid, cid_to_string, generate_car
    logger.info("Successfully imported functions from car_library.py") # Use configured logger
except ImportError as e:
    logger.critical(f"Failed to import from car_library.py: {e}. Local CID/CAR generation will fail.") # Use configured logger
    # Define dummy functions so the app doesn't crash immediately
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

# --- Google Gemini LLM & Live Client Setup ---
client = None # For standard Chat models
live_client = None # For Live API using v1alpha
quick_chat = None # Will be initialized globally now
slow_chat = None
query_chat = None

# --- System Prompts (Modified for CID in user message) ---
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

# --- LLM/Client Initialization ---
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.critical("API Key missing.")
    else:
        logger.info("Using GOOGLE_API_KEY.")
        try:
            client = genai.Client(api_key=api_key)
            logger.info("Std Client init.")
        except Exception as e:
            logger.error(f"Standard Client init failed: {e}")
            client = None
        try:
            live_client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
            _ = live_client.aio.live.connect
            logger.info("v1alpha Client init.")
        except Exception as e:
             logger.error(f"v1alpha Client init failed ({e}).")
             live_client = None

        if client: # Initialize chat models
             # Initialize quick_chat globally
             quick_config = genai_types.GenerateContentConfig(
                 temperature=0.1, top_p=0.95, top_k=40, max_output_tokens=2048,
                 system_instruction=quick_llm_system_prompt # Use the updated system prompt
             )
             slow_config = genai_types.GenerateContentConfig(
                 temperature=0.3, top_p=0.95, top_k=40, max_output_tokens=4096,
                 system_instruction=slow_llm_system_prompt
             )
             query_config = genai_types.GenerateContentConfig(
                 temperature=0.3, top_p=0.95, top_k=40, max_output_tokens=2048,
                 system_instruction=query_llm_system_prompt
             )
             quick_model_name=os.getenv("QUICK_LLM_MODEL","gemini-2.0-flash")
             slow_model_name=os.getenv("SLOW_LLM_MODEL","gemini-2.5-pro-exp-03-25")
             query_model_name=os.getenv("QUERY_LLM_MODEL","gemini-1.5-pro")

             quick_chat = client.chats.create(model=quick_model_name, config=quick_config) # Initialize here
             slow_chat = client.chats.create(model=slow_model_name, config=slow_config)
             query_chat = client.chats.create(model=query_model_name, config=query_config)
             logger.info(f"LLM Chat models initialized - Quick: {quick_model_name}, Slow: {slow_model_name}, Query: {query_model_name}")
        else:
             logger.warning("Chat models not initialized.")
except Exception as e:
     logger.error(f"Client init error: {e}", exc_info=True)
     client=live_client=quick_chat=slow_chat=query_chat=None
     logger.warning("Proceeding with potentially limited LLM/Live functionality.")


# --- Global State ---
client_buffers = {}
accumulated_graph = Graph()
EX = URIRef("http://example.org/")
accumulated_graph.bind("rdf", RDF)
accumulated_graph.bind("rdfs", RDFS)
accumulated_graph.bind("owl", OWL)
accumulated_graph.bind("xsd", XSD)
accumulated_graph.bind("ex", EX)
SENTENCE_CHUNK_SIZE = 1
SLOW_LLM_CHUNK_SIZE = 5
FAST_LLM_TIMEOUT = 20
SLOW_LLM_TIMEOUT = 60

# --- Helper Functions ---
def extract_label(uri_or_literal):
    """Helper to get a readable label from URI or Literal for display."""
    if isinstance(uri_or_literal, URIRef):
        try:
            prefix, namespace, name = accumulated_graph.compute_qname(uri_or_literal, generate=False)
            return f"{prefix}:{name}" if prefix else name
        except:
            if '#' in uri_or_literal:
                return uri_or_literal.split('#')[-1]
            return uri_or_literal.split('/')[-1]
    elif isinstance(uri_or_literal, Literal):
        return str(uri_or_literal)
    else:
        return str(uri_or_literal)

def graph_to_visjs(graph):
    """Converts an rdflib Graph to Vis.js nodes and edges format, focusing on instances
       and adding specific styling for provenance elements."""
    nodes_data = {}
    edges = []
    instance_uris = set()
    schema_properties_to_ignore = {RDF.type, RDFS.subClassOf, RDFS.domain, RDFS.range, OWL.inverseOf, OWL.equivalentClass, OWL.equivalentProperty}
    schema_classes_to_ignore = {OWL.Class, RDFS.Class, RDF.Property, OWL.ObjectProperty, OWL.DatatypeProperty, RDFS.Resource, OWL.Thing}
    schema_prefixes = (str(RDF), str(RDFS), str(OWL), str(XSD))

    # *** Define URIs used for provenance using string concatenation ***
    # Ensure EX is defined globally like: EX = URIRef("http://example.org/")
    TRANSCRIPTION_TYPE = URIRef(str(EX) + "Transcription") # Construct full URI
    PROVENANCE_PREDICATE = URIRef(str(EX) + "sourceTranscriptionCID") # Construct full URI

    # --- Pass 1: Identify instance URIs ---
    # (No changes needed here)
    for s, p, o in graph:
        s_str, p_str, o_str = str(s), str(p), str(o)
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


    # --- Pass 2: Create nodes for identified instances ---
    # (No changes needed here)
    for uri in instance_uris:
        if URIRef(uri) not in schema_classes_to_ignore and not uri.startswith(schema_prefixes):
             nodes_data[uri] = {"id": uri, "label": extract_label(URIRef(uri)), "title": f"URI: {uri}\n", "group": "Instance"} # Default group


    # --- Pass 3: Add edges and properties, apply provenance styling ---
    # (Rest of the function remains the same as the previous corrected version)
    for s, p, o in graph:
        s_str, p_str, o_str = str(s), str(p), str(o)
        if s_str in nodes_data:
            node = nodes_data[s_str] # Get the node dict to update
            if o_str in nodes_data and isinstance(o, URIRef) and \
               p not in schema_properties_to_ignore and \
               not p_str.startswith(schema_prefixes):
                 edge_label = extract_label(p)
                 edge_id = f"{s_str}_{p_str}_{o_str}"
                 edge_data = { "id": edge_id, "from": s_str, "to": o_str, "label": edge_label, "title": f"Predicate: {edge_label}", "arrows": "to" }
                 if p == PROVENANCE_PREDICATE:
                     edge_data["dashes"] = True
                     # edge_data["label"] = "source" # Optional short label
                 edges.append(edge_data)
            elif p == RDF.type and isinstance(o, URIRef):
                if o == TRANSCRIPTION_TYPE:
                     node['group'] = "Transcription"
                     node['label'] = "Txn: " + extract_label(s)
                elif o not in schema_classes_to_ignore and not o_str.startswith(schema_prefixes):
                     type_label = extract_label(o); node['title'] += f"Type: {type_label}\n"; type_suffix = f" ({type_label})";
                     if type_suffix not in node['label'] and node['label'] != type_label: node['label'] += type_suffix
                     if node.get('group') != "Transcription": node['group'] = type_label
            elif isinstance(o, Literal):
                prop_label = extract_label(p); lit_label = extract_label(o);
                node['title'] += f"{prop_label}: {lit_label}\n"
                if p == RDFS.label:
                    if node.get('group') != "Transcription": node['label'] = lit_label

    # --- Pass 4: Create final node list and deduplicate edges ---
    final_nodes = []
    for node in nodes_data.values(): node['title'] = node['title'].strip(); final_nodes.append(node)
    unique_edges_set = set(); unique_edges = []
    for edge in edges:
        if 'from' in edge and 'to' in edge:
            edge_key = (edge['from'], edge['to'], edge.get('label'))
            if edge_key not in unique_edges_set: unique_edges.append(edge); unique_edges_set.add(edge_key)
        else: logger.warning(f"[System] Skipping malformed edge in graph_to_visjs: {edge}")

    return {"nodes": final_nodes, "edges": unique_edges}

def process_turtle_data(turtle_data, sid):
    """Process Turtle data string, add new triples to the shared accumulated_graph."""
    log_prefix = f"[SID:{sid}]" if sid else "[System]"
    if not turtle_data:
        return False
    try:
        turtle_data = turtle_data.strip()
        if turtle_data.startswith("```turtle"):
             turtle_data = turtle_data[len("```turtle"):].strip()
        elif turtle_data.startswith("```"):
             turtle_data = turtle_data[len("```"):].strip()
        if turtle_data.endswith("```"):
             turtle_data = turtle_data[:-len("```")].strip()
        if not turtle_data:
            return False
        prefixes = "\n".join(f"@prefix {p}: <{n}> ." for p, n in accumulated_graph.namespaces())
        full_turtle_for_parsing = prefixes + "\n" + turtle_data
        temp_graph = Graph()
        temp_graph.parse(data=full_turtle_for_parsing, format="turtle")
        new_triples_count = 0
        for triple in temp_graph:
            if triple not in accumulated_graph:
                 accumulated_graph.add(triple)
                 new_triples_count += 1
        if new_triples_count > 0:
            logger.info(f"{log_prefix} Added {new_triples_count} triples. Total: {len(accumulated_graph)}")
            return True
        else:
             logger.info(f"{log_prefix} No new triples added.")
             return False
    except Exception as e:
         logger.error(f"{log_prefix} Turtle parse error: {e}", exc_info=False)
         logger.error(f"{log_prefix} Turtle data: {turtle_data}", exc_info=False) # Log problematic data
         return False

def update_graph_visualization():
    """Generates Vis.js data from the shared graph and broadcasts it to all clients."""
    try:
        vis_data = graph_to_visjs(accumulated_graph)
        socketio.emit('update_graph', vis_data)
        logger.info(f"[System] Graph updated ({len(client_buffers)} clients).")
    except Exception as e:
         logger.error(f"[System] Graph update error: {e}", exc_info=True)

# --- NEW: Local CAR/CID Generation Helper ---
def generate_car_and_cid(text_data: str) -> tuple[str | None, bytes | None]:
    """Generates an IPFS CID (raw) and CAR file data locally using car_library."""
    try:
        logger.debug("Generating CAR/CID locally...")
        # Use functions imported from car_library.py
        car_bytes = generate_car(text_data) # Creates the full CAR file bytes
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

# --- NEW: Lighthouse CAR Upload Helper (Runs in Thread) ---
def upload_car_to_lighthouse(car_data: bytes, cid_str: str, sid: str):
    """Uploads CAR data to Lighthouse storage."""
    log_prefix = f"[SID:{sid}|UploadThread]"
    if not LIGHTHOUSE_API_KEY:
        logger.error(f"{log_prefix} Lighthouse API Key missing. Cannot upload CAR {cid_str}.")
        return
    if not car_data:
        logger.error(f"{log_prefix} No CAR data for {cid_str}. Cannot upload.")
        return

    upload_url = "https://node.lighthouse.storage/api/v0/add" # Check if this is still correct endpoint for CARs
    headers = {"Authorization": f"Bearer {LIGHTHOUSE_API_KEY}"}
    # Lighthouse V2 API for CAR upload might expect 'application/car' Content-Type
    # files = {'file': (f'{cid_str}.car', car_data, 'application/car')}
    # Let's try with octet-stream first as per common IPFS upload patterns
    files = {'file': (f'{cid_str}.car', car_data, 'application/octet-stream')}

    try:
        logger.info(f"{log_prefix} Uploading CAR {cid_str} ({len(car_data)} bytes) to Lighthouse...")
        response = requests.post(upload_url, headers=headers, files=files, timeout=180) # Increased timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        uploaded_cid = response_data.get('Hash')
        logger.info(f"{log_prefix} Uploaded CAR. Lighthouse CID: {uploaded_cid} (Local CID: {cid_str})")
        # It's normal for Lighthouse Hash to differ if it re-imports/re-hashes
        # if uploaded_cid != cid_str: logger.warning(f"{log_prefix} Lighthouse CID mismatch! Local: {cid_str}, Lighthouse: {uploaded_cid}")

    except requests.exceptions.RequestException as e:
        logger.error(f"{log_prefix} Lighthouse CAR upload failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"{log_prefix} Unexpected error during Lighthouse CAR upload for {cid_str}: {e}", exc_info=True)


# --- LLM Processing Functions ---
def process_with_quick_llm(text_chunk, sid, transcription_cid=None):
    """Processes text chunk with Quick LLM using persistent chat session."""
    log_prefix = f"[SID:{sid}]"
    # Use the globally initialized quick_chat object
    if not quick_chat:
        logger.error(f"{log_prefix} Quick LLM Chat Session unavailable.")
        socketio.emit('error', {'message': 'RDF generation service unavailable.'}, room=sid)
        return

    user_message = f"[{transcription_cid}] {text_chunk}"

    logger.info(f"{log_prefix} Processing with Quick LLM (CID: {transcription_cid}): '{text_chunk[:100]}...'")

    try:
        # Send message using the persistent quick_chat session
        response = quick_chat.send_message(user_message)
        turtle_response = response.text
        logger.info(f"{log_prefix} Quick LLM response received (for CID: {transcription_cid}).")
        logger.info(f"{log_prefix} Quick LLM response: '{turtle_response}'")

        triples_added = process_turtle_data(turtle_response, sid)
        if triples_added:
            if sid in client_buffers:
                 client_buffers[sid]['quick_llm_results'].append(turtle_response)
                 logger.info(f"{log_prefix} Added Quick result. Buffer: {len(client_buffers[sid]['quick_llm_results'])}")
                 check_slow_llm_chunk(sid)
            else:
                 logger.warning(f"{log_prefix} Client buffer missing.")
            update_graph_visualization()
        else:
             logger.info(f"{log_prefix} Quick LLM done, no new triples (for CID: {transcription_cid}).")

    except Exception as e:
        logger.error(f"{log_prefix} Error in Quick LLM (CID: {transcription_cid}): {e}", exc_info=True)
        socketio.emit('error', {'message': f'Error processing text: {e}'}, room=sid)

def process_with_slow_llm(combined_quick_results_turtle, sid):
    """Processes combined Turtle results with the Slow LLM for deeper analysis."""
    log_prefix = f"[SID:{sid}]"
    if not slow_chat:
        logger.error(f"{log_prefix} Slow LLM unavailable.")
        return # Don't emit error here, already logged

    logger.info(f"{log_prefix} Processing {len(combined_quick_results_turtle.splitlines())} lines with Slow LLM.")
    try:
        current_graph_turtle=accumulated_graph.serialize(format='turtle')
        MAX_GRAPH_CONTEXT_SLOW=10000
        if len(current_graph_turtle) > MAX_GRAPH_CONTEXT_SLOW:
            logger.warning(f"{log_prefix} Truncated slow context.")
            current_graph_turtle=current_graph_turtle[-MAX_GRAPH_CONTEXT_SLOW:]
        slow_llm_input = f"Existing Graph:\n```turtle\n{current_graph_turtle}\n```\n\nNew Info:\n```turtle\n{combined_quick_results_turtle}\n```\n\nAnalyze..."
        response = slow_chat.send_message(slow_llm_input)
        turtle_response = response.text
        logger.info(f"{log_prefix} Slow LLM response.")
        triples_added = process_turtle_data(turtle_response, sid)
        if triples_added:
            update_graph_visualization()
        else:
            logger.info(f"{log_prefix} Slow LLM done, no new triples.")
    except Exception as e:
        logger.error(f"{log_prefix} Error in Slow LLM: {e}", exc_info=True)
        socketio.emit('error', {'message': f'Analysis error: {e}'}, room=sid)

# --- Timeout and Chunking Logic ---
def flush_sentence_buffer(sid):
    """Forces processing of the sentence buffer due to timeout."""
    log_prefix = f"[SID:{sid}]"
    state = client_buffers.get(sid)
    if not state:
        return
    state['fast_llm_timer'] = None
    if not state['sentence_buffer']:
        return
    count = len(state['sentence_buffer'])
    logger.info(f"{log_prefix} Fast timeout flushing {count}.")
    sentences = list(state['sentence_buffer'])
    state['sentence_buffer'].clear()
    text = " ".join(sentences)
    # Note: CID context is lost on timeout flush with persistent chat
    socketio.start_background_task(process_with_quick_llm, text, sid, transcription_cid=None)

def flush_quick_llm_results(sid):
    """Forces processing of the quick LLM results buffer due to timeout."""
    log_prefix = f"[SID:{sid}]"
    state = client_buffers.get(sid)
    if not state:
        return
    state['slow_llm_timer'] = None
    if not state['quick_llm_results']:
        return
    count = len(state['quick_llm_results'])
    logger.info(f"{log_prefix} Slow timeout flushing {count}.")
    results = list(state['quick_llm_results'])
    state['quick_llm_results'].clear()
    text = "\n\n".join(results)
    socketio.start_background_task(process_with_slow_llm, text, sid)

def schedule_fast_llm_timeout(sid):
    """Schedules or reschedules the fast LLM timeout for a specific client."""
    log_prefix = f"[SID:{sid}]"
    state = client_buffers.get(sid)
    if not state:
        return
    if state.get('fast_llm_timer'):
        try:
            state['fast_llm_timer'].cancel()
        except Exception:
            pass # Ignore errors on cancel
    timer = Timer(FAST_LLM_TIMEOUT, flush_sentence_buffer, args=[sid])
    timer.daemon = True
    timer.start()
    state['fast_llm_timer'] = timer
    logger.info(f"{log_prefix} Scheduled fast timeout ({FAST_LLM_TIMEOUT}s).")

def schedule_slow_llm_timeout(sid):
    """Schedules or reschedules the slow LLM timeout for a specific client."""
    log_prefix = f"[SID:{sid}]"
    state = client_buffers.get(sid)
    if not state:
        return
    if state.get('slow_llm_timer'):
        try:
            state['slow_llm_timer'].cancel()
        except Exception:
            pass # Ignore errors on cancel
    timer = Timer(SLOW_LLM_TIMEOUT, flush_quick_llm_results, args=[sid])
    timer.daemon = True
    timer.start()
    state['slow_llm_timer'] = timer
    logger.info(f"{log_prefix} Scheduled slow timeout ({SLOW_LLM_TIMEOUT}s).")

def check_fast_llm_chunk(sid):
     """Checks if the sentence buffer is full and processes it if necessary."""
     # Bypassed when processing each transcription individually with CID
     pass

def check_slow_llm_chunk(sid):
    """Checks if the quick LLM results buffer is full and processes it."""
    log_prefix = f"[SID:{sid}]"
    state = client_buffers.get(sid)
    if not state:
        return
    count = len(state['quick_llm_results'])
    if count >= SLOW_LLM_CHUNK_SIZE:
        logger.info(f"{log_prefix} Slow chunk size reached.")
        if state.get('slow_llm_timer'):
            try:
                state['slow_llm_timer'].cancel()
                state['slow_llm_timer'] = None
            except Exception:
                pass # Ignore errors on cancel
        results = list(state['quick_llm_results'])
        state['quick_llm_results'].clear()
        text = "\n\n".join(results)
        socketio.start_background_task(process_with_slow_llm, text, sid)
    elif count > 0 and not state.get('slow_llm_timer'):
        schedule_slow_llm_timeout(sid)


# --- Live API Interaction ---

def handle_transcription_result(text, sid):
    """Processes text, generates CAR/CID, starts upload, triggers LLM processing."""
    log_prefix = f"[SID:{sid}]"
    if sid not in client_buffers:
        logger.warning(f"{log_prefix} Transcription result for unknown SID.")
        return
    text = text.strip()
    if not text:
        return

    logger.info(f"{log_prefix} Received Transcription: '{text[:100]}...'")

    # 1. Generate CAR and CID locally
    transcription_cid = cid_to_string(generate_cid(text.encode('utf-8')))

    # 2. Start async upload if successful
    if transcription_cid:
        logger.info(f"{log_prefix} Starting async CAR upload thread for CID: {transcription_cid}")
        upload_thread = threading.Thread( target=upload_car_to_lighthouse, args=(text, transcription_cid, sid), daemon=True )
        upload_thread.start()
    else:
        logger.warning(f"{log_prefix} Failed to generate CAR/CID. Proceeding without CID.")
        transcription_cid = None

    # 3. Trigger Quick LLM processing IMMEDIATELY with the text and CID
    logger.info(f"{log_prefix} Starting background task for Quick LLM (CID: {transcription_cid})")
    socketio.start_background_task(process_with_quick_llm, text, sid, transcription_cid=transcription_cid)

# Helper to safely put status on the queue from asyncio thread
def put_status_update(status_queue, update_dict):
    """Safely puts status update messages onto the thread-safe queue."""
    try:
        if status_queue:
            status_queue.put_nowait(update_dict)
    except QueueFull:
        logger.warning(f"[System|StatusQueue] Full, dropping: {update_dict.get('event')}")
    except Exception as e:
        logger.error(f"[System|StatusQueue] Error putting status: {e}")


async def live_api_sender(sid, session, audio_queue, status_queue):
    """Async task (in worker thread) sending audio and putting errors on status queue."""
    log_prefix = f"[SID:{sid}|Sender]"
    logger.info(f"{log_prefix} Starting...")
    is_active = True
    while is_active:
        try:
            msg = audio_queue.get(block=True, timeout=1.0)
            if msg is None:
                logger.info(f"{log_prefix} Term signal.")
                is_active = False
                audio_queue.task_done()
                break
            # Check flag before processing message content
            if not client_buffers.get(sid, {}).get('is_receiving_audio'):
                logger.info(f"{log_prefix} Client stopped (flag).")
                is_active = False
                audio_queue.task_done()
                break
            if session:
                await session.send(input=msg)
                await asyncio.sleep(0.001) # Yield control
            else:
                logger.warning(f"{log_prefix} Session invalid.")
                await asyncio.sleep(0.1)
            audio_queue.task_done()
        except QueueEmpty:
            if not client_buffers.get(sid, {}).get('is_receiving_audio'):
                logger.info(f"{log_prefix} Client stopped (wait).")
                is_active = False
            continue
        except asyncio.CancelledError:
            logger.info(f"{log_prefix} Cancelled.")
            is_active = False
        except Exception as e:
            logger.error(f"{log_prefix} Error: {e}", exc_info=True)
            is_active = False
            put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Audio Send Error: {e}'}})
    logger.info(f"{log_prefix} Stopped.")

async def live_api_receiver(sid, session, status_queue):
    """Async task (in worker thread) receiving transcriptions and putting them on status queue."""
    log_prefix = f"[SID:{sid}|Receiver]"
    logger.info(f"{log_prefix} Starting...")
    is_active = True
    current_segment = ""
    while is_active:
        try:
            if not client_buffers.get(sid, {}).get('is_receiving_audio'):
                 logger.info(f"{log_prefix} Client stopped (flag).")
                 is_active = False
                 break
            if not session:
                 logger.warning(f"{log_prefix} Session invalid.")
                 await asyncio.sleep(0.5)
                 continue

            turn = session.receive()
            async for response in turn:
                if not client_buffers.get(sid, {}).get('is_receiving_audio'):
                    is_active = False
                    break
                if text := response.text:
                    current_segment += text
                    if text.endswith(('.', '?', '!')) or len(current_segment) > 100:
                        segment = current_segment.strip()
                        current_segment = ""
                        if segment:
                            put_status_update(status_queue, {'event': 'new_transcription', 'data': {'text': segment}})

            if not client_buffers.get(sid, {}).get('is_receiving_audio'):
                 is_active = False
                 break
            if current_segment.strip() and is_active:
                 segment = current_segment.strip()
                 current_segment = ""
                 put_status_update(status_queue, {'event': 'new_transcription', 'data': {'text': segment}})
            await asyncio.sleep(0.01) # Yield

        except asyncio.CancelledError:
            logger.info(f"{log_prefix} Cancelled.")
            is_active = False
        except google_exceptions.StreamClosedError:
             logger.info(f"{log_prefix} API stream closed.")
             is_active = False
        except Exception as e:
            logger.error(f"{log_prefix} Error: {e}", exc_info=True)
            is_active = False
            put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Receive Error: {e}'}})
            if current_segment.strip():
                logger.info(f"{log_prefix} Putting final segment on error.")
                put_status_update(status_queue, {'event': 'new_transcription', 'data': {'text': current_segment.strip()}})

    if current_segment.strip():
        logger.info(f"{log_prefix} Putting final segment after loop.")
        put_status_update(status_queue, {'event': 'new_transcription', 'data': {'text': current_segment.strip()}})
    logger.info(f"{log_prefix} Stopped.")


def run_async_session_manager(sid):
    """Wrapper function to run the asyncio manager in a separate thread."""
    log_prefix = f"[SID:{sid}|AsyncRunner]"
    logger.info(f"{log_prefix} Thread started.")
    state = client_buffers.get(sid)
    audio_queue = state.get('audio_queue')
    status_queue = state.get('status_queue')
    if not state or not audio_queue or not status_queue:
         logger.error(f"{log_prefix} State/Queues missing!")
         return
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    logger.info(f"{log_prefix} Created new asyncio loop.")
    try:
        loop.run_until_complete(manage_live_session(sid, audio_queue, status_queue))
    except Exception as e:
        logger.error(f"{log_prefix} Unhandled error: {e}", exc_info=True)
        if sid in client_buffers:
             client_buffers[sid]['is_receiving_audio'] = False
             client_buffers[sid]['live_session_thread'] = None
        put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Critical session error: {e}'}}) # Use helper
    finally:
        try:
            logger.info(f"{log_prefix} Closing loop.")
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            logger.info(f"{log_prefix} Loop closed.")
        except Exception as close_err:
            logger.error(f"{log_prefix} Error closing loop: {close_err}", exc_info=True)
        logger.info(f"{log_prefix} Thread finished.")
        if sid in client_buffers:
             client_buffers[sid]['live_session_thread'] = None
             client_buffers[sid]['is_receiving_audio'] = False


async def manage_live_session(sid, audio_queue, status_queue):
    """Manages the Google Live API session within the asyncio thread."""
    log_prefix = f"[SID:{sid}|Manager]"
    logger.info(f"{log_prefix} Async manager starting.")
    state = client_buffers.get(sid)
    session_object = None
    if not state or not live_client:
        logger.error(f"{log_prefix} State/Live client missing!")
        put_status_update(status_queue, {'event': 'error', 'data': {'message': 'Server configuration error for live session.'}})
        return
    try:
        logger.info(f"{log_prefix} Connecting to Google Live API...")
        async with live_client.aio.live.connect(model=GOOGLE_LIVE_API_MODEL, config=LIVE_API_CONFIG) as session:
            session_object = session
            logger.info(f"{log_prefix} Connected.")
            if sid in client_buffers:
                 client_buffers[sid]['live_session_object'] = session_object
            else:
                 logger.warning(f"{log_prefix} Client gone during connect.")
                 return

            put_status_update(status_queue, {'event': 'audio_started', 'data': {}}) # Signal start via queue

            async with asyncio.TaskGroup() as tg:
                logger.info(f"{log_prefix} Creating async tasks.")
                receiver_task = tg.create_task(live_api_receiver(sid, session_object, status_queue))
                sender_task = tg.create_task(live_api_sender(sid, session_object, audio_queue, status_queue))
                logger.info(f"{log_prefix} Async tasks running (Receiver started first).")
            logger.info(f"{log_prefix} Async TaskGroup finished.")
    except asyncio.CancelledError:
        logger.info(f"{log_prefix} Cancelled.")
    except Exception as e:
        logger.error(f"{log_prefix} Error in session: {e}", exc_info=True)
        put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Live session error: {e}'}}) # Use helper
    finally:
        logger.info(f"{log_prefix} Cleaning up.")
        if sid in client_buffers:
             client_buffers[sid]['is_receiving_audio'] = False
             client_buffers[sid]['live_session_object'] = None
        if audio_queue:
            try:
                 audio_queue.put_nowait(None)
            except QueueFull:
                 pass
            except Exception as e:
                 logger.warning(f"{log_prefix} Error putting None on audio queue during cleanup: {e}")

        put_status_update(status_queue, {'event': 'audio_stopped', 'data': {'message': 'Session terminated.'}}) # Use helper


# --- SocketIO Event Handlers ---

def status_queue_poller(sid):
    """Eventlet background task polling status queue and emitting results/status."""
    log_prefix = f"[SID:{sid}|Poller]"
    logger.info(f"{log_prefix} Starting.")
    should_run = True
    while should_run:
        if sid not in client_buffers:
            logger.info(f"{log_prefix} Client disconnected.")
            should_run = False
            break

        state = client_buffers.get(sid)
        if not state:
             logger.warning(f"{log_prefix} State missing.")
             should_run = False
             break

        status_queue = state.get('status_queue')
        if not status_queue:
             logger.error(f"{log_prefix} Status queue missing!")
             should_run = False
             break
        try:
            update = status_queue.get_nowait()
            event = update.get('event')
            data = update.get('data', {})
            if event == 'new_transcription':
                 handle_transcription_result(data.get('text',''), sid)
            elif event in ['audio_started', 'audio_stopped', 'error']:
                logger.debug(f"{log_prefix} Emitting '{event}'.")
                socketio.server.emit(event, data, to=sid)
            else:
                logger.warning(f"{log_prefix} Unknown status event: {event}")
            status_queue.task_done()
        except QueueEmpty:
            eventlet.sleep(0.05)
        except Exception as e:
             logger.error(f"{log_prefix} Error processing status queue: {e}", exc_info=True)
             eventlet.sleep(0.5)
    logger.info(f"{log_prefix} Stopped.")


@socketio.on('connect')
def handle_connect():
    sid = request.sid; log_prefix = f"[SID:{sid}]"
    logger.info(f"{log_prefix} Client connected.")
    client_buffers[sid] = {
        'sentence_buffer': [], 'quick_llm_results': [], 'fast_llm_timer': None, 'slow_llm_timer': None,
        'audio_queue': ThreadSafeQueue(maxsize=50), 'status_queue': ThreadSafeQueue(maxsize=50),
        'live_session_thread': None, 'live_session_object': None, 'is_receiving_audio': False,
        'status_poller_task': None
    }
    try:
        vis_data = graph_to_visjs(accumulated_graph)
        emit('update_graph', vis_data, room=sid) # room=sid OK here
        logger.info(f"{log_prefix} State initialized, graph sent.")
        poller_task = socketio.start_background_task(status_queue_poller, sid)
        client_buffers[sid]['status_poller_task'] = poller_task
        logger.info(f"{log_prefix} Started status queue poller.")
    except Exception as e:
         logger.error(f"{log_prefix} Error during connect: {e}", exc_info=True)
         emit('error', {'message': f'Setup error: {e}'}, room=sid)


@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid; log_prefix = f"[SID:{sid}]"
    logger.info(f"{log_prefix} Client disconnected.")
    if sid in client_buffers:
        state = client_buffers[sid]; state['is_receiving_audio'] = False
        audio_queue = state.get('audio_queue')
        if audio_queue:
            try:
                audio_queue.put_nowait(None)
            except QueueFull:
                pass
            except Exception as e:
                logger.warning(f"{log_prefix} Error signalling audio queue: {e}")
        thread = state.get('live_session_thread')
        if thread and thread.is_alive():
            logger.info(f"{log_prefix} Waiting for session thread...")
            thread.join(timeout=1.0)
        poller_task = state.get('status_poller_task')
        if poller_task:
            logger.info(f"{log_prefix} Poller task will stop on next check.")
            # No need to kill explicitly

        if state.get('fast_llm_timer'): state['fast_llm_timer'].cancel()
        if state.get('slow_llm_timer'): state['slow_llm_timer'].cancel()
        del client_buffers[sid]
        logger.info(f"{log_prefix} State cleaned up.")
    else:
        logger.warning(f"{log_prefix} Disconnect for unknown SID.")


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    sid = request.sid; log_prefix = f"[SID:{sid}]"
    if sid not in client_buffers: return
    state = client_buffers[sid]
    if not state.get('is_receiving_audio'): return
    if not isinstance(data, bytes): logger.error(f"{log_prefix} Audio not bytes."); return
    audio_queue = state.get('audio_queue')
    if not audio_queue: logger.error(f"{log_prefix} Audio queue missing!"); return
    try:
        msg = {"data": data, "mime_type": "audio/pcm"}
        audio_queue.put_nowait(msg)
    except QueueFull:
        logger.warning(f"{log_prefix} Audio queue full.")
    except Exception as e:
         logger.error(f"{log_prefix} Error queuing audio: {e}", exc_info=True)


@socketio.on('start_audio')
def handle_start_audio():
    sid = request.sid; log_prefix = f"[SID:{sid}]"
    if sid not in client_buffers: logger.error(f"{log_prefix} 'start' from unknown SID."); return
    if not live_client: logger.error(f"{log_prefix} Live client unavailable."); emit('error', {'message': 'Live service unavailable.'}, room=sid); return
    state = client_buffers[sid]
    thread = state.get('live_session_thread')
    if thread and thread.is_alive(): logger.warning(f"{log_prefix} Already started."); return
    logger.info(f"{log_prefix} Received 'start_audio'. Starting session manager thread.")
    state['is_receiving_audio'] = True
    status_queue = state.get('status_queue')
    if status_queue:
        while not status_queue.empty():
            try:
                status_queue.get_nowait()
                status_queue.task_done()
            except QueueEmpty:
                break
            except Exception:
                 break # Stop clearing on error
    thread = threading.Thread(target=run_async_session_manager, args=(sid,), daemon=True)
    state['live_session_thread'] = thread
    thread.start()


@socketio.on('stop_audio')
def handle_stop_audio():
    sid = request.sid; log_prefix = f"[SID:{sid}]"
    if sid not in client_buffers: logger.warning(f"{log_prefix} 'stop' from unknown SID."); return
    state = client_buffers[sid]
    if not state.get('is_receiving_audio'): logger.warning(f"{log_prefix} Already stopped."); return
    logger.info(f"{log_prefix} Received 'stop_audio'. Signalling thread.")
    state['is_receiving_audio'] = False
    audio_queue = state.get('audio_queue')
    if audio_queue:
        try:
            audio_queue.put_nowait(None) # Signal stop
        except QueueFull:
            pass
        except Exception as e:
            logger.warning(f"{log_prefix} Error signalling queue on stop: {e}")
    # 'audio_stopped' signal now comes via the status_queue_poller


@socketio.on('query_graph')
def handle_query_graph(data):
    sid = request.sid; log_prefix = f"[SID:{sid}]"
    query_text = data.get('query', '').strip()
    if not query_text: logger.warning(f"{log_prefix} Empty query."); return
    if not query_chat: logger.error(f"{log_prefix} Query LLM unavailable."); return
    logger.info(f"{log_prefix} Received query: '{query_text}'")
    try:
        current_graph_turtle = accumulated_graph.serialize(format='turtle'); MAX_GRAPH_CONTEXT_QUERY = 15000
        if len(current_graph_turtle) > MAX_GRAPH_CONTEXT_QUERY: logger.warning(f"{log_prefix} Truncated query context."); current_graph_turtle = current_graph_turtle[-MAX_GRAPH_CONTEXT_QUERY:]
        query_prompt = f"Knowledge Graph:\n```turtle\n{current_graph_turtle}\n```\n\nUser Query: \"{query_text}\"\n\n---\nBased *only* on the graph..."
        def run_query_task():
            task_log_prefix = f"[SID:{sid}|QueryTask]"
            try:
                logger.info(f"{task_log_prefix} Sending query...")
                response = query_chat.send_message(query_prompt)
                answer = response.text
                logger.info(f"{task_log_prefix} Query response received.")
                socketio.emit('query_result', {'answer': answer, 'error': False}, room=sid) # room=sid OK here
            except Exception as e:
                 logger.error(f"{task_log_prefix} Query LLM error: {e}", exc_info=True)
                 socketio.emit('query_result', {'answer': f"Error: {e}", 'error': True}, room=sid) # room=sid OK here
        emit('query_result', {'answer': "Processing query...", 'processing': True}, room=sid) # room=sid OK here
        socketio.start_background_task(run_query_task)
    except Exception as e:
        logger.error(f"{log_prefix} Error preparing query: {e}", exc_info=True)


# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main GRAPH VIEWER page."""
    logger.info("[System] Serving viewer.html (for /)")
    return render_template('viewer.html') # Default to viewer

@app.route('/mobile')
def mobile():
    """Serves the MOBILE INTERFACE page."""
    logger.info("[System] Serving mobile.html (for /mobile)")
    return render_template('mobile.html')


# --- Main Execution ---
if __name__ == '__main__':
    logger.info("--- Starting VoxGraph Server (V14 - Persistent Quick Chat + Syntax Fix) ---") # Version marker
    logger.info(f"Configuration: SentenceChunk={SENTENCE_CHUNK_SIZE}, SlowLLMChunk={SLOW_LLM_CHUNK_SIZE}, FastTimeout={FAST_LLM_TIMEOUT}s, SlowTimeout={SLOW_LLM_TIMEOUT}s")
    llm_status = {
        "LiveAPI Client": "Available" if live_client else "Unavailable",
        "QuickLLM": "Available" if quick_chat else "Unavailable", # Check persistent chat
        "SlowLLM": "Available" if slow_chat else "Unavailable",
        "QueryLLM": "Available" if query_chat else "Unavailable"
    }
    logger.info(f"Service Status: {json.dumps(llm_status)}")
    if not live_client:
        logger.critical("Live API Client is UNAVAILABLE. Live transcription will not function.")
    else:
        logger.info("Live API Client is Available.")
    if "Unavailable" in llm_status.values():
        logger.warning("One or more LLM services unavailable.")

    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Attempting to start server on http://0.0.0.0:{port}")
    try:
        # Run using eventlet WSGI server recommended for Flask-SocketIO
        socketio.run(app, debug=True, host='0.0.0.0', port=port, use_reloader=False)
    except OSError as e:
         if "Address already in use" in str(e):
              logger.error(f"FATAL: Port {port} is already in use.")
         else:
              logger.error(f"FATAL: Failed to start server: {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
        logger.error(f"FATAL: Unexpected startup error: {e}", exc_info=True)
        sys.exit(1)