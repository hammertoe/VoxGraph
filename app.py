# -*- coding: utf-8 -*-
# Step 1: Import and enable eventlet monkey patching FIRST
import eventlet
# *** MODIFICATION: Tell eventlet NOT to patch the standard threading module ***
eventlet.monkey_patch(thread=False) # Allows standard threading for asyncio isolation

# Step 2: Now import all other modules
import os
import sys
import logging
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
try:
    # Check if the punkt tokenizer data is available
    nltk.data.find('tokenizers/punkt')
except LookupError:
    # Download the punkt tokenizer data if not found
    logging.info("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    logging.info("NLTK 'punkt' tokenizer downloaded.")
from nltk.tokenize import sent_tokenize

# --- Configuration & Setup ---
load_dotenv()

# Configure logging - REMOVED [SID:%(sid)s] from global format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Google AI Live API Specific Config ---
# Use model name from env var or default matching client example
GOOGLE_LIVE_API_MODEL = os.getenv("GOOGLE_LIVE_MODEL", "models/gemini-2.0-flash-exp")
LIVE_API_SAMPLE_RATE = 16000
LIVE_API_CHANNELS = 1
LIVE_API_CONFIG = genai_types.GenerateContentConfig(
    response_modalities=[genai_types.Modality.TEXT],
    system_instruction=genai_types.Content(
        parts=[
            genai_types.Part(
                text="""
                    You are a transcription assistant.
                    Transcribe the audio input accurately, preserving meaning.
                    Format transcription as complete sentences when possible.
                    Return ONLY the transcription text.
                    """
            )
        ]
    )
)

# Create Flask app and socketio
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Google Gemini LLM & Live Client Setup ---
client = None # For standard Chat models
live_client = None # For Live API using v1alpha
quick_chat = None
slow_chat = None
query_chat = None

# --- System Prompts (Define BEFORE use) ---
quick_llm_system_prompt = """
You will convert transcribed speech into an RDF knowledge graph using Turtle syntax.
Return only the new RDF Turtle triples representing entities and relationships mentioned in the text.
Use the 'ex:' prefix for examples (e.g., <http://example.org/>).

Follow these steps:
1. Identify entities (people, places, concepts, times, organizations, etc.).
2. Create URIs for entities using the ex: prefix and CamelCase (e.g., ex:EntityName). Use existing URIs if entities are mentioned again.
3. Identify relationships between entities (e.g., ex:worksAt, ex:locatedIn, ex:discussedConcept).
4. Identify properties of entities (e.g., rdfs:label, ex:hasValue, ex:occurredOnDate). Use appropriate datatypes for literals (e.g., "value"^^xsd:string, "123"^^xsd:integer, "2024-01-01"^^xsd:date).
5. Format as valid Turtle triples. Ensure prefixes are defined if you use them, although the parsing context will have common ones.
6. Output ONLY the Turtle syntax for the NEW information found in the text chunk. Do not repeat triples that would likely have been generated from previous text. Do not include explanatory text, markdown fences, or comments outside the Turtle syntax.

Example Input Text: "Acme Corporation announced a new project called Phoenix starting next Tuesday. Alice Johnson will lead it."

Example Output Format:
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:AcmeCorporation a ex:Organization ;
    ex:announcedProject ex:ProjectPhoenix .

ex:ProjectPhoenix a ex:Project ;
    rdfs:label "Phoenix"^^xsd:string ;
    ex:ledBy ex:AliceJohnson ;
    ex:startDate "YYYY-MM-DD"^^xsd:date . # Replace YYYY-MM-DD with actual date

ex:AliceJohnson a ex:Person .

Note: Keep your response strictly to the Turtle syntax. Focus only on triples derived directly from the input text.
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
You are a knowledge graph query assistant. You will answer user queries based *strictly* on the provided RDF knowledge graph (in Turtle format).

Follow these steps:
1. Analyze the user's query to understand the information requested.
2. Carefully examine the provided 'Knowledge Graph' section.
3. Identify relevant entities (URIs starting typically with `ex:`) and relationships (predicates like `rdf:type`, `rdfs:label`, `ex:relatesTo`, etc.) within the graph that address the query.
4. Synthesize the information found *only* within the graph into a clear, concise natural language answer.
5. **Crucially:** Explain *how* you arrived at the answer by referencing the specific entities and relationships (using their qnames like `ex:EntityName` or full URIs if needed) from the graph that support your conclusion.
6. If the information needed to answer the query is *not present* in the provided graph, explicitly state that the graph does not contain the answer. Do not make assumptions or infer information beyond what is explicitly stated in the triples.

Example Query: "Who is leading Project Phoenix?"
Example Graph Snippet:
ex:ProjectPhoenix a ex:Project ; ex:ledBy ex:AliceJohnson .
ex:AliceJohnson a ex:Person ; rdfs:label "Alice Johnson" .

Example Answer:
Based on the knowledge graph:
- The entity `ex:ProjectPhoenix` has a relationship `ex:ledBy` pointing to `ex:AliceJohnson`.
- `ex:AliceJohnson` is defined as an `ex:Person`.
Therefore, the graph indicates that Alice Johnson (`ex:AliceJohnson`) is leading Project Phoenix (`ex:ProjectPhoenix`).
"""

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.critical("CRITICAL: GOOGLE_API_KEY environment variable not set. LLM/Live features will FAIL.")
    else:
        logger.info("Using GOOGLE_API_KEY from environment variable.")
        # Standard client for Chat models (if needed)
        try:
            client = genai.Client(api_key=api_key)
            logger.info("Standard Google GenAI Client initialized.")
        except Exception as e_std:
            logger.error(f"Failed to initialize standard Google GenAI Client: {e_std}", exc_info=True)
            client = None

        # Client for Live API using v1alpha (confirmed working pattern)
        try:
            live_client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
            # Check if the expected async interface exists
            _ = live_client.aio.live.connect # Check for the specific method used in client script
            logger.info("Google GenAI Client initialized with v1alpha for Live API.")
        except Exception as e_live:
             logger.error(f"Failed to initialize Google GenAI client with v1alpha support ({e_live}). Live transcription will fail.", exc_info=True)
             live_client = None

        # Initialize Chat models only if standard client succeeded
        if client:
            # Correctly initialize GenerateContentConfig with keyword arguments
            quick_config = genai_types.GenerateContentConfig(
                temperature=0.1,
                top_p=0.95,
                top_k=40,
                max_output_tokens=2048,
                system_instruction=quick_llm_system_prompt # Use the defined variable
            )
            slow_config = genai_types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.95,
                top_k=40,
                max_output_tokens=4096,
                system_instruction=slow_llm_system_prompt # Use the defined variable
            )
            query_config = genai_types.GenerateContentConfig(
                temperature=0.3,
                top_p=0.95,
                top_k=40,
                max_output_tokens=2048,
                system_instruction=query_llm_system_prompt # Use the defined variable
            )

            # Get model names from environment or use defaults
            quick_model_name = os.getenv("QUICK_LLM_MODEL", "gemini-2.0-flash")
            slow_model_name = os.getenv("SLOW_LLM_MODEL", "gemini-2.5-pro-exp-03-25")
            query_model_name = os.getenv("QUERY_LLM_MODEL", "gemini-1.5-pro")

            # Create chat sessions
            quick_chat = client.chats.create(model=quick_model_name, config=quick_config)
            slow_chat = client.chats.create(model=slow_model_name, config=slow_config)
            query_chat = client.chats.create(model=query_model_name, config=query_config)
            logger.info(f"LLM Chat models initialized - Quick: {quick_model_name}, Slow: {slow_model_name}, Query: {query_model_name}")
        else:
            logger.warning("Chat models not initialized due to standard client failure.")

except Exception as e:
    logger.error(f"General error during Google Gemini client initialization: {e}", exc_info=True)
    client = live_client = quick_chat = slow_chat = query_chat = None
    logger.warning("Proceeding with potentially limited LLM/Live functionality.")


# --- Global State Management ---
client_buffers = {}
# Structure includes audio_queue, status_queue, and live_session_thread
# client_buffers = { 'sid': { 'sentence_buffer': [], 'quick_llm_results': [], 'fast_llm_timer': None, 'slow_llm_timer': None,
#                            'audio_queue': ThreadSafeQueue, 'status_queue': ThreadSafeQueue, 'live_session_thread': threading.Thread,
#                            'is_receiving_audio': False, 'live_session_object': <Google Live Session Object>,
#                            'status_poller_task': <GreenThread> } }
accumulated_graph = Graph()
EX = URIRef("http://example.org/")
accumulated_graph.bind("rdf", RDF); accumulated_graph.bind("rdfs", RDFS); accumulated_graph.bind("owl", OWL); accumulated_graph.bind("xsd", XSD); accumulated_graph.bind("ex", EX)
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
            return uri_or_literal.split('#')[-1] if '#' in uri_or_literal else uri_or_literal.split('/')[-1]
    elif isinstance(uri_or_literal, Literal):
        return str(uri_or_literal)
    else:
        return str(uri_or_literal)

def graph_to_visjs(graph):
    """Converts an rdflib Graph to Vis.js nodes and edges format, focusing on instances."""
    nodes_data = {} # Use dict for easier node property updates: {uri_str: node_dict}
    edges = []
    instance_uris = set() # URIs identified as instances to be visualized as nodes
    schema_properties_to_ignore = {RDF.type, RDFS.subClassOf, RDFS.domain, RDFS.range, OWL.inverseOf, OWL.equivalentClass, OWL.equivalentProperty}
    schema_classes_to_ignore = {OWL.Class, RDFS.Class, RDF.Property, OWL.ObjectProperty, OWL.DatatypeProperty, RDFS.Resource, OWL.Thing}
    schema_prefixes = (str(RDF), str(RDFS), str(OWL), str(XSD))

    for s, p, o in graph: # Pass 1: Identify instance URIs
        s_str, p_str, o_str = str(s), str(p), str(o)
        if p == RDF.type and isinstance(s, URIRef) and isinstance(o, URIRef) and o not in schema_classes_to_ignore and not s_str.startswith(schema_prefixes) and not o_str.startswith(schema_prefixes): instance_uris.add(s_str)
        elif isinstance(s, URIRef) and isinstance(o, URIRef) and p not in schema_properties_to_ignore and not s_str.startswith(schema_prefixes) and not o_str.startswith(schema_prefixes) and not p_str.startswith(schema_prefixes): instance_uris.add(s_str); instance_uris.add(o_str)
        elif isinstance(s, URIRef) and isinstance(o, Literal) and not s_str.startswith(schema_prefixes) and not p_str.startswith(schema_prefixes): instance_uris.add(s_str)

    for uri in instance_uris: # Pass 2: Create nodes for identified instances
        if URIRef(uri) not in schema_classes_to_ignore and not uri.startswith(schema_prefixes): nodes_data[uri] = {"id": uri, "label": extract_label(URIRef(uri)), "title": f"URI: {uri}\n", "group": "Instance"}

    for s, p, o in graph: # Pass 3: Add edges and properties to the created instance nodes
        s_str, p_str, o_str = str(s), str(p), str(o)
        if s_str in nodes_data:
            node = nodes_data[s_str]
            if o_str in nodes_data and isinstance(o, URIRef) and p not in schema_properties_to_ignore and not p_str.startswith(schema_prefixes):
                 edge_id = f"{s_str}_{p_str}_{o_str}"; edges.append({"id": edge_id, "from": s_str, "to": o_str, "label": extract_label(p), "title": f"Predicate: {extract_label(p)}", "arrows": "to"})
            elif p == RDF.type and isinstance(o, URIRef) and o not in schema_classes_to_ignore and not o_str.startswith(schema_prefixes):
                type_label = extract_label(o); node['title'] += f"Type: {type_label}\n"; type_suffix = f" ({type_label})";
                if type_suffix not in node['label'] and node['label'] != type_label: node['label'] += type_suffix; node['group'] = type_label
            elif isinstance(o, Literal):
                prop_label = extract_label(p); lit_label = extract_label(o); node['title'] += f"{prop_label}: {lit_label}\n";
                if p == RDFS.label: node['label'] = lit_label

    final_nodes = [] # Pass 4: Create final node list and deduplicate edges
    for node in nodes_data.values(): node['title'] = node['title'].strip(); final_nodes.append(node)
    unique_edges_set = set(); unique_edges = []
    for edge in edges:
        if 'from' in edge and 'to' in edge: # Check essential keys exist
            edge_key = (edge['from'], edge['to'], edge.get('label'))
            if edge_key not in unique_edges_set: unique_edges.append(edge); unique_edges_set.add(edge_key)
        else: logger.warning(f"[System] Skipping malformed edge in graph_to_visjs: {edge}")
    return {"nodes": final_nodes, "edges": unique_edges}


def process_turtle_data(turtle_data, sid):
    """Process Turtle data string, add new triples to the shared accumulated_graph."""
    log_prefix = f"[SID:{sid}]" if sid else "[System]" # Manual SID prefix
    if not turtle_data: logger.warning(f"{log_prefix} Empty Turtle data received."); return False
    try:
        turtle_data = turtle_data.strip() # Strip whitespace and potential markdown fences
        if turtle_data.startswith("```turtle"): turtle_data = turtle_data[len("```turtle"):].strip()
        elif turtle_data.startswith("```"): turtle_data = turtle_data[len("```"):].strip()
        if turtle_data.endswith("```"): turtle_data = turtle_data[:-len("```")].strip()
        if not turtle_data: logger.warning(f"{log_prefix} Turtle data empty after stripping."); return False

        prefixes = "\n".join(f"@prefix {p}: <{n}> ." for p, n in accumulated_graph.namespaces())
        full_turtle_for_parsing = prefixes + "\n" + turtle_data
        temp_graph = Graph(); temp_graph.parse(data=full_turtle_for_parsing, format="turtle")
        new_triples_count = 0
        for triple in temp_graph:
            if triple not in accumulated_graph: accumulated_graph.add(triple); new_triples_count += 1
        if new_triples_count > 0: logger.info(f"{log_prefix} Added {new_triples_count} new triples. Total: {len(accumulated_graph)}"); return True
        else: logger.info(f"{log_prefix} No new triples added from Turtle data."); return False
    except Exception as e:
        logger.error(f"{log_prefix} Error parsing Turtle data: {e}", exc_info=False) # Reduce noise maybe
        problem_data_preview = turtle_data[:500] + ('...' if len(turtle_data) > 500 else '')
        logger.error(f"{log_prefix} Problematic Turtle snippet:\n---\n{problem_data_preview}\n---")
        return False


def update_graph_visualization():
    """Generates Vis.js data from the shared graph and broadcasts it to all clients."""
    try:
        vis_data = graph_to_visjs(accumulated_graph)
        socketio.emit('update_graph', vis_data) # socketio emit is threadsafe
        connected_clients = len(client_buffers)
        logger.info(f"[System] Graph visualization updated ({connected_clients} clients).") # Manual prefix
    except Exception as e:
        logger.error(f"[System] Error generating/broadcasting graph: {e}", exc_info=True)
        socketio.emit('error', {'message': f'Error generating graph viz: {e}'})


# --- LLM Processing Functions (Run via socketio background tasks) ---

def process_with_quick_llm(text_chunk, sid):
    """Processes a text chunk with the Quick LLM to extract RDF triples."""
    log_prefix = f"[SID:{sid}]" # Manual prefix
    if not quick_chat: logger.error(f"{log_prefix} Quick LLM unavailable."); socketio.emit('error', {'message': 'RDF generation unavailable.'}, room=sid); return
    logger.info(f"{log_prefix} Processing text with Quick LLM: '{text_chunk[:100]}...'")
    try:
        response = quick_chat.send_message(text_chunk)
        turtle_response = response.text
        logger.info(f"{log_prefix} Quick LLM response received.")
        triples_added = process_turtle_data(turtle_response, sid) # Pass sid for logging within
        if triples_added:
            if sid in client_buffers: client_buffers[sid]['quick_llm_results'].append(turtle_response); logger.info(f"{log_prefix} Added Quick result. Buffer: {len(client_buffers[sid]['quick_llm_results'])}"); check_slow_llm_chunk(sid)
            else: logger.warning(f"{log_prefix} Client buffer missing after quick LLM.")
            update_graph_visualization() # Update graph for all clients
        else: logger.info(f"{log_prefix} Quick LLM processing complete, no new triples added.")
    except Exception as e:
        logger.error(f"{log_prefix} Error in Quick LLM processing: {e}", exc_info=True)
        socketio.emit('error', {'message': f'Error processing text: {e}'}, room=sid)

def process_with_slow_llm(combined_quick_results_turtle, sid):
    """Processes combined Turtle results with the Slow LLM for deeper analysis."""
    log_prefix = f"[SID:{sid}]" # Manual prefix
    if not slow_chat: logger.error(f"{log_prefix} Slow LLM unavailable."); socketio.emit('error', {'message': 'Graph analysis unavailable.'}, room=sid); return
    logger.info(f"{log_prefix} Processing {len(combined_quick_results_turtle.splitlines())} lines with Slow LLM.")
    try:
        current_graph_turtle = accumulated_graph.serialize(format='turtle')
        MAX_GRAPH_CONTEXT_SLOW = 10000
        if len(current_graph_turtle) > MAX_GRAPH_CONTEXT_SLOW: logger.warning(f"{log_prefix} Truncated graph context for Slow LLM."); current_graph_turtle = current_graph_turtle[-MAX_GRAPH_CONTEXT_SLOW:]
        slow_llm_input = f"Existing Graph:\n```turtle\n{current_graph_turtle}\n```\n\nNew Info:\n```turtle\n{combined_quick_results_turtle}\n```\n\nAnalyze..." # Keep full prompt
        response = slow_chat.send_message(slow_llm_input)
        turtle_response = response.text
        logger.info(f"{log_prefix} Slow LLM response received.")
        triples_added = process_turtle_data(turtle_response, sid) # Pass sid
        if triples_added: update_graph_visualization() # Update graph for all clients
        else: logger.info(f"{log_prefix} Slow LLM analysis done, no new triples.")
    except Exception as e:
        logger.error(f"{log_prefix} Error in Slow LLM processing: {e}", exc_info=True)
        socketio.emit('error', {'message': f'Error during analysis: {e}'}, room=sid)


# --- Timeout and Chunking Logic ---

def flush_sentence_buffer(sid):
    """Forces processing of the sentence buffer due to timeout."""
    log_prefix = f"[SID:{sid}]"; state = client_buffers.get(sid)
    if not state: logger.warning(f"{log_prefix} flush_sentence_buffer for unknown SID."); return
    state['fast_llm_timer'] = None # Mark timer inactive
    if not state['sentence_buffer']: logger.info(f"{log_prefix} Fast timeout, buffer empty."); return
    count = len(state['sentence_buffer']); logger.info(f"{log_prefix} Fast timeout flushing {count} sentences.")
    sentences = list(state['sentence_buffer']); state['sentence_buffer'].clear()
    text = " ".join(sentences)
    # Use socketio's background task runner (integrates with eventlet)
    socketio.start_background_task(process_with_quick_llm, text, sid)

def flush_quick_llm_results(sid):
    """Forces processing of the quick LLM results buffer due to timeout."""
    log_prefix = f"[SID:{sid}]"; state = client_buffers.get(sid)
    if not state: logger.warning(f"{log_prefix} flush_quick_llm_results for unknown SID."); return
    state['slow_llm_timer'] = None # Mark timer inactive
    if not state['quick_llm_results']: logger.info(f"{log_prefix} Slow timeout, buffer empty."); return
    count = len(state['quick_llm_results']); logger.info(f"{log_prefix} Slow timeout flushing {count} results.")
    results = list(state['quick_llm_results']); state['quick_llm_results'].clear()
    text = "\n\n".join(results)
    # Use socketio's background task runner
    socketio.start_background_task(process_with_slow_llm, text, sid)


def schedule_fast_llm_timeout(sid):
    """Schedules or reschedules the fast LLM timeout for a specific client."""
    log_prefix = f"[SID:{sid}]"; state = client_buffers.get(sid)
    if not state: logger.warning(f"{log_prefix} Cannot schedule fast timeout for unknown SID."); return
    if state.get('fast_llm_timer'):
        try: state['fast_llm_timer'].cancel()
        except Exception as e: logger.error(f"{log_prefix} Error cancelling fast timer: {e}")
    timer = Timer(FAST_LLM_TIMEOUT, flush_sentence_buffer, args=[sid]); timer.daemon = True; timer.start()
    state['fast_llm_timer'] = timer; logger.info(f"{log_prefix} Scheduled fast timeout ({FAST_LLM_TIMEOUT}s).")

def schedule_slow_llm_timeout(sid):
    """Schedules or reschedules the slow LLM timeout for a specific client."""
    log_prefix = f"[SID:{sid}]"; state = client_buffers.get(sid)
    if not state: logger.warning(f"{log_prefix} Cannot schedule slow timeout for unknown SID."); return
    if state.get('slow_llm_timer'):
        try: state['slow_llm_timer'].cancel()
        except Exception as e: logger.error(f"{log_prefix} Error cancelling slow timer: {e}")
    timer = Timer(SLOW_LLM_TIMEOUT, flush_quick_llm_results, args=[sid]); timer.daemon = True; timer.start()
    state['slow_llm_timer'] = timer; logger.info(f"{log_prefix} Scheduled slow timeout ({SLOW_LLM_TIMEOUT}s).")


def check_fast_llm_chunk(sid):
    """Checks if the sentence buffer is full and processes it if necessary."""
    log_prefix = f"[SID:{sid}]"; state = client_buffers.get(sid)
    if not state: return
    count = len(state['sentence_buffer'])
    if count >= SENTENCE_CHUNK_SIZE:
        logger.info(f"{log_prefix} Sentence chunk size ({count}/{SENTENCE_CHUNK_SIZE}) reached.")
        if state.get('fast_llm_timer'):
            try: state['fast_llm_timer'].cancel(); state['fast_llm_timer'] = None
            except Exception as e: logger.error(f"{log_prefix} Error cancelling fast timer: {e}")
        sentences = list(state['sentence_buffer']); state['sentence_buffer'].clear()
        text = " ".join(sentences)
        logger.info(f"{log_prefix} Starting background task for Quick LLM (chunk size).")
        socketio.start_background_task(process_with_quick_llm, text, sid)
    elif count > 0 and not state.get('fast_llm_timer'): schedule_fast_llm_timeout(sid)

def check_slow_llm_chunk(sid):
    """Checks if the quick LLM results buffer is full and processes it."""
    log_prefix = f"[SID:{sid}]"; state = client_buffers.get(sid)
    if not state: return
    count = len(state['quick_llm_results'])
    if count >= SLOW_LLM_CHUNK_SIZE:
        logger.info(f"{log_prefix} Slow chunk size ({count}/{SLOW_LLM_CHUNK_SIZE}) reached.")
        if state.get('slow_llm_timer'):
            try: state['slow_llm_timer'].cancel(); state['slow_llm_timer'] = None
            except Exception as e: logger.error(f"{log_prefix} Error cancelling slow timer: {e}")
        results = list(state['quick_llm_results']); state['quick_llm_results'].clear()
        text = "\n\n".join(results)
        logger.info(f"{log_prefix} Starting background task for Slow LLM (chunk size).")
        socketio.start_background_task(process_with_slow_llm, text, sid)
    elif count > 0 and not state.get('slow_llm_timer'): schedule_slow_llm_timeout(sid)


# --- Live API Interaction Functions (Running in Asyncio Thread) ---

# handle_transcription_result runs in main eventlet context now
def handle_transcription_result(text, sid):
    """Processes transcribed text. Called by the status poller."""
    log_prefix = f"[SID:{sid}]" # Manual prefix
    if sid not in client_buffers: logger.warning(f"{log_prefix} Transcription result for unknown SID."); return
    text = text.strip();
    if not text: return # Ignore empty
    logger.info(f"{log_prefix} Processing Transcription Result: '{text[:100]}...'")
    try: sentences = sent_tokenize(text)
    except Exception as e: logger.error(f"{log_prefix} NLTK error: {e}", exc_info=True); sentences = [text]
    if not sentences: logger.warning(f"{log_prefix} Zero sentences."); return
    state = client_buffers[sid]; state['sentence_buffer'].extend(sentences)
    logger.info(f"{log_prefix} Added {len(sentences)} sentence(s). Buffer: {len(state['sentence_buffer'])}")
    check_fast_llm_chunk(sid) # Trigger RDF processing pipeline

# Helper to safely put status on the queue from asyncio thread
def put_status_update(status_queue, update_dict):
    """Safely puts status update messages onto the thread-safe queue."""
    try:
        if status_queue: status_queue.put_nowait(update_dict)
    except QueueFull: logger.warning(f"[System|StatusQueue] Full, dropping: {update_dict.get('event')}")
    except Exception as e: logger.error(f"[System|StatusQueue] Error putting status: {e}")


async def live_api_sender(sid, session, audio_queue, status_queue):
    """Async task (in worker thread) sending audio and putting errors on status queue."""
    log_prefix = f"[SID:{sid}|Sender]"; logger.info(f"{log_prefix} Starting...")
    is_active = True
    while is_active:
        try:
            msg = audio_queue.get(block=True, timeout=1.0)
            if msg is None: logger.info(f"{log_prefix} Term signal."); is_active = False; audio_queue.task_done(); break
            # Check flag before processing message content
            if not client_buffers.get(sid, {}).get('is_receiving_audio'): logger.info(f"{log_prefix} Client stopped (flag)."); is_active = False; audio_queue.task_done(); break
            if session:
                await session.send(input=msg)
                # Use slightly longer sleep to ensure yield
                await asyncio.sleep(0.001) # Yield control
            else: logger.warning(f"{log_prefix} Session invalid."); await asyncio.sleep(0.1)
            audio_queue.task_done()
        except QueueEmpty:
            if not client_buffers.get(sid, {}).get('is_receiving_audio'): logger.info(f"{log_prefix} Client stopped (wait)."); is_active = False
            continue
        except asyncio.CancelledError: logger.info(f"{log_prefix} Cancelled."); is_active = False
        except Exception as e:
            logger.error(f"{log_prefix} Error: {e}", exc_info=True); is_active = False
            # Put error on status queue
            put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Audio Send Error: {e}'}})
    logger.info(f"{log_prefix} Stopped.")

async def live_api_receiver(sid, session, status_queue):
    """Async task (in worker thread) receiving transcriptions and putting them on status queue."""
    log_prefix = f"[SID:{sid}|Receiver]"; logger.info(f"{log_prefix} Starting...")
    is_active = True; current_segment = ""
    while is_active:
        try:
            # Check flag before potentially blocking on receive
            if not client_buffers.get(sid, {}).get('is_receiving_audio'): logger.info(f"{log_prefix} Client stopped (flag)."); is_active = False; break
            if not session: logger.warning(f"{log_prefix} Session invalid."); await asyncio.sleep(0.5); continue

            # Receive call might block, ensure flag check after waking
            turn = session.receive()
            async for response in turn:
                # Check flag again after async iteration wakes up
                if not client_buffers.get(sid, {}).get('is_receiving_audio'): is_active = False; break
                if text := response.text:
                    current_segment += text
                    if text.endswith(('.', '?', '!')) or len(current_segment) > 100:
                        segment = current_segment.strip(); current_segment = ""
                        if segment: put_status_update(status_queue, {'event': 'new_transcription', 'data': {'text': segment}})

            # Check flag again after turn finishes
            if not client_buffers.get(sid, {}).get('is_receiving_audio'): is_active = False; break

            if current_segment.strip() and is_active:
                 segment = current_segment.strip(); current_segment = ""
                 put_status_update(status_queue, {'event': 'new_transcription', 'data': {'text': segment}})

            # Explicit small sleep if receive doesn't block long
            await asyncio.sleep(0.01) # Adjust sleep time as needed (e.g., 0.01 or 0.05)

        except asyncio.CancelledError: logger.info(f"{log_prefix} Cancelled."); is_active = False
        except google_exceptions.StreamClosedError: logger.info(f"{log_prefix} API stream closed."); is_active = False
        except Exception as e:
            logger.error(f"{log_prefix} Error: {e}", exc_info=True); is_active = False
            put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Receive Error: {e}'}})
            if current_segment.strip(): logger.info(f"{log_prefix} Putting final segment on error."); put_status_update(status_queue, {'event': 'new_transcription', 'data': {'text': current_segment.strip()}})

    if current_segment.strip(): logger.info(f"{log_prefix} Putting final segment after loop."); put_status_update(status_queue, {'event': 'new_transcription', 'data': {'text': current_segment.strip()}})
    logger.info(f"{log_prefix} Stopped.")


def run_async_session_manager(sid):
    """Wrapper function to run the asyncio manager in a separate thread."""
    log_prefix = f"[SID:{sid}|AsyncRunner]"; logger.info(f"{log_prefix} Thread started.")
    state = client_buffers.get(sid); audio_queue = state.get('audio_queue'); status_queue = state.get('status_queue')
    if not state or not audio_queue or not status_queue: logger.error(f"{log_prefix} State/Queues missing!"); return
    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); logger.info(f"{log_prefix} Created new asyncio loop.")
    try: loop.run_until_complete(manage_live_session(sid, audio_queue, status_queue))
    except Exception as e:
        logger.error(f"{log_prefix} Unhandled error: {e}", exc_info=True)
        if sid in client_buffers: client_buffers[sid]['is_receiving_audio'] = False; client_buffers[sid]['live_session_thread'] = None
        put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Critical session error: {e}'}}) # Use helper
    finally:
        try: logger.info(f"{log_prefix} Closing loop."); loop.run_until_complete(loop.shutdown_asyncgens()); loop.close(); logger.info(f"{log_prefix} Loop closed.")
        except Exception as close_err: logger.error(f"{log_prefix} Error closing loop: {close_err}", exc_info=True)
        logger.info(f"{log_prefix} Thread finished.")
        if sid in client_buffers: client_buffers[sid]['live_session_thread'] = None; client_buffers[sid]['is_receiving_audio'] = False


async def manage_live_session(sid, audio_queue, status_queue):
    """Manages the Google Live API session within the asyncio thread."""
    log_prefix = f"[SID:{sid}|Manager]"; logger.info(f"{log_prefix} Async manager starting.")
    state = client_buffers.get(sid); session_object = None
    if not state or not live_client: logger.error(f"{log_prefix} State/Live client missing!"); return
    try:
        logger.info(f"{log_prefix} Connecting to Google Live API...")
        async with live_client.aio.live.connect(model=GOOGLE_LIVE_API_MODEL, config=LIVE_API_CONFIG) as session:
            session_object = session; logger.info(f"{log_prefix} Connected.")
            if sid in client_buffers: client_buffers[sid]['live_session_object'] = session_object
            else: logger.warning(f"{log_prefix} Client gone during connect."); return

            put_status_update(status_queue, {'event': 'audio_started', 'data': {}}) # Signal start via queue

            async with asyncio.TaskGroup() as tg:
                logger.info(f"{log_prefix} Creating async tasks.")
                # Start receiver FIRST, then sender
                receiver_task = tg.create_task(live_api_receiver(sid, session_object, status_queue))
                sender_task = tg.create_task(live_api_sender(sid, session_object, audio_queue, status_queue))
                logger.info(f"{log_prefix} Async tasks running (Receiver started first).")
            logger.info(f"{log_prefix} Async TaskGroup finished.")
    except asyncio.CancelledError: logger.info(f"{log_prefix} Cancelled.")
    except Exception as e:
        logger.error(f"{log_prefix} Error in session: {e}", exc_info=True)
        put_status_update(status_queue, {'event': 'error', 'data': {'message': f'Live session error: {e}'}}) # Use helper
    finally:
        logger.info(f"{log_prefix} Cleaning up.")
        if sid in client_buffers: client_buffers[sid]['is_receiving_audio'] = False; client_buffers[sid]['live_session_object'] = None
        if audio_queue:
            try:
                 audio_queue.put_nowait(None) # Ensure termination signal sent
            except QueueFull:
                 pass # Ignore if already full
            except Exception as e: # Catch other potential queue errors
                 logger.warning(f"{log_prefix} Error putting None on audio queue during cleanup: {e}")

        put_status_update(status_queue, {'event': 'audio_stopped', 'data': {'message': 'Session terminated.'}}) # Use helper


# --- SocketIO Event Handlers (Running in Eventlet Context) ---

def status_queue_poller(sid):
    """Eventlet background task to poll the status queue for a client."""
    log_prefix = f"[SID:{sid}|Poller]"; logger.info(f"{log_prefix} Starting.")
    should_run = True
    while should_run:
        # Check disconnect first
        if sid not in client_buffers:
            logger.info(f"{log_prefix} Client disconnected, stopping poller.")
            should_run = False
            break

        # Poller should keep running as long as the client *might* be connected
        # It will process any final messages put on the queue during cleanup.

        state = client_buffers.get(sid) # Get current state inside loop, check existence
        if not state: # Should be caught by above check, but safety first
             logger.warning(f"{log_prefix} Client buffer missing unexpectedly, stopping poller.")
             should_run = False
             break

        status_queue = state.get('status_queue')
        if not status_queue: logger.error(f"{log_prefix} Status queue missing!"); should_run = False; break

        try:
            update = status_queue.get_nowait() # Non-blocking get
            event = update.get('event'); data = update.get('data', {})
            if event == 'new_transcription':
                handle_transcription_result(data.get('text',''), sid)
            elif event in ['audio_started', 'audio_stopped', 'error']:
                logger.debug(f"{log_prefix} Emitting '{event}' based on status queue.")
                # *** FIX: Use socketio.server.emit for background tasks ***
                socketio.server.emit(event, data, to=sid)
            else:
                logger.warning(f"{log_prefix} Unknown event type from status queue: {event}")
            status_queue.task_done()
        except QueueEmpty:
             eventlet.sleep(0.05) # Yield if queue is empty
        except Exception as e:
             logger.error(f"{log_prefix} Error processing status queue: {e}", exc_info=True); eventlet.sleep(0.5)
    logger.info(f"{log_prefix} Stopped.")


@socketio.on('connect')
def handle_connect():
    """Handles a new client connection."""
    sid = request.sid; log_prefix = f"[SID:{sid}]"
    logger.info(f"{log_prefix} Client connected.")
    client_buffers[sid] = {
        'sentence_buffer': [], 'quick_llm_results': [], 'fast_llm_timer': None, 'slow_llm_timer': None,
        'audio_queue': ThreadSafeQueue(maxsize=50), 'status_queue': ThreadSafeQueue(maxsize=50), # Add status queue
        'live_session_thread': None, 'live_session_object': None, 'is_receiving_audio': False,
        'status_poller_task': None # Add poller task reference
    }
    try:
        vis_data = graph_to_visjs(accumulated_graph)
        emit('update_graph', vis_data, room=sid) # room=sid is OK here (in request context)
        logger.info(f"{log_prefix} State initialized, graph sent.")
        # Start the status queue poller
        poller_task = socketio.start_background_task(status_queue_poller, sid)
        client_buffers[sid]['status_poller_task'] = poller_task
        logger.info(f"{log_prefix} Started status queue poller.")
    except Exception as e: logger.error(f"{log_prefix} Error during connect: {e}", exc_info=True); emit('error', {'message': f'Setup error: {e}'}, room=sid)


@socketio.on('disconnect')
def handle_disconnect():
    """Handles a client disconnection."""
    sid = request.sid; log_prefix = f"[SID:{sid}]"
    logger.info(f"{log_prefix} Client disconnected.")
    if sid in client_buffers:
        state = client_buffers[sid]; state['is_receiving_audio'] = False # Signal flag FIRST
        audio_queue = state.get('audio_queue')
        if audio_queue:
            try: audio_queue.put_nowait(None) # Signal sender stop
            except QueueFull: pass
            except Exception as e: logger.warning(f"{log_prefix} Error signalling audio queue: {e}")
        thread = state.get('live_session_thread')
        if thread and thread.is_alive(): logger.info(f"{log_prefix} Waiting for session thread..."); thread.join(timeout=1.0);

        poller_task = state.get('status_poller_task')
        # *** Remove poller task kill - let it exit naturally based on client_buffers check ***
        # if poller_task:
        #      try:
        #          logger.info(f"{log_prefix} Killing status poller task.")
        #          poller_task.kill() # This caused error
        #      except Exception as e:
        #          logger.error(f"{log_prefix} Error killing poller task: {e}", exc_info=True) # Log if kill fails

        if state.get('fast_llm_timer'): state['fast_llm_timer'].cancel()
        if state.get('slow_llm_timer'): state['slow_llm_timer'].cancel()
        del client_buffers[sid]; logger.info(f"{log_prefix} State cleaned up.")
    else: logger.warning(f"{log_prefix} Disconnect for unknown SID.")


@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Receives audio chunk, puts it on the thread-safe audio queue."""
    # (Implementation remains the same)
    sid = request.sid; log_prefix = f"[SID:{sid}]"
    if sid not in client_buffers: return
    state = client_buffers[sid]
    if not state.get('is_receiving_audio'): return
    if not isinstance(data, bytes): logger.error(f"{log_prefix} Audio not bytes."); return
    audio_queue = state.get('audio_queue')
    if not audio_queue: logger.error(f"{log_prefix} Audio queue missing!"); return
    try: msg = {"data": data, "mime_type": "audio/pcm"}; audio_queue.put_nowait(msg)
    except QueueFull: logger.warning(f"{log_prefix} Audio queue full.")
    except Exception as e: logger.error(f"{log_prefix} Error queuing audio: {e}", exc_info=True)


@socketio.on('start_audio')
def handle_start_audio():
    """Starts the background thread to manage the asyncio session."""
    # (Implementation remains the same)
    sid = request.sid; log_prefix = f"[SID:{sid}]"
    if sid not in client_buffers: logger.error(f"{log_prefix} 'start' from unknown SID."); return
    if not live_client: logger.error(f"{log_prefix} Live client unavailable."); emit('error', {'message': 'Live service unavailable.'}, room=sid); return
    state = client_buffers[sid]
    thread = state.get('live_session_thread')
    if thread and thread.is_alive(): logger.warning(f"{log_prefix} Already started."); return
    logger.info(f"{log_prefix} Received 'start_audio'. Starting session manager thread.")
    state['is_receiving_audio'] = True
    status_queue = state.get('status_queue') # Clear status queue
    if status_queue:
        while not status_queue.empty():
            try: status_queue.get_nowait(); status_queue.task_done()
            except QueueEmpty: break
            except Exception: break # Stop clearing on error
    thread = threading.Thread(target=run_async_session_manager, args=(sid,), daemon=True)
    state['live_session_thread'] = thread; thread.start()


@socketio.on('stop_audio')
def handle_stop_audio():
    """Signals the background thread to stop the session."""
    # (Implementation remains the same)
    sid = request.sid; log_prefix = f"[SID:{sid}]"
    if sid not in client_buffers: logger.warning(f"{log_prefix} 'stop' from unknown SID."); return
    state = client_buffers[sid]
    if not state.get('is_receiving_audio'): logger.warning(f"{log_prefix} Already stopped."); return
    logger.info(f"{log_prefix} Received 'stop_audio'. Signalling thread.")
    state['is_receiving_audio'] = False
    audio_queue = state.get('audio_queue')
    if audio_queue:
        try: audio_queue.put_nowait(None) # Signal stop
        except QueueFull: pass
        except Exception as e: logger.warning(f"{log_prefix} Error signalling queue on stop: {e}")
    # 'audio_stopped' signal now comes via the status_queue_poller


@socketio.on('query_graph')
def handle_query_graph(data):
    """Handles a natural language query about the graph."""
    # (Implementation remains the same)
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
            try: logger.info(f"{task_log_prefix} Sending query..."); response = query_chat.send_message(query_prompt); answer = response.text; logger.info(f"{task_log_prefix} Query response received."); socketio.emit('query_result', {'answer': answer, 'error': False}, room=sid)
            except Exception as e: logger.error(f"{task_log_prefix} Query LLM error: {e}", exc_info=True); socketio.emit('query_result', {'answer': f"Error: {e}", 'error': True}, room=sid)
        emit('query_result', {'answer': "Processing query...", 'processing': True}, room=sid)
        socketio.start_background_task(run_query_task)
    except Exception as e: logger.error(f"{log_prefix} Error preparing query: {e}", exc_info=True)


# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main GRAPH VIEWER page."""
    logger.info("[System] Serving viewer.html (for /)")
    # Rename your original index.html to viewer.html in the templates folder
    return render_template('viewer.html')

@app.route('/mobile')
def mobile():
    """Serves the MOBILE INTERFACE page."""
    logger.info("[System] Serving mobile.html (for /mobile)")
    # Create a new mobile.html file in the templates folder
    return render_template('mobile.html')

# --- Main Execution ---
if __name__ == '__main__':
    logger.info("--- Starting VoxGraph Server (Threaded Asyncio V9 - Emit Fix) ---") # Version marker
    logger.info(f"Configuration: SentenceChunk={SENTENCE_CHUNK_SIZE}, SlowLLMChunk={SLOW_LLM_CHUNK_SIZE}, FastTimeout={FAST_LLM_TIMEOUT}s, SlowTimeout={SLOW_LLM_TIMEOUT}s")
    llm_status = {
        "LiveAPI Client": "Available" if live_client else "Unavailable",
        "QuickLLM": "Available" if quick_chat else "Unavailable",
        "SlowLLM": "Available" if slow_chat else "Unavailable",
        "QueryLLM": "Available" if query_chat else "Unavailable"
    }
    logger.info(f"Service Status: {json.dumps(llm_status)}")
    if not live_client:
        logger.critical("Live API Client is UNAVAILABLE. Live transcription will not function.")
    else:
        logger.info("Live API Client is Available.")
    if "Unavailable" in [llm_status['QuickLLM'], llm_status['SlowLLM'], llm_status['QueryLLM']]:
        logger.warning("One or more standard LLM services unavailable.")

    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Attempting to start server on http://0.0.0.0:{port}")
    try:
        # Run using eventlet WSGI server recommended for Flask-SocketIO
        socketio.run(app, debug=True, host='0.0.0.0', port=port, use_reloader=False)
    except OSError as e:
         if "Address already in use" in str(e):
              logger.error(f"FATAL: Port {port} is already in use. Please stop the other process or choose a different port.")
         else:
              logger.error(f"FATAL: Failed to start server: {e}", exc_info=True)
         sys.exit(1) # Exit if server cannot start
    except Exception as e:
        logger.error(f"FATAL: An unexpected error occurred during server startup: {e}", exc_info=True)
        sys.exit(1)