# Step 1: Import and enable eventlet monkey patching FIRST
import eventlet
eventlet.monkey_patch()

# Step 2: Now import all other modules
import os
import sys # Added for potential exit on missing API key
import logging
import json
import time
import re
from threading import Thread, Timer # Timer is used for timeouts
# Removed Queue as state is managed per client in a dictionary
import numpy as np # Keep if used elsewhere, otherwise can remove

# Step 3: Import Flask and related libraries
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room, leave_room # Added join/leave for potential future use

# Step 4: Import the rest of the dependencies
# Google Gemini API imports
from google import genai as genai
from google.genai import types
# `types` might not be needed if not directly using specific types from it
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
    logger.info("NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    logger.info("NLTK 'punkt' tokenizer downloaded.")
from nltk.tokenize import sent_tokenize

# --- Configuration & Setup ---
load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Add a filter to inject SID into log records if available
class SidFilter(logging.Filter):
    def filter(self, record):
        # If request context is active and has a sid, add it
        if request and hasattr(request, 'sid'):
            record.sid = request.sid
        else:
            # Try to get sid from explicit arguments if passed (for background tasks)
            # This requires passing sid explicitly to logging calls in background tasks
            # or modifying the logger setup further. For now, default to 'N/A'.
             record.sid = getattr(record, 'sid', 'N/A') # Get sid if passed, else N/A
        return True

logger = logging.getLogger(__name__)
# logger.addFilter(SidFilter()) # Disabled for now, SID added manually in logs

# Create Flask app and socketio
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)  # Secret key for session management
# Ensure CORS allows connections from any origin (adjust if needed for security)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Google Gemini LLM Setup ---
# System prompts for the LLMs (Complete prompts included)
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

# Initialize Gemini clients
quick_chat = None
slow_chat = None
query_chat = None

try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Using a dummy key locally might work for testing *without* actual LLM calls
        # but will fail if calls are made. A real key is needed for functionality.
        # api_key = "YOUR_API_KEY" # Replace with your actual key if needed temporarily
        logger.critical("CRITICAL: GOOGLE_API_KEY environment variable not set. LLM features will likely fail.")
        # Consider exiting if the key is essential:
        # sys.exit("GOOGLE_API_KEY environment variable not set. Cannot initialize LLMs.")
        client = genai.Client(api_key="dummy_key") # Configure with dummy to maybe allow app run
    else:
        logger.info("Using GOOGLE_API_KEY from environment variable.")
        client = genai.Client(api_key=api_key)

    # Set up generation config for quick model
    quick_config = types.GenerateContentConfig(
        temperature=0.1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048,
        system_instruction=quick_llm_system_prompt
    )

    # Set up generation config for slow model
    slow_config = types.GenerateContentConfig(
        temperature=0.3,
        top_p=0.95,
        top_k=40,
        max_output_tokens=4096, # Allow larger output for analysis
        system_instruction=slow_llm_system_prompt
    )

    # Set up generation config for query model
    query_config = types.GenerateContentConfig(
        temperature=0.3, # Allow some flexibility for natural language answers
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048,
        system_instruction=query_llm_system_prompt
    )

    # Create chat sessions - these maintain conversation history if needed,
    # though for stateless processing per chunk/query, sending full context each time is also fine.
    quick_model_name = "gemini-2.0-flash"
    quick_chat = client.chats.create(
        model=quick_model_name,
        config=quick_config,
    )
    slow_model_name = "gemini-2.5-pro-exp-03-25"
    slow_chat = client.chats.create(
        model=slow_model_name,
        config=slow_config,
    )
    query_model_name = "gemini-1.5-pro"
    query_chat = client.chats.create(
        model=query_model_name,
        config=query_config,
    )


    logger.info(f"LLM models initialized - Quick: {quick_model_name}, Slow: {slow_model_name}, Query: {query_model_name}")

except Exception as e:
    logger.error(f"Error initializing Google Gemini clients: {e}", exc_info=True)
    # Set chats to None so checks later will prevent errors
    quick_chat = None
    slow_chat = None
    query_chat = None
    logger.warning("Proceeding without full LLM functionality.")

# --- Global State Management ---
# Dictionary to store state for each connected client (identified by session ID)
# Each client gets their own buffers and timers.
client_buffers = {}
# Example structure:
# client_buffers = {
#     'client_sid_1': {
#         'sentence_buffer': ['Sentence 1.', 'Sentence 2.'],
#         'quick_llm_results': ['<...> rdf data ...', '<...> more rdf ...'],
#         'fast_llm_timer': <Timer object or None>,
#         'slow_llm_timer': <Timer object or None>
#     },
#     'client_sid_2': { ... }
# }

# The RDF graph is shared across all clients. Modifications from any client update this single graph.
accumulated_graph = Graph()
# Define common namespaces and bind them
EX = URIRef("http://example.org/")
accumulated_graph.bind("rdf", RDF)
accumulated_graph.bind("rdfs", RDFS)
accumulated_graph.bind("owl", OWL)
accumulated_graph.bind("xsd", XSD)
accumulated_graph.bind("ex", EX)

# Config for chunk sizes and timeouts (can be global or potentially adjusted per client later)
SENTENCE_CHUNK_SIZE = 1  # Number of sentences to collect before quick LLM processing
SLOW_LLM_CHUNK_SIZE = 5  # Number of quick LLM results to collect before slow LLM processing
FAST_LLM_TIMEOUT = 20  # Seconds to wait before flushing sentence buffer to fast LLM
SLOW_LLM_TIMEOUT = 60  # Seconds to wait before flushing quick LLM results to slow LLM

# Global timers are removed, they are now managed per client in client_buffers

# --- Helper Functions ---

def extract_label(uri_or_literal):
    """Helper to get a readable label from URI or Literal for display."""
    if isinstance(uri_or_literal, URIRef):
        try:
            # Try to compute a QName (prefix:localname) if prefix is bound
            prefix, namespace, name = accumulated_graph.compute_qname(uri_or_literal, generate=False)
            return f"{prefix}:{name}" if prefix else name # Return qname or just local name
        except:
            # Fallback: Split by the last '#' or '/'
            if '#' in uri_or_literal:
                return uri_or_literal.split('#')[-1]
            return uri_or_literal.split('/')[-1]
    elif isinstance(uri_or_literal, Literal):
        # Just return the string value of the literal
        return str(uri_or_literal)
    else:
        # Fallback for other types
        return str(uri_or_literal)

def graph_to_visjs(graph):
    """Converts an rdflib Graph to Vis.js nodes and edges format."""
    nodes = []
    edges = []
    node_ids = set()  # Keep track of added node URIs/IDs to avoid duplicates

    # Define schema URIs to potentially filter out or style differently in visualization
    # These are typically less interesting to see as explicit nodes in the main graph view
    schema_node_uris = {
        RDF.type, RDFS.Class, OWL.Class, OWL.ObjectProperty, OWL.Thing, RDFS.Resource,
        OWL.DatatypeProperty, RDFS.subClassOf, RDFS.Literal, RDFS.domain, RDFS.range,
        XSD.string, XSD.integer, XSD.boolean, XSD.date, XSD.dateTime # Common datatypes
    }
    schema_prefixes = (str(RDF), str(RDFS), str(OWL), str(XSD))

    # Iterate through all triples in the graph
    for s, p, o in graph:
        subject_str = str(s)
        predicate_str = str(p)
        object_str = str(o)

        # --- Add Nodes ---
        # Add subject node if it's a URI, not already added, and not a schema URI/prefix
        if isinstance(s, URIRef) and subject_str not in node_ids and s not in schema_node_uris and not subject_str.startswith(schema_prefixes):
            nodes.append({
                "id": subject_str,
                "label": extract_label(s), # Initial label from URI
                "title": f"URI: {subject_str}\n" # Tooltip starts with URI, add newline
            })
            node_ids.add(subject_str)

        # Add object node if it's a URI, not already added, and not a schema URI/prefix
        if isinstance(o, URIRef) and object_str not in node_ids and o not in schema_node_uris and not object_str.startswith(schema_prefixes):
             nodes.append({
                 "id": object_str,
                 "label": extract_label(o), # Initial label from URI
                 "title": f"URI: {object_str}\n" # Tooltip starts with URI
             })
             node_ids.add(object_str)

        # --- Add Edges and Properties to Nodes ---
        # If subject is a visualized node...
        if subject_str in node_ids:
            # Case 1: Object is a URI (relationship edge or rdf:type)
            if isinstance(o, URIRef):
                # If the object is also a visualized node...
                if object_str in node_ids:
                    # Add a relationship edge (excluding specific schema predicates like type/subClass)
                    if p not in {RDF.type, RDFS.subClassOf, RDFS.domain, RDFS.range}:
                        edges.append({
                            "from": subject_str,
                            "to": object_str,
                            "label": extract_label(p),
                            "title": f"Predicate: {extract_label(p)}", # Tooltip for edge
                            "arrows": "to"
                        })
                    # Handle rdf:type separately to add info to node label/title
                    elif p == RDF.type and o not in schema_node_uris and not object_str.startswith(schema_prefixes):
                         # Add type information to the subject node's tooltip and maybe label
                         for node in nodes:
                             if node["id"] == subject_str:
                                 type_label = extract_label(o)
                                 # Add to title (tooltip)
                                 node['title'] += f"Type: {type_label}\n"
                                 # Optionally append type to label for clarity, avoid duplication
                                 type_suffix = f" ({type_label})"
                                 if type_suffix not in node['label'] and node['label'] != type_label:
                                      node['label'] += type_suffix
                                 break
                # Else (object is a URI but not visualized, e.g., a schema class):
                # Optionally add this info to the subject node's title
                elif p == RDF.type: # e.g., node rdf:type RDFS.Class
                     for node in nodes:
                         if node["id"] == subject_str:
                              node['title'] += f"Schema Type: {extract_label(o)}\n"
                              break

            # Case 2: Object is a Literal (property)
            elif isinstance(o, Literal):
                # Add literal value as property to the subject node's title (tooltip)
                for node in nodes:
                    if node["id"] == subject_str:
                        prop_label = extract_label(p)
                        lit_label = extract_label(o)
                        # Append property to tooltip
                        node['title'] += f"{prop_label}: {lit_label}\n"
                        # Special handling for labels: Use rdfs:label as the primary node label
                        if p == RDFS.label:
                            node['label'] = lit_label # Overwrite default URI label
                        # You could add similar logic for 'name' or other primary display properties
                        elif predicate_str.endswith("Name") or predicate_str.endswith("name"):
                             if node['label'] == extract_label(s): # Only overwrite if label is still default
                                 node['label'] = lit_label
                        break

    # --- Add Hierarchy Edges ---
    # Add rdfs:subClassOf as distinct hierarchical edges
    for s, p, o in graph.triples((None, RDFS.subClassOf, None)):
        subject_str = str(s)
        object_str = str(o)
        # Only add edge if both subject and object are visualized nodes
        if subject_str in node_ids and object_str in node_ids:
            edges.append({
                "from": subject_str,
                "to": object_str,
                "label": "subClassOf",
                "arrows": "to",
                "color": {"color": "#AAAAAA", "highlight": "#777777"}, # Style differently
                "dashes": True, # Use dashed lines for hierarchy
                "title": "rdfs:subClassOf"
            })

    # --- Deduplicate Edges ---
    # Prevents visual clutter if the same relationship is asserted multiple times
    unique_edges_set = set()
    unique_edges = []
    for edge in edges:
        # Create a tuple key: (from_node, to_node, edge_label)
        edge_key = (edge['from'], edge['to'], edge.get('label'))
        if edge_key not in unique_edges_set:
            unique_edges.append(edge)
            unique_edges_set.add(edge_key)

    # Final cleanup on node titles (remove trailing newline)
    for node in nodes:
        node['title'] = node['title'].strip()

    return {"nodes": nodes, "edges": unique_edges}

def process_turtle_data(turtle_data, sid):
    """Process Turtle data string, add new triples to the shared accumulated_graph."""
    # Log with SID for context
    log_prefix = f"[SID:{sid}]" if sid else "[System]"

    if not turtle_data:
        logger.warning(f"{log_prefix} Empty Turtle data received, nothing to process.")
        return False # Indicate nothing was added

    try:
        # Strip potential markdown code fences and leading/trailing whitespace
        turtle_data = turtle_data.strip()
        if turtle_data.startswith("```turtle"):
            turtle_data = turtle_data[len("```turtle"):].strip()
        elif turtle_data.startswith("```"): # Handle generic code fence
             turtle_data = turtle_data[len("```"):].strip()
        if turtle_data.endswith("```"):
            turtle_data = turtle_data[:-len("```")].strip()

        # Check if data is empty after stripping
        if not turtle_data:
             logger.warning(f"{log_prefix} Turtle data was empty after stripping markdown fences.")
             return False

        # Define prefixes string based on the current main graph to provide context for parsing the fragment
        # This helps rdflib resolve prefixes used in the incoming turtle_data
        prefixes = "\n".join(f"@prefix {p}: <{n}> ." for p, n in accumulated_graph.namespaces())
        full_turtle_for_parsing = prefixes + "\n" + turtle_data

        # Create a temporary graph to parse into. This avoids polluting the main graph if parsing fails.
        temp_graph = Graph()
        temp_graph.parse(data=full_turtle_for_parsing, format="turtle")

        # Count how many *new* unique triples are added to the main graph
        new_triples_count = 0
        for triple in temp_graph:
            # Check if the triple does NOT already exist in the main graph
            if triple not in accumulated_graph:
                accumulated_graph.add(triple)
                new_triples_count += 1

        # Log information about the update
        if new_triples_count > 0:
            logger.info(f"{log_prefix} Added {new_triples_count} new triples to the graph. Total graph size: {len(accumulated_graph)}")
            return True # Indicate that new triples were added
        else:
            logger.info(f"{log_prefix} No new triples added from the received Turtle data (data parsed OK, but triples already existed or were empty).")
            return False # Indicate nothing new was added

    except Exception as e:
        logger.error(f"{log_prefix} Error parsing Turtle data: {e}", exc_info=True)
        # Log the problematic data for debugging (limit length)
        problem_data_preview = turtle_data[:500] + ('...' if len(turtle_data) > 500 else '')
        logger.error(f"{log_prefix} Problematic Turtle data (first 500 chars):\n---\n{problem_data_preview}\n---")
        return False # Indicate failure/no change

def update_graph_visualization():
    """Generates Vis.js data from the shared graph and broadcasts it to all clients."""
    try:
        # Convert the current accumulated graph to Vis.js format
        vis_data = graph_to_visjs(accumulated_graph)
        # Broadcast the updated graph data over SocketIO to all connected clients
        # The event name 'update_graph' should be listened for by the frontend JavaScript
        socketio.emit('update_graph', vis_data)
        connected_clients = len(client_buffers) # Get current count from our state management dict
        logger.info(f"Graph visualization updated and broadcast to all ({connected_clients}) clients.")
    except Exception as e:
        logger.error(f"Error generating or broadcasting graph visualization: {e}", exc_info=True)
        # Optionally emit an error event to clients
        socketio.emit('error', {'message': f'Error generating graph visualization: {e}'})

# --- LLM Processing Functions (Run as Background Tasks) ---

def process_with_quick_llm(text_chunk, sid):
    """Processes a text chunk with the Quick LLM to extract RDF triples."""
    log_prefix = f"[SID:{sid}]"
    if not quick_chat:
        logger.error(f"{log_prefix} Quick LLM not available. Cannot process text: '{text_chunk[:50]}...'")
        # Emit error back to the specific client that sent the data
        socketio.emit('error', {'message': 'RDF generation service (Quick LLM) is unavailable.'}, room=sid)
        return

    logger.info(f"{log_prefix} Processing text chunk with Quick LLM: '{text_chunk[:100]}...'")
    try:
        # Send the text chunk to the LLM
        # Ensure the LLM understands it's potentially part of a larger conversation/context if necessary,
        # although the prompt encourages focusing only on the current chunk.
        response = quick_chat.send_message(text_chunk)
        turtle_response = response.text

        logger.info(f"{log_prefix} Quick LLM response received.")
        # Log a snippet of the response for debugging
        debug_turtle = turtle_response.strip()
        if len(debug_turtle) > 200: debug_turtle = debug_turtle[:200] + "..."
        logger.debug(f"{log_prefix} Quick LLM Turtle Output (raw snippet):\n---\n{debug_turtle}\n---")

        # Process the Turtle data received from the LLM
        triples_added = process_turtle_data(turtle_response, sid)

        if triples_added:
            # If new triples were successfully added to the graph:
            # 1. Add the raw Turtle output to this client's buffer for potential Slow LLM processing
            if sid in client_buffers:
                client_buffers[sid]['quick_llm_results'].append(turtle_response)
                buffer_size = len(client_buffers[sid]['quick_llm_results'])
                logger.info(f"{log_prefix} Added Quick LLM result to buffer for Slow LLM. Buffer size: {buffer_size}")
                # Check if this addition fills the chunk for the slow LLM
                check_slow_llm_chunk(sid) # Important: check *after* adding result
            else:
                 logger.warning(f"{log_prefix} Client buffer not found after successful quick LLM processing. Cannot store result for slow LLM.")

            # 2. Update the graph visualization for *all* connected clients
            update_graph_visualization()
        else:
             # Log if the LLM responded but processing yielded no *new* triples
             logger.info(f"{log_prefix} Quick LLM processing complete, but no new triples were added to the graph.")

    except Exception as e:
        logger.error(f"{log_prefix} Error during Quick LLM processing or subsequent handling: {e}", exc_info=True)
        socketio.emit('error', {'message': f'Error processing text with Quick LLM: {e}'}, room=sid)


def process_with_slow_llm(combined_quick_results_turtle, sid):
    """Processes combined Turtle results from Quick LLM with the Slow LLM for deeper analysis."""
    log_prefix = f"[SID:{sid}]"
    if not slow_chat:
        logger.error(f"{log_prefix} Slow LLM not available. Cannot perform advanced analysis.")
        socketio.emit('error', {'message': 'Graph analysis service (Slow LLM) is unavailable.'}, room=sid)
        return

    logger.info(f"{log_prefix} Processing {len(combined_quick_results_turtle.splitlines())} lines of previous Turtle results with Slow LLM.")
    try:
        # Serialize the current state of the *entire* accumulated graph to provide context
        # Be mindful of token limits for the LLM. Truncate if necessary.
        current_graph_turtle = accumulated_graph.serialize(format='turtle')
        MAX_GRAPH_CONTEXT_SLOW = 10000 # Adjust based on LLM limits and typical graph size
        if len(current_graph_turtle) > MAX_GRAPH_CONTEXT_SLOW:
             # Simple tail truncation; more sophisticated sampling/summarization could be used
             current_graph_turtle = current_graph_turtle[-MAX_GRAPH_CONTEXT_SLOW:]
             logger.warning(f"{log_prefix} Truncated current graph context (last {MAX_GRAPH_CONTEXT_SLOW} bytes) for Slow LLM input due to size.")

        # Construct the detailed prompt for the Slow LLM
        slow_llm_input = f"""
        Existing Knowledge Graph (Turtle format, potentially partial):
        ```turtle
        {current_graph_turtle}
        ```

        New Information/Triples to Analyze (from previous processing steps):
        ```turtle
        {combined_quick_results_turtle}
        ```

        Based on the existing graph and the new information, identify higher-level concepts, implicit relationships, categories, or inconsistencies.
        Return ONLY new Turtle triples that enhance the graph with these insights. Do not repeat triples already present in the 'Existing Knowledge Graph' section above or trivially derived from the 'New Information'.
        Use the 'ex:' prefix. Focus on adding value through abstraction, inference, and connection. Output only valid Turtle.
        """

        # Send the combined context and new data to the Slow LLM
        response = slow_chat.send_message(slow_llm_input)
        turtle_response = response.text

        logger.info(f"{log_prefix} Slow LLM response received.")
        # Log a snippet for debugging
        debug_turtle = turtle_response.strip()
        if len(debug_turtle) > 200: debug_turtle = debug_turtle[:200] + "..."
        logger.debug(f"{log_prefix} Slow LLM Turtle Output (raw snippet):\n---\n{debug_turtle}\n---")

        # Process the Turtle data received from the Slow LLM
        triples_added = process_turtle_data(turtle_response, sid)

        if triples_added:
            # If the Slow LLM generated new, valid triples:
            # 1. Update the graph visualization for all clients
            update_graph_visualization()
        else:
            # Log if Slow LLM ran but produced no new insights / triples
            logger.info(f"{log_prefix} Slow LLM analysis complete, but no new triples were added to the graph.")

    except Exception as e:
        logger.error(f"{log_prefix} Error during Slow LLM processing or subsequent handling: {e}", exc_info=True)
        socketio.emit('error', {'message': f'Error during graph analysis with Slow LLM: {e}'}, room=sid)


# --- Timeout and Chunking Logic (Operates on Per-Client State) ---

def flush_sentence_buffer(sid):
    """Forces processing of the sentence buffer for a specific client SID due to timeout."""
    log_prefix = f"[SID:{sid}]"
    if sid not in client_buffers:
        # This might happen if the client disconnected just before the timer fired
        logger.warning(f"{log_prefix} flush_sentence_buffer called for disconnected or unknown SID.")
        return

    # Access the specific client's state
    state = client_buffers[sid]
    # Mark that the timer is no longer active (it just fired)
    state['fast_llm_timer'] = None

    if not state['sentence_buffer']:
        logger.info(f"{log_prefix} Fast LLM timeout reached, but sentence buffer is empty. No action needed.")
        return

    sentence_count = len(state['sentence_buffer'])
    logger.info(f"{log_prefix} Fast LLM timeout reached. Flushing {sentence_count} sentences from buffer.")

    # Important: Create a copy of the buffer content to process
    sentences_to_process = list(state['sentence_buffer'])
    # Clear the client's buffer *before* starting the background task
    state['sentence_buffer'].clear()

    # Combine the sentences into a single text chunk
    combined_text = " ".join(sentences_to_process)

    # Start the Quick LLM processing in a background task so it doesn't block the server
    logger.info(f"{log_prefix} Starting background task for Quick LLM processing (timeout flush).")
    socketio.start_background_task(process_with_quick_llm, combined_text, sid)

def flush_quick_llm_results(sid):
    """Forces processing of the quick LLM results buffer for a specific client SID due to timeout."""
    log_prefix = f"[SID:{sid}]"
    if sid not in client_buffers:
        logger.warning(f"{log_prefix} flush_quick_llm_results called for disconnected or unknown SID.")
        return

    state = client_buffers[sid]
    state['slow_llm_timer'] = None # Mark timer as inactive

    if not state['quick_llm_results']:
        logger.info(f"{log_prefix} Slow LLM timeout reached, but quick results buffer is empty.")
        return

    results_count = len(state['quick_llm_results'])
    logger.info(f"{log_prefix} Slow LLM timeout reached. Flushing {results_count} quick LLM results.")

    # Copy the results and clear the buffer
    results_to_process = list(state['quick_llm_results'])
    state['quick_llm_results'].clear()

    # Combine the Turtle snippets into one block for the Slow LLM prompt
    # Ensure they are separated by newlines for clarity in the prompt
    combined_turtle_text = "\n\n".join(results_to_process) # Use double newline as separator

    # Start the Slow LLM processing in a background task
    logger.info(f"{log_prefix} Starting background task for Slow LLM processing (timeout flush).")
    socketio.start_background_task(process_with_slow_llm, combined_turtle_text, sid)


def schedule_fast_llm_timeout(sid):
    """Schedules or reschedules the fast LLM timeout for a specific client."""
    log_prefix = f"[SID:{sid}]"
    if sid not in client_buffers:
         logger.warning(f"{log_prefix} Attempted to schedule fast LLM timeout for unknown SID.")
         return

    state = client_buffers[sid]

    # Cancel any existing timer for this client first
    if state.get('fast_llm_timer'):
        try:
            state['fast_llm_timer'].cancel()
            logger.debug(f"{log_prefix} Cancelled existing fast LLM timer.")
        except Exception as e:
            logger.error(f"{log_prefix} Error cancelling existing fast timer: {e}")


    # Create a new Timer object. When it expires, it will call flush_sentence_buffer(sid)
    new_timer = Timer(FAST_LLM_TIMEOUT, flush_sentence_buffer, args=[sid])
    new_timer.daemon = True # Allow the main program to exit even if timers are pending
    new_timer.start()

    # Store the new timer object in the client's state
    state['fast_llm_timer'] = new_timer
    logger.info(f"{log_prefix} Scheduled fast LLM processing timeout in {FAST_LLM_TIMEOUT} seconds.")


def schedule_slow_llm_timeout(sid):
    """Schedules or reschedules the slow LLM timeout for a specific client."""
    log_prefix = f"[SID:{sid}]"
    if sid not in client_buffers:
         logger.warning(f"{log_prefix} Attempted to schedule slow LLM timeout for unknown SID.")
         return

    state = client_buffers[sid]

    # Cancel existing timer
    if state.get('slow_llm_timer'):
        try:
            state['slow_llm_timer'].cancel()
            logger.debug(f"{log_prefix} Cancelled existing slow LLM timer.")
        except Exception as e:
             logger.error(f"{log_prefix} Error cancelling existing slow timer: {e}")


    # Create and start a new timer targeting the slow LLM flush function
    new_timer = Timer(SLOW_LLM_TIMEOUT, flush_quick_llm_results, args=[sid])
    new_timer.daemon = True
    new_timer.start()

    # Store the new timer in the client's state
    state['slow_llm_timer'] = new_timer
    logger.info(f"{log_prefix} Scheduled slow LLM processing timeout in {SLOW_LLM_TIMEOUT} seconds.")


def check_fast_llm_chunk(sid):
    """Checks if the sentence buffer for a client is full and processes it if necessary."""
    log_prefix = f"[SID:{sid}]"
    if sid not in client_buffers:
        logger.warning(f"{log_prefix} check_fast_llm_chunk called for unknown SID.")
        return

    state = client_buffers[sid]
    buffer_size = len(state['sentence_buffer'])

    if buffer_size >= SENTENCE_CHUNK_SIZE:
        logger.info(f"{log_prefix} Sentence chunk size ({buffer_size}/{SENTENCE_CHUNK_SIZE}) reached. Processing now.")

        # Cancel any pending timeout timer since we are processing due to chunk size
        if state.get('fast_llm_timer'):
            try:
                state['fast_llm_timer'].cancel()
                state['fast_llm_timer'] = None # Clear the timer from state
                logger.debug(f"{log_prefix} Cancelled pending fast LLM timer due to chunk processing.")
            except Exception as e:
                 logger.error(f"{log_prefix} Error cancelling fast timer during chunk check: {e}")


        # Process the chunk (similar logic to flush_sentence_buffer)
        sentences_to_process = list(state['sentence_buffer'])
        state['sentence_buffer'].clear()
        combined_text = " ".join(sentences_to_process)

        logger.info(f"{log_prefix} Starting background task for Quick LLM processing (chunk size reached).")
        socketio.start_background_task(process_with_quick_llm, combined_text, sid)

    elif buffer_size > 0:
        # If the buffer has content but isn't full, ensure a timeout is scheduled.
        # This handles cases where input stops before a chunk is full.
        if not state.get('fast_llm_timer'):
            logger.debug(f"{log_prefix} Buffer has {buffer_size} sentences, scheduling fast LLM timeout.")
            schedule_fast_llm_timeout(sid)
    # If buffer_size is 0, do nothing (no data to process, no need for a timer)


def check_slow_llm_chunk(sid):
    """Checks if the quick LLM results buffer for a client is full and processes it."""
    log_prefix = f"[SID:{sid}]"
    if sid not in client_buffers:
        logger.warning(f"{log_prefix} check_slow_llm_chunk called for unknown SID.")
        return

    state = client_buffers[sid]
    buffer_size = len(state['quick_llm_results'])

    if buffer_size >= SLOW_LLM_CHUNK_SIZE:
        logger.info(f"{log_prefix} Slow LLM chunk size ({buffer_size}/{SLOW_LLM_CHUNK_SIZE}) reached. Processing now.")

        # Cancel pending slow timer
        if state.get('slow_llm_timer'):
            try:
                state['slow_llm_timer'].cancel()
                state['slow_llm_timer'] = None
                logger.debug(f"{log_prefix} Cancelled pending slow LLM timer due to chunk processing.")
            except Exception as e:
                 logger.error(f"{log_prefix} Error cancelling slow timer during chunk check: {e}")


        # Process the chunk (similar logic to flush_quick_llm_results)
        results_to_process = list(state['quick_llm_results'])
        state['quick_llm_results'].clear()
        combined_turtle_text = "\n\n".join(results_to_process)

        logger.info(f"{log_prefix} Starting background task for Slow LLM processing (chunk size reached).")
        socketio.start_background_task(process_with_slow_llm, combined_turtle_text, sid)

    elif buffer_size > 0:
        # If the buffer has results but isn't full, ensure a timeout is running
        if not state.get('slow_llm_timer'):
            logger.debug(f"{log_prefix} Quick results buffer has {buffer_size} items, scheduling slow LLM timeout.")
            schedule_slow_llm_timeout(sid)
    # If buffer_size is 0, do nothing


# --- SocketIO Event Handlers ---

@socketio.on('connect')
def handle_connect():
    """Handles a new client connection."""
    sid = request.sid
    # Use sid in logging for clarity when multiple clients connect
    log_prefix = f"[SID:{sid}]"
    logger.info(f"{log_prefix} Client connected.")

    # Initialize the state for this new client
    client_buffers[sid] = {
        'sentence_buffer': [],
        'quick_llm_results': [],
        'fast_llm_timer': None, # No timers active initially
        'slow_llm_timer': None
    }
    logger.info(f"{log_prefix} Initialized state buffers and timers.")

    # Send the current state of the knowledge graph to the newly connected client
    # This allows the visualization to be populated immediately on connection.
    try:
        vis_data = graph_to_visjs(accumulated_graph)
        # Emit only to the connecting client using 'room=sid'
        emit('update_graph', vis_data, room=sid)
        logger.info(f"{log_prefix} Sent current graph state ({len(vis_data.get('nodes',[]))} nodes, {len(vis_data.get('edges',[]))} edges) to newly connected client.")
    except Exception as e:
        logger.error(f"{log_prefix} Error generating or sending initial graph state: {e}", exc_info=True)
        emit('error', {'message': f'Error loading initial graph state: {e}'}, room=sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Handles a client disconnection."""
    sid = request.sid
    log_prefix = f"[SID:{sid}]"
    logger.info(f"{log_prefix} Client disconnected.")

    # Clean up the state associated with the disconnected client
    if sid in client_buffers:
        state = client_buffers[sid]
        # Cancel any running timers for this client
        if state.get('fast_llm_timer'):
            try:
                state['fast_llm_timer'].cancel()
                logger.info(f"{log_prefix} Cancelled fast LLM timer on disconnect.")
            except Exception as e:
                 logger.error(f"{log_prefix} Error cancelling fast timer on disconnect: {e}")

        if state.get('slow_llm_timer'):
            try:
                state['slow_llm_timer'].cancel()
                logger.info(f"{log_prefix} Cancelled slow LLM timer on disconnect.")
            except Exception as e:
                 logger.error(f"{log_prefix} Error cancelling slow timer on disconnect: {e}")

        # Remove the client's state from the dictionary
        del client_buffers[sid]
        logger.info(f"{log_prefix} Cleaned up state and timers.")
    else:
         # This shouldn't normally happen if connect logic is correct
         logger.warning(f"{log_prefix} Disconnect event received for SID not found in active buffers.")

@socketio.on('transcription_chunk')
def handle_transcription_chunk(data):
    """Handles incoming transcription text chunks from a client (e.g., the CLI)."""
    sid = request.sid
    log_prefix = f"[SID:{sid}]"

    # Ensure the client is known (should have connected previously)
    if sid not in client_buffers:
         logger.error(f"{log_prefix} Received transcription chunk from unknown or disconnected SID. Ignoring.")
         # Optionally, send an error back to the SID, though it might not be listening
         # emit('error', {'message': 'Server state lost for your session, please reconnect if needed.'}, room=sid)
         return

    # Extract text, default to empty string if 'text' key is missing
    text = data.get('text', '').strip()
    if not text:
        logger.warning(f"{log_prefix} Received empty transcription chunk.")
        return

    logger.info(f"{log_prefix} Received transcription chunk: '{text[:100]}...'")

    # Tokenize the incoming text into sentences
    try:
        # Use NLTK's sentence tokenizer
        sentences = sent_tokenize(text)
        if not sentences: # Handle case where tokenization results in empty list
             logger.warning(f"{log_prefix} Tokenization resulted in zero sentences for text: '{text[:50]}...'")
             return # Nothing to add
    except Exception as e:
        # Fallback if NLTK fails: treat the whole chunk as one sentence
        logger.error(f"{log_prefix} NLTK sentence tokenization failed: {e}. Treating chunk as one sentence.", exc_info=True)
        sentences = [text]

    # Add the extracted sentences to this client's sentence buffer
    state = client_buffers[sid]
    state['sentence_buffer'].extend(sentences)
    buffer_size = len(state['sentence_buffer'])
    logger.info(f"{log_prefix} Added {len(sentences)} sentence(s) to buffer. New buffer size: {buffer_size}")

    # Check if the buffer is now full enough to trigger fast LLM processing
    check_fast_llm_chunk(sid)


@socketio.on('query_graph')
def handle_query_graph(data):
    """Handles a natural language query about the graph from a client."""
    sid = request.sid
    log_prefix = f"[SID:{sid}]"

    query_text = data.get('query', '').strip()
    if not query_text:
        logger.warning(f"{log_prefix} Received empty query.")
        emit('query_result', {'answer': "Please provide a query text.", 'error': True}, room=sid)
        return

    # Check if the Query LLM is available
    if not query_chat:
        logger.error(f"{log_prefix} Query LLM is not available. Cannot process query: '{query_text}'")
        emit('query_result', {'answer': "Sorry, the knowledge graph query service is currently unavailable.", 'error': True}, room=sid)
        return

    logger.info(f"{log_prefix} Received query: '{query_text}'")

    # Prepare the context for the Query LLM
    try:
        # Serialize the current accumulated graph
        # Again, be mindful of token limits. Truncate if the graph is very large.
        current_graph_turtle = accumulated_graph.serialize(format='turtle')
        MAX_GRAPH_CONTEXT_QUERY = 15000 # Adjust as needed
        if len(current_graph_turtle) > MAX_GRAPH_CONTEXT_QUERY:
             logger.warning(f"{log_prefix} Query Graph context size ({len(current_graph_turtle)} bytes) exceeds limit ({MAX_GRAPH_CONTEXT_QUERY}). Truncating.")
             # Simple tail truncation - might lose relevant older info
             current_graph_turtle = current_graph_turtle[-MAX_GRAPH_CONTEXT_QUERY:]

        # Construct the prompt for the Query LLM
        query_prompt = f"""
        Knowledge Graph:
        ```turtle
        {current_graph_turtle}
        ```

        User Query: "{query_text}"

        ---
        Based *only* on the 'Knowledge Graph' provided above, answer the 'User Query'.
        Explain your reasoning by referencing specific entities (like `ex:EntityName`) and relationships from the graph.
        If the graph does not contain the information needed, state that clearly. Do not invent information.
        """

        # Define the function to execute the LLM query (will run in background)
        def run_query_task():
            task_log_prefix = f"[SID:{sid}|QueryTask]"
            try:
                logger.info(f"{task_log_prefix} Sending query to Query LLM...")
                # Send the constructed prompt to the query LLM
                response = query_chat.send_message(query_prompt)
                answer = response.text
                logger.info(f"{task_log_prefix} Query LLM response received.")
                # Emit the result back to the specific client who asked
                socketio.emit('query_result', {'answer': answer, 'error': False}, room=sid)
            except Exception as e:
                logger.error(f"{task_log_prefix} Error during Query LLM processing: {e}", exc_info=True)
                # Emit an error message back to the client
                socketio.emit('query_result', {'answer': f"An error occurred while processing your query: {e}", 'error': True}, room=sid)

        # Acknowledge the query immediately (optional, good for UX)
        emit('query_result', {'answer': "Processing your query against the knowledge graph...", 'processing': True}, room=sid)

        # Start the query execution in a background task
        logger.info(f"{log_prefix} Starting background task for graph query.")
        socketio.start_background_task(run_query_task)

    except Exception as e:
         # Handle errors during graph serialization or prompt preparation
         logger.error(f"{log_prefix} Error preparing data for graph query: {e}", exc_info=True)
         emit('query_result', {'answer': f"An internal error occurred before processing the query: {e}", 'error': True}, room=sid)


# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page for the web interface."""
    # Assumes you have an 'index.html' file inside a 'templates' directory
    # in the same location as app.py
    logger.info("Serving index.html")
    return render_template('index.html')

# --- Main Execution ---
if __name__ == '__main__':
    logger.info("--- Starting VoxGraph Server ---")

    # Log critical configuration settings at startup
    logger.info(f"Configuration: SentenceChunk={SENTENCE_CHUNK_SIZE}, SlowLLMChunk={SLOW_LLM_CHUNK_SIZE}, FastTimeout={FAST_LLM_TIMEOUT}s, SlowTimeout={SLOW_LLM_TIMEOUT}s")

    # Log the status of initialized LLMs
    llm_status = {
        "QuickLLM (RDF Extraction)": "Available" if quick_chat else "Unavailable",
        "SlowLLM (Analysis)": "Available" if slow_chat else "Unavailable",
        "QueryLLM (Querying)": "Available" if query_chat else "Unavailable"
    }
    logger.info(f"LLM Status: {json.dumps(llm_status)}")
    if "Unavailable" in llm_status.values():
        logger.warning("One or more LLMs are unavailable. Related functionality will be limited or disabled.")

    # Get port from environment or default to 5001
    port = int(os.environ.get('PORT', 5001))
    logger.info(f"Attempting to start server on http://0.0.0.0:{port}")

    # Run the Flask-SocketIO development server
    # - debug=True enables auto-reloading and verbose error pages (disable in production)
    # - use_reloader=False is often recommended with eventlet/gevent to prevent issues
    # - host='0.0.0.0' makes the server accessible externally (use '127.0.0.1' for local only)
    try:
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

    # Note for production deployment:
    # Use a production-grade WSGI server like Gunicorn with the eventlet worker:
    # Example: gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5001 app:app