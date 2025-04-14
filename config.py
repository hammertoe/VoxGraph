# -*- coding: utf-8 -*-
import os
import logging
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()
logger = logging.getLogger(__name__) # Use root logger temporarily for early config logging

# --- Essential API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LIGHTHOUSE_API_KEY = os.getenv("LIGHTHOUSE_API_KEY")

# --- Google AI Model Configuration ---
# Base Models (for Chat/RDF/Query)
QUICK_LLM_MODEL = os.getenv("QUICK_LLM_MODEL", "models/gemini-2.0-flash")
SLOW_LLM_MODEL = os.getenv("SLOW_LLM_MODEL", "models/gemini-2.5-pro-exp-03-25")
QUERY_LLM_MODEL = os.getenv("QUERY_LLM_MODEL", "models/gemini-1.5-pro")

# Live API Model (for Transcription)
GOOGLE_LIVE_API_MODEL = os.getenv("GOOGLE_LIVE_MODEL", "models/gemini-2.0-flash-exp")
LIVE_API_SAMPLE_RATE = 16000
LIVE_API_CHANNELS = 1
LIVE_API_VERSION = "v1alpha" # Required for live client

# --- LLM Prompts ---
QUICK_LLM_SYSTEM_PROMPT = """
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

SLOW_LLM_SYSTEM_PROMPT = """
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

QUERY_LLM_SYSTEM_PROMPT = """
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

# --- Processing Constants ---
SLOW_LLM_CHUNK_SIZE = 5
FAST_LLM_TIMEOUT = 20 # Less relevant with utterance-based approach
SLOW_LLM_TIMEOUT = 60
MAX_GRAPH_CONTEXT_SLOW = 10000 # Character limit for graph context passed to slow LLM
MAX_GRAPH_CONTEXT_QUERY = 15000 # Character limit for graph context passed to query LLM
MAX_AUDIO_QUEUE_SIZE = 50
MAX_STATUS_QUEUE_SIZE = 50
AUDIO_TERMINATION_WAIT = 3.0 # Seconds to wait for audio thread termination
AUDIO_MANAGER_MAX_RETRIES = 5
AUDIO_MANAGER_RETRY_BASE_DELAY = 1.0 # Seconds

# --- RDF Namespace Configuration ---
from rdflib import URIRef
from rdflib.namespace import RDF, RDFS, OWL, XSD
EX = URIRef("http://example.org/")
NAMESPACES = {
    "rdf": RDF,
    "rdfs": RDFS,
    "owl": OWL,
    "xsd": XSD,
    "ex": EX
}
TRANSCRIPTION_TYPE_URI = URIRef(str(EX) + "Transcription") # Construct full URI
PROVENANCE_PREDICATE_URI = URIRef(str(EX) + "derivedFromTranscript") # Construct full URI

# --- Sanity Checks ---
if not GOOGLE_API_KEY:
    logger.critical("CRITICAL: GOOGLE_API_KEY environment variable is missing. LLM features will be disabled.")
if not LIGHTHOUSE_API_KEY:
    # This is less critical, maybe just a warning
    logger.warning("WARNING: LIGHTHOUSE_API_KEY environment variable is missing. CAR file archiving will be disabled.")
