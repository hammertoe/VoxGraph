# -*- coding: utf-8 -*-
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD
import config
from logger_setup import logger, get_log_prefix

# Define namespaces locally for clarity, using config values
RDF_NS = config.NAMESPACES['rdf']
RDFS_NS = config.NAMESPACES['rdfs']
OWL_NS = config.NAMESPACES['owl']
XSD_NS = config.NAMESPACES['xsd']
EX_NS = config.NAMESPACES['ex'] # This is a URIRef already

SCHEMA_URIS = (str(RDF_NS), str(RDFS_NS), str(OWL_NS), str(XSD_NS))
SCHEMA_PROPERTIES_TO_IGNORE = {
    RDF.type, RDFS.subClassOf, RDFS.domain, RDFS.range, OWL.inverseOf,
    OWL.equivalentClass, OWL.equivalentProperty
}
SCHEMA_CLASSES_TO_IGNORE = {
    OWL.Class, RDFS.Class, RDF.Property, OWL.ObjectProperty,
    OWL.DatatypeProperty, RDFS.Resource, OWL.Thing
}

def extract_label(graph: Graph, uri_or_literal):
    """Helper to get a readable label from URI or Literal using the graph's namespaces."""
    if isinstance(uri_or_literal, URIRef):
        try:
            # Try to compute QName first
            prefix, namespace, name = graph.compute_qname(uri_or_literal, generate=False)
            return f"{prefix}:{name}" if prefix else name
        except:
            # Fallback to splitting URI
            local_part = uri_or_literal.fragment or uri_or_literal.split('/')[-1]
            return local_part if local_part else str(uri_or_literal) # Handle empty local part
    elif isinstance(uri_or_literal, Literal):
        return f'"{str(uri_or_literal)}"{f"^^<{uri_or_literal.datatype}>" if uri_or_literal.datatype else ""}{f"@{uri_or_literal.language}" if uri_or_literal.language else ""}'
    else:
        return str(uri_or_literal)

def graph_to_visjs(graph: Graph):
    """Converts an rdflib Graph to Vis.js nodes/edges, styling provenance."""
    nodes_data = {} # Store node info keyed by URI string
    edges = []
    instance_uris = set() # Keep track of things that are instances (not schema)

    # --- Pass 1: Identify instance URIs ---
    # An instance is a subject or object that isn't clearly schema-related
    for s, p, o in graph:
        s_str, p_str, o_str = str(s), str(p), str(o)

        # If s has type that's not a schema class, s is an instance
        if p == RDF.type and isinstance(s, URIRef) and isinstance(o, URIRef) and \
           o not in SCHEMA_CLASSES_TO_IGNORE and not s_str.startswith(SCHEMA_URIS) and \
           not o_str.startswith(SCHEMA_URIS):
             instance_uris.add(s_str)

        # If s and o are URIs and p isn't a schema property, they are likely instances
        elif isinstance(s, URIRef) and isinstance(o, URIRef) and \
             p not in SCHEMA_PROPERTIES_TO_IGNORE and \
             not s_str.startswith(SCHEMA_URIS) and \
             not o_str.startswith(SCHEMA_URIS) and \
             not p_str.startswith(SCHEMA_URIS):
                 instance_uris.add(s_str)
                 instance_uris.add(o_str)

        # If s has a literal value via a non-schema property, s is an instance
        elif isinstance(s, URIRef) and isinstance(o, Literal) and \
             p not in SCHEMA_PROPERTIES_TO_IGNORE and \
             not s_str.startswith(SCHEMA_URIS) and \
             not p_str.startswith(SCHEMA_URIS):
                 instance_uris.add(s_str)

    # --- Pass 2: Create nodes for identified instances ---
    for uri_str in instance_uris:
        uri = URIRef(uri_str)
        # Double check it's not a schema class itself
        if uri not in SCHEMA_CLASSES_TO_IGNORE and not uri_str.startswith(SCHEMA_URIS):
             node_label = extract_label(graph, uri)
             nodes_data[uri_str] = {
                 "id": uri_str,
                 "label": node_label,
                 "title": f"URI: {uri_str}\n", # Tooltip starts with URI
                 "group": "Instance" # Default group
             }

    # --- Pass 3: Add edges and refine node properties/groups ---
    for s, p, o in graph:
        s_str, p_str = str(s), str(p)

        # Process only if subject is an instance node we've created
        if s_str in nodes_data:
            node = nodes_data[s_str]

            # --- Handle Object Properties (Edges) ---
            if isinstance(o, URIRef):
                o_str = str(o)
                # Check if object is also an instance node and property is relevant
                if o_str in nodes_data and \
                   p not in SCHEMA_PROPERTIES_TO_IGNORE and \
                   not p_str.startswith(SCHEMA_URIS):

                    edge_label = extract_label(graph, p)
                    edge_id = f"{s_str}_{p_str}_{o_str}" # Simple edge ID

                    edge_data = {
                        "id": edge_id,
                        "from": s_str,
                        "to": o_str,
                        "label": edge_label,
                        "title": f"Predicate: {edge_label}\nURI: {p_str}",
                        "arrows": "to"
                    }

                    # Special styling for provenance links
                    if p == config.PROVENANCE_PREDICATE_URI:
                        edge_data["dashes"] = True
                        edge_data["color"] = 'lightgray'
                        # edge_data["label"] = "derived from" # Keep full label for clarity

                    edges.append(edge_data)

                # --- Handle RDF Type ---
                elif p == RDF.type:
                    type_uri = o
                    type_label = extract_label(graph, type_uri)
                    node['title'] += f"Type: {type_label}\n"

                    # Set group and potentially label based on type
                    if type_uri == config.TRANSCRIPTION_TYPE_URI:
                        node['group'] = "Transcription"
                        # Add Txn prefix if not already there
                        if not node['label'].startswith("Txn:"):
                            node['label'] = "Txn: " + node['label']
                    elif type_uri not in SCHEMA_CLASSES_TO_IGNORE and not str(type_uri).startswith(SCHEMA_URIS):
                        # Use type label as group, unless already Transcription
                        if node.get('group') != "Transcription":
                            node['group'] = type_label
                        # Append type to label if different and not already there
                        type_suffix = f" ({type_label})"
                        if type_suffix not in node['label'] and node['label'] != type_label:
                             node['label'] += type_suffix

            # --- Handle Datatype Properties (Literals) ---
            elif isinstance(o, Literal):
                prop_label = extract_label(graph, p)
                lit_label = extract_label(graph, o) # Gets formatted literal string
                node['title'] += f"{prop_label}: {lit_label}\n"

                # Use rdfs:label as primary node label if available and not a Transcription
                if p == RDFS.label and node.get('group') != "Transcription":
                    node['label'] = str(o) # Use plain string value for label


    # --- Pass 4: Finalize nodes and deduplicate edges ---
    final_nodes = list(nodes_data.values())
    for node in final_nodes:
        node['title'] = node['title'].strip() # Clean up tooltip whitespace

    # Deduplicate edges based on from, to, and label
    unique_edges_set = set()
    unique_edges = []
    for edge in edges:
        # Ensure required fields exist before creating key
        if 'from' in edge and 'to' in edge:
            edge_key = (edge['from'], edge['to'], edge.get('label'))
            if edge_key not in unique_edges_set:
                unique_edges.append(edge)
                unique_edges_set.add(edge_key)
        else:
            # This should ideally not happen if logic above is correct
            logger.warning(f"[GraphUtils] Skipping malformed edge in graph_to_visjs: {edge}")

    return {"nodes": final_nodes, "edges": unique_edges}


def process_turtle_data(turtle_data: str, graph: Graph, session_id: str, sid: str) -> bool:
    """
    Parses Turtle data string and adds new triples to the provided rdflib Graph.
    Returns True if new triples were added, False otherwise.
    """
    log_prefix = get_log_prefix(session_id, sid, "RDFParser")
    if not turtle_data:
        logger.warning(f"{log_prefix} Received empty Turtle data.")
        return False

    added_count = 0
    try:
        turtle_data = turtle_data.strip()
        # Remove common markdown fences
        if turtle_data.startswith("```turtle"):
            turtle_data = turtle_data[len("```turtle"):].strip()
        elif turtle_data.startswith("```"):
             turtle_data = turtle_data[len("```"):].strip()
        if turtle_data.endswith("```"):
             turtle_data = turtle_data[:-len("```")].strip()

        if not turtle_data:
            logger.warning(f"{log_prefix} Turtle data became empty after stripping fences.")
            return False

        # Use a temporary graph for parsing to isolate new triples
        temp_graph = Graph()
        # Bind namespaces from the main graph to the temp graph for context
        for prefix, ns_uri in graph.namespaces():
            temp_graph.bind(prefix, ns_uri)

        # Parse the data into the temporary graph
        temp_graph.parse(data=turtle_data, format="turtle")

        # Add triples from the temp graph to the main graph if they don't exist
        initial_len = len(graph)
        graph += temp_graph # Efficiently add all triples from temp_graph
        added_count = len(graph) - initial_len

        if added_count > 0:
            logger.info(f"{log_prefix} Added {added_count} new triples. Session total: {len(graph)}")
            return True
        else:
            logger.info(f"{log_prefix} Parsed Turtle data, but no *new* triples were added to the session graph.")
            return False

    except Exception as e:
        # Log the specific error and the problematic data
        logger.error(f"{log_prefix} Turtle parsing error: {e}", exc_info=False) # exc_info=False to avoid huge trace for syntax errors
        logger.error(f"{log_prefix} Problematic Turtle data snippet:\n---\n{turtle_data[:500]}...\n---")
        return False

def update_graph_visualization(socketio, session_id: str, graph: Graph):
    """Generates Vis.js data from the graph and broadcasts it to the session room."""
    log_prefix = get_log_prefix(session_id, None, "VisUpdate")
    if not graph:
        logger.warning(f"{log_prefix} Cannot update visualization, graph is missing.")
        return

    try:
        vis_data = graph_to_visjs(graph)
        # Emit only to the specific session room
        socketio.emit('update_graph', vis_data, room=session_id)
        logger.info(f"{log_prefix} Graph update sent to session room '{session_id}'.")
    except Exception as e:
        logger.error(f"{log_prefix} Failed to generate or send graph visualization: {e}", exc_info=True)
