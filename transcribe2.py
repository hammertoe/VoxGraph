import asyncio
import hashlib
import os
import re
import sys
import traceback

from collections import deque

import pyaudio
from dotenv import load_dotenv

from google import genai

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
RDF_WINDOW = 5

TRANS_MODEL = "models/gemini-2.0-flash-live-001"
RDF_MODEL = "models/gemini-2.0-flash-live-001"

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("Error: GOOGLE_API_KEY not found in .env file.")
    sys.exit(1)

client = genai.Client(api_key=API_KEY, http_options={"api_version": "v1beta"})

QUICK_LLM_SYSTEM_PROMPT = """
You are a live audio transcription service. Transcribe the user's speech accurately and verbatim. Do not add any extra commentary, summaries, or introductory phrases.
Double check the transcription for accuracy and ensure it is in the correct format.
"""

RDF_SYSTEM_PROMPT = """
You are a transcription processing service that accepts a textual transcript as input. Each input begins with a unique identifier followed by the transcript text. Process the transcript text exactly as provided and create a knowledge graph in Turtle syntax using the DBpedia Ontology provided below. Do not add any extra commentary, summaries, or introductory phrases. Only return what is asked for below.

Return a knowledge graph in Turtle syntax. The knowledge graph should include the following:

1. Identify the unique identifier provided at the start of the transcript and use that for the transcript node’s URI (e.g., ex:transcription_<UNIQUEID>).
2. Identify the full Transcript Text that follows the unique identifier.
3. Create a new transcription node using a URI that incorporates the unique identifier. This transcription node MUST include:
   - The property ex:transcriptionText containing the full transcript text.
4. Identify entities (people, places, concepts, times, organizations, etc.) within the transcript.
5. Create URIs for entities using the ex: prefix and CamelCase (e.g., ex:EntityName). Use existing URIs if the same entities are mentioned again.
6. Identify relationships between entities using properties from the DBpedia Ontology where applicable (for example, dbo:organisationFounded, dbo:locatedIn, dbo:abstract, dbo:ledBy, etc.).
7. Identify properties of entities (e.g., rdfs:label, dbo:name, dbo:birthDate, ex:hasValue). Use appropriate datatypes for literals (e.g., "value"^^xsd:string, "123"^^xsd:integer, "2024-01-01"^^xsd:date).
8. For significant entities or statements derived directly from the Transcript Text, add a triple linking them to the transcription node using a relation such as ex:derivedFromTranscript.
9. Now go back and check again as you probably have missed some entities or relationships.
10. Use the following format for the output:

Example:
```
ex:AliceJohnson ex:derivedFromTranscript ex:transcription_<UNIQUEID> .
ex:ProjectPhoenix ex:derivedFromTranscript ex:transcription_<UNIQUEID> .
```

Format your output as valid Turtle triples. Output ONLY Turtle syntax and do not repeat triples.

Example Input User Message:  
"000001 Acme Corporation announced Project Phoenix. Alice Johnson leads it."

Example Output Format:
```
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .
@prefix dbo: <http://dbpedia.org/ontology/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

ex:transcription_000001 a ex:Transcription ;
    ex:transcriptionText "Acme Corporation announced Project Phoenix. Alice Johnson leads it." .

ex:AcmeCorporation a dbo:Organisation ;
    dbo:announcedProject ex:ProjectPhoenix ;
    ex:derivedFromTranscript ex:transcription_000001 .

ex:ProjectPhoenix a dbo:Project ;
    rdfs:label "Phoenix"^^xsd:string ;
    dbo:ledBy ex:AliceJohnson ;
    ex:derivedFromTranscript ex:transcription_000001 .

ex:AliceJohnson a dbo:Person ;
    ex:derivedFromTranscript ex:transcription_000001 .
```

DBPEDIA ONTOLOGY:

"""# + open("ontology.ttl.txt").read()

# --- Updated CONFIG with system_instruction ---
TRANS_CONFIG = {
    "response_modalities": ["TEXT"],
    # Explicitly instruct the model to perform transcription only
    "system_instruction": QUICK_LLM_SYSTEM_PROMPT,
    "activity_handling": 'NO_INTERRUPTION',
    "turn_coverage": 'TURN_INCLUDES_ONLY_ACTIVITY',
}

RDF_CONFIG = {
    "response_modalities": ["TEXT"],
    "system_instruction": RDF_SYSTEM_PROMPT,
}


pya = pyaudio.PyAudio()

def extract_complete_sentences(buffer: str):
    """
    Extracts complete sentences from a text buffer.
    
    A complete sentence is defined as one that ends with a punctuation mark (., !, or ?).
    Any text following the last complete sentence (i.e. an incomplete sentence) is
    returned as part of the remaining buffer.
    
    Parameters:
        buffer (str): The current text buffer containing parts or whole sentences.
    
    Returns:
        tuple: A pair where the first element is a list of complete sentences,
               and the second element is the leftover incomplete text.
    """
    # This regex matches as little text as needed until it finds one of . ! ?,
    # followed by one or more whitespace characters or end-of-string.
    pattern = re.compile(r'(.*?[.!?])(?:\s+|$)', re.DOTALL)
    
    complete_sentences = []
    last_match_end = 0

    for match in pattern.finditer(buffer):
        # Each match.group(1) ends in a punctuation character.
        sentence = match.group(1)
        complete_sentences.append(sentence)
        last_match_end = match.end()
    
    # The remaining buffer is the text after the last matched complete sentence.
    remaining_buffer = buffer[last_match_end:]
    
    return complete_sentences, remaining_buffer

class AudioLoop:
    def __init__(self):
        self.audio_in_queue = None
        self.out_queue = None
        self.trans_session = None
        self.rdf_session = None
        self.audio_stream = None  # Keep track of the audio stream
        self.buffer = ""  # Buffer to hold incoming text
        self.rdf_buffer = deque(maxlen=RDF_WINDOW)  # Buffer to hold RDF text

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            try:
                await self.trans_session.send(input=msg)
                #print(f"\nSent audio data to server: {len(msg['data'])} bytes")  # Added print statement
            except Exception as e:
                print(f"\nError sending audio: {e}")
                break  # Exit the loop if there's an error sending

    async def send_rdf(self):
        while True:
            msg = await self.rdf_queue.get()
            self.rdf_buffer.append(msg)
            para = " ".join(self.rdf_buffer)
            trans_id = hashlib.md5(para.encode('utf-8')).hexdigest()[:8]
            try:
                await self.rdf_session.send_client_content(
                    turns={"role": "user", "parts": [{"text": trans_id + " " + para}]},
                )
                async for response in self.rdf_session.receive():
                    if ttl := response.text:
                        # Using sys.stdout.write and flush for potentially smoother printing
                        sys.stdout.write(ttl)
                        sys.stdout.flush()

            except Exception as e:
                print(f"\nError sending rdf: {e}")
                break 

    async def listen_audio(self):
        try:
            mic_info = pya.get_default_input_device_info()
            print(f"Using microphone: {mic_info['name']}") # Added print statement
            self.audio_stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                input_device_index=mic_info["index"],
                frames_per_buffer=CHUNK_SIZE,
            )
        except Exception as e:
            print(f"\nError opening audio stream: {e}")
            return  # Exit the task if the stream can't be opened

        print("Listening... Speak into the microphone.") # Added status message

        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            try:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
            except IOError as e:
                 # Handle Input overflowed gracefully if not in debug mode
                if e.errno == pyaudio.paInputOverflowed and not __debug__:
                    print("Warning: Input overflowed. Some audio may have been lost.", file=sys.stderr)
                    continue # Continue reading audio
                else:
                    print(f"\nError reading audio: {e}")
                    break # Exit loop on other IOErrors
            except Exception as e:
                print(f"\nError reading audio: {e}")
                break  # Exit the loop if there's another read error


    async def receive_text(self):
        while True:
            try:
                turn = self.trans_session.receive()
                async for response in turn:
                    if text := response.text:
                        # Using sys.stdout.write and flush for potentially smoother printing
                        #sys.stdout.write(text)
                        #sys.stdout.flush()
                        self.buffer += text
                        # Extract complete sentences from the buffer
                        complete_sentences, self.buffer = extract_complete_sentences(self.buffer)
                        for sentence in complete_sentences:
                            # Print each complete sentence
                            print(sentence)
                            self.rdf_queue.put_nowait(sentence)
                            print(self.rdf_queue)   
            except Exception as e:
                print(f"\nError receiving text: {e}")
                break  # Exit the loop if there's an error receiving


    async def run(self):
        try:
            async with (
                client.aio.live.connect(model=TRANS_MODEL, config=TRANS_CONFIG) as trans_session, 
                client.aio.live.connect(model=RDF_MODEL, config=RDF_CONFIG) as rdf_session, 
                asyncio.TaskGroup() as tg,
            ):
                self.trans_session = trans_session
                self.rdf_session = rdf_session
                self.out_queue = asyncio.Queue(maxsize=10) # Slightly increased queue size
                self.rdf_queue = asyncio.Queue(maxsize=10) # Slightly increased queue size

                tg.create_task(self.send_realtime())
                tg.create_task(self.send_rdf())
                listen_task = tg.create_task(self.listen_audio())
                tg.create_task(self.receive_text())

                await listen_task # Wait specifically for listen_audio to potentially finish first
                await asyncio.Future()  # Run until interrupted or an error occurs in tasks

        except asyncio.CancelledError:
            print("\nTranscription stopped.")
        except ExceptionGroup as eg:
             print("\nAn error occurred within task group:")
             for i, exc in enumerate(eg.exceptions):
                 print(f"  Error {i+1}: {exc}")
                 traceback.print_exception(type(exc), exc, exc.__traceback__)
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            traceback.print_exc()
        finally:
            print("Cleaning up audio stream...")
            if self.audio_stream and self.audio_stream.is_active():
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            pya.terminate() # Terminate PyAudio instance
            print("Cleanup complete.")


if __name__ == "__main__":
    # Add basic check for API key before starting
    if not API_KEY:
        print("Error: GOOGLE_API_KEY not found. Please set it in the .env file.")
    else:
        main = AudioLoop()
        try:
            asyncio.run(main.run())
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
