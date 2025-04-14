# -*- coding: utf-8 -*-
import google.generativeai as genai
from google.generativeai import types as genai_types
from google.api_core import exceptions as google_exceptions
import config
from logger_setup import logger, get_log_prefix

class LLMHandler:
    """Handles initialization and interaction with Google Gemini models."""

    def __init__(self):
        self.base_client = None
        self.live_client = None
        self._initialize_clients()

    def _initialize_clients(self):
        """Initializes the base and live API clients using the API key."""
        log_prefix = get_log_prefix(component="LLMInit")
        if not config.GOOGLE_API_KEY:
            logger.critical(f"{log_prefix} GOOGLE_API_KEY is missing. LLM features disabled.")
            return

        logger.info(f"{log_prefix} GOOGLE_API_KEY found. Initializing Gemini clients.")
        try:
            # Base client (v1) for chat models
            self.base_client = genai.Client(api_key=config.GOOGLE_API_KEY)
            # Simple check: list models to verify connection/key
            _ = self.base_client.models.list()
            logger.info(f"{log_prefix} Standard base Client (v1) initialized successfully.")
        except Exception as e:
            logger.error(f"{log_prefix} Standard base Client initialization failed: {e}", exc_info=True)
            self.base_client = None

        try:
            # Live client (v1alpha) for streaming transcription
            self.live_client = genai.Client(
                api_key=config.GOOGLE_API_KEY,
                http_options={"api_version": config.LIVE_API_VERSION}
            )
            # Check if live connection attribute exists
            _ = getattr(self.live_client.aio.live, 'connect')
            logger.info(f"{log_prefix} Live Client ({config.LIVE_API_VERSION}) initialized successfully.")
        except AttributeError:
             logger.error(f"{log_prefix} Failed to access live API functionality. Ensure '{config.LIVE_API_VERSION}' API version is supported.")
             self.live_client = None
        except Exception as e:
             logger.error(f"{log_prefix} Live Client ({config.LIVE_API_VERSION}) initialization failed: {e}", exc_info=True)
             self.live_client = None

        if not self.base_client or not self.live_client:
            logger.warning(f"{log_prefix} Proceeding with potentially limited functionality due to client initialization issues.")

    def is_base_client_available(self) -> bool:
        """Check if the base client for chat models is initialized."""
        return self.base_client is not None

    def is_live_client_available(self) -> bool:
        """Check if the live client for transcription is initialized."""
        return self.live_client is not None

    def initialize_session_chats(self, session_state: dict, session_id: str):
        """
        Initializes the quick, slow, and query chat objects for a session state dictionary.
        Returns True if at least one chat model was successfully initialized or already existed.
        """
        log_prefix = get_log_prefix(session_id, component="ChatInit")
        if not self.base_client:
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
                    system_instruction=config.QUICK_LLM_SYSTEM_PROMPT
                )
                session_state['quick_chat'] = self.base_client.chats.create(
                    model=config.QUICK_LLM_MODEL, config=quick_config
                )
                logger.info(f"{log_prefix} Initialized Quick LLM Chat: {config.QUICK_LLM_MODEL}")
            except Exception as e:
                logger.error(f"{log_prefix} Failed to initialize Quick LLM ({config.QUICK_LLM_MODEL}): {e}", exc_info=True)
                session_state['quick_chat'] = None

        # Initialize Slow Chat if needed
        if session_state.get('slow_chat') is None:
            try:
                slow_config = genai_types.GenerateContentConfig(
                    temperature=0.3, top_p=0.95, top_k=40, max_output_tokens=4096,
                    system_instruction=config.SLOW_LLM_SYSTEM_PROMPT
                )
                session_state['slow_chat'] = self.base_client.chats.create(
                    model=config.SLOW_LLM_MODEL, config=slow_config
                )
                logger.info(f"{log_prefix} Initialized Slow LLM Chat: {config.SLOW_LLM_MODEL}")
            except Exception as e:
                logger.error(f"{log_prefix} Failed to initialize Slow LLM ({config.SLOW_LLM_MODEL}): {e}", exc_info=True)
                session_state['slow_chat'] = None

        # Initialize Query Chat if needed
        if session_state.get('query_chat') is None:
            try:
                query_config = genai_types.GenerateContentConfig(
                    temperature=0.3, top_p=0.95, top_k=40, max_output_tokens=2048,
                    system_instruction=config.QUERY_LLM_SYSTEM_PROMPT
                )
                session_state['query_chat'] = self.base_client.chats.create(
                    model=config.QUERY_LLM_MODEL, config=query_config
                )
                logger.info(f"{log_prefix} Initialized Query LLM Chat: {config.QUERY_LLM_MODEL}")
            except Exception as e:
                logger.error(f"{log_prefix} Failed to initialize Query LLM ({config.QUERY_LLM_MODEL}): {e}", exc_info=True)
                session_state['query_chat'] = None

        # Return True if any chat model is available
        return bool(session_state.get('quick_chat') or session_state.get('slow_chat') or session_state.get('query_chat'))

    def get_live_api_config(self) -> genai_types.GenerateContentConfig:
        """Returns the configuration object for the Live API."""
        # Simplified system instruction for transcription focus
        system_instruction = genai_types.Content(parts=[genai_types.Part(
            text="You are a highly accurate real-time transcription assistant. Focus on transcribing the spoken words clearly."
        )])
        return genai_types.GenerateContentConfig(
            response_modalities=[genai_types.Modality.TEXT],
            system_instruction=system_instruction
        )

# Create a single instance to be used across the application
llm_handler_instance = LLMHandler()
