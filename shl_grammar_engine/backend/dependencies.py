import logging
from functools import lru_cache

from .analyzer import CommunicationAnalyzer
from .asr import SpeechRecognition
from .grammar import GrammarScorer

LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_speech_recognizer() -> SpeechRecognition:
    LOGGER.info("Initializing SpeechRecognition singleton")
    return SpeechRecognition()


@lru_cache(maxsize=1)
def get_grammar_scorer() -> GrammarScorer:
    LOGGER.info("Initializing GrammarScorer singleton")
    return GrammarScorer()


@lru_cache(maxsize=1)
def get_analyzer() -> CommunicationAnalyzer:
    LOGGER.info("Initializing CommunicationAnalyzer singleton")
    return CommunicationAnalyzer(get_speech_recognizer(), get_grammar_scorer())
