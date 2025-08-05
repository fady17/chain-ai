# src/minichain/text_splitters/streaming_arabic.py
import re
from typing import Iterator, List, Optional

class StreamingArabicSentenceSplitter:
    """
    An optimized, stateful text splitter for Arabic text that processes streams
    and yields complete sentences. Designed for mission-critical voice agents.
    (This version uses the user's correct buffer management logic)
    """
    DEFAULT_SEPARATORS = [".", "!", "?", "؟", "؛", "\u061F", "\u061B"]
    
    def __init__(self, separators: Optional[List[str]] = None, max_buffer_size: int = 10000, min_sentence_length: int = 3):
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.max_buffer_size = max_buffer_size
        self.min_sentence_length = min_sentence_length
        
        separator_pattern = "|".join(re.escape(sep) for sep in self.separators)
        self.split_pattern = re.compile(f'({separator_pattern})')
        
        self.buffer = ""
        self._sentences_yielded = 0
        
    def _is_valid_sentence(self, sentence: str) -> bool:
        sentence = sentence.strip()
        if len(sentence) < self.min_sentence_length:
            return False
        return any(char.isalpha() for char in sentence)
    
    def _process_buffer(self) -> Iterator[str]:
        """
        Processes the internal buffer and yields any complete sentences.
        This is a stateful operation that modifies self.buffer.
        """
        # Split by separators, keeping the separators themselves.
        # e.g., "Hello world. How are you?" -> ['Hello world', '.', ' How are you?']
        parts = self.split_pattern.split(self.buffer)
        
        # If there are no separators, the list has only one part.
        if len(parts) <= 1:
            # Check for safety flush if buffer is too large
            if len(self.buffer) > self.max_buffer_size:
                if self._is_valid_sentence(self.buffer):
                    yield self.buffer.strip()
                    self._sentences_yielded += 1
                self.buffer = "" # Clear buffer after forced flush
            return

        # We can safely yield all but the last part, as it's an incomplete sentence.
        # We pair up the text and its terminator.
        # e.g., parts = ['Hello world', '.', ' How are you', '?', '']
        
        # We iterate up to the second-to-last element because the last one is the new buffer.
        processed_len = 0
        for i in range(0, len(parts) - 1, 2):
            sentence = parts[i] + parts[i+1]
            if self._is_valid_sentence(sentence):
                yield sentence.strip()
                self._sentences_yielded += 1
            processed_len += len(sentence)
        
        # The new buffer is what's left over.
        self.buffer = self.buffer[processed_len:]
    
    def add_chunk(self, chunk: str) -> Iterator[str]:
        self.buffer += chunk
        # This loop is crucial. A single chunk might contain multiple sentences.
        # _process_buffer yields one sentence at a time and updates the buffer,
        # so we need to keep calling it until it's empty.
        while True:
            # We create a list of sentences found in the current buffer
            sentences_in_chunk = list(self._process_buffer())
            if not sentences_in_chunk:
                # No more complete sentences in the buffer, break the loop
                break
            # Yield all the complete sentences we just found
            for sentence in sentences_in_chunk:
                yield sentence

    def flush(self) -> Iterator[str]:
        if self._is_valid_sentence(self.buffer):
            yield self.buffer.strip()
            self._sentences_yielded += 1
        self.buffer = ""

    def reset(self):
        self.buffer = ""
        self._sentences_yielded = 0