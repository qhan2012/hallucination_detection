# hallucination_detection/debug_logger.py
"""
Simple debug-level logging utility.
"""

# Debug level constants
DEBUG_NONE = 0
DEBUG_ERROR = 1
DEBUG_WARNING = 2
DEBUG_INFO = 3
DEBUG_VERBOSE = 4

# Global debug level. Adjust as needed.
DEBUG_LEVEL = DEBUG_WARNING

def set_debug_level(level: int):
    global DEBUG_LEVEL
    DEBUG_LEVEL = level

def debug_print(level: int, message: str):
    if DEBUG_LEVEL >= level:
        print(f"[DBG] {message}")
