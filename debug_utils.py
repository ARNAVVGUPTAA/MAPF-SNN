#!/usr/bin/env python3
"""
Debug utilities for controlling debug output throughout the MAPF-GNN training system.
"""

from config import config

def debug_print(*args, **kwargs):
    """
    Print debug messages only when debug flag is enabled.
    
    Usage:
        debug_print("This is a debug message")
        debug_print(f"Variable value: {var}")
    """
    if config.get('debug', False):
        print(*args, **kwargs)

def is_debug_enabled():
    """
    Check if debug mode is enabled.
    
    Returns:
        bool: True if debug mode is enabled, False otherwise
    """
    return config.get('debug', False)
