# SPDX-License-Identifier: Apache-2.0
import logging
from typing import Optional
import torch

class RoutedExpertsCapturer:
    def __init__(self):
        pass
    
    @classmethod
    def create(cls):
        return cls()
        
    def init_buffer(self, *args, **kwargs):
        pass
        
    def capture(self, *args, **kwargs):
        pass
        
    def clear_buffer(self):
        pass

class RoutedExpertsReader:
    def __init__(self):
        pass
        
    @classmethod
    def create(cls):
        return cls()
        
    def attach_buffer(self, *args, **kwargs):
        pass
        
    def read(self, *args, **kwargs):
        return None
