# SPDX-License-Identifier: Apache-2.0
from enum import Enum

class SchedulingPolicy(Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"

class RequestQueue:
    def __init__(self):
        self.queue = []
    def __len__(self): return len(self.queue)
    def __bool__(self): return len(self.queue) > 0
    def add_request(self, request): self.queue.append(request)
    def prepend_request(self, request): self.queue.insert(0, request)
    def prepend_requests(self, requests):
        if hasattr(requests, 'queue'):
            self.queue = requests.queue + self.queue
        else:
            self.queue = list(requests) + self.queue
    def pop_request(self): return self.queue.pop(0)
    def peek_request(self): return self.queue[0]
    def remove_requests(self, requests):
        for r in requests:
            if r in self.queue:
                self.queue.remove(r)

def create_request_queue(policy):
    return RequestQueue()
