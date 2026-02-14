import unittest
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

# Try to import torch, skip tests if not available
try:
    import torch
    from vllm.entrypoints.output_processors.verl import VerlOutputProcessor
    from vllm.outputs import RequestOutput, CompletionOutput
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class TestVerlOutputProcessor(unittest.TestCase):
    def setUp(self):
        if not TORCH_AVAILABLE:
            self.skipTest("Torch not available")

    def test_process_outputs(self):
        processor = VerlOutputProcessor()
        
        # Create mock request outputs
        req1 = MagicMock(spec=RequestOutput)
        out1 = MagicMock(spec=CompletionOutput)
        out1.token_ids = [1, 2, 3]
        out1.cumulative_logprob = -0.5
        out1.logprobs = None
        req1.outputs = [out1]
        
        req2 = MagicMock(spec=RequestOutput)
        out2 = MagicMock(spec=CompletionOutput)
        out2.token_ids = [4, 5]
        out2.cumulative_logprob = -0.3
        out2.logprobs = None
        req2.outputs = [out2]
        
        request_outputs = [req1, req2]
        
        # Run processor
        result = processor.process_outputs(request_outputs)
        
        # Verify keys
        self.assertIn("token_ids", result)
        self.assertIn("cumulative_logprob", result)
        
        # Verify token_ids tensor shape and content (padded)
        token_ids = result["token_ids"]
        self.assertIsInstance(token_ids, torch.Tensor)
        self.assertEqual(token_ids.shape, (2, 3)) # Max len is 3
        
        # Check padding (assuming 0 padding)
        expected_token_ids = torch.tensor([
            [1, 2, 3],
            [4, 5, 0]
        ], dtype=torch.long)
        
        self.assertTrue(torch.equal(token_ids, expected_token_ids))
        
        # Verify logprobs
        cum_logprobs = result["cumulative_logprob"]
        self.assertIsInstance(cum_logprobs, torch.Tensor)
        self.assertEqual(cum_logprobs.shape, (2,))
        self.assertTrue(torch.allclose(cum_logprobs, torch.tensor([-0.5, -0.3])))

    def test_empty_outputs(self):
        processor = VerlOutputProcessor()
        result = processor.process_outputs([])
        self.assertEqual(result, {})

if __name__ == '__main__':
    unittest.main()
