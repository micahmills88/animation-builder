"""Wrapper that runs the upstream text-encoder server with a single-worker
Gradio queue.

Why this exists: upstream's `kimodo.scripts.run_text_encoder_server` calls
`demo.launch()` without configuring a queue. Gradio 6's default behavior spins
up a multiprocessing worker pool, and those workers crash on
`rebuild_cuda_tensor` when receiving the task payload — the LLM2Vec model lives
on CUDA in the main process and its tensor handles can't be reconstructed in a
spawned worker. Result: every Kimodo embedding request hangs forever.

Fix: monkey-patch `gradio.Blocks.launch` to call `queue(default_concurrency_limit=1)`
first. With concurrency_limit=1, Gradio serializes requests through a single
in-process handler, so CUDA tensors stay in the process that owns them.

Mounted into the text-encoder container by our docker-compose.yml; the service's
`command:` is overridden to run this wrapper instead of `kimodo.scripts.run_text_encoder_server`.
"""

import gradio as gr
from gradio.blocks import Blocks

_orig_launch = Blocks.launch


def _patched_launch(self, *args, **kwargs):
    self.queue(default_concurrency_limit=1, max_size=64)
    return _orig_launch(self, *args, **kwargs)


Blocks.launch = _patched_launch

# Now hand off to upstream's main.
from kimodo.scripts import run_text_encoder_server  # noqa: E402

if __name__ == "__main__":
    run_text_encoder_server.main()
