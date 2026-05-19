# Nginx

Nginx can be used as a generic HTTP reverse proxy in front of FastInference,
but multi-container load balancing is not a maintained project-level deployment
profile.

First validate one local server:

```bash
uv run python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3.5-9B-AWQ \
  --host 0.0.0.0 \
  --port 8000
```

A minimal reverse-proxy location is:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

For production traffic, keep the FastInference process model simple unless the
target workload has been validated with the correctness and performance gates in
this repository.
