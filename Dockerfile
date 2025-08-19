FROM nvcr.io/nvidia/tritonserver:24.05-py3

# Install runtime deps for Python backend
RUN pip install --no-cache-dir pillow numpy

# Copy your model repository
COPY models /models

# (Optional) quick health check script
RUN echo '#!/usr/bin/env bash\ntritonserver --model-repository=/models --exit-on-error=false --disable-auto-complete-config' > /entrypoint.sh \
 && chmod +x /entrypoint.sh

EXPOSE 8000 8001 8002
CMD ["/entrypoint.sh"]
