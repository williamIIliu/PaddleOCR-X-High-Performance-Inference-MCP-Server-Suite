import time
from prometheus_client import Counter, Histogram
 
# 定义监控指标
REQUEST_COUNT = Counter('ocr_requests_total', 'Total OCR requests')
REQUEST_LATENCY = Histogram('ocr_request_latency_seconds', 'OCR request latency')
 
@REQUEST_LATENCY.time()
def process_with_monitoring(image_path):
    REQUEST_COUNT.inc()
    start_time = time.time()
    result = pipeline.predict(image_path)
    latency = time.time() - start_time
    return result, latency