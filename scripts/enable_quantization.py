"""
Enable INT8 scalar quantization on the live Qdrant collection's dense vectors.
~4x smaller quantized copy kept in RAM (fast search) with originals on disk for
rescoring — less memory, faster retrieval, negligible recall loss. Applied
in-place to the existing collection (no re-embed).
"""
from __future__ import annotations
import os, time
from dotenv import load_dotenv
load_dotenv("/Users/ashutosh/PycharmProjects/RAGPipeline/scholar-rag/.env")
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

COLL = "scholar_rag"
c = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=120)

print("points:", c.count(COLL).count)
print("enabling INT8 scalar quantization (quantile 0.99, always_ram)…")
c.update_collection(
    collection_name=COLL,
    quantization_config=qm.ScalarQuantization(
        scalar=qm.ScalarQuantizationConfig(
            type=qm.ScalarType.INT8,
            quantile=0.99,
            always_ram=True,   # quantized vectors in RAM → fast search
        )
    ),
)
# wait for optimizer to (re)build with quantization
for _ in range(60):
    info = c.get_collection(COLL)
    st = info.status
    print("  status:", st)
    if str(st).lower().endswith("green"):
        break
    time.sleep(5)

info = c.get_collection(COLL)
qcfg = info.config.quantization_config
print("quantization_config now:", qcfg)
print("done — retrieval will use the quantized vectors for the first pass, "
      "rescoring with originals for accuracy.")
