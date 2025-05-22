import logging
import os

log = logging.getLogger(__file__)

import redis as RedisLib

REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
redis = RedisLib.Redis(host=REDIS_HOST, decode_responses=True)
