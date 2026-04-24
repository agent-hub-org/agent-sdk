import re

SAFE_SESSION_RE = re.compile(r'^[a-zA-Z0-9\-]{1,64}$')
