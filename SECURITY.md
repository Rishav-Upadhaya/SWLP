# Security Policy

Please report vulnerabilities privately through
[GitHub Security Advisories](https://github.com/Rishav-Upadhaya/SWLP/security/advisories/new),
not public issues. Fixes land on the latest release only.

`swlp serve` binds to `127.0.0.1` by default and has no authentication. Anyone
who can reach the port can run prompts on your machine; only pass
`--host 0.0.0.0` (or another address) on a trusted network or behind an
authenticating reverse proxy.
