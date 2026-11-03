# Security

No trained public model is currently released. Public inference artifacts use SafeTensors and checksums; checksums establish integrity relative to a manifest, not publisher authenticity. Native checkpoints are local recovery artifacts; only load trusted checkpoints, even with weights_only enabled. Safe formats do not prevent memory exhaustion from untrusted inputs.

Private vulnerability reporting is currently disabled on the remote. Maintainers must enable it before public release; do not publish exploit details, secrets or private data in issues. A non-sensitive request to establish a private reporting channel may be opened. No private contact address has been designated. Reporting activation requires separate remote authorization.

Preflight scans known credential/private-key patterns and forbidden files, with redacted diagnostics. It cannot prove the absence of all secrets. Review source and artifact inventories manually. Dependency metadata inventories are not a security or license certification.
