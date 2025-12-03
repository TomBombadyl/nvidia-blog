# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| < Latest| :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue. Instead, please send an email to the repository maintainers with details about the vulnerability.

**Please include:**
- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- The location of the affected code
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

We will acknowledge receipt of your report within 48 hours and provide a more detailed response within 7 days indicating the next steps in handling your report.

## Security Considerations

This service:
- Accesses publicly available RSS feeds from NVIDIA's official blogs
- Does not store or cache blog content locally
- Uses Google Cloud Platform services (Vertex AI, Cloud Run)
- Requires proper GCP authentication and service account permissions

**Important:** Ensure that:
- GCP service account credentials are never committed to the repository
- Environment variables containing sensitive information are properly secured
- The `.env` file is in `.gitignore` and never committed
