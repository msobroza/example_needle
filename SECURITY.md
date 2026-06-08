# Security Policy

## Supported versions

The latest released minor version receives security fixes.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.
Use GitHub's [private vulnerability reporting][advisories] for this repository,
or contact the maintainers directly.

We aim to acknowledge reports within a few business days and to provide a fix
or mitigation timeline after triage.

## Scope notes

- This project loads third-party model weights when a real retriever backend is
  used. Only load models from sources you trust.

[advisories]: https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability
