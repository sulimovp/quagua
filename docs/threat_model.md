# Threat model and non-goals

## What the evaluation assumes

The bundled evaluation imagines an analyst who sees **only encoded features** (and labels for the utility models where applicable) and tries to infer **original features or sensitive attributes** with standard sklearn models. That is a deliberately narrow, scriptable attacker: it is useful for comparing encoders on equal footing, not for modeling a motivated insider with side information, linkage attacks, or full database access.

## Non-goals

- **Differential privacy** — not implemented here. Noise mechanisms, privacy budgets, and formal composition are out of scope unless you add them elsewhere.
- **Cryptographic guarantees** — encoders are not encryption schemes. Keys, if any, are not managed as in production crypto.
- **Regulatory sign-off** — no warranty that use of this code satisfies HIPAA, GDPR Article 25, or similar frameworks.

## Responsible use

Report evaluation numbers with the dataset, split strategy, seeds, and model list. If you open-source a derivative, keep the same honesty about limits so downstream teams do not mistake experimental scores for certifications.
