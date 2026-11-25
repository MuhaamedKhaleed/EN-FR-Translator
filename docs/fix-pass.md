# Complete-input inference follow-up, 25 November 2026

Translate no longer silently truncates long source text at max-length. Tokenization first checks complete source length including special tokens. Overlength input produces an actionable error to split text explicitly or raise the supported limit. No incomplete translation is returned as if it covered the full document. The same max-length still bounds generated tokens; reaching that bound is not proof the target text is complete. Do not use for unattended legal/medical document translation.

This change fixes ordinary-use reliability, not model quality. Tests include long source rejection and full successful tiny-model CLI/demo paths. Existing real Marian source and16sentence development metrics are unchanged. No new full fine-tuning or original notebook training reproduction is claimed.
