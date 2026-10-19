# Memory-limited attempt, not a scientific result

Stopped deliberately after the process footprint reached 20.9 GB (peak 21.5 GB)
on a 24 GB machine and stack sampling showed a wait for Metal command completion.
The last logged step was 48 of epoch 1. No epoch checkpoint or evaluation completed.
Process termination was verified before any replacement run. The original source,
initial summary and stdout log are retained. This is not evidence about model quality.

The identified avoidable allocation is full-vocabulary logits for all passage tokens
when only response-token losses are used. A response-suffix implementation must match
full-forward losses and gradients before retrying the same experimental protocol.
