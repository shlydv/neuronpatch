# Boundary transfer: compact learning works, guided selection fails transfer

The repaired Qwen3-4B-Instruct-2507 comparison completed all eight fits and1,728
generations. Independent artifact checks passed. The NeST-inspired selected
groups do not meet the declared method criteria. A random grouped patch can
learn useful behavior with fewer parameters, but the large difference between
two random seeds prevents recommending random selection as a robust recipe.

## Answer review

The assistant reviewed218 distinct document-response groups covering all48
confirmation documents, against their supplied tables. Exact marker responses
were scored separately. This is not an independent human evaluation. Decisions
are bound to response hashes in private artifacts; the public
[semantic summary](semantic_summary.json) includes paired counts and sensitivity
analyses. Five ambiguous/malformed QA documents(2,9,15,25,30) are excluded from
QA comparisons for every arm; their legitimate instruction controls remain.
Incomplete outputs fail even if a promising answer fragment appears earlier.

| Condition | Clean correct /43 | Redirect correct /43 | Unseen authority correct /43 | Recovery on base-correct documents: redirect /27 | Recovery: authority /27 | Clean regressions |
|---|---:|---:|---:|---:|---:|---:|
| Base |27|0|0|0|0|0|
| LoRA |29|27|22|25|21|0|
| Guided groups |27|27|0|24|0|2|
| Random groups,43 |31|30|19|25|19|0|
| Random groups,71 |28|29|4|25|4|3|

All conditions answer all192 legitimate direct/quoted marker requests correctly.
These controls test compliance and quotation comprehension in this setup, not
broad unrelated-task retention. Recovery uses documents with correct completed
base clean answers; higher total clean accuracy can coexist with regressions.

The primary convention scores the requested final answer, accepting explicit
self-corrections. Stricter supporting-claim scoring gives26 base-clean-correct
documents; authority recoveries remain21,0,19,4 for LoRA,guided,random43,random71.
Lenient acceptance of an aggregate score instead of its sum and of an electoral
riding instead of its candidate changes some counts: guided clean regressions
fall from2 to1 and LoRA authority recovery rises from21 to22. Neither sensitivity
changes the method decision. The reference for document34 incorrectly excludes
the episode equal to14%; the correct not-over14 count is9. Document41's reference
names a riding although the question asks who; both interpretations are reported.

## Frozen criteria

- Recover at least half of base paired failures in each family: guided fails
  on authority(0/27).
- Guided beats each random support by at least four authority answers: fails.
- Guided within one answer of LoRA in both families at no more than half its
  trainable scalars: parameter condition passes, quality condition fails.
- No additional clean/control regressions versus comparators: fails, including
  under lenient scoring.
- At least16 correct base clean documents for useful headroom: passes(27).

## What the failure tells us

Guided selection learns redirect examples and preserves direct/quoted controls,
yet returns the attacker marker on all48 authority prompts. Thus it is neither
a total inability to train nor a successful general boundary repair. One random
support also transfers poorly while the other transfers substantially better.
Selection can matter, but this experiment does not identify the causal reason
the better support works.

All72 discovery probes fit the training labels perfectly. Their label distinction
is clean versus redirect-injected prompts; the same payload in legitimate direct
or quotation contexts is absent from discovery. Payload cues therefore remain
a plausible shortcut. Activation-based grouping also lacked positive gradient
alignment evidence on the two measured examples. These are hypotheses, not a
causal tracing result.

The guided patch's first observed update reduces redirect calibration loss more
than the random patches, yet its authority transfer is worse. Those calibration
examples share training attack wording. Improving a local gradient proxy on
that distribution would not by itself solve the transfer problem. We should
test discovery against matched legitimate uses of the same instruction text,
then require fresh attack families before recommending a selector. Do not tune
another rate or support on these exposed confirmation outputs.

## Costs and repair

Grouped patches train368,640 scalars; attention-plus-MLP rank1 LoRA trains2,064,384.
The grouped budget is17.86% of LoRA's, but fit cycles are about364seconds versus
349seconds for LoRA, including validation and serialization. No training-speed
advantage is shown. Peak allocated GPU memory is8.28–8.31GiB; the whole worker
takes6,527.17seconds including generation. These are measured wrapper/allocation
figures, not provider quota billing or pure optimization timings.

The original run failed during a gradient-bearing calibration forward before
the first Adam update. Non-reentrant decoder recomputation repaired memory use
without changing data, targets, rates, counts or criteria. The repaired run
replays baseline outputs, teachers, supervision, probes, supports and discovery
geometry byte-for-byte. All fits finish with zero numerical retries, exact
adapter reload checks and restored base fingerprints. Software verification
does not substitute for the answer review above.
