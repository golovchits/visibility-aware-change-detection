### Video A vs video B

| Object | Truth | Naive | Ours |
|---|---|---|---|
| notebook | removed | removed | removed |
| pencil case | moved | moved | **not checked + added** (wrong) |
| flashcards | added | added | added |
| water bottle | not checked | **removed** (wrong) | not checked |
| laptop | unchanged | unchanged | unchanged |

Naive gets 4 of 5 right and reports 4 changes. Ours gets 4 of 5 right and reports 3 changes.

### Video B vs video C

| Object | Truth | Naive | Ours |
|---|---|---|---|
| laptop | unchanged | unchanged | unchanged |
| pencil case | unchanged | unchanged | unchanged |
| flashcards | unchanged | unchanged | unchanged |
| water bottle | not checked | **added** (wrong) | not checked |

Naive gets 3 of 4 right and reports 1 change. Ours gets 4 of 4 right and reports 0 changes.

Video B vs video C has no real changes, so every change reported there is a false alarm.
