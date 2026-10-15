# Design note

**The idea.** The brief asks to tell real change from "differences in camera viewpoint, occlusion, or incomplete observations". A diff of two reconstructions can't. A spot the second video never saw looks exactly like a spot that is now empty. So every object first gets one question. Did the other video look at its spot? Only then is it compared.

**Choices, and what I rejected.**

- *One joint VGGT run on both videos.* Rejected two reconstructions plus an alignment step, which is one more thing to fail. Rejected COLMAP, which is slow and fragile on a plain white table.
- *Position decides if a spot is occupied, size decides if it is the same object.* Rejected labels. The detector called the flashcards "box" in one video and "book", "notebook" and "cards" in another. Matching labels would have made the control pair report a change.
- *Coverage counts each 1 cm cube of surface once.* Rejected counting frames, and with it Dempster's rule, which my thesis used. Twenty near-identical frames are not twenty independent witnesses.
- *Low-confidence depth counts as seen only if it agrees with the expected distance.* VGGT is least sure on plain white table, which is exactly what a removed object leaves behind. I found this on the A vs B notebook, so it is test data reuse, and I say so. On the control pair it adds no false visibility.
- *Template sentences.* Rejected an LLM. Every number in a sentence comes from results.json.

**How I measured it.** Ground truth written while filming. A no-change control pair (B vs C). A naive baseline that is the same pipeline without the visibility test, so the test is the only difference. Thresholds were checked on B vs C only and frozen before the final A vs B run. The one rule shaped by A vs B is the low-confidence depth rule above.

**What didn't work.** The pencil case move. Video B saw 41% of its old spot, under the 50% cut-off, so the system said "not checked" and lost the move. Accuracy on A vs B ties with the baseline at 4 of 5. What the visibility test buys is the kind of mistake. It never says "removed" without looking, and it raised no false alarms on the control pair, where the baseline raised one.

**The tradeoff I am unsure about.** That 50% cut-off. Anything from 5% to 90% passes the control pair, so it can't tell me where the cut-off belongs. A lower one would have caught the pencil case and would decide on thinner evidence everywhere else. I left it alone, because moving it after seeing A vs B would be tuning on the test. A second scene would settle it.

**Working with AI.** I built this with Claude Code and logged its mistakes as I went. It caught a real flaw in my first presence rule, which would have got two of five objects wrong. It also proposed rules without checking the data (low-confidence depth as unseen, labels as identity), and it drifted toward making A vs B come out right, which I stopped.
