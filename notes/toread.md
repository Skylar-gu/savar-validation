

The big picture: We built a trick that figures out which internal patterns an AI weather-model actually relies on — and how much to trust that answer — without ever having the answer key. These experiments stress-test whether the trick keeps working as we make the setup more realistic.

- R1 (overlapping patterns): We made the AI's internal patterns physically smear into each other, the way real climate patterns do, and the trick still correctly picked out the good patterns from the bad ones.
- R5 (harder training): We retrained the AI the tougher way that top real-world models (like GraphCast) are trained, and even though that reshuffled its internals, the trick still worked.
- R3 (several variables at once): We fed the system multiple linked measurements together (think temperature and pressure), and the trick worked cleanly there too.
- R4 (three times bigger): We tripled the number of internal patterns to test realistic scale — the flexible method for finding the patterns held up nicely (the old rigid method fell apart), but the trust-check stumbled, most likely because the candidates we compared weren't on an even playing field.
- R4 re-test (running now): We're redoing R4 with all the candidates put on equal footing, to see whether that stumble was just an unfair comparison — if it was, the trick should pass this time.
- E4-final (up next): We'll pool every experiment together to build a single "trust dial" — turning how strongly our two independent checks agree into how accurate the recovered answer probably is — which is the gauge we'd ultimately read off on a real model where there's no answer key at all.

Scoreboard so far: trick works in 3 of 4 tests (overlap ✅, harder training ✅, multiple variables ✅), with the fourth (bigger scale) looking like a fixable fairness issue rather than a real failure — which the re-test now running should settle. I'll report its result and move straight into building the trust dial.
