This is is the initial planning/architecture file for the project

Problem:
Guitar audio-to-tab tools generally treat the fretboard like a piano keyboard: given a
pitch, they pick a valid (string, fret) position and move on. But on a guitar, most
pitches can be played in 2-6 different places. Picking positions independently, note by
note, routinely produces tab that is technically "correct" (it plays the right notes) but
physically unreasonable — big jumps up and down the neck for a phrase a real guitarist
would play in one hand position.

Useability: 
A user can put in an audio file, either purely guitar or a light mix
where the guitar will then be extracted from the file into a midi file. From there
the notes will be put into a readable guitar tab that is optimized for playability.

Minimizing fret-position jumps between consecutive notes
Avoiding unnecessary string changes when a same-string option exists nearby
Keeping runs within a compact hand span (~4 frets) where possible
Preferring position clusters (e.g., playing a run across 2-3 strings near fret 5-8)
over sliding the same string up and down

----------------------------------------------------------------------------------------
Cost function (initial design, to be tuned against real tabs)

For a transition from position (str_a, fret_a) to (str_b, fret_a)... (str_b, fret_b):


fret_jump_cost = w1 * |fret_b - fret_a| (ignore jumps for open strings, fret 0 is "free")
string_change_cost = w2 * |str_b - str_a| (moving to an adjacent string is cheap;
skipping strings is more expensive since it changes the whole hand angle)
span_violation_cost = w3 * max(0, span_of_recent_window - MAX_HAND_SPAN) — a soft
penalty that grows if maintaining a compact hand-span window across a short lookback
becomes impossible
open_string_bonus = small negative cost (discount) for using an open string, since
it requires no fretting hand movement at all


Initial weights (w1=1.0, w2=0.5, w3=2.0) are a starting guess — Stage 6 tunes these
against real, hand-verified tabs.

-----------------------------------------------------------------------------------------
Non-goals (for now)
Polyphony / chords are out of scope for v1 — this assumes a monophonic note sequence
(single note at a time), which covers solo lines, melodies, and single-note runs.
Chord voicing is a natural but separate extension.
Bends, slides, hammer-ons/pull-offs as notated techniques are not detected — only
the underlying pitch sequence and position choice.


-----------------------------------------------------------------------------------------
Build stages
Note extraction (pitch/onset detection → note list) — librosa.pyin
Candidate generation (pitch → all valid (string, fret) positions, standard tuning)
Cost function + DP/Viterbi solver over the candidate graph
Tab rendering (position sequence → ASCII tab)
Source separation pre-processing (Demucs) for full-mix input — future work
Validation against real tabs, cost-function tuning — ongoing