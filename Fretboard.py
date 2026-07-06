"""
Fretboard model for standard-tuned 6-string guitar.

For given a MIDI pitch, enumerate every playable
(string, fret) position that produces that pitch.
"""

from dataclasses import dataclass

# Standard tuning, string 6 (low E) to string 1 (high e), as MIDI note numbers 
# for the open strings.
STANDARD_TUNING_MIDI = {
    6: 40,  # E2
    5: 45,  # A2
    4: 50,  # D3
    3: 55,  # G3
    2: 59,  # B3
    1: 64,  # E4
}

DEFAULT_MAX_FRET = 22


@dataclass(frozen=True)
class Position:
    """A single playable position for a note: which string, which fret."""

    string: int  # 1 (high e) to 6 (low E)
    fret: int  # 0 = open string

    @property
    def is_open(self) -> bool:
        return self.fret == 0

    def __repr__(self) -> str:
        return f"Position(string={self.string}, fret={self.fret})"


def midi_to_candidates(
    midi_pitch: int,
    max_fret: int = DEFAULT_MAX_FRET,
    tuning: dict[int, int] | None = None,
) -> list[Position]:
    """
    Return every (string, fret) position that plays the given MIDI pitch,
    across all six strings, within [0, max_fret].

    Positions are returned in a stable order: string 6 -> string 1 (low to
    high), which keeps downstream DP behavior deterministic when several
    candidates end up with equal cost.
    """
    candidates: list[Position] = []
    tuning_map = tuning or STANDARD_TUNING_MIDI

    for string in sorted(tuning_map.keys(), reverse=True):
        open_pitch = tuning_map[string]
        fret = midi_pitch - open_pitch
        if 0 <= fret <= max_fret:
            candidates.append(Position(string=string, fret=fret))

    return candidates


def note_name_to_midi(note_name: str) -> int:
    """
    Convert a note name like 'E2', 'A#3', 'Bb4' to a MIDI pitch number.
    Octave numbering follows scientific pitch notation (C4 = 60, middle C).
    """
    note_name = note_name.strip()
    pitch_classes = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4,
        "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
        "A#": 10, "Bb": 10, "B": 11,
    }

    # Split into pitch-class letters (+ optional accidental) and octave digits.
    idx = 1
    if len(note_name) > 1 and note_name[1] in ("#", "b"):
        idx = 2
    pitch_class_str = note_name[:idx]
    octave_str = note_name[idx:]

    if pitch_class_str not in pitch_classes:
        raise ValueError(f"Unrecognized pitch class in note name: {note_name!r}")
    if not octave_str.lstrip("-").isdigit():
        raise ValueError(f"Unrecognized octave in note name: {note_name!r}")

    octave = int(octave_str)
    return 12 * (octave + 1) + pitch_classes[pitch_class_str]