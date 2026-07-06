"""
Note extraction.

Takes a mono guitar audio recording and produces a list of discrete notes
(pitch, start time, duration), using librosa's pyin pitch tracker followed
by a post-processing step that collapses a continuous pitch contour into
discrete note events.
"""

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class Note:
    midi_pitch: int
    start_time: float  # seconds
    duration: float  # seconds

    def __repr__(self) -> str:
        return (
            f"Note(midi={self.midi_pitch}, "
            f"start={self.start_time:.3f}s, dur={self.duration:.3f}s)"
        )


def extract_pitch_contour(
    y: np.ndarray,
    sr: int,
    fmin: str = "C2",
    fmax: str = "C6",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run pyin over the audio and return (f0, voiced_flag, times).
    f0 is in Hz, NaN where unvoiced.
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz(fmin),
        fmax=librosa.note_to_hz(fmax),
        sr=sr,
    )
    times = librosa.times_like(f0, sr=sr)
    return f0, voiced_flag, times


def _hz_to_midi_round(hz: float) -> int:
    return int(round(librosa.hz_to_midi(hz)))


def contour_to_notes(
    f0: np.ndarray,
    voiced_flag: np.ndarray,
    times: np.ndarray,
    min_note_duration: float = 0.05,
) -> list[Note]:
    """
    Collapse a per-frame pitch contour into discrete Note events.

    Consecutive voiced frames with the same rounded MIDI pitch are merged
    into one note. A new note starts whenever the rounded pitch changes,
    or after a gap of unvoiced frames. Notes shorter than
    `min_note_duration` are dropped as detection noise.
    """
    notes: list[Note] = []
    current_pitch: int | None = None
    current_start: float | None = None
    frame_dt = times[1] - times[0] if len(times) > 1 else 0.0

    def flush(end_time: float) -> None:
        nonlocal current_pitch, current_start
        if current_pitch is not None and current_start is not None:
            duration = end_time - current_start
            if duration >= min_note_duration:
                notes.append(
                    Note(
                        midi_pitch=current_pitch,
                        start_time=current_start,
                        duration=duration,
                    )
                )
        current_pitch = None
        current_start = None

    for i, t in enumerate(times):
        is_voiced = bool(voiced_flag[i]) and not np.isnan(f0[i])
        if not is_voiced:
            flush(t)
            continue

        pitch = _hz_to_midi_round(f0[i])
        if current_pitch is None:
            current_pitch = pitch
            current_start = t
        elif pitch != current_pitch:
            flush(t)
            current_pitch = pitch
            current_start = t
        # else: same note continues, do nothing

    if len(times) > 0:
        flush(times[-1] + frame_dt)

    return notes


def extract_notes_from_file(
    path: str,
    min_note_duration: float = 0.05,
) -> list[Note]:
    """
    Full Stage 1 pipeline: load a mono audio file and return a Note list.
    """
    y, sr = librosa.load(path, sr=None, mono=True)
    f0, voiced_flag, times = extract_pitch_contour(y, sr)
    return contour_to_notes(f0, voiced_flag, times, min_note_duration)