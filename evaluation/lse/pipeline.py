"""Run pinned upstream face preprocessing with lossless intermediate audio.

The official first AVI conversion omits an audio codec. FFmpeg 6 defaults to
MP3, which pads a 1.091375-second test WAV to 1.188s and duplicates two video
frames. Explicit PCM preserves all 17462 samples and the original 28 frames.
The source checkout, face tracking, crop geometry and video codec stay intact.
"""

import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch


def pcm_command(command):
    if (
        isinstance(command, list)
        and command
        and Path(command[0]).name == "ffmpeg"
        and Path(command[-1]).name == "video.avi"
        and "-qscale:v" in command
        and "-async" in command
    ):
        return [*command[:-1], "-c:a", "pcm_s16le", command[-1]]
    return command


def main():
    source = Path(sys.argv[1]).resolve()
    sys.argv = [str(source), *sys.argv[2:]]
    sys.path.insert(0, str(source.parent))
    original_run = subprocess.run

    def run(command, *args, **kwargs):
        return original_run(pcm_command(command), *args, **kwargs)

    # Scoped to this child process; never modify installed upstream source files.
    with patch.object(subprocess, "run", run):
        runpy.run_path(str(source), run_name="__main__")


if __name__ == "__main__":
    main()
