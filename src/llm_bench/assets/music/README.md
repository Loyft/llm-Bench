# Background music

llm-bench plays a looping **playlist** while the benchmarks run (never during setup or after the
run) and syncs the live screen's VU bars, scope and note pattern to whichever song is playing.

The bundled `keygen-NN.mp3` tracks are free to use and redistribute. They play in filename
order and wrap around; rename or add files to change the order (the file name becomes the
title shown on screen, e.g. `keygen-03.mp3` -> "Keygen 03").

The playlist is picked in this order:

1. `--music PATH` (a file, or a folder of tracks)
2. `$LLM_BENCH_MUSIC` (file or folder)
3. the audio files in this folder
4. the audio files in `~/.config/llm-bench/music/`

Supported: `.mp3 .m4a .wav .ogg .flac .aac`. Playback uses `afplay` (macOS), else `ffplay`,
`mpv` or `cvlc`. The spectrum analysis needs `ffmpeg`; without it (or without tracks) the
visuals fall back to a synthetic animation. Use `--no-music` to turn all of it off.
