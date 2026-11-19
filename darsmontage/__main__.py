"""darsmontage: from a lesson recording and the book PDF to a ready timeline.

    python -m darsmontage --audio lesson.mp3 --pdf book.pdf --turath 197 --vol 3 --out out_dir

For the graphical interface run:  python -m darsmontage.ui
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .align import align, build_plan
from .asr import audio_duration, transcribe
from .book import build_book, load_turath_pages, render_view
from .export import Timeline, camera_placeholder, render_preview
from .sync import camera_track, place_all, probe_video, save
from .voice import VoiceModel, load_audio

VIDEO_EXT = {".mp4", ".mov", ".mts", ".m2ts", ".mxf", ".avi", ".mkv"}
STAGES = ["تفريغ الصوت", "قراءة الكتاب", "مطابقة التسجيل مع الكتاب", "مزامنة الكاميرات",
          "تجهيز صور الصفحات", "كتابة الخط الزمني", "تصيير المعاينة"]


def mmss(t: float) -> str:
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:04.1f}"


def video_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT)
        elif p.suffix.lower() in VIDEO_EXT:
            out.append(p)
    return out


def resolve_friendly(video: Path, out: Path) -> Path:
    """A copy of the video that DaVinci Resolve can open, when the original is not one.

    Resolve does not read MKV/WebM files or Opus/Vorbis sound (typical of videos
    downloaded from the web). The picture is copied untouched into a MOV file and
    only the sound is rewritten, so this is quick and loses no quality.
    """
    import subprocess

    codecs = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True).stdout.split()
    if video.suffix.lower() in {".mp4", ".mov", ".mxf"} and not {"opus", "vorbis"} & set(codecs):
        return video
    # A short plain name: Resolve failed to find the copy under a long Arabic file name.
    target = out / "lesson_video.mov"
    if not target.exists():
        print("      converting the video to a file Resolve can open…", file=sys.stderr, flush=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", "0:v:0", "-map", "0:a:0",
                        "-c:v", "copy", "-c:a", "pcm_s16le", "-ar", "48000", str(target)], check=True)
    return target


def main() -> None:
    ap = argparse.ArgumentParser(prog="darsmontage", description=__doc__.splitlines()[0])
    ap.add_argument("--audio", required=True, type=Path, help="master audio recording of the lesson")
    ap.add_argument("--pdf", required=True, type=Path, help="the book PDF shown on screen")
    ap.add_argument("--out", required=True, type=Path, help="output folder for this lesson")
    ap.add_argument("--turath", type=int, required=True, help="turath.io book id (text source)")
    ap.add_argument("--vol", default="1", help="volume of the book this PDF contains")
    ap.add_argument("--pages", help="printed pages to search, e.g. 1-120 (default: whole volume)")
    ap.add_argument("--page-offset", type=int, default=0,
                    help="PDF page number minus printed page number")
    ap.add_argument("--asr", choices=["whisper", "elevenlabs"], default="whisper")
    ap.add_argument("--window", type=float, default=0.55,
                    help="share of the page height visible at once (0.55 = about half a page)")
    ap.add_argument("--background", type=Path,
                    help="picture behind the page (your design with the book title and logo)")
    ap.add_argument("--overlay-size", type=float, default=1.0,
                    help="height of the page card as a share of the frame height")
    ap.add_argument("--overlay-pos", choices=["left", "center", "right"], default="center",
                    help="where the page card sits on the background")
    ap.add_argument("--cam-main", type=Path, nargs="*", default=[],
                    help="main camera clips of this lesson (files or a folder)")
    ap.add_argument("--cam-backup", type=Path, nargs="*", default=[],
                    help="backup camera clips, used only where the main camera is missing")
    ap.add_argument("--fps", type=float, default=25.0,
                    help="timeline frame rate when no camera clips are given")
    ap.add_argument("--cache", type=Path, help="where downloaded book texts are kept")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--json-progress", action="store_true",
                    help="print machine-readable progress lines on stdout (used by the interface)")
    args = ap.parse_args()

    def stage(n: int, p: float | None = None) -> None:
        if p is None:
            print(f"[{n + 1}/{len(STAGES)}] {STAGES[n]}…", file=sys.stderr, flush=True)
        if args.json_progress:
            print(json.dumps({"stage": n, "p": p}), flush=True)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    first = last = None
    if args.pages:
        first, last = (int(x) for x in args.pages.split("-"))

    stage(0)
    asr = transcribe(args.audio, out / f"asr_{args.asr}.json", backend=args.asr,
                     progress=lambda f: stage(0, f))
    words = asr["words"]
    duration = audio_duration(args.audio)

    stage(1)
    text = load_turath_pages(args.turath, args.vol, args.cache or out.parent / "_books")
    book = build_book(args.pdf, text, args.page_offset, first, last, args.window)

    stage(2)
    result = align(words, book, voice=VoiceModel(load_audio(args.audio)))
    plan = build_plan(words, book, result, duration)
    plan["voice_accuracy"] = result["voice_accuracy"]
    if result["voice_accuracy"] is not None:
        print(f"      voice check: tells reader from commentator in "
              f"{result['voice_accuracy']:.0%} of test windows", file=sys.stderr)

    cameras: list[dict] = []
    sync_lines: list[str] = []
    fps = args.fps
    main_clips, backup_clips = video_files(args.cam_main), video_files(args.cam_backup)
    plan["cameras"] = []
    if main_clips or backup_clips:
        stage(3)
        placed = place_all(args.audio, main_clips + backup_clips)
        save(placed, out / "sync.json")
        names = {str(c) for c in main_clips}
        main = [p for p in placed if p.path in names]
        backup = [p for p in placed if p.path not in names]
        cameras = camera_track(main, backup, duration)
        if main or backup:
            fps = (main or backup)[0].fps
        for p in placed:
            note = "" if p.ok else "   NOT SYNCED: place by hand"
            if p.ok and abs(p.drift_ms) > 1000 / p.fps:
                note = f"   drifts {p.drift_ms:+.0f} ms over the clip: check its end"
            sync_lines.append(f"{Path(p.path).name}: starts at {mmss(max(0, p.offset))}"
                              f" (confidence {p.confidence:.2f}){note}")
            plan["cameras"].append({
                "name": Path(p.path).name, "camera": "main" if p.path in names else "backup",
                "offset": p.offset, "duration": p.duration, "ok": p.ok,
                "confidence": p.confidence, "drift_ms": p.drift_ms})

    # A finished video given as the recording: it is the picture as well as the sound.
    ready_video = args.audio.suffix.lower() in VIDEO_EXT and not cameras
    timeline_audio = args.audio
    if ready_video:
        info = probe_video(args.audio)
        fps = info["fps"]
        timeline_audio = resolve_friendly(args.audio, out)
        # Stop two frames short of the file's real end: a clip asking for even one frame
        # more than the file holds is reported by Resolve as "file not found".
        import subprocess
        lengths = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=duration", "-of", "csv=p=0",
             str(timeline_audio)], capture_output=True, text=True).stdout.split()
        real = min([float(x) for x in lengths if x.replace(".", "").isdigit()] + [duration])
        clip_end = real - 2 / fps
        cameras = [{"path": str(timeline_audio), "t0": 0.0, "t1": clip_end, "in": 0.0,
                    "duration": clip_end}]
        # The sound goes into the timeline as its own WAV file. Resolve imports a
        # separate sound file reliably, and refused the sound of the video file itself.
        wav = out / "lesson_audio.wav"
        if not wav.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.audio), "-vn",
                            "-c:a", "pcm_s16le", "-ar", "48000", str(wav)], check=True)
        timeline_audio = wav

    stage(4)
    for n, v in enumerate(plan["views"]):
        v["file"] = str(render_view(book, v["page"], v["stop"], out / "views",
                                    scale=args.overlay_size, pos=args.overlay_pos,
                                    background=args.background))
        if n % 10 == 0:
            stage(4, n / max(1, len(plan["views"])))

    stage(5)
    tl = Timeline(args.audio.stem, fps)
    sound_end = cameras[0]["t1"] if ready_video else duration
    tl.add_audio(timeline_audio, 0.0, sound_end, 0.0, sound_end)
    for c in cameras:
        tl.add_video(1, Path(c["path"]), c["t0"], c["t1"], src_in=c["in"], src_duration=c["duration"],
                     has_audio=c.get("has_audio", False))
    for v in plan["views"]:
        tl.add_video(2, Path(v["file"]), v["t0"], v["t1"], still=True)
    tl.write_fcp7_xml(out / "timeline.xml")
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")

    read = [s for s in plan["segments"] if s["kind"] == "read"]
    lines = [f"reading {sum(s['t1'] - s['t0'] for s in read) / 60:.1f} min of {duration / 60:.1f} min, "
             f"{len(read)} passages, {len(plan['views'])} page views", ""]
    for s in plan["segments"]:
        if s["kind"] == "read":
            how = "" if s["by"] == "text" else "  [by voice only: check]"
            lines.append(f"{mmss(s['t0'])} - {mmss(s['t1'])}  BOOK  p.{s['pages'][0]}-{s['pages'][-1]}"
                         f"   {s['first_words']} … {s['last_words']}{how}")
        else:
            lines.append(f"{mmss(s['t0'])} - {mmss(s['t1'])}  camera")
    if sync_lines:
        lines += ["", "camera clips:"] + sync_lines
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(lines[0], file=sys.stderr)

    if not args.no_preview:
        stage(6)
        render_preview(plan, args.audio, camera_placeholder(out / "views" / "camera_placeholder.png"),
                       out / "preview.mp4")
    print(f"done: {out}", file=sys.stderr)
    if args.json_progress:
        print(json.dumps({"done": True}), flush=True)


if __name__ == "__main__":
    main()
