"""darsmontage: from a lesson recording and the book PDF to a ready timeline.

    python -m darsmontage --audio lesson.mp3 --pdf book.pdf --turath 197 --vol 3 --out out_dir

For the graphical interface run:  python -m darsmontage.ui
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from .align import align, build_plan
from .asr import audio_duration, stamp, transcribe
from .book import build_book, guess_page_offset, load_turath_pages, render_view

VERSION = "0.2"
VIEW_VERSION = 1     # raise when the way page pictures are drawn changes
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
    if not made_from(target, video):
        print("      converting the video to a file Resolve can open…", file=sys.stderr, flush=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", "0:v:0", "-map", "0:a:0",
                        "-c:v", "copy", "-c:a", "pcm_s16le", "-ar", "48000", str(target)], check=True)
        mark_made_from(target, video)
    return target


def made_from(target: Path, source: Path) -> bool:
    """True when `target` exists and was produced from this very `source` file."""
    note = target.with_name(target.name + ".source.json")
    try:
        return target.exists() and json.loads(note.read_text(encoding="utf-8")) == stamp(source)
    except (OSError, ValueError):
        return False


def mark_made_from(target: Path, source: Path) -> None:
    target.with_name(target.name + ".source.json").write_text(
        json.dumps(stamp(source), ensure_ascii=False), encoding="utf-8")


def checks(plan: dict, args, text: dict[int, str], repaired: int) -> list[str]:
    """Things about this result a person should look at, in plain words."""
    notes: list[str] = []
    read = [s for s in plan["segments"] if s["kind"] == "read"]
    share = sum(s["t1"] - s["t0"] for s in read) / max(1.0, plan["duration"])
    if share < 0.03:
        notes.append("لم يُعثر تقريبًا على قراءة تطابق الكتاب. تحقق من أن الكتاب والمجلد المختارين هما المقروءان "
                     "في هذا الدرس، ومن أن الصفحات المقروءة موجودة في ملف الـPDF.")
    guess, margin = guess_page_offset(args.pdf, text)
    if margin > 0.15 and guess != args.page_offset:
        notes.append(f"فرق ترقيم الصفحات المُدخل ({args.page_offset}) يخالف المكتشَف من الملف ({guess}). "
                     "إن صحّ المكتشَف فالصفحات المعروضة الآن ليست الصفحات المقروءة.")
    elif margin <= 0.15:
        notes.append("تعذّر التأكد آليًّا من فرق ترقيم الصفحات. راجع المعاينة في قسم الكتاب، "
                     "فقد تكون طبعة الملف غير طبعة النص.")
    by_text = [s for s in read if s["by"] == "text"]
    for a, b in zip(by_text, by_text[1:]):
        if b["pages"][0] < a["pages"][-1] - 2:
            notes.append(f"عند {mmss(b['t0'])} رجعت القراءة من صفحة {a['pages'][-1]} إلى صفحة {b['pages'][0]}. راجع هذا الموضع.")
        elif b["pages"][0] > a["pages"][-1] + 10:
            notes.append(f"عند {mmss(b['t0'])} قفزت القراءة من صفحة {a['pages'][-1]} إلى صفحة {b['pages'][0]}. راجع هذا الموضع.")
    accuracy = plan.get("voice_accuracy")
    if read and accuracy is None:
        notes.append("لم يتيسّر تعلّم صوتَي القارئ والمعلّق من هذا الدرس، فاعتُمد على مطابقة النص وحدها.")
    elif accuracy is not None and accuracy < 0.8:
        notes.append(f"تمييز صوت القارئ من المعلّق ضعيف في هذا الدرس ({accuracy:.0%}). راجع حدود المقاطع بعناية أكبر.")
    voiced = sum(1 for s in read if s["by"] == "voice")
    if voiced:
        notes.append(f"{voiced} مقطع حُسب قراءةً بالصوت وحده دون مطابقة نص. هي معلَّمة في القائمة.")
    lost = [c["name"] for c in plan.get("cameras", []) if not c["ok"]]
    if lost:
        notes.append("مقاطع كاميرا لم تُزامَن وتحتاج وضعًا يدويًّا: " + "، ".join(lost))
    if repaired:
        notes.append(f"أُعيد تفريغ {repaired} موضع أسقطه التفريغ الأول.")
    return notes


def explain(exc: Exception) -> str:
    """A failure in words the person running the tool can act on."""
    import subprocess

    name = type(exc).__name__
    if isinstance(exc, FileNotFoundError) and getattr(exc, "filename", None) in ("ffmpeg", "ffprobe"):
        return "برنامج ffmpeg غير مثبّت أو غير موجود في مسار النظام."
    if isinstance(exc, FileNotFoundError):
        return f"ملف غير موجود: {exc.filename}"
    if isinstance(exc, subprocess.CalledProcessError):
        return ("تعذّر على ffmpeg قراءة أحد الملفات أو كتابته. تأكد من سلامة ملف التسجيل، "
                "ومن أن ملفات المخرجات غير مفتوحة في برنامج آخر.")
    if isinstance(exc, PermissionError):
        return f"لا يمكن الكتابة في: {exc.filename}. أغلق الملف إن كان مفتوحًا في برنامج آخر."
    if "requests" in type(exc).__module__:
        return "تعذّر الاتصال بالإنترنت لتنزيل نص الكتاب أو للتفريغ السحابي."
    if isinstance(exc, LessonError):
        return str(exc)
    return f"خطأ غير متوقع ({name}): {exc}"


class LessonError(Exception):
    """A problem with the lesson's inputs, already worded for the user."""


def main() -> None:
    json_mode = "--json-progress" in sys.argv
    try:
        run()
    except Exception as exc:  # noqa: BLE001 - every failure is reported in plain words
        import traceback
        traceback.print_exc()
        message = explain(exc)
        print(f"\nERROR: {message}", file=sys.stderr, flush=True)
        if json_mode:
            print(json.dumps({"error": message}, ensure_ascii=False), flush=True)
        sys.exit(1)


def run() -> None:
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
    if not words:
        raise LessonError("لم يُستخرج أي كلام من التسجيل. تأكد من أن الملف فيه صوت مسموع.")

    stage(1)
    text = load_turath_pages(args.turath, args.vol, args.cache or out.parent / "_books")
    if not text:
        raise LessonError(f"المجلد «{args.vol}» لا نص له في هذا الكتاب.")
    try:
        book = build_book(args.pdf, text, args.page_offset, first, last, args.window)
    except (ValueError, IndexError):
        book = None
    if book is None or not book.words:
        raise LessonError("لا توجد صفحات مشتركة بين ملف الـPDF ونص الكتاب بفرق الترقيم المُدخل. "
                          "استعمل «اكتشاف تلقائي» في قسم الكتاب.")

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
        if not made_from(wav, args.audio):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.audio), "-vn",
                            "-c:a", "pcm_s16le", "-ar", "48000", str(wav)], check=True)
            mark_made_from(wav, args.audio)
        timeline_audio = wav

    stage(4)
    # Page pictures live in a folder named after everything that shapes them, so a
    # changed PDF, page numbering, size or background can never reuse old pictures.
    look = json.dumps([stamp(args.pdf), args.page_offset, args.window, args.overlay_size,
                       args.overlay_pos, stamp(args.background) if args.background else None,
                       VIEW_VERSION], sort_keys=True)
    views_dir = out / "views" / hashlib.sha1(look.encode()).hexdigest()[:10]
    if (out / "views").exists():
        for old in (out / "views").iterdir():
            if old != views_dir:
                shutil.rmtree(old) if old.is_dir() else old.unlink()
    for n, v in enumerate(plan["views"]):
        v["file"] = str(render_view(book, v["page"], v["stop"], views_dir,
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
    plan["warnings"] = checks(plan, args, text, asr.get("repaired_gaps", 0))
    # What produced this result, so it can be explained or reproduced later.
    plan["run"] = {
        "tool_version": VERSION,
        "inputs": {"audio": stamp(args.audio), "pdf": stamp(args.pdf)},
        "settings": {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v)
                     for k, v in vars(args).items() if k not in ("json_progress", "no_preview")},
        "transcript": asr.get("backend"),
    }
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
    if plan["warnings"]:
        lines += ["", "notes:"] + ["- " + w for w in plan["warnings"]]
        for w in plan["warnings"]:
            print("      note: " + w, file=sys.stderr)
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(lines[0], file=sys.stderr)

    if not args.no_preview:
        stage(6)
        render_preview(plan, args.audio, camera_placeholder(out / "camera_placeholder.png"),
                       out / "preview.mp4")
    print(f"done: {out}", file=sys.stderr)
    if args.json_progress:
        print(json.dumps({"done": True}), flush=True)


if __name__ == "__main__":
    main()
