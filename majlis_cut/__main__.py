"""majlis_cut: from a lesson recording and the book PDF to a ready timeline.

    python -m majlis_cut --audio lesson.mp3 --pdf book.pdf --turath 197 --vol 3 --out out_dir

For the graphical interface run:  python -m majlis_cut.ui
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from . import __version__ as VERSION
from .align import align, apply_edits, build_plan, fill_gaps
from .asr import audio_duration, stamp, transcribe
from .book import build_book, guess_page_offset, load_turath_pages, read_pdf_text, render_view

VIEW_VERSION = 2     # raise when the way page pictures are drawn changes (2: transparent without a background)
from .export import Timeline, camera_placeholder, render_preview
from .highlight import DEFAULT_COLOR, STYLES, clip_scale, make_clips, parse_color
from .sync import camera_track, place_all, probe_video, save
from .voice import VoiceModel, load_audio

VIDEO_EXT = {".mp4", ".mov", ".mts", ".m2ts", ".mxf", ".avi", ".mkv"}
STAGES = ["تفريغ الصوت", "قراءة الكتاب", "مطابقة التسجيل مع الكتاب", "مزامنة الكاميرات",
          "تجهيز صور الصفحات", "تحديد الكلمات المقروءة", "كتابة الخط الزمني", "تصيير المعاينة"]


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


def tidy_old_layout(out: Path, res: Path, work: Path) -> None:
    """Move what an earlier version left at the top of the lesson folder into the new layout."""
    for f in out.glob("asr_*"):
        f.replace(work / f.name)
    for name in ("lesson_video.mov", "lesson_audio.wav"):
        for f in (out / name, out / (name + ".source.json")):
            if f.exists():
                f.replace(res / f.name)
    for name in ("timeline.xml", "plan.json", "report.txt", "preview.mp4", "sync.json",
                 "camera_placeholder.png"):
        (out / name).unlink(missing_ok=True)
    if (out / "views").is_dir():
        shutil.rmtree(out / "views")


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


def checks(plan: dict, args, text: dict[int, str], repaired: int) -> list[dict]:
    """Things about this result a person should look at, in plain words.
    Each note may carry the time in the lesson it is about."""
    found: list[dict] = []

    class notes:                      # noqa: N801 - keeps the calls below short
        @staticmethod
        def append(text: str, t: float | None = None) -> None:
            found.append({"text": text, "t": t})
    read = [s for s in plan["segments"] if s["kind"] == "read"]
    share = sum(s["t1"] - s["t0"] for s in read) / max(1.0, plan["duration"])
    if share < 0.03:
        notes.append("لم يُعثر تقريبًا على قراءة تطابق الكتاب. تحقق من أن الكتاب والمجلد المختارين هما المقروءان "
                     "في هذا الدرس، ومن أن الصفحات المقروءة موجودة في ملف الـPDF.")
    guess, margin = guess_page_offset(args.pdf, text) if args.turath else (args.page_offset, 1.0)
    if margin > 0.15 and guess != args.page_offset:
        notes.append(f"فرق ترقيم الصفحات المُدخل ({args.page_offset}) يخالف المكتشَف من الملف ({guess}). "
                     "إن صحّ المكتشَف فالصفحات المعروضة الآن ليست الصفحات المقروءة.")
    elif margin <= 0.15:
        notes.append("تعذّر التأكد آليًّا من فرق ترقيم الصفحات. راجع المعاينة في قسم الكتاب، "
                     "فقد تكون طبعة الملف غير طبعة النص.")
    by_text = [s for s in read if s["by"] == "text"]
    for a, b in zip(by_text, by_text[1:]):
        if b["pages"][0] < a["pages"][-1] - 2:
            notes.append(f"عند {mmss(b['t0'])} رجعت القراءة من صفحة {a['pages'][-1]} إلى صفحة {b['pages'][0]}. راجع هذا الموضع.", b["t0"])
        elif b["pages"][0] > a["pages"][-1] + 10:
            notes.append(f"عند {mmss(b['t0'])} قفزت القراءة من صفحة {a['pages'][-1]} إلى صفحة {b['pages'][0]}. راجع هذا الموضع.", b["t0"])
    accuracy = plan.get("voice_accuracy")
    if read and accuracy is None:
        notes.append("لم تُقَس دقة تمييز صوتَي القارئ والمعلّق في هذا الدرس (مقاطعه قليلة أو قصيرة). "
                     "راجع حدود المقاطع بعناية أكبر.")
    elif accuracy is not None and accuracy < 0.8:
        notes.append(f"تمييز صوت القارئ من المعلّق ضعيف في هذا الدرس ({accuracy:.0%}). راجع حدود المقاطع بعناية أكبر.")
    for s in read:
        if s["by"] == "voice":
            notes.append(f"المقطع عند {mmss(s['t0'])} حُسب قراءةً بالصوت وحده دون مطابقة نص. تأكد منه.", s["t0"])
    lost = [c["name"] for c in plan.get("cameras", []) if not c["ok"]]
    if lost:
        notes.append("مقاطع كاميرا لم تُزامَن وتحتاج وضعًا يدويًّا: " + "، ".join(lost))
    if repaired:
        notes.append(f"أُعيد تفريغ {repaired} موضع أسقطه التفريغ الأول.")
    return found


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


def highlight_views(plan: dict, args, book, words: list[dict], pos, books: Path, views_dir: Path,
                    res: Path, fps: float, color: tuple[int, int, int], progress) -> list[str]:
    """Highlight clips (V3) and ink pictures (V4) for every page view.

    The words' places come from reading the shown pages with the Windows reader;
    without it only the line being read is highlighted. Returns notes for the report."""
    from .book import ocr_cache
    from .ocr import available, ocr_pages

    notes: list[str] = []
    pages = sorted({v["page"] for v in plan["views"]})
    ocr: dict[int, list] = {}
    reason = available()
    if reason:
        notes.append(f"تحديد الكلمة يحتاج قارئ الصفحات في ويندوز، فحُدِّد السطر وحده. ({reason})")
    elif pages:
        ocr = ocr_pages(args.pdf, pages, ocr_cache(args.pdf, books), lambda f: progress(0.3 * f), boxes=True)
    for v in plan["views"]:
        v["ink"] = str(render_view(book, v["page"], v["stop"], views_dir, scale=args.overlay_size,
                                   pos=args.overlay_pos, ink_only=True))
    # Named after the colour and style too, so changing them keeps nothing stale.
    out_dir = res / "highlight" / f"{views_dir.name}_{args.highlight_style}_{'%02x%02x%02x' % color}"
    if (res / "highlight").exists():
        for old in (res / "highlight").iterdir():
            if old != out_dir:
                shutil.rmtree(old) if old.is_dir() else old.unlink()
    found = make_clips(plan["views"], words, pos, book, ocr, color, fps, out_dir,
                       scale=args.overlay_size, place=args.overlay_pos, style=args.highlight_style,
                       progress=lambda f: progress(0.3 + 0.65 * f))
    line_only = [p for p in found["line_only"] if ocr]
    if line_only:
        shown = "، ".join(str(p) for p in line_only[:12]) + ("…" if len(line_only) > 12 else "")
        notes.append(f"في الصفحات {shown} لم تُطابَق كلمات الكتاب بصورة الصفحة، فحُدِّد السطر وحده. "
                     "قد تكون طبعة الملف غير طبعة النص، أو الصفحة غير واضحة.")
    plan["highlight"] = {"style": args.highlight_style, "color": "#%02x%02x%02x" % color,
                         "mb": round(found["bytes"] / 1e6, 1),
                         "line_only": found["line_only"] if ocr else "all"}
    return notes


def preview_highlights(views: list[dict], words, pos, book, args, color, books: Path,
                       height: int = 720) -> list[dict]:
    """The views' highlights drawn at the preview's size."""
    from .book import ocr_cache, view_frame
    from .highlight import highlight_for, page_boxes, view_events
    from .ocr import available, ocr_pages

    pages = sorted({v["page"] for v in views if v.get("highlight")})
    ocr = {} if available() or not pages else ocr_pages(args.pdf, pages, ocr_cache(args.pdf, books), boxes=True)
    boxes = {p: page_boxes(book, p, ocr.get(p, [])) for p in pages}
    out = []
    for v in views:
        if not v.get("highlight"):
            continue
        frame = view_frame(book, v["page"], v["stop"], scale=args.overlay_size, pos=args.overlay_pos)
        events = view_events(v, words, pos, book, boxes[v["page"]], frame)
        out.append({"t0": v["t0"], "t1": v["t1"], "ink": v["ink"],
                    "hl": highlight_for(args.highlight_style, events, v["t1"] - v["t0"], color,
                                        scale=height / 1080)})
    return out


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
    ap = argparse.ArgumentParser(prog="majlis-cut", description=__doc__.splitlines()[0])
    ap.add_argument("--audio", required=True, type=Path, help="master audio recording of the lesson")
    ap.add_argument("--pdf", required=True, type=Path, help="the book PDF shown on screen")
    ap.add_argument("--out", required=True, type=Path, help="output folder for this lesson")
    ap.add_argument("--turath", type=int,
                    help="turath.io book id (text source). Without it the text is read off the PDF")
    ap.add_argument("--vol", default="1", help="volume of the book this PDF contains")
    ap.add_argument("--pages", help="printed pages to search, e.g. 1-120 (default: whole volume)")
    ap.add_argument("--page-offset", type=int, default=0,
                    help="PDF page number minus printed page number")
    ap.add_argument("--asr", choices=["whisper", "elevenlabs"], default="whisper")
    ap.add_argument("--window", type=float, default=0.55,
                    help="share of the page height visible at once (0.55 = about half a page)")
    ap.add_argument("--background", type=Path,
                    help="picture behind the page (your design with the book title and logo)")
    ap.add_argument("--no-paper", action="store_true",
                    help="remove the page's paper: only the ink, on a transparent ground")
    ap.add_argument("--overlay-size", type=float, default=1.0,
                    help="height of the page card as a share of the frame height")
    ap.add_argument("--overlay-pos", choices=["left", "center", "right"], default="center",
                    help="where the page card sits on the background")
    ap.add_argument("--highlight", action="store_true",
                    help="highlight the word being read and its line (a clip on V3, the ink on V4)")
    ap.add_argument("--highlight-color", default=DEFAULT_COLOR,
                    help="colour of the highlight, e.g. #4096ff")
    ap.add_argument("--highlight-style", choices=STYLES, default="ribbon",
                    help="ribbon (colour under the words), glass (a pane of liquid glass) "
                         "or glass-tail (glass fading out behind the word)")
    ap.add_argument("--cam-main", type=Path, nargs="*", default=[],
                    help="main camera clips of this lesson (files or a folder)")
    ap.add_argument("--cam-backup", type=Path, nargs="*", default=[],
                    help="backup camera clips, used only where the main camera is missing")
    ap.add_argument("--fill-with-book", action="store_true",
                    help="where no camera clip and no page would be on screen, keep the last page shown")
    ap.add_argument("--fps", type=float, default=25.0,
                    help="timeline frame rate when no camera clips are given")
    ap.add_argument("--cache", type=Path, help="where downloaded book texts are kept")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--json-progress", action="store_true",
                    help="print machine-readable progress lines on stdout (used by the interface)")
    args = ap.parse_args()
    highlight_color = parse_color(args.highlight_color) if args.highlight else None

    def stage(n: int, p: float | None = None) -> None:
        if p is None:
            print(f"[{n + 1}/{len(STAGES)}] {STAGES[n]}…", file=sys.stderr, flush=True)
        if args.json_progress:
            print(json.dumps({"stage": n, "p": p}), flush=True)

    out: Path = args.out
    res, rev, work = out / "resolve", out / "review", out / "work"
    for d in (res, rev, work):
        d.mkdir(parents=True, exist_ok=True)
    tidy_old_layout(out, res, work)
    first = last = None
    if args.pages:
        first, last = (int(x) for x in args.pages.split("-"))

    stage(0)
    asr = transcribe(args.audio, work / f"asr_{args.asr}.json", backend=args.asr,
                     progress=lambda f: stage(0, f))
    words = asr["words"]
    duration = audio_duration(args.audio)
    if not words:
        raise LessonError("لم يُستخرج أي كلام من التسجيل. تأكد من أن الملف فيه صوت مسموع.")

    stage(1)
    books = args.cache or out.parent / "_books"
    if args.turath:
        text = load_turath_pages(args.turath, args.vol, books)
        if not text:
            raise LessonError(f"المجلد «{args.vol}» لا نص له في هذا الكتاب.")
    else:
        from .ocr import available
        if available():
            raise LessonError(available())
        args.page_offset = 0          # pages are counted as they come in the PDF
        text = read_pdf_text(args.pdf, books, first, last, progress=lambda f: stage(1, f))
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
    edits_file = work / "edits.json"
    edits = json.loads(edits_file.read_text(encoding="utf-8")) if edits_file.exists() else []
    apply_edits(plan, edits, words)
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
        save(placed, work / "sync.json")
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
        timeline_audio = resolve_friendly(args.audio, res)
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
        wav = res / "lesson_audio.wav"
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
                       args.no_paper, VIEW_VERSION], sort_keys=True)
    views_dir = res / "pages" / hashlib.sha1(look.encode()).hexdigest()[:10]
    if (res / "pages").exists():
        for old in (res / "pages").iterdir():
            if old != views_dir:
                shutil.rmtree(old) if old.is_dir() else old.unlink()
    for n, v in enumerate(plan["views"]):
        v["file"] = str(render_view(book, v["page"], v["stop"], views_dir,
                                    scale=args.overlay_size, pos=args.overlay_pos,
                                    background=args.background, paper=not args.no_paper))
        if n % 10 == 0:
            stage(4, n / max(1, len(plan["views"])))

    hl_notes: list[str] = []
    if highlight_color:
        stage(5)
        hl_notes = highlight_views(plan, args, book, words, result["pos"], books, views_dir, res, fps,
                                   highlight_color, lambda f: stage(5, f))
    elif (res / "highlight").exists():
        shutil.rmtree(res / "highlight")             # turned off since the last run

    stage(6)
    sound_end = cameras[0]["t1"] if ready_video else duration
    # What goes on the page track: the reading, and with --fill-with-book every empty place too.
    shown = plan["views"]
    if args.fill_with_book:
        shown = fill_gaps(plan["views"], cameras, sound_end,
                          keep_empty=[e for e in edits if e["to"] == "talk"])
        plan["filled"] = sum(v["t1"] - v["t0"] for v in shown) - sum(v["t1"] - v["t0"] for v in plan["views"])
    tl = Timeline(args.audio.stem, fps)
    tl.add_audio(timeline_audio, 0.0, sound_end, 0.0, sound_end)
    for c in cameras:
        tl.add_video(1, Path(c["path"]), c["t0"], c["t1"], src_in=c["in"], src_duration=c["duration"],
                     has_audio=c.get("has_audio", False))
    for v in shown:
        tl.add_video(2, Path(v["file"]), v["t0"], v["t1"], still=True)
    # The ribbon goes between the page and its ink, so letters stay on top of it. The
    # glass carries the bent letters itself and goes over the page, with no ink track.
    style = args.highlight_style
    cs = clip_scale(style)
    hl_size = None if cs == 1.0 else (round(1920 * cs), round(1080 * cs))
    for v in plan["views"]:
        if v.get("highlight"):
            tl.add_video(3, Path(v["highlight"]), v["t0"], v["t1"], src_duration=v["highlight_s"],
                         size=hl_size, alpha=True)
            if style == "ribbon":
                tl.add_video(4, Path(v["ink"]), v["t0"], v["t1"], still=True)
    tl.write_fcp7_xml(res / "timeline.xml")
    plan["warnings"] = checks(plan, args, text, asr.get("repaired_gaps", 0))
    plan["warnings"] += [{"text": note, "t": None} for note in hl_notes]
    # What produced this result, so it can be explained or reproduced later.
    plan["run"] = {
        "tool_version": VERSION,
        "inputs": {"audio": stamp(args.audio), "pdf": stamp(args.pdf)},
        "settings": {k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v)
                     for k, v in vars(args).items() if k not in ("json_progress", "no_preview")},
        "transcript": asr.get("backend"),
    }
    (work / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")

    read = [s for s in plan["segments"] if s["kind"] == "read"]
    lines = [f"reading {sum(s['t1'] - s['t0'] for s in read) / 60:.1f} min of {duration / 60:.1f} min, "
             f"{len(read)} passages, {len(plan['views'])} page views", ""]
    if plan.get("highlight"):
        hl = plan["highlight"]
        lines[1:1] = [f"highlight ({hl['style']}, {hl['color']}): {hl['mb']} MB of clips on V3"
                      + (", the ink on V4" if hl["style"] == "ribbon" else "")]
    if args.fill_with_book:
        lines[1:1] = [f"the page also fills {plan['filled'] / 60:.1f} min where nothing else is on screen"]
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
        lines += ["", "notes:"] + ["- " + w["text"] for w in plan["warnings"]]
        for w in plan["warnings"]:
            print("      note: " + w["text"], file=sys.stderr)
    (rev / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    print(lines[0], file=sys.stderr)

    if not args.no_preview:
        stage(7)
        highlights = None
        if highlight_color:
            highlights = preview_highlights(plan["views"], words, result["pos"], book, args, highlight_color, books)
        render_preview(shown, plan["duration"], args.audio,
                       camera_placeholder(work / "camera_placeholder.png"), rev / "preview.mp4",
                       highlights=highlights)
    print(f"done: {out}", file=sys.stderr)
    if args.json_progress:
        print(json.dumps({"done": True}), flush=True)


if __name__ == "__main__":
    main()
