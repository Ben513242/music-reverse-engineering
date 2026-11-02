"""``mrep`` command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mrep import __version__
from mrep.config import PROFILES, load_config
from mrep.domain.project import Project
from mrep.logging import configure_logging, get_logger
from mrep.manifest import PROJECT_COMPLETE, PROJECT_PARTIAL, Manifest

STAGE_TARGETS = {
    "run": None,
    "analyze": ["global_analysis"],
    "separate": ["separation", "classification"],
    "transcribe": ["transcription", "fusion"],
    "score": ["scoring"],
    "reconstruct": ["reconstruction"],
}


def _add_common(p: argparse.ArgumentParser, project_based: bool = False) -> None:
    if project_based:
        p.add_argument("project", help="project directory (output/<song>) or an audio file")
    else:
        p.add_argument("audio", help="input audio (.wav / .flac / .mp3)")
    p.add_argument("--profile", default=None, help=f"configuration profile: {', '.join(PROFILES)} or a yaml path")
    p.add_argument("--device", default=None, help="auto | cpu | cuda | mps")
    p.add_argument("--output", "-o", default=None, help="output root directory (default: config output_root)")
    p.add_argument("--config", action="append", default=[], help="extra yaml override file (repeatable)")
    p.add_argument("--set", action="append", default=[], help="override a key: a.b.c=value (repeatable)")
    p.add_argument("--resume", action="store_true", help="skip stages that are already complete")
    p.add_argument("--force-stage", action="append", default=[], help="re-run this stage and everything downstream (repeatable)")
    p.add_argument("--log-level", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mrep", description="MREP - Music Reverse Engineering Pipeline")
    parser.add_argument("--version", action="version", version=f"mrep {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd, help_text in (
        ("run", "run the full pipeline"),
        ("analyze", "ingest + normalize + global analysis (tempo, key, structure)"),
        ("separate", "... + source separation and classification"),
    ):
        _add_common(sub.add_parser(cmd, help=help_text))
    for cmd, help_text in (
        ("transcribe", "transcription + fusion on an existing project (or audio file)"),
        ("score", "MIDI / MusicXML on an existing project"),
        ("reconstruct", "render the reconstruction of an existing project"),
    ):
        _add_common(sub.add_parser(cmd, help=help_text), project_based=True)
    info = sub.add_parser("info", help="print the manifest summary of a project")
    info.add_argument("project")
    info.add_argument("--json", action="store_true")
    synth = sub.add_parser("synth", help="generate a synthetic demo song (+ ground truth) for testing")
    synth.add_argument("output", help="output .wav path")
    synth.add_argument("--bpm", type=float, default=120.0)
    synth.add_argument("--bars", type=int, default=8)
    synth.add_argument("--sr", type=int, default=44100)
    synth.add_argument("--truth", default=None, help="write ground-truth json here (default: <output>.truth.json)")
    sub.add_parser("profiles", help="list configuration profiles")
    ad = sub.add_parser("adapters", help="list registered adapters")
    ad.add_argument("--kind", default=None)
    models = sub.add_parser("models", help="manage model files for adapters (explicit downloads only)")
    msub = models.add_subparsers(dest="models_cmd", required=True)
    dl = msub.add_parser("download", help="download/prepare the models of one adapter")
    dl.add_argument("kind")
    dl.add_argument("name")
    return parser


def _resolve_project(args: argparse.Namespace, cfg: dict) -> tuple[Project, Path | None, Manifest | None]:
    target = Path(getattr(args, "audio", None) or args.project)
    if target.is_dir():
        project = Project.open(target)
        if not project.manifest_path.exists():
            raise SystemExit(f"{target} is not an MREP project (no manifest.json)")
        manifest = Manifest.load(project.manifest_path)
        return project, Path(manifest.source["path"]) if manifest.source.get("path") else None, manifest
    output_root = Path(args.output or cfg.get("output_root", "output"))
    project = Project.for_audio(target, output_root)
    manifest = Manifest.load(project.manifest_path) if project.manifest_path.exists() else None
    return project, target, manifest


def cmd_pipeline(args: argparse.Namespace) -> int:
    from mrep.pipeline.runner import build_runner

    target = Path(getattr(args, "audio", None) or args.project)
    manifest_cfg = None
    if target.is_dir() and (target / "manifest.json").exists() and not (args.profile or args.config or args.set):
        manifest_cfg = Manifest.load(target / "manifest.json").config or None
    cfg = manifest_cfg or load_config(args.profile or "default", [Path(p) for p in args.config], args.set, args.device)
    if args.device:
        cfg["device"] = args.device
    project, source, manifest = _resolve_project(args, cfg)
    project.ensure_dirs()
    level = args.log_level or cfg.get("logging", {}).get("level", "INFO")
    configure_logging(level, project.logs_dir / "mrep.log.jsonl" if cfg.get("logging", {}).get("json_file", True) else None)
    log = get_logger("cli")
    targets = STAGE_TARGETS[args.command]
    resume = args.resume or args.command in ("transcribe", "score", "reconstruct")
    runner = build_runner(project, cfg, source, manifest)
    log.info("mrep %s: %s -> %s (profile=%s, device=%s)", args.command, source or project.root, project.root, cfg.get("profile"), runner.ctx.device)
    man = runner.run(targets=targets, resume=resume, force=args.force_stage)
    print(json.dumps(man.summary(), indent=2))
    print(f"\nProject status: {man.status}\nManifest: {project.manifest_path}")
    if man.status == PROJECT_COMPLETE:
        return 0
    return 2 if man.status == PROJECT_PARTIAL else 1


def cmd_info(args: argparse.Namespace) -> int:
    project = Project.open(Path(args.project))
    man = Manifest.load(project.manifest_path)
    if args.json:
        print(json.dumps(man.to_dict(), indent=2, default=str))
    else:
        print(json.dumps(man.summary(), indent=2))
        for stage, rec in man.stages.items():
            for err in rec.errors:
                print(f"  [{stage}] {err.get('where')}: {err.get('type')}: {err.get('message')}")
            for w in rec.warnings:
                print(f"  [{stage}] warning: {w}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    from mrep.io.audio import write_audio
    from mrep.io.json import write_json
    from mrep.utils.synthetic import demo_song

    audio, truth = demo_song(sr=args.sr, bpm=args.bpm, bars=args.bars)
    out = Path(args.output)
    write_audio(out, audio, args.sr)
    truth_path = Path(args.truth) if args.truth else out.with_suffix(out.suffix + ".truth.json")
    write_json(truth_path, truth)
    print(f"wrote {out} ({truth['duration']:.1f}s) and {truth_path}")
    return 0


def cmd_adapters(args: argparse.Namespace) -> int:
    from mrep.adapters import list_adapters

    for kind, name in list_adapters(args.kind):
        print(f"{kind:<20} {name}")
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from mrep.adapters import create_adapter
    from mrep.adapters.base import AdapterUnavailable

    try:
        adapter = create_adapter(args.kind, args.name, {}, "cpu")
        adapter.ensure_models(download=True)
    except AdapterUnavailable as exc:
        print(f"cannot prepare {args.kind}/{args.name}: {exc}")
        return 1
    print(f"{args.kind}/{args.name}: models ready ({'none required' if not adapter.requires_models else adapter.model})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in STAGE_TARGETS:
        return cmd_pipeline(args)
    if args.command == "info":
        return cmd_info(args)
    if args.command == "synth":
        return cmd_synth(args)
    if args.command == "profiles":
        for p in PROFILES:
            print(p)
        return 0
    if args.command == "adapters":
        return cmd_adapters(args)
    if args.command == "models":
        return cmd_models(args)
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
