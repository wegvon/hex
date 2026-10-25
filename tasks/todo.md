# Remote ASR (Qwen3-ASR on Rem) backend for HEX — Plan

Goal: dictate with HEX while transcription runs on Rem's Qwen3-ASR-1.7B instead of a
local GGUF / Apple Speech model. Everything else (shortcut, HUD, Modes, corrections,
OpenCode post-processing, paste, history) stays unchanged.

## Target server (from localai repo, docs/operations/model-services/qwen3-asr-rem-2026-10-15.md)

- Rem `127.0.0.1:7794`, official Qwen Flask streaming demo (vLLM 0.14, qwen-asr 0.0.6), GPU1.
- Reached from the Mac via SSH relay `http://127.0.0.1:17794`; public `https://qasr.turel.se` (no key).
- **Not OpenAI-compatible.** Session protocol:
  1. `POST /api/start` → `{"session_id"}`
  2. `POST /api/chunk?session_id=…` body = raw **16 kHz mono f32 LE PCM**, `application/octet-stream` → `{"language","text"}` (full text so far)
  3. `POST /api/finish?session_id=…` → `{"language","text"}` final, session freed
- Language is forced to Turkish server-side; HEX language setting cannot change it.
- Limits: `max_num_seqs=1` (single user), 8192-token context per session, sessions GC'd after 10 min idle,
  first chunk after a restart is cold (~6 s), warm chunks ~0.15 s.

## Findings in HEX that shape the design

- No engine trait: backends are enum variants. macOS → `transcription::Transcriber`
  (`Gguf | AppleSpeech`); Linux/Windows → concrete `local_transcriber::LocalTranscriber`.
- Engine gets the whole clip on key release as **16 kHz mono f32** — byte-identical to what
  the server wants (no WAV, no resampling).
- `stream_options() == None` ⇒ batch path. HEX already has a LiveStream path for GGUF
  streaming while the key is held — a natural phase-2 fit for `/api/chunk`.
- No HTTP crate: shell out to system `curl`, bodies on stdin never argv (AGENTS.md rule);
  `dictation_processor::run_command` gives deadline + cancel polling.
- Gotchas: `available()`/`validate()` reject non-GGUF; onboarding forces a download unless
  `is_installed()` is true; `load_from` fails on an invalid selection.

## Design (phase 1: batch on release)

- Catalog: `TranscriptionModelId::QwenServer` + `ModelRuntime::Remote`, size label "Server",
  `available() = true`, `is_installed() = endpoint configured`, language shown as server-defined.
- Config: `AppSettings.remote_asr { endpoint (default http://127.0.0.1:17794), timeout_secs }`.
  Kept out of `TranscriptionSelection` so equality/recents/reload logic stays untouched.
- `src/remote_transcriber.rs` (shared): `transcribe(&[f32], &RemoteAsrConfig, cancel) -> Result<String>`
  → start → chunk(s) (split into ≤30 s pieces) → finish → trim `text`. On any failure after
  start, best-effort finish to free the server session.
- macOS wiring: `Transcriber::Remote` + all exhaustive match sites.
- Preparation: no download; prepare = health round-trip (start + 0.5 s silence + finish), which
  also warms the cold model. Onboarding skips download when remote is selected.
- UI: when Remote is selected, Settings shows the Endpoint field + Test connection.
- Errors: HUD error, paste nothing.

## Phase 2 (after phase 1 is proven)

- Live streaming: open the session on key-down, push chunks while held, `finish` on release
  → near-zero post-release latency. Maps onto HEX's existing LiveStream path.
- Linux/Windows wiring.

## Tasks (phase 1, macOS)

- [x] 0. Branch `remote-asr` from `dev` (02ee88e = wegvon/hex dev); Linux toolchain + deps
- [x] 1. `remote_transcriber.rs` + 11 tests against a fake HTTP server + opt-in live test
- [x] 2. `AppSettings.remote_asr` (serde default, round-trip test) + `hex settings remote_asr.endpoint` with validation
- [x] 3. Catalog: `QwenServer` / `ModelRuntime::Remote`, macOS-only availability, catalog test
- [x] 4. macOS wiring: `Transcriber::Remote`, all runtime match sites (parakeet, local_api, preparation)
- [x] 5. Preparation runs one silent server session (checks + warms); no download, onboarding sees it installed
- [x] 6. Settings UI: "Server address" field (shown on macOS), translations in all 5 UI languages
- [x] 7. Docs: AGENTS.md architecture, docs/features/dictation.md (`dictate.qwen-server`)
- [ ] 8. macOS compile + tests on CI (fork Actions), then real dictation on the Mac
- [x] 9. Review section

## Decisions (2026-10-25)

- Platforms phase 1: macOS only — default, user had no preference
- Unreachable server: HUD error, paste nothing — default, user had no preference
- Fork: develop here first; push to the user's GitHub once named — default
- Protocol: Rem's Qwen session API only (no OpenAI-compat layer until needed — simplicity first)

## Review (2026-10-25)

Verified here (Linux x86_64):
- `cargo fmt --check`, `cargo clippy --bin voice-control --tests -D warnings`: clean
- `cargo test --bin voice-control`: 632 passed, 0 failed (incl. i18n coverage, CLI key coverage)
- Live: `real_server_transcribes_a_clip` against https://qasr.turel.se — 6.3 s Turkish clip →
  "Merhaba bugün hava çok güzel. Toplantıya gün sabah üçte yapalım." in 3.0 s (espeak-ng voice)
- Raw protocol: whole clip in one `/api/chunk` works; warm cost ≈ 1.8 s chunk + 0.7 s finish for 6.3 s

Not verified yet:
- macOS compile/tests (macOS-only code paths: `transcription.rs`, `parakeet.rs`, preparation) → CI on fork
- Real dictation on the Mac through the 127.0.0.1:17794 relay

Findings:
- Server decodes in 1 s steps sequentially → post-release latency grows ~linearly with speech length.
  Phase 2 (stream chunks while the key is held) leaves only `/api/finish` after release.
- Full suite caught a real gap: every persisted setting needs a `hex settings` key → added.
- Upstream dev moved to bb3d348 (Scrawl capture engine); fork dev still 02ee88e. Sync is a separate decision.
