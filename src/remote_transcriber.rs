//! Transcription on a Qwen3-ASR server: the official Qwen streaming demo
//! (`qwen-asr` with vLLM) that serves one session per utterance.
//!
//! ```ts
//! POST /api/start                    → { session_id }
//! POST /api/chunk?session_id=…       ← 16 kHz mono little-endian f32 PCM
//!                                    → { language, text }   text so far
//! POST /api/finish?session_id=…      → { language, text }   final, session freed
//! ```
//!
//! Requests go through the platform's curl like every other Hex request;
//! audio travels on stdin, never in arguments. The server is the person's
//! own: Hex sends it nothing but dictation audio.

use std::io::Write;
use std::process::{Command, Stdio};
use std::thread;

use color_eyre::eyre::{Result, WrapErr, bail, eyre};
use serde::Deserialize;

/// Audio per `/api/chunk` request. The server decodes in one-second steps
/// either way; bounded bodies keep one request well inside its time limit.
const CHUNK_SAMPLES: usize = 30 * 16_000;
const CONNECT_TIMEOUT_SECS: u64 = 5;
/// The first chunk after a server restart compiles kernels (~6 s observed).
const REQUEST_TIMEOUT_SECS: u64 = 60;
const CLEANUP_TIMEOUT_SECS: u64 = 5;
/// Half a second of silence: enough for a full start → chunk → finish round
/// trip that also warms a cold server, short enough to cost no GPU time.
const CHECK_SAMPLES: usize = 8_000;

#[derive(Debug)]
pub struct QwenServer {
    base: url::Url,
}

#[derive(Deserialize)]
struct Started {
    session_id: String,
}

#[derive(Deserialize)]
struct Transcribed {
    #[serde(default)]
    text: String,
}

impl QwenServer {
    pub fn new(endpoint: &str) -> Result<Self> {
        Ok(Self {
            base: endpoint_url(endpoint)?,
        })
    }

    /// The server configured in settings.
    #[cfg(target_os = "macos")]
    pub fn configured() -> Result<Self> {
        Self::new(&crate::app_settings::remote_asr_settings().endpoint)
    }

    /// Proves the server answers a whole session, warming it on the way.
    pub fn check(&self) -> Result<()> {
        self.transcribe(&[0.0; CHECK_SAMPLES]).map(drop)
    }

    /// Transcribes normalized 16 kHz mono audio in one server session.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        let started: Started = self.post("/api/start", None, None, REQUEST_TIMEOUT_SECS)?;
        let session = started.session_id;
        let result = samples
            .chunks(CHUNK_SAMPLES)
            .try_for_each(|chunk| {
                self.post::<Transcribed>(
                    "/api/chunk",
                    Some(&session),
                    Some(pcm_bytes(chunk)),
                    REQUEST_TIMEOUT_SECS,
                )
                .map(drop)
            })
            .and_then(|()| {
                self.post::<Transcribed>("/api/finish", Some(&session), None, REQUEST_TIMEOUT_SECS)
            });
        match result {
            Ok(finished) => Ok(finished.text.trim().to_owned()),
            Err(error) => {
                // Free the server's single session slot before reporting.
                let _ = self.post::<Transcribed>(
                    "/api/finish",
                    Some(&session),
                    None,
                    CLEANUP_TIMEOUT_SECS,
                );
                Err(error)
            }
        }
    }

    fn post<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        session: Option<&str>,
        body: Option<Vec<u8>>,
        timeout_secs: u64,
    ) -> Result<T> {
        let mut url = self.base.clone();
        url.set_path(&format!("{}{path}", self.base.path().trim_end_matches('/')));
        if let Some(session) = session {
            url.query_pairs_mut().append_pair("session_id", session);
        }
        let mut command = curl();
        command
            .args(["--silent", "--show-error", "--request", "POST"])
            .args(["--proto", "=http,https", "--max-redirs", "0"])
            .arg("--connect-timeout")
            .arg(CONNECT_TIMEOUT_SECS.to_string())
            .arg("--max-time")
            .arg(timeout_secs.to_string())
            .args(["--write-out", "\n%{http_code}"]);
        if is_loopback(&url) {
            // A proxy in the environment must never see local audio.
            command.args(["--noproxy", "*"]);
        }
        if body.is_some() {
            command
                .args(["--header", "Content-Type: application/octet-stream"])
                .args(["--data-binary", "@-"]);
        }
        command.arg(url.as_str());
        let output = run(command, body).wrap_err_with(|| format!("Qwen server {path} failed"))?;
        if !output.status.success() {
            let detail = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            bail!(
                "Qwen server at {} is unreachable: {detail}",
                self.base.as_str().trim_end_matches('/')
            );
        }
        let (status, body) = split_http_status(output.stdout)
            .ok_or_else(|| eyre!("Qwen server {path} returned no HTTP status"))?;
        if !(200..300).contains(&status) {
            let detail = serde_json::from_slice::<serde_json::Value>(&body)
                .ok()
                .and_then(|value| value.get("error")?.as_str().map(str::to_owned))
                .unwrap_or_else(|| {
                    String::from_utf8_lossy(&body)
                        .trim()
                        .chars()
                        .take(200)
                        .collect()
                });
            bail!("Qwen server {path} answered HTTP {status}: {detail}");
        }
        serde_json::from_slice(&body)
            .wrap_err_with(|| format!("Qwen server {path} returned unexpected JSON"))
    }
}

/// Accepts an `http(s)://host[:port][/prefix]` server root.
pub fn endpoint_url(endpoint: &str) -> Result<url::Url> {
    let url = url::Url::parse(endpoint.trim()).wrap_err("invalid Qwen server address")?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        bail!("Qwen server address must look like http://host:port");
    }
    Ok(url)
}

fn is_loopback(url: &url::Url) -> bool {
    match url.host() {
        Some(url::Host::Domain(domain)) => domain.eq_ignore_ascii_case("localhost"),
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        None => false,
    }
}

fn pcm_bytes(samples: &[f32]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect()
}

fn curl() -> Command {
    // macOS always has /usr/bin/curl; elsewhere (NixOS) it comes from PATH.
    let system = std::path::Path::new("/usr/bin/curl");
    let mut command = Command::new(if cfg!(target_os = "macos") || system.exists() {
        system
    } else {
        std::path::Path::new("curl")
    });
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    command
}

fn run(mut command: Command, body: Option<Vec<u8>>) -> Result<std::process::Output> {
    command.stdin(if body.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    let mut child = command.spawn().wrap_err("could not start curl")?;
    let writer = body.map(|body| {
        let mut stdin = child.stdin.take().expect("curl stdin is piped");
        thread::spawn(move || stdin.write_all(&body))
    });
    let output = child
        .wait_with_output()
        .wrap_err("could not wait for curl")?;
    if let Some(writer) = writer {
        // curl closes its input early when the server rejects the request;
        // the HTTP status explains that better than a broken pipe.
        let written = writer
            .join()
            .map_err(|_| eyre!("curl input writer panicked"))?;
        if output.status.success() {
            written.wrap_err("could not send audio to curl")?;
        }
    }
    Ok(output)
}

/// curl appends `\n<status>` after the body (`--write-out`).
fn split_http_status(mut stdout: Vec<u8>) -> Option<(u16, Vec<u8>)> {
    let newline = stdout.iter().rposition(|byte| *byte == b'\n')?;
    let status = std::str::from_utf8(&stdout[newline + 1..])
        .ok()?
        .trim()
        .parse()
        .ok()?;
    stdout.truncate(newline);
    Some((status, stdout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    #[derive(Clone, Debug)]
    struct Seen {
        target: String,
        content_type: Option<String>,
        body: Vec<u8>,
    }

    /// Answers each request with the next canned `(status, body)`; records
    /// what curl sent.
    fn server(replies: Vec<(u16, &'static str)>) -> (String, Arc<Mutex<Vec<Seen>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        thread::spawn(move || {
            for (status, reply) in replies {
                let Ok((stream, _)) = listener.accept() else {
                    return;
                };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                let target = request_line.split(' ').nth(1).unwrap_or("").to_owned();
                let mut length = 0;
                let mut content_type = None;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    let line = line.trim_end();
                    if line.is_empty() {
                        break;
                    }
                    let (name, value) = line.split_once(':').unwrap();
                    match name.to_ascii_lowercase().as_str() {
                        "content-length" => length = value.trim().parse().unwrap(),
                        "content-type" => content_type = Some(value.trim().to_owned()),
                        _ => {}
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                log.lock().unwrap().push(Seen {
                    target,
                    content_type,
                    body,
                });
                let mut stream = stream;
                write!(
                    stream,
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                    reply.len()
                )
                .unwrap();
            }
        });
        (address, seen)
    }

    const START: (u16, &str) = (200, r#"{"session_id":"abc"}"#);

    #[test]
    fn transcribes_one_session_and_sends_raw_little_endian_pcm() {
        let (address, seen) = server(vec![
            START,
            (200, r#"{"language":"Turkish","text":"mer"}"#),
            (200, r#"{"language":"Turkish","text":"  merhaba dünya \n"}"#),
        ]);
        let samples = [0.5_f32, -0.25, 1.0];
        let text = QwenServer::new(&address)
            .unwrap()
            .transcribe(&samples)
            .unwrap();
        assert_eq!(text, "merhaba dünya");
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 3);
        assert_eq!(seen[0].target, "/api/start");
        assert_eq!(seen[1].target, "/api/chunk?session_id=abc");
        assert_eq!(
            seen[1].content_type.as_deref(),
            Some("application/octet-stream")
        );
        assert_eq!(seen[1].body, pcm_bytes(&samples));
        assert_eq!(seen[1].body.len(), samples.len() * 4);
        assert_eq!(&seen[1].body[..4], &0.5_f32.to_le_bytes());
        assert_eq!(seen[2].target, "/api/finish?session_id=abc");
        assert!(seen[2].body.is_empty());
    }

    #[test]
    fn splits_long_audio_into_bounded_chunks() {
        let (address, seen) = server(vec![
            START,
            (200, r#"{"text":"a"}"#),
            (200, r#"{"text":"a b"}"#),
            (200, r#"{"text":"a b"}"#),
        ]);
        let samples = vec![0.1; CHUNK_SAMPLES + 10];
        let text = QwenServer::new(&address)
            .unwrap()
            .transcribe(&samples)
            .unwrap();
        assert_eq!(text, "a b");
        let seen = seen.lock().unwrap();
        assert_eq!(seen[1].body.len(), CHUNK_SAMPLES * 4);
        assert_eq!(seen[2].body.len(), 10 * 4);
        assert_eq!(seen[3].target, "/api/finish?session_id=abc");
    }

    #[test]
    fn keeps_an_endpoint_path_prefix() {
        let (address, seen) = server(vec![START, (200, "{}"), (200, r#"{"text":"ok"}"#)]);
        let text = QwenServer::new(&format!("{address}/asr/"))
            .unwrap()
            .transcribe(&[0.0])
            .unwrap();
        assert_eq!(text, "ok");
        assert_eq!(seen.lock().unwrap()[0].target, "/asr/api/start");
    }

    #[test]
    fn a_failed_chunk_reports_the_server_error_and_frees_the_session() {
        let (address, seen) = server(vec![
            START,
            (400, r#"{"error":"invalid session_id"}"#),
            (200, r#"{"text":""}"#),
        ]);
        let error = QwenServer::new(&address)
            .unwrap()
            .transcribe(&[0.0])
            .unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("HTTP 400"), "{message}");
        assert!(message.contains("invalid session_id"), "{message}");
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 3);
        assert_eq!(seen[2].target, "/api/finish?session_id=abc");
    }

    #[test]
    fn a_server_crash_reports_its_status_without_a_session() {
        let (address, seen) = server(vec![(500, "Internal Server Error")]);
        let message = format!(
            "{:#}",
            QwenServer::new(&address)
                .unwrap()
                .transcribe(&[0.0])
                .unwrap_err()
        );
        assert!(message.contains("HTTP 500"), "{message}");
        assert!(message.contains("Internal Server Error"), "{message}");
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[test]
    fn rejects_unexpected_json() {
        let (address, _) = server(vec![(200, "not json")]);
        let message = format!(
            "{:#}",
            QwenServer::new(&address)
                .unwrap()
                .transcribe(&[0.0])
                .unwrap_err()
        );
        assert!(message.contains("unexpected JSON"), "{message}");
    }

    #[test]
    fn an_absent_server_is_unreachable() {
        let address = {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            format!("http://{}", listener.local_addr().unwrap())
        };
        let message = format!(
            "{:#}",
            QwenServer::new(&address)
                .unwrap()
                .transcribe(&[0.0])
                .unwrap_err()
        );
        assert!(message.contains("unreachable"), "{message}");
    }

    #[test]
    fn silence_sends_nothing() {
        let server = QwenServer::new("http://127.0.0.1:9").unwrap();
        assert_eq!(server.transcribe(&[]).unwrap(), "");
    }

    #[test]
    fn check_runs_a_whole_session() {
        let (address, seen) = server(vec![START, (200, "{}"), (200, r#"{"text":""}"#)]);
        QwenServer::new(&address).unwrap().check().unwrap();
        let seen = seen.lock().unwrap();
        assert_eq!(seen[1].body.len(), CHECK_SAMPLES * 4);
    }

    #[test]
    fn validates_server_addresses() {
        for valid in [
            "http://127.0.0.1:17794",
            "http://rem.lan:7794/",
            "https://qasr.example.com",
            "  http://localhost:17794  ",
        ] {
            assert!(endpoint_url(valid).is_ok(), "{valid}");
        }
        for invalid in [
            "",
            "127.0.0.1:17794",
            "ftp://host",
            "http://user:secret@host",
            "http://host/?x=1",
            "file:///tmp/socket",
        ] {
            assert!(endpoint_url(invalid).is_err(), "{invalid}");
        }
    }

    /// Live check against a real server, opt-in:
    /// `HEX_QWEN_SERVER=http://127.0.0.1:17794 HEX_QWEN_AUDIO=clip.f32 cargo test
    /// real_server -- --ignored` with 16 kHz mono f32 LE audio of speech.
    #[test]
    #[ignore = "needs a Qwen3-ASR server and an audio clip"]
    fn real_server_transcribes_a_clip() {
        let server = QwenServer::new(&std::env::var("HEX_QWEN_SERVER").unwrap()).unwrap();
        let bytes = std::fs::read(std::env::var("HEX_QWEN_AUDIO").unwrap()).unwrap();
        let samples: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        server.check().unwrap();
        let started = std::time::Instant::now();
        let text = server.transcribe(&samples).unwrap();
        eprintln!(
            "{:.1} s audio -> {:?} in {:.2} s",
            samples.len() as f64 / 16_000.0,
            text,
            started.elapsed().as_secs_f64()
        );
        assert!(!text.is_empty());
    }

    #[test]
    fn only_loopback_skips_proxies() {
        assert!(is_loopback(&endpoint_url("http://127.0.0.1:1").unwrap()));
        assert!(is_loopback(&endpoint_url("http://LOCALHOST:1").unwrap()));
        assert!(is_loopback(&endpoint_url("http://[::1]:1").unwrap()));
        assert!(!is_loopback(&endpoint_url("http://10.0.0.5:1").unwrap()));
        assert!(!is_loopback(
            &endpoint_url("https://qasr.example.com").unwrap()
        ));
    }
}
