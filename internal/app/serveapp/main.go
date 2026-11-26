// Command serve is an OpenAI- and Anthropic-compatible HTTP server for goinfer
// models: pure stdlib net/http, no dependencies. It speaks /v1/chat/completions,
// /v1/completions, /v1/responses, /v1/embeddings, /v1/models, and the Anthropic
// Messages API (/v1/messages, /v1/messages/count_tokens), and TypeSafe's decisions
// shape (/v1/systemone, answered by label scoring) — enough for Open WebUI,
// LangChain, the OpenAI SDKs, Claude Code, jevx, and anything else that points at an
// OpenAI, Anthropic or TypeSafe base URL — including streaming (SSE) and
// `response_format: json_schema` constrained decoding (the model physically
// cannot emit non-conforming JSON; see the constrain package).
//
// A generative (decoder) model is served via -model; an embedding (encoder)
// model via -embed-model. Either or both may be loaded in one process — like
// running llama.cpp/vLLM with a model per task, but without a separate router.
// Endpoints are registered for whatever is loaded.
//
//	go run ./cmd/serve --model ~/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
//	go run ./cmd/serve --embed-model ~/models/coderankembed       # /v1/embeddings only
//	# then point a client at http://localhost:8080/v1
//	curl localhost:8080/v1/chat/completions -d '{"model":"local",
//	  "messages":[{"role":"user","content":"hi"}]}'
package serveapp

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"maps"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/townsendmerino/goinfer/decoder"
	"github.com/townsendmerino/goinfer/internal/app/connect"
	"github.com/townsendmerino/goinfer/internal/app/pullcmd"
	"github.com/townsendmerino/goinfer/internal/app/serveapp/jobs"
	"github.com/townsendmerino/goinfer/internal/guard/servecheck"
	"github.com/townsendmerino/goinfer/pull"
)

// restoreSessions loads each model's saved KV sessions from -session-dir and turns on tiering when it is asked for.
// A no-op without a session directory or with sessions off.
func (srv *server) restoreSessions(cfg config) {
	if cfg.sessionDir == "" || cfg.kvSessions <= 0 {
		return
	}
	for _, lm := range srv.modelList() {
		sub := sessionSubdir(cfg.sessionDir, lm.fp)
		lm.sessions.Load(sub)
		if cfg.kvIdleDemote > 0 {
			lm.sessions.EnableTiering(sub, cfg.kvIdleDemote, cfg.kvDemotedMax)
		}
	}
}

// sessionNoticeOut is where a session checkpoint that could not be written is said; a test swaps it.
var sessionNoticeOut io.Writer = os.Stderr

// checkpointSessions saves each model's KV sessions to -session-dir at shutdown. The caller has already cancelled
// the generations in flight; a model one of them still holds after five seconds is skipped, not waited for.
func (srv *server) checkpointSessions(cfg config) {
	if cfg.sessionDir == "" || cfg.kvSessions <= 0 {
		return
	}
	deadline := time.Now().Add(5 * time.Second)
	for _, lm := range srv.modelList() {
		// A generation mid-forward, not yet at a context check, can briefly still hold lm.sessMu.
		if !tryLockUntil(&lm.sessMu, deadline) {
			fmt.Fprintf(os.Stderr, "shutdown: model %q still busy — skipping its session checkpoint\n", lm.name)
			continue
		}
		if err := lm.sessions.Save(sessionSubdir(cfg.sessionDir, lm.fp)); err != nil {
			fmt.Fprintf(sessionNoticeOut, "shutdown: model %q: session checkpoint not written (%v); its sessions will not restore\n", lm.name, err)
		}
		lm.sessions.RemoveColdFiles() // the cold tier is in-process; clear its scratch
		lm.sessMu.Unlock()
	}
}

// routes builds the server's route table: every endpoint behind its auth check, halt gate, inflight cap and body cap.
// authKey "" disables the auth check. With cfg.logRequests set, each generation request writes its line to stderr. It returns
// the mux and the text body cap, which the admin socket's routes share. The listener's own concerns (the Host guard, TLS,
// timeouts, signals) are the caller's.
func (srv *server) routes(cfg config, authKey string, stderr io.Writer) (*http.ServeMux, int64) {
	var reqLog io.Writer
	if cfg.logRequests {
		reqLog = stderr
	}
	// Every POST body is size-bounded: the chat/messages endpoints carry base64 image_url data, so they get the larger
	// vision cap; the rest a few MB. auth wraps a handler with the optional shared-secret check (no-op when authKey is "");
	// every route below goes through it so a set key protects the whole surface.
	auth := func(h http.HandlerFunc) http.HandlerFunc { return requireAuth(authKey, h) }
	// inflight is the global concurrency cap over the inference POST handlers (the pre-queue stage), shared across all of
	// them. GET/health and admin stay uncapped so an operator's health probe is always answered and never consumes a slot.
	// Order: auth outermost (reject bad auth without taking a slot), then the inflight gate, then the body cap.
	var inflight chan struct{}
	if cfg.maxInflight > 0 {
		inflight = make(chan struct{}, cfg.maxInflight)
	}
	inf := func(h http.HandlerFunc) http.HandlerFunc { return limitInflight(inflight, h) }
	// Request-body caps: the largest servable text prompt is ctx tokens x the longest token's bytes, x4 for JSON
	// structure/escaping, derived per the largest served model's context window, floored at fixed minimums so a
	// small-context model keeps a usable body budget (the per-request tokenization guard, not this cap, protects it), and
	// overridable with -max-body-bytes. Reported on the startup line.
	textCap, visionCap, embedCap, fileCap := srv.resolveBodyCaps(cfg.maxBodyBytes)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", auth(srv.handleModels))
	// Operator surface for the resolved compute paths — same fields as the /v1/models vendor
	// extension, on a payload with no OpenAI-schema contract to break. See handleHealth.
	mux.HandleFunc("GET /health", auth(srv.handleHealth))
	// haltGate runs AFTER auth (a bad key is still rejected during a halt) and BEFORE inf (a halt must not wait for an
	// inflight slot). /admin/* and /health are deliberately NOT wrapped in it: an operator must always be able to resume and
	// check status.
	// These routes are registered whether or not a model is loaded at startup: a server started with only --web,
	// --allow-admin or --admin-socket loads its model later, and the mux is built once. Every handler resolves its model
	// through resolveAndLock, which answers an unknown or absent model with the OpenAI-shaped 404 "model not found (served:
	// ..)".
	// -log-requests wraps the four generation routes OUTERMOST, so a request the auth, halt or queue gates turned away is
	// logged with its status too.
	rl := func(h http.HandlerFunc) http.HandlerFunc { return srv.traceRequests(reqLog, h) }
	mux.HandleFunc("POST /v1/chat/completions", rl(auth(srv.haltGate(inf(maxBytes(visionCap, srv.handleChat))))))
	mux.HandleFunc("POST /v1/completions", rl(auth(srv.haltGate(inf(maxBytes(textCap, srv.handleCompletions))))))
	mux.HandleFunc("POST /v1/responses", rl(auth(srv.haltGate(inf(maxBytes(textCap, srv.handleResponses))))))
	mux.HandleFunc("POST /v1/messages", rl(auth(srv.haltGate(inf(maxBytes(visionCap, srv.handleMessages))))))
	// Speech to text for a Whisper --model (speech.go). The body is audio, so it has its own cap: -max-body-bytes, else 256 MiB (over two hours of 16 kHz mono 16-bit).
	audioCap := cfg.maxBodyBytes
	if audioCap <= 0 {
		audioCap = 256 << 20
	}
	mux.HandleFunc("POST /v1/audio/transcriptions", rl(auth(srv.haltGate(inf(maxBytes(audioCap, srv.handleTranscriptions))))))
	mux.HandleFunc("POST /v1/messages/count_tokens", auth(srv.haltGate(inf(maxBytes(textCap, srv.handleCountTokens)))))
	// POST /v1/systemone is TypeSafe's wire shape, served by label scoring (internal/decide), so jevx and the TypeSafe SDKs
	// work against goinfer through their base-URL override.
	mux.HandleFunc("POST /v1/systemone", auth(srv.haltGate(inf(maxBytes(textCap, srv.handleSystemOne)))))
	// Submitting a job starts new admission, so POST /v1/jobs gets the same haltGate/inf/maxBytes stack as every other POST
	// above. Polling (GET), the event stream and cancel (DELETE) are not new inference work: haltGate would 503 a client
	// trying to learn that its job was halted, and inf's cap bounds pre-queue decode and tokenization, which these do not
	// do. They get auth only.
	mux.HandleFunc("POST /v1/jobs", auth(srv.haltGate(inf(maxBytes(textCap, srv.handleCreateJob)))))
	mux.HandleFunc("GET /v1/jobs/{id}", auth(srv.handleGetJob))
	mux.HandleFunc("GET /v1/jobs/{id}/events", auth(srv.handleJobEvents))
	mux.HandleFunc("DELETE /v1/jobs/{id}", auth(srv.handleCancelJob))
	// The two batch APIs, over the same job store. POST /v1/files is an upload, not inference, so like /web/models/pull it
	// gets auth + maxBytes only (haltGate and inf bound decode concurrency, not upload I/O). POST /v1/batches and POST
	// /v1/messages/batches queue real generation work (one job per line, the /v1/jobs pipeline) and get the full stack.
	// Every GET and every .../cancel gets auth only, as /v1/jobs' own GET/DELETE do.
	// Bound concurrent 64 MiB file uploads independently of inference inflight slots.
	const maxConcurrentFileUploads = 8
	fileUploadSem := make(chan struct{}, maxConcurrentFileUploads)
	mux.HandleFunc("POST /v1/files", auth(limitInflight(fileUploadSem, maxBytes(fileCap, srv.handleCreateFile))))
	mux.HandleFunc("GET /v1/files/{id}", auth(srv.handleGetFile))
	mux.HandleFunc("GET /v1/files/{id}/content", auth(srv.handleGetFileContent))
	mux.HandleFunc("POST /v1/batches", auth(srv.haltGate(inf(maxBytes(textCap, srv.handleCreateBatch)))))
	mux.HandleFunc("GET /v1/batches/{id}", auth(srv.handleGetBatch))
	mux.HandleFunc("POST /v1/batches/{id}/cancel", auth(srv.handleCancelBatch))
	mux.HandleFunc("POST /v1/messages/batches", auth(srv.haltGate(inf(maxBytes(textCap, srv.handleCreateMessageBatch)))))
	mux.HandleFunc("GET /v1/messages/batches/{id}", auth(srv.handleGetMessageBatch))
	mux.HandleFunc("GET /v1/messages/batches/{id}/results", auth(srv.handleGetMessageBatchResults))
	mux.HandleFunc("POST /v1/messages/batches/{id}/cancel", auth(srv.handleCancelMessageBatch))
	// Registered unconditionally: with no embedding model, handleEmbeddings returns a JSON error naming -embed-model rather
	// than a bare 404, so an SDK sees "unconfigured", not "wrong URL".
	mux.HandleFunc("POST /v1/embeddings", auth(srv.haltGate(inf(maxBytes(embedCap, srv.handleEmbeddings,
		// The per-input / per-batch bounds are per-DIMENSION and multiply out past any body cap, so
		// name all three: a client within both per-dimension limits can still exceed the total.
		fmt.Sprintf("this route also limits each request to %d inputs of at most %d bytes each; "+
			"the body cap bounds their total", maxEmbedInputs, maxEmbedInputBytes))))))
	// /admin/* lives on EITHER the TCP listener (auth + -allow-admin) OR the admin socket when -admin-socket is set, never
	// both, so a request against the surface that was not chosen 404s instead of being refused: a 403 would confirm the
	// surface exists. See admin_socket.go for the socket side.
	if cfg.adminSocket == "" {
		registerAdminRoutes(mux, srv, textCap, func(h http.HandlerFunc) http.HandlerFunc { return auth(srv.requireAdmin(h)) })
	}
	if cfg.web {
		// "GET /{$}" matches the root path EXACTLY; a bare "GET /" is a catch-all that would answer every unknown GET with the
		// UI page instead of a 404.
		// UNAUTHENTICATED on purpose: the page embeds no secrets (handleWebUI) and is the only place a user can type the key in,
		// so auth() here would make the key unenterable whenever -api-key is set. auth stays on the routes below, which act.
		mux.HandleFunc("GET /{$}", srv.handleWebUI)
		// The page's own CSS and JS, unauthenticated for the same reason as the page.
		// {file} is one path segment, so "GET /ui/{file}" cannot become a catch-all.
		mux.HandleFunc("GET /ui/{file}", srv.handleWebAsset)
		// sameOrigin: on the key-free loopback default auth() alone is a no-op, and these routes act (pull starts a caller-named
		// multi-gigabyte download). See sameOriginGuard in webui.go.
		sameOrigin := sameOriginGuard(splitHosts(cfg.allowedHosts))
		mux.HandleFunc("POST /web/models/list", sameOrigin(auth(maxBytes(textCap, srv.handleWebList))))
		// Live search-as-you-type suggestions over the repo box (pull.Search): read-only, same origin/auth stack as list.
		mux.HandleFunc("POST /web/models/search", sameOrigin(auth(maxBytes(textCap, srv.handleWebSearch))))
		// Not wrapped in inf(): the inflight gate bounds INFERENCE, and a download that runs
		// for minutes must not occupy one of those slots. handleWebPull is single-flighted on
		// its own (pullState), which is the bound that actually fits it.
		mux.HandleFunc("POST /web/models/pull", sameOrigin(auth(maxBytes(textCap, srv.handleWebPull))))
		// Load what the pull just downloaded. NOT the admin load, which takes any caller-named path and stays behind
		// -allow-admin: this one is confined to regular .gguf files under the pull cache (webLoadPath) and gets the same stack
		// as pull, since it is at least as heavy an action. Not in inf() either, for the pull's reason.
		mux.HandleFunc("POST /web/models/load", sameOrigin(auth(maxBytes(textCap, srv.handleWebLoad))))
		// Unload a currently-loaded model. It needs no path policy: the only names it can act on are ones GET /v1/models already
		// publishes, so it reuses unloadByName directly. Same stack as load.
		mux.HandleFunc("POST /web/models/unload", sameOrigin(auth(maxBytes(textCap, srv.handleWebUnload))))
		// What the pull cache holds on disk, for the page's "On disk" card. Read-only, and it names only paths under the pull
		// cache, the ones the load route above would accept anyway.
		mux.HandleFunc("GET /web/models/cache", sameOrigin(auth(srv.handleWebCache)))
	}
	return mux, textCap
}

func Main() {
	if dispatchSubcommand() {
		return
	}
	cfg, authKey, addr, tlsCert, tlsKey, versionOnly := parseServeFlags()
	if versionOnly {
		return
	}
	srv, mux, textCap, closeAdminSock := startServer(cfg, authKey)

	// The timeouts bound slow-header (slowloris), slow-body dribble and idle keep-alive connections. ReadTimeout is the
	// whole-request read deadline (60s: generous for a 32 MiB vision body on a slow link); it bounds only the request READ,
	// so a long SSE stream is unaffected. WriteTimeout stays 0 on purpose: a write deadline would truncate a legitimate
	// stream.
	// srvCtx is the server-lifetime context. BaseContext makes every request's r.Context() a child of it, so cancelling it at
	// shutdown cancels every in-flight generation (drive derives its context from r.Context()). Without that,
	// httpSrv.Shutdown waits for a long streaming generation but never cancels it, and the shutdown runs past its 30s
	// timeout. sessMu is held only briefly (around sessions.acquire), so the checkpoint loop does not deadlock on a running
	// generation, but Shutdown still waits for the handler to return: cancelling is what bounds it. tryLockUntil's deadline
	// is the remaining bound on sessMu itself.
	srvCtx, srvCancel := srv.initLifeContext()
	defer srvCancel(errServerShutdown)
	httpSrv := &http.Server{
		Addr:              *addr,
		Handler:           serveHandler(mux, *addr, authKey, cfg.allowedHosts),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       60 * time.Second,
		IdleTimeout:       120 * time.Second,
		BaseContext:       func(net.Listener) context.Context { return srvCtx },
	}

	stopHaltPoll, done := watchShutdown(srv, cfg, httpSrv, srvCancel, closeAdminSock)

	// SIGUSR1 halts, SIGUSR2 resumes, so a supervisor can pull this switch without a socket or an HTTP client.
	// Separate from the one-shot SIGINT/SIGTERM channel above: these repeat for the life
	// of the process, so they get their own Notify and a loop. Platform-specific (haltsignal_unix.go /
	// haltsignal_windows.go): SIGUSR1/SIGUSR2 are undefined identifiers on Windows.
	startHaltSignalLoop(srv)

	// -halt-file is polled in its own goroutine; stopHaltPoll is closed alongside stopDemote at shutdown. It is best-effort
	// background work with no result main waits on.
	if cfg.haltFile != "" {
		go haltFilePoller(srv, cfg.haltFile, stopHaltPoll)
	}

	useTLS := *tlsCert != ""
	printServing(srv, cfg, addr, authKey, useTLS, textCap)
	var listenErr error
	if useTLS {
		listenErr = httpSrv.ListenAndServeTLS(*tlsCert, *tlsKey)
	} else {
		listenErr = httpSrv.ListenAndServe()
	}
	if listenErr != nil && !errors.Is(listenErr, http.ErrServerClosed) {
		fmt.Fprintf(os.Stderr, "server: %v\n", listenErr)
		os.Exit(1)
	}
	<-done // let the shutdown handler finish checkpointing before we exit
}

// dispatchSubcommand runs a subcommand named by the first argument and exits, or prints the version. It also installs
// the help header. It reports whether Main is done; it returns false when the arguments are for a server.
func dispatchSubcommand() bool {
	// Subcommand dispatch, before flag.Parse so `pull` gets its own flag set. Shares one
	// implementation with `goinfer-chat pull` (internal/pullcmd) rather than repeating the
	// flags and error messages — serve users need a model on disk for exactly the same
	// reason chat users do.
	if len(os.Args) > 1 && os.Args[1] == "pull" {
		os.Exit(pullcmd.Run(os.Args[2:]))
	}
	// `check` drives a RUNNING server through a harness's conversation. A client rather than
	// an embedded server, so it exercises whatever is actually serving with the flags its
	// operator chose — which is what a harness meets.
	if len(os.Args) > 1 && os.Args[1] == "check" {
		os.Exit(servecheck.Run(os.Args[2:], filepath.Base(os.Args[0])))
	}
	if len(os.Args) > 1 && os.Args[1] == "connect" {
		os.Exit(connect.Run(os.Args[2:], filepath.Base(os.Args[0]), os.Stdout))
	}
	// `status|ls|cancel|halt|resume` are a CLIENT talking to a RUNNING server's admin socket (-admin-socket), not the
	// server itself: the same dispatch shape as pull/check above.
	if len(os.Args) > 1 && isAdminCLICmd(os.Args[1]) {
		os.Exit(runAdminCLI(os.Args[1], os.Args[2:], filepath.Base(os.Args[0])))
	}
	// A skimmable help header, printed before the flag dump. The per-flag text stays long on purpose: each paragraph is a
	// disclosure a measurement earned, and deleting disclosures to shorten the page is how a trade-off stops being
	// disclosed. This adds a map ABOVE it rather than trimming it. Do not shorten the flag help to fix a long --help.
	flag.Usage = func() {
		out := flag.CommandLine.Output()
		// The examples use the name the binary was actually INVOKED as, not a hardcoded "goinfer-serve": `go install.
		// ../cmd/serve@latest` installs a binary called `serve`.
		self := filepath.Base(os.Args[0])
		fmt.Fprintf(out, `%[1]s — OpenAI/Anthropic-compatible local inference server.

  %[1]s --model <file.gguf|dir|hf:owner/repo:quant>     serve one model
  %[1]s --model m.gguf --web                            + browser UI at /
  %[1]s check                                           drive a RUNNING server, per-feature verdicts
  %[1]s pull <ref>                                      fetch a model, sha256-verified
  %[1]s --version                                       version + the backends COMPILED IN

%[3]s

The flags people actually reach for:

  --model      what to serve; repeatable as name=path to serve several at once
  --backend    auto (default: the first of cuda, metal that finds its device, else cpu) | cpu | cuda | metal | webgpu — --version says which this binary has
  --quant      int4 (default) | int8int8 | int4mix | f32 — see docs/quantization.md
  --ctx        KV capacity in positions
  --addr       listen address (loopback by default)
  --api-key    required to bind anywhere but loopback

  --stream-weights   RUN A MODEL BIGGER THAN YOUR RAM. Pages weights from disk instead of
                     holding them resident; the one flag to know before loading something large.

All %[2]d flags, with the trade-offs each one makes, follow.

`, self, countFlags(), pull.CacheHelp())
		flag.PrintDefaults()
	}

	// --version answers "what is in this binary" WITHOUT a model, so it is handled here and works on a binary whose other
	// required flags are absent; it is registered below as well so it appears in --help.
	if len(os.Args) > 1 && isVersionArg(os.Args[1]) {
		fmt.Print(versionReport(filepath.Base(os.Args[0])))
		return true
	}
	return false
}

// parseServeFlags parses and checks the server's command line, exiting 2 on a refusal. It returns the config, the
// resolved API key and the listen flags; versionOnly reports that --version was printed and Main is done.
func parseServeFlags() (config, string, *string, *string, *string, bool) {
	sf := registerFlags(flag.CommandLine)
	flag.Parse()
	markGivenFlags(flag.CommandLine, &sf.cfg)
	cfg, addr, apiKey, tlsCert, tlsKey, showVersion, lf := sf.cfg, sf.addr, sf.apiKey, sf.tlsCert, sf.tlsKey, sf.showVersion, sf.lf
	if *showVersion {
		fmt.Print(versionReport(filepath.Base(os.Args[0])))
		return config{}, "", nil, nil, nil, true
	}
	if cfg.noSelfTest {
		decoder.SkipSelfTests() // before the first Load, which is what runs the cpu self-test
	}
	// Every argument is a --flag, so anything flag.Parse left in flag.Args() is a typo: refuse it rather than fall through
	// silently.
	if args := flag.Args(); len(args) > 0 {
		fmt.Fprintf(os.Stderr, "%s: unrecognized argument %q\n\nknown subcommands: pull <ref>, check, connect <agent>, --version. Or pass --model <file.gguf|dir|hf:owner/repo:quant>.\n",
			filepath.Base(os.Args[0]), args[0])
		os.Exit(2)
	}
	cfg.load = *lf
	if err := cfg.load.Validate(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(2)
	}
	if l := cfg.load.BackendLine(); l != "" {
		fmt.Fprintln(os.Stderr, l)
	}
	if err := requireAutoBackend(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(2)
	}
	// -web counts as a reason to start with no model: fetching one is the whole point of
	// the UI's Models tab, and requiring a model in order to go and get a model is a
	// bootstrap the user cannot satisfy.
	if len(cfg.models) == 0 && cfg.embedPath == "" && !cfg.allowAdmin && !cfg.web && cfg.adminSocket == "" {
		fmt.Fprintln(os.Stderr, "error: need at least one of --model, --embed-model, --web, --allow-admin, or --admin-socket")
		flag.Usage()
		os.Exit(2)
	}
	// Resolve the optional shared secret (flag wins over env). -allow-admin exposes an arbitrary-path model load + sidecar
	// write, so it must not run unauthenticated.
	authKey := *apiKey
	if authKey == "" {
		authKey = os.Getenv("GOINFER_API_KEY")
	}
	if cfg.allowAdmin && authKey == "" {
		fmt.Fprintln(os.Stderr, "error: -allow-admin requires -api-key (or $GOINFER_API_KEY) — admin load/unload must be authenticated")
		os.Exit(2)
	}
	// Mirrors the -allow-admin check above: the -addr help invites 0.0.0.0 exposure, and a non-loopback bind with no key
	// would start fully open to the network. Loopback stays key-free by default (no auth friction for the single-user desktop
	// case).
	if !addrIsLoopback(*addr) && authKey == "" {
		fmt.Fprintf(os.Stderr, "error: -addr %s is not loopback-only — requires -api-key (or $GOINFER_API_KEY), or bind to 127.0.0.1 instead\n", *addr)
		os.Exit(2)
	}
	if (*tlsCert == "") != (*tlsKey == "") {
		fmt.Fprintln(os.Stderr, "error: -tls-cert and -tls-key must be set together")
		os.Exit(2)
	}
	if err := cfg.validate(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(2)
	}
	cfg.kvShareOff = kvShareOffFor(cfg.kvShare, authKey, *addr, cfg.allowedHosts)
	var kvErr error
	if cfg.kvHostBytes, cfg.kvHostBytesHow, kvErr = resolveKVHostBytes(cfg.kvHostBytesFlag, decoder.HostRAMBytes()); kvErr != nil {
		fmt.Fprintln(os.Stderr, kvErr)
		os.Exit(2)
	}
	cfg.kvDemote = kvDemoteAsked(cfg.kvHostBytesFlag)
	if cfg.kvShareOff && cfg.kvShare != "off" {
		fmt.Fprintln(os.Stderr, "KV sharing: off — this server answers the network without -api-key, so one client's time to first token must not reveal what another sent; set -api-key to share a common prompt between conversations")
	}
	return cfg, authKey, addr, tlsCert, tlsKey, false
}

// startServer builds the server, restores its sessions, builds its routes and starts the admin socket if one is
// set, exiting 1 on a failure. The returned close func is a no-op when -admin-socket is unset.
func startServer(cfg config, authKey string) (*server, *http.ServeMux, int64, func()) {
	srv, err := newServer(resolveGemma3VisionQuant(cfg, os.Stderr)) // Gemma 3 on CUDA: float32 if the card holds it beside the decoder, else int8, before the plan prices the tower
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}
	srv.restoreSessions(cfg)

	mux, textCap := srv.routes(cfg, authKey, os.Stderr)

	// The admin socket. closeAdminSock is a no-op when -admin-socket is unset, so the
	// shutdown handler below can call it unconditionally.
	closeAdminSock := func() {}
	if cfg.adminSocket != "" {
		var err error
		closeAdminSock, err = startAdminSocket(srv, cfg.adminSocket, textCap)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error: -admin-socket %s: %v\n", cfg.adminSocket, err)
			os.Exit(1)
		}
		fmt.Fprintf(os.Stderr, "admin socket: %s (mode 0600, no api-key — file permissions are the auth; /admin/* is NOT registered on the TCP listener while this is set)\n", cfg.adminSocket)
	}
	return srv, mux, textCap, closeAdminSock
}

// watchShutdown starts the idle-session demoter and the SIGINT/SIGTERM handler that drains and checkpoints the
// server. It returns the halt-poll stop channel, which the handler closes for the -halt-file poller, and a channel
// that closes when the checkpoint is written.
func watchShutdown(srv *server, cfg config, httpSrv *http.Server, srvCancel context.CancelCauseFunc, closeAdminSock func()) (chan struct{}, chan struct{}) {
	// Tiered KV: a background ticker demotes idle sessions to disk. It takes sessMu, the
	// lock the request path takes around acquire and checkin, and skips any session a
	// generation has checked out, so it never touches a running generation's session;
	// stopDemote halts it before the shutdown checkpoint runs.
	stopDemote := make(chan struct{})
	if cfg.kvIdleDemote > 0 {
		go demoteLoop(srv, cfg.kvIdleDemote, stopDemote)
	}
	// stopHaltPoll stops the -halt-file poller; declared here so the shutdown handler below can close it alongside stopDemote.
	stopHaltPoll := make(chan struct{})

	// Graceful shutdown: on SIGINT/SIGTERM, stop accepting, drain in-flight
	// generations, then checkpoint the KV sessions to -session-dir (if set).
	// done closes once that's complete so main waits for the save before exit.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	done := make(chan struct{})
	go func() {
		defer close(done)
		<-sig
		fmt.Fprintln(os.Stderr, "\nshutting down…")
		// A SECOND signal during the drain force-exits instead of being swallowed by the buffered channel, so Ctrl-C twice
		// always kills the server, not only SIGKILL.
		go func() {
			<-sig
			fmt.Fprintln(os.Stderr, "second signal — forcing exit")
			os.Exit(1)
		}()
		close(stopDemote)            // stop demoting before we checkpoint
		close(stopHaltPoll)          // stop polling -halt-file; nothing left to react to it after shutdown
		closeAdminSock()             // close + unlink the admin socket, if one was started
		srvCancel(errServerShutdown) // cancel in-flight generations (via BaseContext) so they release lm.mu
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(ctx)
		// What the CPU batcher did over this server's life: that decode tokens actually ran in batched steps, and how wide.
		for _, lm := range srv.modelList() {
			if lm.model != nil && lm.model.CPUBatchActive() {
				st := lm.model.CPUBatchStats()
				fmt.Fprintf(os.Stderr, "cpu batch %q: %d runs, %d batched steps (%d tokens), %d solo tokens, %d straggler runs, steps by size %v\n",
					lm.name, st.Runs, st.Steps, st.StepTokens, st.SoloTokens, st.StragglerRuns, st.StepSizes[:max(2, lm.concurrent+1)])
			}
			if lm.model != nil && lm.model.ResidentConcurrency() > 1 {
				// The resident batcher's counters, the same shape.
				st := lm.model.ResidentBatchStats()
				fmt.Fprintf(os.Stderr, "resident batch %q: %d runs, %d batched steps (%d tokens), %d solo tokens, %d straggler runs, steps by size %v\n",
					lm.name, st.Runs, st.Steps, st.StepTokens, st.SoloTokens, st.StragglerRuns, st.StepSizes[:max(2, lm.concurrent+1)])
				// The tokens a lone generation ran with the resident held (the chained decode, holdSolo): the record that
				// the greedy and sampled chains reach serve.
				fmt.Fprintf(os.Stderr, "resident batch %q held: %d tokens in %d holds\n", lm.name, st.HeldTokens, st.Holds)
				fmt.Fprintf(os.Stderr, "resident batch %q time: runs %.3f s, exclusive %.3f s (prefill %d passes %.3f s, bookkeeping %.3f s)\n",
					lm.name, float64(st.RunNs)/1e9, float64(st.ExclusiveNs)/1e9, st.PrefillPasses, float64(st.PrefillNs)/1e9,
					float64(st.ExclusiveNs-st.PrefillNs)/1e9)
			}
		}
		srv.checkpointSessions(cfg)
	}()
	return stopHaltPoll, done
}

// printServing writes the serving line, the banner, the two security warnings and the body caps to stderr.
func printServing(srv *server, cfg config, addr *string, authKey string, useTLS bool, textCap int64) {
	scheme := "http"
	if useTLS {
		scheme = "https"
	}
	fmt.Fprintf(os.Stderr, "goinfer serving on %s://%s [%s]\n", scheme, *addr, srv.endpointSummary())
	for _, line := range serverBanner(srv, cfg) {
		fmt.Fprintf(os.Stderr, "  %s\n", line)
	}
	// No -api-key: every route above is unauthenticated. Binding to loopback keeps
	// other MACHINES out, but not other TABS — any page open in a browser on this
	// machine can still fetch()/POST to it while it's running (the request is sent
	// regardless of CORS; CORS only gates whether the page can READ the response),
	// the classic "localhost drive-by" pattern. This is a deliberate product default
	// (no auth friction for the common single-user desktop case), not an oversight —
	// but it needs to be visible, not just documented, since most users never read
	// -h before running the one-liner from the README.
	if authKey == "" {
		fmt.Fprintln(os.Stderr, "warning: no -api-key set — any web page open in your browser can silently send requests to this API while it's running. Set -api-key (or $GOINFER_API_KEY) to require authentication.")
	}
	// A non-loopback bind with no TLS sends -api-key (a bearer token, every request)
	// and every prompt/completion in cleartext to anyone on the network path between
	// client and server — the -addr help text itself invites 0.0.0.0 exposure, so
	// this needs to be loud, not just in -h. Loopback is exempt: traffic never
	// leaves the machine, so there is no network path to sniff.
	if !useTLS && !addrIsLoopback(*addr) {
		fmt.Fprintln(os.Stderr, "warning: serving non-loopback with no TLS — -api-key and every prompt/completion travel in cleartext on the network. Set -tls-cert/-tls-key, or put a TLS-terminating reverse proxy in front.")
	}
	capSrc := "derived from context window"
	if cfg.maxBodyBytes > 0 {
		capSrc = "-max-body-bytes"
	}
	_, visionCap, embedCap, fileCap := srv.resolveBodyCaps(cfg.maxBodyBytes)
	fmt.Fprintf(os.Stderr, "request body cap: %s (text) / %s (vision) / %s (embeddings) / %s (batch files) [%s]\n", humanBytes(textCap), humanBytes(visionCap), humanBytes(embedCap), humanBytes(fileCap), capSrc)
}

// newServer loads the configured model(s): a decoder (with tokenizer, chat
// template, and KV sessions) and/or an encoder (with its tokenizer for token
// counting). At least one must be configured.
func newServer(cfg config) (*server, error) {
	if cfg.unloadDrainWait <= 0 {
		cfg.unloadDrainWait = 5 * time.Second // floor; the flag defaults here too. ?wait=false is the per-request path to an immediate 202.
	}
	// 256 matches responseStore's own bound (newResponseStore(256), just below) — no dedicated
	// -job-cap flag yet; a job past this many jobs old is only evicted once terminal (evictLocked).
	jobs, err := jobs.NewStore(cfg.jobDir, 256)
	if err != nil {
		return nil, fmt.Errorf("-job-dir %q: %w", cfg.jobDir, err)
	}
	s := &server{
		models:    map[string]*loadedModel{},
		speech:    map[string]*speechModel{},
		liveness:  map[io.Closer]*modelLiveness{},
		draining:  map[string]struct{}{},
		cfg:       cfg,
		responses: newResponseStore(256),
		gens:      newGenerationRegistry(),
		jobs:      jobs,
		// 256 matches the bounds above: no dedicated -file-cap/-batch-cap flag, and a record past this many old is only evicted
		// once terminal.
		files:   newFileStore(256, cfg.maxFileStoreBytes),
		batches: newBatchStore(256),
	}
	if !cfg.noCarryIDs {
		if s.idsKey, err = loadIDsKey(cfg.idsKeyFile); err != nil {
			return nil, err
		}
		if w := idsKeyFileWarning(runtime.GOOS, cfg.idsKeyFile); w != "" {
			fmt.Fprintln(os.Stderr, w)
		}
	}
	if cfg.logTokenIDs != "" {
		if s.tokenLog, err = openTokenIDLog(cfg.logTokenIDs); err != nil {
			return nil, err
		}
	}
	for _, spec := range cfg.models {
		if isWhisperDir(spec.path) { // an encoder-decoder speech model: its own registry and route (speech.go)
			sm, err := loadSpeech(spec, cfg)
			if err != nil {
				return nil, err
			}
			if _, dup := s.models[sm.name]; dup || s.speech[sm.name] != nil {
				return nil, fmt.Errorf("duplicate served model name %q (use --model name=path to disambiguate)", sm.name)
			}
			s.speech[sm.name] = sm
			continue
		}
		// An `hf:`/`demo:` spec is fetched (or found in the cache) inside loadDecoder (modelload.Resolve), before anything else,
		// so the served name derives from the real filename. A plain path is returned untouched, so no existing --model changes
		// meaning: that is what lets a reference form be added to a Hard-tier flag.
		// Startup load: context.Background() (a Ctrl-C during startup already ends the process). The admin load path passes the
		// request context so a disconnect cancels it.
		lm, err := loadDecoder(context.Background(), spec, cfg)
		if err != nil {
			return nil, err
		}
		if _, dup := s.models[lm.name]; dup {
			return nil, fmt.Errorf("duplicate served model name %q (use --model name=path to disambiguate)", lm.name)
		}
		s.models[lm.name] = lm
		s.retainLocked(lm.livenessKey()) // liveness refs (startup is single-threaded; no lock contention)
	}
	if err := s.loadAdapters(cfg); err != nil {
		return nil, err
	}
	if cfg.embedPath != "" {
		// An hf: reference is fetched (or found in the cache) first: hf:<repo>:safetensors as an encoder checkpoint, checked
		// against what loadEncoder can load before any weight byte; hf:<repo>:<quant> as a GGUF, the decoder-as-embedder.
		p, err := resolveEmbedModel(context.Background(), cfg.embedPath)
		if err != nil {
			return nil, fmt.Errorf("-embed-model %s: %w", cfg.embedPath, err)
		}
		cfg.embedPath = p
		if err := s.loadEncoder(cfg); err != nil {
			return nil, err
		}
	}
	if err := s.loadVisionTower(cfg); err != nil {
		return nil, err
	}
	share := &visionShare{n: len(s.models)} // loadVisionTower gives every startup entry the same tower
	for _, lm := range s.models {
		lm.visionShare = share
	}
	// Decided last, once adapters and vision towers are attached: a vision model runs one generation at a time.
	for _, lm := range s.models {
		if line := lm.setConcurrency(cfg); line != "" {
			fmt.Fprintf(os.Stderr, "%q %s\n", lm.name, line) // decided only now, so not part of its load banner
		}
	}
	// Armed last, after every startup load that could itself grow swap has finished: the guard's baseline should be steady
	// state after startup, not a mid-load reading that counts every startup byte as its own delta.
	s.startSwapGuard()
	return s, nil
}

// endpointSummary describes the registered endpoints for the startup banner.
func (s *server) endpointSummary() string {
	var parts []string
	if len(s.models) > 0 {
		names := slices.Sorted(maps.Keys(s.models))
		parts = append(parts, fmt.Sprintf("chat:[%s]", strings.Join(names, " ")))
	}
	if s.embed != nil {
		parts = append(parts, fmt.Sprintf("embeddings:%q", s.embedID))
	}
	return strings.Join(parts, " | ")
}

// tryLockUntil acquires mu, giving up at deadline instead of blocking forever, so the shutdown checkpoint can never
// deadlock on a generation that outlived the drain. Returns false if the lock was not taken by the deadline.
func tryLockUntil(mu *sync.Mutex, deadline time.Time) bool {
	for {
		if mu.TryLock() {
			return true
		}
		if !time.Now().Before(deadline) {
			return false
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// modelList snapshots the registry under regMu. Background sweeps (demote, shutdown checkpoint) must iterate this, not
// range s.models directly: admin load/unload mutate the map under regMu (admin.go), and a concurrent map iteration plus
// write is a runtime-fatal panic, not just a race. The returned slice is a copy of the pointers; the caller still locks
// each loadedModel via its own lm.mu.
func (s *server) modelList() []*loadedModel {
	s.regMu.RLock()
	defer s.regMu.RUnlock()
	out := make([]*loadedModel, 0, len(s.models))
	for _, lm := range s.models {
		out = append(out, lm)
	}
	return out
}

// demoteLoop periodically demotes idle KV sessions across all models to disk (tiered KV). It polls at a fraction of the
// idle threshold (clamped to [5s, 1m]) and takes each model's lock per sweep, so it stalls no in-flight generation and
// skips a busy model until its lock is free. Returns when stop is closed.
func demoteLoop(srv *server, idle time.Duration, stop <-chan struct{}) {
	period := min(max(idle/4, 5*time.Second), time.Minute)
	t := time.NewTicker(period)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case <-t.C:
			for _, lm := range srv.modelList() {
				lm.sessMu.Lock()
				if n := lm.sessions.DemoteIdle(); n > 0 {
					fmt.Fprintf(os.Stderr, "tiered-kv: demoted %d idle session(s) for %q\n", n, lm.name)
				}
				lm.sessMu.Unlock()
			}
		}
	}
}
