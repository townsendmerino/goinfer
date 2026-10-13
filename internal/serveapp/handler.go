package serveapp

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/townsendmerino/goinfer/decoder"
)

// listenerFlags are the goinfer-serve flags that configure its listener, its process or its operator surfaces. They
// have no meaning for a handler served in process, so NewHandler refuses them by name.
var listenerFlags = []string{"addr", "tls-cert", "tls-key", "allowed-hosts", "admin-socket", "allow-admin", "web", "halt-file", "halt-exit-code", "version"}

// NewHandler loads the models the arguments name and returns goinfer-serve's HTTP API as a handler, with no
// listener: the routes, the auth check, the halt gate and the body caps are the ones the binary serves. args are
// goinfer-serve's own flags ("-model", "m.gguf", "-backend", "metal"), parsed by the same flag set, so a setting means
// here what it means on the command line. The flags that configure the binary's listener and process (listenerFlags)
// are refused. -api-key is honoured when given; the environment is not read for one.
//
// Close on the returned io.Closer cancels the generations in flight, checkpoints sessions when -session-dir is set,
// and unloads every model. The handler must not be used after it.
func NewHandler(args []string) (http.Handler, io.Closer, error) {
	fs := flag.NewFlagSet("goinfer", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	sf := registerFlags(fs)
	if err := fs.Parse(args); err != nil {
		return nil, nil, err
	}
	if rest := fs.Args(); len(rest) > 0 {
		return nil, nil, fmt.Errorf("unrecognized argument %q: every argument is a goinfer-serve flag", rest[0])
	}
	var refused []string
	fs.Visit(func(f *flag.Flag) {
		for _, name := range listenerFlags {
			if f.Name == name {
				refused = append(refused, "-"+name)
			}
		}
	})
	if len(refused) > 0 {
		return nil, nil, fmt.Errorf("%s configure goinfer-serve's listener or process and do not apply to an in-process handler", strings.Join(refused, ", "))
	}
	markGivenFlags(fs, &sf.cfg)
	cfg := sf.cfg
	if cfg.noSelfTest {
		decoder.SkipSelfTests()
	}
	cfg.load = *sf.lf
	if err := cfg.load.Validate(); err != nil {
		return nil, nil, err
	}
	if err := requireAutoBackend(cfg); err != nil {
		return nil, nil, err
	}
	if len(cfg.models) == 0 && cfg.embedPath == "" {
		return nil, nil, errors.New("need at least one of -model and -embed-model")
	}
	if err := cfg.validate(); err != nil {
		return nil, nil, err
	}
	srv, err := newServer(resolveGemma3VisionQuant(cfg, os.Stderr))
	if err != nil {
		return nil, nil, err
	}
	srv.restoreSessions(cfg)
	var reqLog io.Writer
	if cfg.logRequests {
		reqLog = os.Stderr
	}
	mux, _ := srv.routes(cfg, *sf.apiKey, reqLog)
	// Every request's context is a child of life, as the binary's are of its server context, so Close cancels the
	// generations in flight instead of waiting for them.
	life, stop := context.WithCancel(context.Background())
	h := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx, cancel := context.WithCancel(r.Context())
		defer cancel()
		defer context.AfterFunc(life, cancel)()
		mux.ServeHTTP(w, r.WithContext(ctx))
	})
	return h, &handlerCloser{srv: srv, cfg: cfg, stop: stop}, nil
}

type handlerCloser struct {
	srv  *server
	cfg  config
	stop context.CancelFunc
}

// Close ends the handler's life: see NewHandler. It reports the models still draining when the wait ran out.
func (c *handlerCloser) Close() error {
	c.stop()
	c.srv.checkpointSessions(c.cfg)
	var busy []string
	for _, lm := range c.srv.modelList() {
		if status, _, ok := c.srv.unloadByName(lm.name, c.cfg.unloadDrainWait); ok && status != http.StatusOK {
			busy = append(busy, lm.name)
		}
	}
	if len(busy) > 0 {
		return fmt.Errorf("goinfer: %s still had requests in flight after %s; their memory is released when those finish", strings.Join(busy, ", "), c.cfg.unloadDrainWait)
	}
	return nil
}
