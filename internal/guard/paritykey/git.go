package paritykey

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// Git reads commits' files through one `git cat-file --batch` process in the repository at Dir ("" is the current
// directory). Close it when done.
type Git struct {
	dir string
	cmd *exec.Cmd
	in  io.WriteCloser
	out *bufio.Reader
}

// OpenGit starts the reader.
func OpenGit(dir string) (*Git, error) {
	cmd := exec.Command("git", "cat-file", "--batch")
	cmd.Dir = dir
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &Git{dir: dir, cmd: cmd, in: in, out: bufio.NewReaderSize(out, 1<<20)}, nil
}

// Close stops the reader.
func (g *Git) Close() error {
	g.in.Close()
	return g.cmd.Wait()
}

// At is a Source over one commit.
func (g *Git) At(commit string) Source { return gitCommit{g, commit} }

type gitCommit struct {
	g      *Git
	commit string
}

func (c gitCommit) ReadFile(path string) ([]byte, error) {
	if _, err := fmt.Fprintf(c.g.in, "%s:%s\n", c.commit, path); err != nil {
		return nil, err
	}
	hdr, err := c.g.out.ReadString('\n')
	if err != nil {
		return nil, err
	}
	f := strings.Fields(hdr)
	if len(f) == 2 && f[1] == "missing" {
		return nil, fmt.Errorf("%s:%s: %w", c.commit, path, os.ErrNotExist)
	}
	if len(f) != 3 {
		return nil, fmt.Errorf("git cat-file %s:%s: %q", c.commit, path, hdr)
	}
	n, err := strconv.Atoi(f[2])
	if err != nil {
		return nil, err
	}
	b := make([]byte, n+1)
	if _, err := io.ReadFull(c.g.out, b); err != nil {
		return nil, err
	}
	return b[:n], nil
}

func (c gitCommit) ListDir(path string) ([]string, error) {
	cmd := exec.Command("git", "ls-tree", "--name-only", c.commit+":"+path)
	cmd.Dir = c.g.dir
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("%s:%s: %w", c.commit, path, os.ErrNotExist)
	}
	return strings.Fields(string(out)), nil
}
