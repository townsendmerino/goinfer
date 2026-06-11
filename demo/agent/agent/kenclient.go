package agent

import (
	"context"
	"fmt"
	"os/exec"
	"strings"

	sdk "github.com/modelcontextprotocol/go-sdk/mcp"
)

// kenClient wraps an MCP stdio session to a ken server subprocess (the
// go-stdlib demo binary, or any ken-mcp). The JSON the model emits in the
// DECIDE phase becomes a tools/call on this session — the same wire format
// any MCP agent (Claude, Cursor, …) would send ken.
type kenClient struct {
	sess  *sdk.ClientSession
	tools []string
	topK  int
}

// dialKen spawns the server binary and performs the MCP handshake.
func dialKen(ctx context.Context, bin string, topK int) (*kenClient, error) {
	cli := sdk.NewClient(&sdk.Implementation{Name: "stdlib-agent", Version: "0.1.0"}, nil)
	sess, err := cli.Connect(ctx, &sdk.CommandTransport{Command: exec.Command(bin)}, nil)
	if err != nil {
		return nil, fmt.Errorf("connect %s: %w", bin, err)
	}
	kc := &kenClient{sess: sess, topK: topK}
	if res, err := sess.ListTools(ctx, nil); err == nil {
		for _, t := range res.Tools {
			kc.tools = append(kc.tools, t.Name)
		}
	}
	return kc, nil
}

// search forwards a query to ken's `search` tool and returns the markdown
// result block (file:line-cited chunks, semble-compatible shape).
func (k *kenClient) search(ctx context.Context, query string) (string, error) {
	res, err := k.sess.CallTool(ctx, &sdk.CallToolParams{
		Name: "search",
		Arguments: map[string]any{
			"query": query,
			"top_k": k.topK,
		},
	})
	if err != nil {
		return "", err
	}
	if res.IsError {
		return "", fmt.Errorf("ken search returned an error: %s", contentText(res.Content))
	}
	text := contentText(res.Content)
	if strings.TrimSpace(text) == "" {
		return "", fmt.Errorf("ken search returned no content")
	}
	return text, nil
}

func (k *kenClient) close() {
	if k.sess != nil {
		_ = k.sess.Close()
	}
}

// contentText concatenates the text parts of an MCP content slice.
func contentText(content []sdk.Content) string {
	var b strings.Builder
	for _, c := range content {
		if t, ok := c.(*sdk.TextContent); ok {
			b.WriteString(t.Text)
		}
	}
	return b.String()
}
