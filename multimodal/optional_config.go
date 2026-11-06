package multimodal

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
)

// readOptionalConfig reads a checkpoint file a preprocess config may do without; found is false when it is absent. Any other failure
// is an error: defaults in place of a config that exists would size images differently with no word to anyone.
func readOptionalConfig(dir, name string) (raw []byte, found bool, err error) {
	raw, err = os.ReadFile(filepath.Join(dir, name))
	if errors.Is(err, fs.ErrNotExist) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	return raw, true, nil
}
