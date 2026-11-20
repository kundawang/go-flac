package pcm

import (
	"bytes"
	"io"
	"testing"
)

// FuzzCheckpointRestore feeds mutated checkpoint bytes (and, through the
// trailing argument, a mutated resume offset) to NewDecoderFromCheckpoint
// against a fixed valid stream. The checkpoint parser and the restore path
// treat the bytes as untrusted: they must never panic, and a successful
// restore must decode the remaining stream without a panic either. The stream
// is small, so draining the resumed decoder stays bounded.
func FuzzCheckpointRestore(f *testing.F) {
	cfg := Config{SampleRate: 44100, Channels: 2, BitDepth: 16}
	stream, _ := encodeCheckpointStreamF(cfg, 2*4096+13)

	dec, err := NewDecoder(bytes.NewReader(stream))
	if err != nil {
		f.Fatal(err)
	}
	buf := make([]byte, 256<<10)
	if _, err := dec.Read(buf); err != nil {
		f.Fatal(err)
	}
	valid, err := dec.Checkpoint()
	if err != nil {
		f.Fatal(err)
	}
	f.Add(valid)
	f.Add([]byte{})
	f.Add([]byte(checkpointMagic))
	mut := bytes.Clone(valid)
	mut[30] ^= 0xFF // frame offset low byte region
	f.Add(mut)

	f.Fuzz(func(t *testing.T, data []byte) {
		resumed, err := NewDecoderFromCheckpoint(bytes.NewReader(stream), data)
		if err != nil {
			return
		}
		// A restore that succeeded must drain without a panic; errors from the
		// mutated position (CRC, truncation, MD5) are all legitimate.
		_, _ = io.ReadAll(resumed)
	})
}

// encodeCheckpointStreamF is encodeCheckpointStream for the fuzz target, which
// has no *testing.T at seed time.
func encodeCheckpointStreamF(cfg Config, samples int) (stream, pcmBytes []byte) {
	pcmBytes = genPCM(cfg, samples)
	var sb seekBuffer
	enc, err := NewEncoder(&sb, cfg)
	if err != nil {
		panic(err)
	}
	if _, err := enc.Write(pcmBytes); err != nil {
		panic(err)
	}
	if err := enc.Close(); err != nil {
		panic(err)
	}
	return sb.Bytes(), pcmBytes
}
