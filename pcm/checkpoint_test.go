package pcm

import (
	"bytes"
	"encoding/binary"
	"errors"
	"io"
	"testing"

	flac "github.com/tphakala/go-flac"
	"github.com/tphakala/go-flac/internal/crc"
)

// readOneFrame reads exactly one frame's worth of PCM: Decoder.Read decodes a
// single frame per call when p can hold it, so a large enough p leaves the
// decoder at a frame boundary on return.
func readOneFrame(t *testing.T, d *Decoder, p []byte) (int, error) {
	t.Helper()
	n, err := d.Read(p)
	if n == 0 && err == nil {
		t.Fatal("Read returned 0 bytes without an error")
	}
	return n, err
}

// encodeCheckpointStream encodes genPCM output for cfg, exercising the
// streaming encoder so metadata sections (SEEKTABLE, VORBIS_COMMENT) land in
// the stream when cfg requests them.
func encodeCheckpointStream(t *testing.T, cfg Config, samples int) (stream, pcmBytes []byte) {
	t.Helper()
	pcmBytes = genPCM(cfg, samples)
	var sb seekBuffer
	enc, err := NewEncoder(&sb, cfg)
	if err != nil {
		t.Fatalf("NewEncoder: %v", err)
	}
	if _, err := enc.Write(pcmBytes); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err := enc.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	return sb.Bytes(), pcmBytes
}

// decodeStreamAll decodes stream to EOF and returns the PCM.
func decodeStreamAll(t *testing.T, stream []byte, opts ...DecoderOption) []byte {
	t.Helper()
	dec, err := NewDecoder(bytes.NewReader(stream), opts...)
	if err != nil {
		t.Fatalf("NewDecoder: %v", err)
	}
	out, err := io.ReadAll(dec)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	return out
}

// checkFrameSync verifies the parsed checkpoint offset points at a frame sync
// in the source (or exactly at end of stream).
func checkFrameSync(t *testing.T, stream []byte, cp *checkpoint) {
	t.Helper()
	if cp.frameOffset == int64(len(stream)) {
		return // end-of-stream checkpoint
	}
	if cp.frameOffset+2 > int64(len(stream)) || stream[cp.frameOffset] != 0xFF || stream[cp.frameOffset+1]&0xFC != 0xF8 {
		t.Fatalf("checkpoint offset %d does not point at a frame sync", cp.frameOffset)
	}
}

// TestCheckpointResumeMatchesContinuous checkpoints a decode at every frame
// boundary and verifies that a decoder resumed from each checkpoint yields
// exactly the remaining PCM of a continuous decode, across block counts, bit
// depths, channel counts, compression levels, and metadata sections.
func TestCheckpointResumeMatchesContinuous(t *testing.T) {
	configs := []Config{
		{SampleRate: 44100, Channels: 2, BitDepth: 16, CompressionLevel: 0},
		{SampleRate: 44100, Channels: 2, BitDepth: 16, CompressionLevel: 8,
			SeekTableInterval: 4096, SeekTableMaxPoints: 16,
			Tags: []flac.Tag{{Name: tagTitle, Value: "checkpoint"}}},
		{SampleRate: 48000, Channels: 1, BitDepth: 24, CompressionLevel: 5},
		{SampleRate: 8000, Channels: 2, BitDepth: 8, CompressionLevel: 2},
	}
	const samples = 3*4096 + 100 // several full frames plus a short final frame
	for _, cfg := range configs {
		stream, pcmBytes := encodeCheckpointStream(t, cfg, samples)
		reference := decodeStreamAll(t, stream)
		if !bytes.Equal(reference, pcmBytes) {
			t.Fatalf("%+v: continuous decode does not round-trip", cfg)
		}

		dec, err := NewDecoder(bytes.NewReader(stream))
		if err != nil {
			t.Fatalf("%+v: NewDecoder: %v", cfg, err)
		}
		// A checkpoint before the first frame is a valid stream-start position.
		type savedCkpt struct {
			data     []byte
			consumed int
			decoded  uint64
		}
		var ckpts []savedCkpt
		ck, err := dec.Checkpoint()
		if err != nil {
			t.Fatalf("%+v: Checkpoint at stream start: %v", cfg, err)
		}
		ckpts = append(ckpts, savedCkpt{ck, 0, 0})

		buf := make([]byte, 256<<10)
		consumed := 0
		for {
			n, rerr := readOneFrame(t, dec, buf)
			consumed += n
			if errors.Is(rerr, io.EOF) {
				break
			}
			if rerr != nil {
				t.Fatalf("%+v: Read: %v", cfg, rerr)
			}
			ck, err := dec.Checkpoint()
			if err != nil {
				t.Fatalf("%+v: Checkpoint after %d bytes: %v", cfg, consumed, err)
			}
			cp, err := parseCheckpoint(ck)
			if err != nil {
				t.Fatalf("%+v: parseCheckpoint: %v", cfg, err)
			}
			checkFrameSync(t, stream, cp)
			ckpts = append(ckpts, savedCkpt{ck, consumed, cp.decoded})
		}
		// A checkpoint at end of stream is valid too.
		ck, err = dec.Checkpoint()
		if err != nil {
			t.Fatalf("%+v: Checkpoint at end of stream: %v", cfg, err)
		}
		ckpts = append(ckpts, savedCkpt{ck, consumed, 0})

		for i, sc := range ckpts {
			resumed, err := NewDecoderFromCheckpoint(bytes.NewReader(stream), sc.data)
			if err != nil {
				t.Fatalf("%+v: checkpoint %d: NewDecoderFromCheckpoint: %v", cfg, i, err)
			}
			if resumed.Info() != dec.Info() {
				t.Fatalf("%+v: checkpoint %d: Info differs after resume", cfg, i)
			}
			rest, err := io.ReadAll(resumed)
			if err != nil {
				t.Fatalf("%+v: checkpoint %d: Read after resume: %v", cfg, i, err)
			}
			if !bytes.Equal(rest, reference[sc.consumed:]) {
				t.Fatalf("%+v: checkpoint %d: resumed PCM differs from continuous decode (%d of %d bytes)",
					cfg, i, len(rest), len(reference)-sc.consumed)
			}
		}
	}
}

// TestCheckpointNonDestructive verifies that taking a checkpoint does not
// disturb the decoder it was taken from.
func TestCheckpointNonDestructive(t *testing.T) {
	cfg := Config{SampleRate: 44100, Channels: 2, BitDepth: 16, CompressionLevel: 5}
	stream, _ := encodeCheckpointStream(t, cfg, 2*4096+7)
	reference := decodeStreamAll(t, stream)

	dec, err := NewDecoder(bytes.NewReader(stream))
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 256<<10)
	var got []byte
	for {
		n, rerr := readOneFrame(t, dec, buf)
		got = append(got, buf[:n]...)
		if _, err := dec.Checkpoint(); err != nil && !errors.Is(rerr, io.EOF) {
			t.Fatalf("Checkpoint: %v", err)
		}
		if errors.Is(rerr, io.EOF) {
			break
		}
		if rerr != nil {
			t.Fatalf("Read: %v", rerr)
		}
	}
	if !bytes.Equal(got, reference) {
		t.Fatal("decode interleaved with Checkpoint calls differs from continuous decode")
	}
}

// buildVBSStream builds a variable-blocksize stream by hand: STREAMINFO plus
// one constant-subframe stereo frame per entry of blocks, with the coded
// sample number of each frame. It returns the stream and the expected PCM.
// The MD5 is left zero so the decoder's end-of-stream sample-count check (not
// the MD5) guards the total, which exercises the restored decoded count.
func buildVBSStream(t *testing.T, blocks []int) (stream, pcmBytes []byte) {
	t.Helper()
	var total uint64
	maxBlock := 0
	for _, b := range blocks {
		total += uint64(b)
		maxBlock = max(maxBlock, b)
	}

	out := []byte("fLaC")
	out = append(out, 0x80, 0x00, 0x00, 0x22) // last block, STREAMINFO, 34 bytes
	body := make([]byte, 0, 34)
	put16 := func(v int) { body = append(body, byte(v>>8), byte(v)) }
	put24 := func(v int) { body = append(body, byte(v>>16), byte(v>>8), byte(v)) }
	put16(16)       // min block size
	put16(maxBlock) // max block size
	put24(0)        // min frame size
	put24(0)        // max frame size
	var packed uint64
	packed = uint64(44100) << 44
	packed |= uint64(2-1) << 41
	packed |= uint64(16-1) << 36
	packed |= total & 0xFFFFFFFFF
	for i := 7; i >= 0; i-- {
		body = append(body, byte(packed>>(uint(i)*8)))
	}
	body = append(body, make([]byte, 16)...) // MD5 zero: sample-count check instead
	out = append(out, body...)

	var sampleNo uint64
	for fi, bs := range blocks {
		v0 := int16(fi*1000 + 1)
		v1 := -v0
		frame := buildVBSFrame(sampleNo, bs, v0, v1)
		out = append(out, frame...)
		for range bs {
			pcmBytes = append(pcmBytes, byte(v0), byte(uint16(v0)>>8), byte(v1), byte(uint16(v1)>>8))
		}
		sampleNo += uint64(bs)
	}
	return out, pcmBytes
}

// buildVBSFrame builds one variable-blocksize frame (blocking strategy bit set,
// coded sample number) with two constant subframes.
func buildVBSFrame(sampleNo uint64, blockSize int, v0, v1 int16) []byte {
	// sync, variable blocking, bsCode 7, stereo, bps from STREAMINFO
	hdrBody := append(make([]byte, 0, 16), 0xFF, 0xF9, 0x70, 0x10)
	hdrBody = append(hdrBody, codedNumber(sampleNo)...)
	bs := blockSize - 1
	hdrBody = append(hdrBody, byte(bs>>8), byte(bs))
	hdr := make([]byte, len(hdrBody)+1)
	copy(hdr, hdrBody)
	hdr[len(hdrBody)] = crc.Checksum8(hdrBody)

	w := &bitWriter{}
	for _, v := range []int16{v0, v1} {
		w.put(0, 1) // zero-pad bit
		w.put(0, 6) // subframe type CONSTANT
		w.put(0, 1) // wasted bits flag
		w.put(uint64(uint16(v)), 16)
	}
	body := w.bytes()

	full := make([]byte, 0, len(hdr)+len(body)+2)
	full = append(full, hdr...)
	full = append(full, body...)
	c16 := crc.Checksum16(full)
	return append(full, byte(c16>>8), byte(c16))
}

// codedNumber encodes a frame/sample number in the stream's UTF-8-like coding.
func codedNumber(v uint64) []byte {
	switch {
	case v < 0x80:
		return []byte{byte(v)}
	case v < 0x800:
		return []byte{0xC0 | byte(v>>6), 0x80 | byte(v&0x3F)}
	case v < 0x10000:
		return []byte{0xE0 | byte(v>>12), 0x80 | byte((v>>6)&0x3F), 0x80 | byte(v&0x3F)}
	default:
		return []byte{0xF0 | byte(v>>18), 0x80 | byte((v>>12)&0x3F), 0x80 | byte((v>>6)&0x3F), 0x80 | byte(v&0x3F)}
	}
}

// TestCheckpointResumeVariableBlockSize resumes a hand-built variable-blocksize
// stream from every frame boundary and compares against the continuous decode.
func TestCheckpointResumeVariableBlockSize(t *testing.T) {
	// Mixed block sizes push the coded sample number past the 1-byte and 2-byte
	// boundaries (sample numbers 0, 100, 150, 350).
	stream, reference := buildVBSStream(t, []int{100, 50, 200, 7})

	dec, err := NewDecoder(bytes.NewReader(stream))
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 64<<10)
	consumed := 0
	for {
		ck, err := dec.Checkpoint()
		if err != nil {
			t.Fatalf("Checkpoint after %d bytes: %v", consumed, err)
		}
		cp, err := parseCheckpoint(ck)
		if err != nil {
			t.Fatal(err)
		}
		checkFrameSync(t, stream, cp)

		resumed, err := NewDecoderFromCheckpoint(bytes.NewReader(stream), ck)
		if err != nil {
			t.Fatalf("resume after %d bytes: %v", consumed, err)
		}
		rest, err := io.ReadAll(resumed)
		if err != nil {
			t.Fatalf("Read after resume at %d bytes: %v", consumed, err)
		}
		if !bytes.Equal(rest, reference[consumed:]) {
			t.Fatalf("resumed PCM at %d bytes differs from continuous decode", consumed)
		}

		n, rerr := readOneFrame(t, dec, buf)
		consumed += n
		if errors.Is(rerr, io.EOF) {
			break
		}
		if rerr != nil {
			t.Fatalf("Read: %v", rerr)
		}
	}
	if consumed != len(reference) {
		t.Fatalf("decoded %d PCM bytes, want %d", consumed, len(reference))
	}
}

// TestCheckpointMidFrameRejected verifies Checkpoint refuses a position with
// unread samples and accepts the position once the frame is drained.
func TestCheckpointMidFrameRejected(t *testing.T) {
	cfg := Config{SampleRate: 44100, Channels: 2, BitDepth: 16}
	stream, _ := encodeCheckpointStream(t, cfg, 2*4096)

	dec, err := NewDecoder(bytes.NewReader(stream))
	if err != nil {
		t.Fatal(err)
	}
	// Read less than one frame: the decoder sits mid-frame.
	small := make([]byte, 16)
	if _, err := dec.Read(small); err != nil {
		t.Fatalf("Read: %v", err)
	}
	if _, err := dec.Checkpoint(); !errors.Is(err, flac.ErrCheckpointNotFrameBoundary) {
		t.Fatalf("Checkpoint mid-frame: got %v, want ErrCheckpointNotFrameBoundary", err)
	}
	// Drain the rest of the frame; the boundary is reachable again.
	if _, err := io.Copy(io.Discard, io.LimitReader(dec, 4096*2*2-16)); err != nil {
		t.Fatalf("drain: %v", err)
	}
	if _, err := dec.Checkpoint(); err != nil {
		t.Fatalf("Checkpoint at frame boundary after drain: %v", err)
	}
}

// TestCheckpointParseErrors covers the malformed-checkpoint verdicts.
func TestCheckpointParseErrors(t *testing.T) {
	cfg := Config{SampleRate: 44100, Channels: 2, BitDepth: 16}
	stream, _ := encodeCheckpointStream(t, cfg, 2*4096)
	valid := func() []byte {
		dec, err := NewDecoder(bytes.NewReader(stream))
		if err != nil {
			t.Fatal(err)
		}
		ck, err := dec.Checkpoint()
		if err != nil {
			t.Fatal(err)
		}
		return ck
	}()

	t.Run("truncated", func(t *testing.T) {
		for _, n := range []int{0, 7, 8, 65, len(valid) - 1} {
			_, err := NewDecoderFromCheckpoint(bytes.NewReader(stream), valid[:n])
			if !errors.Is(err, flac.ErrCheckpointCorrupt) || !errors.Is(err, io.ErrUnexpectedEOF) {
				t.Fatalf("len %d: got %v, want ErrCheckpointCorrupt wrapping io.ErrUnexpectedEOF", n, err)
			}
		}
	})
	t.Run("bad magic", func(t *testing.T) {
		ck := bytes.Clone(valid)
		ck[0] = 'X'
		if _, err := NewDecoderFromCheckpoint(bytes.NewReader(stream), ck); !errors.Is(err, flac.ErrCheckpointCorrupt) {
			t.Fatalf("got %v, want ErrCheckpointCorrupt", err)
		}
	})
	t.Run("version", func(t *testing.T) {
		ck := bytes.Clone(valid)
		ck[8] = checkpointVersion + 1
		if _, err := NewDecoderFromCheckpoint(bytes.NewReader(stream), ck); !errors.Is(err, flac.ErrCheckpointVersion) {
			t.Fatalf("got %v, want ErrCheckpointVersion", err)
		}
	})
	t.Run("trailing garbage", func(t *testing.T) {
		ck := append(bytes.Clone(valid), 0x00)
		if _, err := NewDecoderFromCheckpoint(bytes.NewReader(stream), ck); !errors.Is(err, flac.ErrCheckpointCorrupt) {
			t.Fatalf("got %v, want ErrCheckpointCorrupt", err)
		}
	})
	t.Run("unknown flags", func(t *testing.T) {
		ck := bytes.Clone(valid)
		ck[9] |= 0x80
		if _, err := NewDecoderFromCheckpoint(bytes.NewReader(stream), ck); !errors.Is(err, flac.ErrCheckpointCorrupt) {
			t.Fatalf("got %v, want ErrCheckpointCorrupt", err)
		}
	})
}

// withFrameOffset returns a copy of ck with the frame offset field adjusted.
func withFrameOffset(ck []byte, delta int64) []byte {
	out := bytes.Clone(ck)
	binary.BigEndian.PutUint64(out[24:32], uint64(int64(binary.BigEndian.Uint64(out[24:32]))+delta))
	return out
}

// TestCheckpointSourceMismatch covers the verdicts when the source no longer
// matches the checkpoint: a changed source, a shifted offset, and a frame that
// fails its CRC at the resume point.
func TestCheckpointSourceMismatch(t *testing.T) {
	cfg := Config{SampleRate: 44100, Channels: 2, BitDepth: 16}
	stream, _ := encodeCheckpointStream(t, cfg, 3*4096)

	// Checkpoint after the first frame.
	dec, err := NewDecoder(bytes.NewReader(stream))
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 256<<10)
	if _, err := readOneFrame(t, dec, buf); err != nil {
		t.Fatal(err)
	}
	ck, err := dec.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	cp, err := parseCheckpoint(ck)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("different audio", func(t *testing.T) {
		other, _ := encodeCheckpointStream(t, Config{SampleRate: 48000, Channels: 2, BitDepth: 16}, 3*4096)
		if _, err := NewDecoderFromCheckpoint(bytes.NewReader(other), ck); !errors.Is(err, flac.ErrCheckpointMismatch) {
			t.Fatalf("got %v, want ErrCheckpointMismatch", err)
		}
	})
	t.Run("changed metadata section", func(t *testing.T) {
		// Same PCM but a VORBIS_COMMENT block shifts every frame offset.
		cfgTags := cfg
		cfgTags.Tags = []flac.Tag{{Name: "TITLE", Value: "changed"}}
		tagged, _ := encodeCheckpointStream(t, cfgTags, 3*4096)
		if _, err := NewDecoderFromCheckpoint(bytes.NewReader(tagged), ck); !errors.Is(err, flac.ErrCheckpointMismatch) {
			t.Fatalf("got %v, want ErrCheckpointMismatch", err)
		}
	})
	t.Run("offset not at frame boundary", func(t *testing.T) {
		if _, err := NewDecoderFromCheckpoint(bytes.NewReader(stream), withFrameOffset(ck, 1)); !errors.Is(err, flac.ErrCheckpointNotFrameBoundary) {
			t.Fatalf("got %v, want ErrCheckpointNotFrameBoundary", err)
		}
	})
	t.Run("frame CRC failure at resume", func(t *testing.T) {
		corrupt := bytes.Clone(stream)
		corrupt[cp.frameOffset+8] ^= 0xFF // inside the resumed frame's header/body
		if _, err := NewDecoderFromCheckpoint(bytes.NewReader(corrupt), ck); !errors.Is(err, flac.ErrCRCMismatch) {
			t.Fatalf("got %v, want ErrCRCMismatch", err)
		}
	})
	t.Run("source truncated before offset", func(t *testing.T) {
		cut := stream[:cp.frameOffset-4]
		if _, err := NewDecoderFromCheckpoint(bytes.NewReader(cut), ck); !errors.Is(err, flac.ErrCheckpointMismatch) {
			t.Fatalf("got %v, want ErrCheckpointMismatch", err)
		}
	})
	t.Run("non-seekable source", func(t *testing.T) {
		type noSeek struct{ io.Reader }
		if _, err := NewDecoderFromCheckpoint(noSeek{bytes.NewReader(stream)}, ck); !errors.Is(err, flac.ErrSeekUnsupported) {
			t.Fatalf("got %v, want ErrSeekUnsupported", err)
		}
	})
}

// TestCheckpointMD5VerifiedAfterResume proves the running MD5 state crosses the
// checkpoint: a resumed decode of an MD5-signed stream still verifies at end of
// stream, which a lost or restarted hash would fail.
func TestCheckpointMD5VerifiedAfterResume(t *testing.T) {
	cfg := Config{SampleRate: 44100, Channels: 2, BitDepth: 16}
	stream, _ := encodeCheckpointStream(t, cfg, 3*4096)
	dec, err := NewDecoder(bytes.NewReader(stream))
	if err != nil {
		t.Fatal(err)
	}
	var zero [16]byte
	if dec.Info().MD5 == zero {
		t.Fatal("test stream must carry a nonzero STREAMINFO MD5")
	}
	buf := make([]byte, 256<<10)
	if _, err := readOneFrame(t, dec, buf); err != nil {
		t.Fatal(err)
	}
	ck, err := dec.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	resumed, err := NewDecoderFromCheckpoint(bytes.NewReader(stream), ck)
	if err != nil {
		t.Fatal(err)
	}
	// A restarted (empty) MD5 state would mismatch the STREAMINFO signature at
	// end of stream; reaching EOF cleanly proves the state was carried over.
	if _, err := io.ReadAll(resumed); err != nil {
		t.Fatalf("resumed decode failed end-of-stream MD5 verification: %v", err)
	}
}

// TestCheckpointAfterSeek verifies a checkpoint taken after a seek keeps the
// seeked policy (no end-of-stream MD5 check) and resumes at the right frame.
func TestCheckpointAfterSeek(t *testing.T) {
	cfg := Config{SampleRate: 44100, Channels: 2, BitDepth: 16}
	const samples = 3 * 4096
	stream, _ := encodeCheckpointStream(t, cfg, samples)

	// Continuous reference: seek into the second frame, then read to the end.
	const target = 4096 + 100
	ref, err := NewDecoder(bytes.NewReader(stream))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ref.SeekToSample(target); err != nil {
		t.Fatal(err)
	}
	refPCM, err := io.ReadAll(ref)
	if err != nil {
		t.Fatal(err)
	}

	// Same seek, read one frame, checkpoint, resume, read the rest.
	dec, err := NewDecoder(bytes.NewReader(stream))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := dec.SeekToSample(target); err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 256<<10)
	n, err := readOneFrame(t, dec, buf)
	if err != nil {
		t.Fatal(err)
	}
	ck, err := dec.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	resumed, err := NewDecoderFromCheckpoint(bytes.NewReader(stream), ck)
	if err != nil {
		t.Fatal(err)
	}
	rest, err := io.ReadAll(resumed)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]byte, 0, n+len(rest))
	got = append(got, buf[:n]...)
	got = append(got, rest...)
	if !bytes.Equal(got, refPCM) {
		t.Fatal("resumed decode after seek differs from continuous decode")
	}
}

// TestCheckpointFromNonSeekableSource verifies a checkpoint taken on a
// non-seekable source (offset relative to the stream start) resumes against a
// seekable reader of the same stream.
func TestCheckpointFromNonSeekableSource(t *testing.T) {
	cfg := Config{SampleRate: 44100, Channels: 2, BitDepth: 16}
	stream, _ := encodeCheckpointStream(t, cfg, 2*4096)
	reference := decodeStreamAll(t, stream)

	type noSeek struct{ io.Reader }
	dec, err := NewDecoder(noSeek{bytes.NewReader(stream)})
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 256<<10)
	n, err := readOneFrame(t, dec, buf)
	if err != nil {
		t.Fatal(err)
	}
	ck, err := dec.Checkpoint()
	if err != nil {
		t.Fatalf("Checkpoint on a non-seekable source: %v", err)
	}
	resumed, err := NewDecoderFromCheckpoint(bytes.NewReader(stream), ck)
	if err != nil {
		t.Fatal(err)
	}
	rest, err := io.ReadAll(resumed)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(rest, reference[n:]) {
		t.Fatal("resumed PCM differs from continuous decode")
	}
}

// TestCheckpointConvertTo verifies a resumed decoder honors fresh options: the
// checkpoint carries no output-format state, so WithConvertTo applies to the
// resumed decode exactly as to a continuous one.
func TestCheckpointConvertTo(t *testing.T) {
	cfg := Config{SampleRate: 44100, Channels: 2, BitDepth: 16}
	stream, _ := encodeCheckpointStream(t, cfg, 2*4096)
	reference := decodeStreamAll(t, stream, WithConvertTo(24))

	dec, err := NewDecoder(bytes.NewReader(stream))
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]byte, 256<<10)
	n, err := readOneFrame(t, dec, buf)
	if err != nil {
		t.Fatal(err)
	}
	ck, err := dec.Checkpoint()
	if err != nil {
		t.Fatal(err)
	}
	resumed, err := NewDecoderFromCheckpoint(bytes.NewReader(stream), ck, WithConvertTo(24))
	if err != nil {
		t.Fatal(err)
	}
	if resumed.OutputBitDepth() != 24 {
		t.Fatalf("OutputBitDepth = %d, want 24", resumed.OutputBitDepth())
	}
	rest, err := io.ReadAll(resumed)
	if err != nil {
		t.Fatal(err)
	}
	// The first frame was read at the native depth; convert it for the prefix.
	native := buf[:n]
	conv, err := NewDecoder(bytes.NewReader(stream), WithConvertTo(24))
	if err != nil {
		t.Fatal(err)
	}
	prefix := make([]byte, 256<<10)
	m, err := conv.Read(prefix)
	if err != nil {
		t.Fatal(err)
	}
	_ = native
	if !bytes.Equal(append(prefix[:m], rest...), reference) {
		t.Fatal("resumed converted PCM differs from continuous converted decode")
	}
}
