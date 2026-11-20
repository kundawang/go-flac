package pcm

import (
	"encoding"
	"encoding/binary"
	"errors"
	"fmt"
	"io"

	flac "github.com/tphakala/go-flac"
)

// Decoder checkpoints let a caller suspend a long decode and resume it later,
// possibly in another process, without re-decoding from the start of the
// stream. Checkpoint captures the decoder's position at a frame boundary as a
// small versioned byte slice; NewDecoderFromCheckpoint rebuilds a decoder at
// that position against a fresh source. The serialized form carries only
// decoder state and a byte offset: the underlying reader and any file handle
// are never serialized, and the resumed decoder reads from the source it is
// given.
//
// The format is versioned (checkpointVersion) and stable within a major
// release; a checkpoint written by an incompatible version is rejected with
// flac.ErrCheckpointVersion rather than misinterpreted.
const checkpointVersion = 1

// checkpointMagic identifies a go-flac decoder checkpoint ("goFlacCp").
const checkpointMagic = "goFlacCp"

const (
	checkpointFlagSeeked    = 1 << 0 // a seek happened before the checkpoint
	checkpointFlagMD5Active = 1 << 1 // the MD5 state is present and live
)

// checkpointHeaderLen is the fixed portion of the serialized checkpoint:
// magic(8) version(1) flags(1) channels(1) bitDepth(1) sampleRate(4)
// totalSamples(8) frameOffset(8) decoded(8) metaLen(8) streamMD5(16)
// md5StateLen(2). The variable MD5 state follows.
const checkpointHeaderLen = 8 + 1 + 1 + 1 + 1 + 4 + 8 + 8 + 8 + 8 + 16 + 2

// checkpoint is the parsed form of a serialized decoder checkpoint.
type checkpoint struct {
	seeked       bool     // a seek happened; end-of-stream MD5 and count checks stay off
	md5Active    bool     // md5State holds a live running hash
	channels     int      // STREAMINFO identity: channels
	bitDepth     int      // STREAMINFO identity: bit depth
	sampleRate   int      // STREAMINFO identity: sample rate
	totalSamples uint64   // STREAMINFO identity: declared total samples
	frameOffset  int64    // absolute source offset of the next frame to decode
	decoded      uint64   // inter-channel samples decoded so far
	metaLen      uint64   // metadata region length; pins the layout the offsets assume
	streamMD5    [16]byte // STREAMINFO MD5 signature
	md5State     []byte   // marshaled crypto/md5 state (only when md5Active)
}

// Checkpoint serializes the decoder's position so a later
// NewDecoderFromCheckpoint can resume decoding from the same frame without
// re-reading the stream from the start. It must be called at a frame boundary:
// after NewDecoder or Reset, after Read or WriteTo has consumed every sample
// of the frames decoded so far, or at end of stream. If decoded samples remain
// unread (a Read returned a short count and the rest were not drained), the
// decoder sits mid-frame and Checkpoint returns an error wrapping
// flac.ErrCheckpointNotFrameBoundary. A decoder in a failed state returns its
// sticky error.
//
// The returned bytes are self-contained and safe to store or transmit: they
// hold the decoder's position, the running MD5 state, and a fingerprint of the
// stream's identity, never the source reader or a file handle. The byte offset
// is absolute for a seekable source; for a non-seekable source it is relative
// to the stream's first byte, and resuming then requires a source positioned
// identically (for example the same file opened from the start).
func (d *Decoder) Checkpoint() ([]byte, error) {
	if d.err != nil {
		return nil, d.err
	}
	if d.br == nil {
		return nil, errors.New("go-flac/pcm: Checkpoint: decoder has no stream (NewDecoder or Reset failed)")
	}
	if len(d.pending) != 0 {
		return nil, fmt.Errorf("go-flac/pcm: Checkpoint: %d unread bytes sit mid-frame: %w", len(d.pending), flac.ErrCheckpointNotFrameBoundary)
	}
	// md5Active mirrors the condition decodeNextFrame hashes under, so the
	// resumed decoder's hash continues the same running state.
	md5Active := !d.seeked && !d.skipMD5
	var md5State []byte
	if md5Active {
		// d.md5 is always a crypto/md5 digest, which implements
		// encoding.BinaryMarshaler; the assertion cannot fail.
		m, ok := d.md5.(encoding.BinaryMarshaler)
		if !ok {
			return nil, errors.New("go-flac/pcm: Checkpoint: MD5 hash state is not serializable")
		}
		var err error
		md5State, err = m.MarshalBinary()
		if err != nil {
			return nil, fmt.Errorf("go-flac/pcm: Checkpoint: serialize MD5 state: %w", err)
		}
	}
	var flags byte
	if d.seeked {
		flags |= checkpointFlagSeeked
	}
	if md5Active {
		flags |= checkpointFlagMD5Active
	}
	out := make([]byte, 0, checkpointHeaderLen+len(md5State))
	out = append(out, checkpointMagic...)
	out = append(out, checkpointVersion, flags, byte(d.info.Channels), byte(d.info.BitDepth))
	out = binary.BigEndian.AppendUint32(out, uint32(d.info.SampleRate))
	out = binary.BigEndian.AppendUint64(out, d.info.TotalSamples)
	out = binary.BigEndian.AppendUint64(out, uint64(d.br.BytesRead()))
	out = binary.BigEndian.AppendUint64(out, d.decoded)
	out = binary.BigEndian.AppendUint64(out, uint64(d.metaLen))
	out = append(out, d.info.MD5[:]...)
	out = binary.BigEndian.AppendUint16(out, uint16(len(md5State)))
	out = append(out, md5State...)
	return out, nil
}

// NewDecoderFromCheckpoint resumes decoding a stream from a checkpoint written
// by Decoder.Checkpoint. r must be a source for the same stream the checkpoint
// was taken on, positioned at the stream's first byte as in NewDecoder, and it
// must be an io.Seeker so the decoder can position itself at the checkpoint
// offset; a non-seekable source fails with flac.ErrSeekUnsupported. The
// checkpoint bytes are untrusted input: a truncated or malformed checkpoint
// fails with flac.ErrCheckpointCorrupt, an incompatible format version with
// flac.ErrCheckpointVersion, and a source whose STREAMINFO, MD5, or metadata
// layout differs from the checkpoint's stream (the source changed, or the
// checkpoint belongs to another stream) with flac.ErrCheckpointMismatch. A
// checkpoint offset that does not land on a frame sync fails with
// flac.ErrCheckpointNotFrameBoundary, and a frame that fails its CRC check at
// the resume point fails with flac.ErrCRCMismatch.
//
// The resumed decoder decodes the frame at the checkpoint offset immediately,
// so those failures surface here rather than on the first Read. From then on
// it yields exactly the PCM a continuous decode would have produced from that
// point, sample for sample, for fixed- and variable-blocksize streams alike.
// When the checkpoint carries a live MD5 state, end-of-stream verification
// still runs over the whole stream (the state covers the frames decoded before
// the checkpoint). A checkpoint taken under SkipMD5Verification cannot verify
// the stream MD5 after resume, so the resumed decoder keeps skipping it even
// without the option. Options (SkipMD5Verification, WithConvertTo) apply to
// this call only, as with NewDecoder.
func NewDecoderFromCheckpoint(r io.Reader, checkpoint []byte, opts ...DecoderOption) (*Decoder, error) {
	cp, err := parseCheckpoint(checkpoint)
	if err != nil {
		return nil, err
	}
	d := &Decoder{}
	if err := d.init("NewDecoderFromCheckpoint", r); err != nil {
		return nil, err
	}
	if err := d.applyOptions("NewDecoderFromCheckpoint", opts); err != nil {
		return nil, err
	}
	if err := d.restore(cp); err != nil {
		return nil, err
	}
	return d, nil
}

// parseCheckpoint decodes and validates the serialized form. The value checks
// are limited to what the format itself guarantees; consistency with a
// concrete source is verified by restore.
func parseCheckpoint(data []byte) (*checkpoint, error) {
	if len(data) < checkpointHeaderLen {
		return nil, fmt.Errorf("go-flac/pcm: checkpoint: %d bytes, want at least %d: %w: %w",
			len(data), checkpointHeaderLen, flac.ErrCheckpointCorrupt, io.ErrUnexpectedEOF)
	}
	if string(data[:len(checkpointMagic)]) != checkpointMagic {
		return nil, fmt.Errorf("go-flac/pcm: checkpoint: bad magic: %w", flac.ErrCheckpointCorrupt)
	}
	if v := data[8]; v != checkpointVersion {
		return nil, fmt.Errorf("go-flac/pcm: checkpoint: format version %d, this build reads %d: %w",
			v, checkpointVersion, flac.ErrCheckpointVersion)
	}
	flags := data[9]
	if flags&^(checkpointFlagSeeked|checkpointFlagMD5Active) != 0 {
		return nil, fmt.Errorf("go-flac/pcm: checkpoint: unknown flags %#x: %w", flags, flac.ErrCheckpointCorrupt)
	}
	md5Len := int(binary.BigEndian.Uint16(data[64:66]))
	switch have := len(data) - checkpointHeaderLen; {
	case have < md5Len:
		return nil, fmt.Errorf("go-flac/pcm: checkpoint: MD5 state truncated (%d of %d bytes): %w: %w",
			have, md5Len, flac.ErrCheckpointCorrupt, io.ErrUnexpectedEOF)
	case have > md5Len:
		return nil, fmt.Errorf("go-flac/pcm: checkpoint: %d trailing bytes: %w", have-md5Len, flac.ErrCheckpointCorrupt)
	}
	cp := &checkpoint{
		seeked:       flags&checkpointFlagSeeked != 0,
		md5Active:    flags&checkpointFlagMD5Active != 0,
		channels:     int(data[10]),
		bitDepth:     int(data[11]),
		sampleRate:   int(binary.BigEndian.Uint32(data[12:16])),
		totalSamples: binary.BigEndian.Uint64(data[16:24]),
		frameOffset:  int64(binary.BigEndian.Uint64(data[24:32])),
		decoded:      binary.BigEndian.Uint64(data[32:40]),
		metaLen:      binary.BigEndian.Uint64(data[40:48]),
		md5State:     data[66:],
	}
	copy(cp.streamMD5[:], data[48:64])
	if cp.md5Active && md5Len == 0 || !cp.md5Active && md5Len != 0 {
		return nil, fmt.Errorf("go-flac/pcm: checkpoint: MD5 state flag and length %d disagree: %w", md5Len, flac.ErrCheckpointCorrupt)
	}
	// The frame offset is absolute and the metadata region sits at the stream
	// start, so a valid offset is never below the metadata length.
	if cp.frameOffset < 0 || cp.frameOffset < int64(cp.metaLen) {
		return nil, fmt.Errorf("go-flac/pcm: checkpoint: frame offset %d below metadata region %d: %w",
			cp.frameOffset, cp.metaLen, flac.ErrCheckpointCorrupt)
	}
	return cp, nil
}

// restore positions an initialized decoder at the checkpoint and validates the
// landing by decoding the first frame. cp was already checked against the
// format invariants; this checks it against the concrete source.
func (d *Decoder) restore(cp *checkpoint) error {
	if !d.seekable {
		// Resuming means positioning the source at the checkpoint offset, which
		// only an io.Seeker can do.
		return fmt.Errorf("go-flac/pcm: NewDecoderFromCheckpoint: resume requires a seekable source: %w", flac.ErrSeekUnsupported)
	}
	// The checkpoint is only meaningful for the stream it was taken on. The
	// STREAMINFO fields and MD5 pin the audio; the metadata region length pins
	// the layout the absolute offset assumes, so an edited metadata section is
	// caught here too instead of landing mid-frame.
	if cp.sampleRate != d.info.SampleRate || cp.channels != d.info.Channels ||
		cp.bitDepth != d.info.BitDepth || cp.totalSamples != d.info.TotalSamples ||
		cp.streamMD5 != d.info.MD5 || cp.metaLen != uint64(d.metaLen) {
		return fmt.Errorf("go-flac/pcm: NewDecoderFromCheckpoint: stream identity differs from the checkpoint: %w", flac.ErrCheckpointMismatch)
	}
	if cp.frameOffset < d.audioStart || cp.frameOffset > d.streamEnd {
		return fmt.Errorf("go-flac/pcm: NewDecoderFromCheckpoint: offset %d outside the audio region [%d, %d]: %w",
			cp.frameOffset, d.audioStart, d.streamEnd, flac.ErrCheckpointMismatch)
	}
	// A frame begins with the 14-bit sync 0b11111111111110, so the first two
	// bytes are 0xFF and 0b111110xx. Checking them here distinguishes "the
	// offset is not a frame boundary" from a frame that merely fails CRC.
	// Fewer than 2 bytes remaining is left to the landing decode below, which
	// reports the cut exactly as a continuous decode would.
	if d.streamEnd-cp.frameOffset >= 2 {
		var sync [2]byte
		if _, err := d.rs.Seek(cp.frameOffset, io.SeekStart); err != nil {
			return fmt.Errorf("go-flac/pcm: NewDecoderFromCheckpoint: seek to checkpoint offset: %w", err)
		}
		if _, err := io.ReadFull(d.rs, sync[:]); err != nil {
			return fmt.Errorf("go-flac/pcm: NewDecoderFromCheckpoint: read frame sync: %w", err)
		}
		if sync[0] != 0xFF || sync[1]&0xFC != 0xF8 {
			return fmt.Errorf("go-flac/pcm: NewDecoderFromCheckpoint: offset %d does not start a frame: %w",
				cp.frameOffset, flac.ErrCheckpointNotFrameBoundary)
		}
	}
	if _, err := d.rs.Seek(cp.frameOffset, io.SeekStart); err != nil {
		return fmt.Errorf("go-flac/pcm: NewDecoderFromCheckpoint: seek to checkpoint offset: %w", err)
	}
	d.br.ResetAt(d.rs, cp.frameOffset)
	d.decoded = cp.decoded
	d.seeked = cp.seeked
	switch {
	case cp.md5Active:
		// d.md5 is a crypto/md5 digest (reset by init); the assertion cannot fail.
		u, ok := d.md5.(encoding.BinaryUnmarshaler)
		if !ok {
			return errors.New("go-flac/pcm: NewDecoderFromCheckpoint: MD5 hash state is not restorable")
		}
		if err := u.UnmarshalBinary(cp.md5State); err != nil {
			return fmt.Errorf("go-flac/pcm: NewDecoderFromCheckpoint: MD5 state: %w: %w", err, flac.ErrCheckpointCorrupt)
		}
	case !cp.seeked:
		// The checkpoint was taken under SkipMD5Verification, so the frames
		// before it were never hashed and the running state cannot be
		// reconstructed. Keep skipping, or the end-of-stream check would
		// compare a hash of only the resumed tail against the STREAMINFO MD5.
		d.skipMD5 = true
	}
	// Decode the first frame now: a source corrupted or shifted since the
	// checkpoint fails its CRC here, at resume time, instead of on the first
	// Read. A clean end of stream (a checkpoint taken after the last frame)
	// runs the end-of-stream checks through decodeNextFrame and is not an
	// error.
	if err := d.decodeNextFrame(); err != nil && !errors.Is(err, io.EOF) {
		return err
	}
	return nil
}
