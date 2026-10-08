# Speech test clips

`librispeech-1272-128104-0000.wav`: LibriSpeech, dev-clean, speaker 1272, chapter 128104, utterance 0000, converted to
16 kHz mono 16-bit PCM WAV with ffmpeg (from the FLAC in `hf-internal-testing/librispeech_asr_dummy`, clean/validation).
Transcript: "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL".

LibriSpeech is licensed CC BY 4.0: V. Panayotov, G. Chen, D. Povey and S. Khudanpur, "LibriSpeech: an ASR corpus based
on public domain audio books", ICASSP 2015 (https://www.openslr.org/12).

## Derived clips for G-S5e (resampling)

Made from `librispeech-1272-128104-0000.wav` by `scripts/make_resample_clips.py`. The tools and versions are in
`librispeech-1272-128104-0000-resample.provenance.json`: ffmpeg 8.1.1, numpy 2.3.5, scipy 1.16.3.

- **`-44k1.wav`:** 44.1 kHz mono, 16-bit, made by ffmpeg's default resampler.
- **`-48k-stereo.wav`:** 48 kHz stereo, 16-bit. The channels are L = s + d and R = s - d, with d = 0.3 times the clip
  time-reversed.
- **`*.ref16k.f32`:** each of the two files back at 16 kHz, made by scipy's `resample_poly` (after the mean downmix,
  for the stereo file). Float32, little-endian.

They are LibriSpeech derivatives under the same CC BY 4.0 license.
