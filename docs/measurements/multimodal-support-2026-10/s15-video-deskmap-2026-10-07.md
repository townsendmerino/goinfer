# S15 Gate 0: how video works in five families, read from transformers' own code (2026-10-07)

This is the desk map behind S15 in `docs/tasks/task-multimodal-support-2026-10.md`.
- **What was read:** the installed transformers 5.16.1 source, by a research agent in the Mac session. One prompt was
  checked by running `Qwen3VLProcessor`'s own token replacement against the shipped chat template.
- **Citation paths:** `T/` is `site-packages/transformers/`, `VPU` is `T/video_processing_utils.py`, and `VU` is
  `T/video_utils.py`. Line numbers are 5.16.1's.
- **Not confirmed:** whether two Qwen quirks below are intended. No other reference was read.

## Qwen2.5-VL
- **Native video.** `Qwen2VLVideoProcessor` (`T/models/auto/video_processing_auto.py:62`); token `<|video_pad|>`
  (`processing_qwen2_5_vl.py:46`); `get_video_features` runs the image tower (`modeling_qwen2_5_vl.py:1063-1070`).
- **Sampling.**
  - `do_sample_frames=False` by default (`video_processing_qwen2_vl.py:129`).
  - With fps: 4-768 frames, floored to even (`:210-213`), indices `arange(0,total,total/n).int()` (`:222`).
  - The budget is per frame: `smart_resize` with factor 28, shortest_edge 128·28², longest_edge 28²·768 (`:117`,
    `:43-69`, `:252-258`).
  - Tokens are grid_t·grid_h·grid_w/4 (`processing_qwen2_5_vl.py:64-67`).
  - `cap_pixels_per_frame` becomes the default in v5.22 (`:324-333`), which changes the counts.
- **Temporal patching.** An odd frame count is padded with the last frame repeated (`:276-279`). The patch layout is
  (channel, temporal, py, px) (`:284-301`). The patch layer is a `Conv3d` with kernel and stride [2,14,14], no bias
  (`modeling_qwen2_5_vl.py:113-121`). Each temporal group is its own vision attention segment (`T/vision_utils.py:62-65`);
  the vision RoPE is (h,w), repeated per t (`:120-125`).
- **Positions.**
  - `second_per_grid_ts = temporal_patch_size / sampled_fps` (`processing_qwen2_5_vl.py:143-153`); fps falls back to
    24 (`VU:109-113`).
  - The t step is `tokens_per_second * int(second_per_grid_t)` (`modeling_qwen2_5_vl.py:1043`). Above 2 sampled fps
    `int()` gives 0, so every group shares one t.
  - `tokens_per_second` is 2 in the local 3B config.
  - After a video, `current_pos` advances by max(h,w)/2, not the t extent (`:1050`).
- **The prompt.** `<|vision_start|><|video_pad|><|vision_end|>` (with an optional `Video N: ` prefix), the pad
  expanded; no timestamps.
- **No DeepStack.**

## Qwen3-VL
- **Native video.** `video_processing_qwen3_vl.py`; `get_video_features` is the image path
  (`modeling_qwen3_vl.py:1028-1035`).
- **Sampling.**
  - `do_sample_frames=True`, fps 2, 4-768 frames (`:121-134`).
  - Count `int(total/video_fps*2)` (`:178-179`), indices `linspace(0,total-1,n).round()` (`:184`).
  - The budget is per video, `t·h·w` against `max_pixels`, factor 32 (`:97-106`); the local 2B caps it at about
    12,288 tokens.
  - Fewer than 2 frames is an error (`:84-85`).
- **The patch layer.** `Conv3d` [2,16,16] with bias (`modeling_qwen3_vl.py:96-104`); the same padding and layout.
- **Timestamps are text.**
  - Each frame pair is `f"<{t:.1f} seconds>"` + `<|vision_start|>` + pads + `<|vision_end|>`
    (`processing_qwen3_vl.py:103-106`), t the pair's mean time (`:178-189`).
  - In the positions each pair is a t=1 grid (`modeling_qwen3_vl.py:966-969`).
- **The doubled wrapper.**
  - The shipped template emits `<|vision_start|><|video_pad|><|vision_end|>`, and the processor replaces only the pad
    (`T/processing_utils.py:879-921`).
  - The result is a doubled outer wrapper, simulated for 30 frames at 10 fps:
    `<|vision_start|><0.3 seconds><|vision_start|>pad×4<|vision_end|>…<|vision_end|>`.
  - `get_rope_index`'s docstring shows no outer pair (`:943`).
- **DeepStack covers video** (`:1189`, `:1198-1218`, `:840-861`).

## Qwen3.5
- **The same processor and template as Qwen3-VL** (`video_processing_auto.py:63`, `processing_auto.py:71`); video token
  id 248057.
- **The tower:** `Conv3d` at `modeling_qwen3_5.py:846-861`; the per-pair grid split at `:1328`.
- **No DeepStack** (`modular_qwen3_5.py:152`, `:442-443`).

## Gemma 4
- **Native video, confirmed.**
  - `Gemma4VideoProcessor` (`video_processing_gemma4.py:155`); token `<|video|>` (`processing_gemma4.py:84-86`).
  - `get_video_features` runs `vision_tower` then `embed_vision` per frame (`modeling_gemma4.py:2465-2490`), scattered
    in with `masked_scatter` (`:2340-2356`).
  - No m-RoPE.
- **Sampling.** 32 frames, `do_sample_frames=True` (`:165-166`); `arange(0,total,total/32).int()`, and fewer than 32
  frames is an error (`VPU:177-185`).
- **Budget.** `max_soft_tokens` 70 per frame, one of {70,140,280,560,1120}; pooling 3, at most 630 patches per frame
  (`:64`, `:168-169`, `:241`).
- **The prompt.** Each frame is `f"{MM:02d}:{SS:02d} <|image>{<|video|>×n}<image|>"`, joined with spaces
  (`processing_gemma4.py:185-189`). The template emits a bare `<|video|>`.

## Gemma 3
- **No video.** No video processor, no `pixel_values_videos`. Frames go in as images (`\n\n<boi>`+256+`<eoi>\n\n`,
  `processing_gemma3.py:55-56`).

## goinfer's own notes
- **The v1 scope:** no video; "no decoding of video containers in v1" (`docs/multimodal.md`); goinfer's `mropePositions`
  matches HF for t=1 only.
- **F5d:** the swapped patch-conv halves are invisible on a still image ("it would matter for video").
- **aikit:** `vision/qwen3_mmproj.go` recombines the split Conv3d halves.

## The peers' same-size pairing
- **llama.cpp #24303 (closed):**
  - 4 images in a row, read as 2, through `mtmd`'s `can_merge_with()` fusing adjacent same-size images into a
    temporal super-frame.
  - Text between the images avoids it; fixed between b10360 and b10520.
- **Ollama #17814 (open, v0.32.13):**
  - Two same-size images to qwen3.5 collapse into one (`prompt_eval` 307 against 565), silently. gemma4 is not
    affected.

## Containers in pure Go
- **Animated GIF:** the standard library's `image/gif.DecodeAll`, which returns frames plus delays. Frames are
  sub-rectangles composited by their disposal method; bound memory with `DecodeConfig` first.
- **mp4/H.264:**
  - Pure-Go demuxers (`abema/go-mp4`, `Eyevinn/mp4ff`; unverified), but no known mature pure-Go H.264 decoder.
  - So it needs cgo (FFmpeg through `go-astiav`, openh264, VideoToolbox or NVDEC) or a WASM decoder under wazero.
