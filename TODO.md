# VSX (PPC64) upstreaming — push notes

Upstreaming the `ppc64-split` branch to libwebp's Gerrit, in chunks.

## How Gerrit push works (NOT a force push)

- Push to `refs/for/main` (the magic review ref), never to `main` directly.
- Gerrit matches each commit to a change by its `Change-Id:` trailer and creates
  a new patchset only where the commit actually changed.
- Pushing a tip walks its parent chain and opens one change per *new* commit;
  commits already on Gerrit with identical hashes get nothing added.

```
git push https://chromium.googlesource.com/webm/libwebp <tip-hash>:refs/for/main
```

## Already on Gerrit (do not need re-pushing)

Hashes below are byte-identical to what's on Gerrit; a re-push adds nothing.

| Change  | Commit     | Subject                              |
|---------|------------|--------------------------------------|
| —       | `b2e17fc9` | build support                        |
| —       | `8b97a5f3` | CPU detection                        |
| 7901972 | `1a118087` | optimizations for the decoder        |
| 7901973 | `6018b0f5` | YUV->RGB conversion                  |
| 7901974 | `a643978d` | fancy upsampling                     |

(Original decoder change 7882704 was abandoned; 7901972 replaced it with a
fresh Change-Id `I262ce82…`.)

## Remaining to push (in order, parents first)

★ = carries a clang/ppc64le build fix folded in.

| # | Commit     | Subject                                        |
|---|------------|------------------------------------------------|
| 1 | `b3a46f2f` | rescaler optimizations                         |
| 2 | `2f002f3b` | alpha processing optimizations                 |
| 3 | `ad86ceb7` | alpha-filter optimizations ★                   |
| 4 | `ef7b7818` | lossless decoding optimizations ★              |
| 5 | `ea0fb4d1` | lossless encoding optimizations                |
| 6 | `90c536cc` | cost optimizations                             |
| 7 | `ceca5816` | distortion (SSIM/SSE) optimizations ★          |
| 8 | `590a7348` | encoder distortion metrics ★                   |
| 9 | `5544aa91` | encoder transforms                             |
|10 | `b1480568` | encoder texture distortion and histogram       |
|11 | `805552b3` | encoder quantization                           |
|12 | `5725c261` | encoder intra predictors                       |
|13 | `2d81681e` | 24-bit RGB/BGR YUV conversion                  |
|14 | `3b9e5103` | RGB565 / RGBA4444 YUV upsampling               |

### Next chunk (suggested: through lossless-decoding)

Finishes the decode/lossless-decode side; includes the alpha-filter + lossless
fixes. Opens 4 new changes stacked on the upsampling change:

```
git push https://chromium.googlesource.com/webm/libwebp ef7b7818:refs/for/main
```

To push everything remaining at once, use the branch tip instead:

```
git push https://chromium.googlesource.com/webm/libwebp ppc64-split:refs/for/main
```

Swap the hash for any row above to push more or less.

## Notes

- The clang fixes (ambiguous `vec_splats`, dead lossless helpers) are folded into
  their feature commits on both `ppc64-split` and `ppc64`; trees are identical.
- Verified on a real POWER9 ppc64le host (clang 22): clean build, bit-exact
  lossless roundtrip.
- If hashes drift after a future rebase, regenerate this list with:
  `git log a643978d..ppc64-split --reverse --format='%h | %s'`
