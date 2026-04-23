# User Image Demo

Place a single demo image here as:

- `demo_input.png`

Then run:

```bash
python -m detectivesam_inference.predict
```

When only a target image is available, the CLI reuses that image as its own reference source. That keeps the demo runnable, but it is not the canonical pairwise evaluation mode the model was trained for.

