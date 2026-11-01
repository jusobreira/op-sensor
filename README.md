# OP Sensor — One Piece Card Game on stream

Chrome extension (Manifest V3) that recognises One Piece Card Game cards in videos
(YouTube, Twitch, Kick, Facebook, or any site you enable) and shows the card in high
resolution beside the mouse, with name/id/rarity, effect text and a price.
Architecture adapted (with the author's permission) from the Star Wars: Unlimited
"SWU Sensor" extension; the card database, embedding model and UI code are new.

## Status: you must build the database once

The extension will not recognise anything until `data/faces.json`, `data/protos.bin`
and `models/embed.onnx` exist. They are generated from the card list + official images:

```bash
pip install torch torchvision transformers timm onnx onnxruntime pillow numpy truststore certifi
git clone --depth 1 https://github.com/buhbbl/punk-records
python tools/build_database.py --punk-records ./punk-records --out .
```

The script prints a simulated top-1 accuracy and writes calibrated thresholds into
`faces.json`. Use `--backbone mobilenetv3` for a smaller/faster model, `--fp16` to halve
the model size, `--metadata-only` to test the card list without any ML.
Re-run after each new set (`git pull` in punk-records; images are cached).

Then: `chrome://extensions` → Developer mode → **Load unpacked** → this folder.

## Using it

Click the round Sensor button on a video, hover a card. Press **O** while hovering to
open the card in the selected store. Popup: language (Auto/EN/PT), price source
(Liga One Piece / TCGplayer / Cardmarket), currency (BRL/USD/EUR), card effect and price
toggles, and "work on any site with a video" (asks Chrome for all-sites permission).

## Known limits

- `models/det.onnx` (card detector) is carried over from the SWU extension. One Piece
  cards share the 63×88 mm format so it may work, but it was trained on SWU; test it on
  real streams. Retraining it needs labelled One Piece stream frames.
- Prices: only TCGplayer USD via the community API optcgapi.com (undocumented schema,
  incomplete coverage, **unverified**). Liga One Piece and Cardmarket have no public price
  API, so choosing them shows the TCGplayer price converted to your currency (marked
  "≈") and **O** opens the real store page. Exchange rates: open.er-api.com.
- Effect text is the official English text (punk-records has no Portuguese).
- The embedding model is generic, not fine-tuned, and alt-art printings look alike:
  expect confusion between a card and its parallel; accuracy on real video is unmeasured.
- Card data © Eiichiro Oda/Shueisha, Toei Animation, Bandai. Not affiliated with them.
