# Space Object Detector

A student-friendly AI project that classifies a space image into four classes:
**planet, star, moon, satellite**.

## Architecture

- **Python + TensorFlow/Keras** — train the image classifier.
- **TensorFlow.js** — run the trained model directly in the browser.
- **HTML/CSS/JavaScript** — user interface.
- **GitHub Pages** — static hosting.
- **Supabase (optional)** — store prediction history in PostgreSQL.

## 1. Prepare the dataset

Create:

```text
dataset/
├── planet/
├── star/
├── moon/
└── satellite/
```

Put labeled images in the matching folders.

## 2. Install dependencies

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Train

```bash
python train_model.py
```

This trains MobileNetV2 using transfer learning and saves a Keras model in `exported_model/`.

## 4. Export to TensorFlow.js

```bash
tensorflowjs_converter --input_format=keras exported_model/space_object_detector.keras model/
```

The `model/` folder should then contain `model.json` plus one or more `.bin` shard files.

## 5. Configure Supabase (optional)

Run `supabase/schema.sql` in the Supabase SQL editor. Then copy `config.example.js` to `config.js` and set:

```js
window.APP_CONFIG = {
  MODEL_URL: "model/model.json",
  CLASS_NAMES: ["planet", "star", "moon", "satellite"],
  SUPABASE_URL: "https://YOUR_PROJECT.supabase.co",
  SUPABASE_PUBLISHABLE_KEY: "YOUR_PUBLISHABLE_KEY"
};
```

Use a publishable/anon client key only in the browser. Never put a Supabase service-role secret in `config.js`.

## 6. Run locally

Because browsers may block model files when opening `index.html` directly, run a local web server:

```bash
python -m http.server 8000
```

Open `http://localhost:8000`.

## 7. Deploy to GitHub Pages

Commit the files to a GitHub repository and enable GitHub Pages for the repository. The project is static and does not require a Node backend for inference.

## Important

The repository includes the full application and training pipeline, but it does **not** include a trained space-object model because that model must be trained on your labeled dataset. Until `model/model.json` and its shard files exist, the web page will show “Model not found”.
