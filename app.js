let model = null;
let selectedFile = null;

const cfg = window.APP_CONFIG || {};

const $ = (id) => document.getElementById(id);

const input = $("imageInput");
const preview = $("preview");
const previewWrap = $("previewWrap");
const predictBtn = $("predictBtn");
const randomBtn = $("randomBtn");
const modelStatus = $("modelStatus");
const errorBox = $("error");

const CLASS_NAMES = [
  "planet",
  "star",
  "nebulae",
  "galaxy"
];

const SPACE_INFO = {
  planet: {
    icon: "🪐",
    fact: "Planets are large celestial bodies that orbit a star or another astronomical object.",
    explore:
      "Planets come in many different types. Some are rocky like Earth, while others are huge gas or ice giants."
  },

  star: {
    icon: "⭐",
    fact: "Stars produce energy through nuclear fusion in their cores. Our Sun is also a star.",
    explore:
      "Stars can have very different sizes, temperatures, and lifetimes. Their light can travel across enormous distances through space."
  },

  nebulae: {
    icon: "🌌",
    fact: "Nebulae are huge clouds of gas and dust. Some nebulae are places where new stars are born.",
    explore:
      "Nebulae can have incredible shapes and colors. They are made mostly of gas and dust spread across huge regions of space."
  },

  galaxy: {
    icon: "🌠",
    fact: "A galaxy can contain billions or even trillions of stars, together with gas, dust, and dark matter.",
    explore:
      "The Milky Way is the galaxy that contains our Solar System. Galaxies can have different shapes, including spiral and elliptical forms."
  }
};


function setError(message) {
  errorBox.textContent = message;
  errorBox.classList.toggle("hidden", !message);
}


function iconFor(label) {
  return SPACE_INFO[label]?.icon || "🔭";
}


function renderClasses(values = []) {

  $("classGrid").innerHTML = CLASS_NAMES.map((name, i) => {

    const value = Number(values[i] || 0);

    return `
      <div class="class-item">
        <div class="class-name">
          ${iconFor(name)} ${name}
        </div>

        <div class="class-value">
          ${(value * 100).toFixed(1)}%
        </div>
      </div>
    `;
  }).join("");
}


function showSpaceInfo(label) {

  const info = SPACE_INFO[label];

  if (!info) return;

  $("factTitle").textContent =
    `${info.icon} Space Fact`;

  $("factText").textContent =
    info.fact;

  $("exploreTitle").textContent =
    `${info.icon} Explore ${label}`;

  $("exploreText").textContent =
    info.explore;

  $("exploreCard").classList.remove("hidden");
}


async function loadModel() {

  try {

    model = await tf.loadGraphModel(
      cfg.MODEL_URL || "model/model.json"
    );

    modelStatus.textContent =
      "Model ready";

    modelStatus.style.color =
      "#8df7c0";

    renderClasses();

    console.log(
      "TensorFlow.js model loaded successfully"
    );

    console.log(
      "Input shape:",
      model.inputs[0].shape
    );

    console.log(
      "Output shape:",
      model.outputs[0].shape
    );

  } catch (err) {

    console.error(err);

    modelStatus.textContent =
      "Model not found";

    setError(
      "Could not load the AI model. Make sure model/model.json and all .bin files are present."
    );
  }
}


function showPreview(src) {

  preview.src = src;

  previewWrap.classList.remove(
    "hidden"
  );

  predictBtn.disabled =
    !model;
}


input.addEventListener(
  "change",
  () => {

    setError("");

    selectedFile =
      input.files?.[0] || null;

    if (!selectedFile) {

      predictBtn.disabled =
        true;

      return;
    }

    showPreview(
      URL.createObjectURL(
        selectedFile
      )
    );
  }
);


async function getRandomImage() {

  const allImages = [];

  for (const className of CLASS_NAMES) {

    try {

      const response =
        await fetch(
          `dataset/${className}/`
        );

      if (!response.ok) continue;

      const html =
        await response.text();

      const parser =
        new DOMParser();

      const doc =
        parser.parseFromString(
          html,
          "text/html"
        );

      const links =
        [...doc.querySelectorAll("a")];

      for (const link of links) {

        const href =
          link.getAttribute("href");

        if (!href) continue;

        if (
          /\.(jpg|jpeg|png|webp|gif)$/i.test(
            href
          )
        ) {

          allImages.push({
            url:
              `dataset/${className}/${href}`,
            name:
              href
          });
        }
      }

    } catch (err) {

      console.warn(
        `Could not read ${className}`,
        err
      );
    }
  }

  if (!allImages.length) {

    throw new Error(
      "No images found in dataset folders."
    );
  }

  return allImages[
    Math.floor(
      Math.random() *
      allImages.length
    )
  ];
}


randomBtn.addEventListener(
  "click",
  async () => {

    randomBtn.disabled =
      true;

    randomBtn.textContent =
      "🔄 Loading...";

    setError("");

    try {

      const image =
        await getRandomImage();

      selectedFile = {
        name: image.name
      };

      showPreview(
        image.url
      );

      $("prediction").textContent =
        "—";

      $("predictionLabel").textContent =
        "Ready to analyze";

      $("confidence").textContent =
        "Confidence: —";

      $("confidenceBar").style.width =
        "0%";

      $("resultIcon").textContent =
        "🔭";

      $("exploreCard")
        .classList.add("hidden");

      $("factTitle").textContent =
        "🌌 Space Fact";

      $("factText").textContent =
        "Click Detect object to analyze this image.";

      renderClasses();

    } catch (err) {

      console.error(err);

      setError(
        "Could not find an image in the dataset."
      );

    } finally {

      randomBtn.disabled =
        false;

      randomBtn.textContent =
        "🎲 Random image";
    }
  }
);


predictBtn.addEventListener(
  "click",
  async () => {

    if (!selectedFile || !model)
      return;

    predictBtn.disabled =
      true;

    randomBtn.disabled =
      true;

    predictBtn.textContent =
      "🔍 Scanning...";

    setError("");

    previewWrap.classList.add(
      "scanning"
    );

    try {

      const image =
        await new Promise(
          (resolve, reject) => {

            const img =
              new Image();

            img.onload = () =>
              resolve(img);

            img.onerror =
              reject;

            img.src =
              preview.src;
          }
        );


      /*
       * MobileNetV2 expects
       * pixel values in [-1, 1].
       */

      const tensor =
        tf.tidy(() =>
          tf.browser
            .fromPixels(image)
            .resizeBilinear([
              224,
              224
            ])
            .toFloat()
            .div(127.5)
            .sub(1)
            .expandDims(0)
        );


      const output =
        model.predict(tensor);

      const prediction =
        Array.isArray(output)
          ? output[0]
          : output;

      const values =
        Array.from(
          await prediction.data()
        );


      tensor.dispose();

      if (
        Array.isArray(output)
      ) {

        output.forEach(
          (t) => t.dispose()
        );

      } else {

        prediction.dispose();
      }


      const maxIndex =
        values.indexOf(
          Math.max(...values)
        );


      const label =
        CLASS_NAMES[maxIndex] ||
        `class ${maxIndex}`;


      const confidence =
        values[maxIndex] || 0;


      $("prediction").textContent =
        label;

      $("predictionLabel").textContent =
        "Detection complete";

      $("confidence").textContent =
        `Confidence: ${(confidence * 100).toFixed(1)}%`;

      $("confidenceBar").style.width =
        `${Math.min(
          confidence * 100,
          100
        )}%`;

      $("resultIcon").textContent =
        iconFor(label);


      renderClasses(
        values
      );

      showSpaceInfo(
        label
      );


      await savePrediction(
        label,
        confidence,
        selectedFile.name
      );


    } catch (err) {

      console.error(err);

      setError(
        "Could not analyze this image."
      );

    } finally {

      previewWrap.classList.remove(
        "scanning"
      );

      predictBtn.disabled =
        !model ||
        !selectedFile;

      randomBtn.disabled =
        false;

      predictBtn.textContent =
        "Detect object";
    }
  }
);


async function savePrediction(
  label,
  confidence,
  fileName
) {

  if (
    !cfg.SUPABASE_URL ||
    !cfg.SUPABASE_PUBLISHABLE_KEY ||
    !window.supabase
  ) {

    return;
  }

  try {

    const client =
      window.supabase.createClient(
        cfg.SUPABASE_URL,
        cfg.SUPABASE_PUBLISHABLE_KEY
      );

    await client
      .from("predictions")
      .insert({
        file_name:
          fileName,

        predicted_class:
          label,

        confidence:
          Number(confidence)
      });

  } catch (err) {

    console.warn(
      "Prediction log skipped:",
      err
    );
  }
}


loadModel();