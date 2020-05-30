import 'bulma/css/bulma.css';

import { load } from '@tensorflow-models/deeplab';
import * as tf from '@tensorflow/tfjs-core';

const modalName = 'pascal';
const state = {};

// deeplab will be storing the model promise
let deeplab = null;
let running = false;

const toggleInvisible = (elementId, force = undefined) => {
  const outputContainer = document.getElementById(elementId);
  outputContainer.classList.toggle('is-invisible', force);
};

const initializeModels = async () => {
  await reloadModal();

  const runner = document.getElementById('run-pascal');

  runner.onclick = async () => {
    if (running) {
      return;
    }

    running = true;
    toggleInvisible('output-card', true);
    toggleInvisible('legend-card', true);

    await tf.nextFrame();
    await runDeeplab(modalName);
    running = false;
  };

  const uploader = document.getElementById('upload-image');
  uploader.addEventListener('change', processImages);

  status('model initialized, waiting for input ... ');
};

const reloadModal = async () => {
  const selector = document.getElementById('quantizationBytes');
  const quantizationBytes = selector
    ? Number(selector.options[selector.selectedIndex].text)
    : 4;

  state.quantizationBytes = quantizationBytes;

  status('Loading the model...');

  const loadingStart = performance.now();
  deeplab = load({ base: modalName, quantizationBytes });

  // await the model to be fully loaded before proceeding
  await deeplab;

  status(`Loaded the model in ${
    ((performance.now() - loadingStart) / 1000).toFixed(2)} s`);
};

const processImage = (file) => {
  console.log(file.type);

  if (!file.type.match('image.*')) {
    return;
  }

  const reader = new FileReader();

  reader.onload = async (event) => {
    const src = event.target.result;

    toggleInvisible('output-card', true);
    toggleInvisible('legend-card', true);

    const image = document.getElementById('input-image');
    image.src = src;

    toggleInvisible('input-card', false);
    status('Image loaded ... ');

    image.onload = async () => {
      running = true;
      await tf.nextFrame();
      await runDeeplab(modalName);
      running = false;
    };
  };

  reader.readAsDataURL(file);
};

const processImages = (event) => {
  const files = event.target.files;

  if (files && files.length > 0) {
    processImage(files[0]);
  }
};

const displaySegmentationMap = (modelName, deeplabOutput) => {
  const { legend, height, width, segmentationMap } = deeplabOutput;
  const canvas = document.getElementById('output-image');
  const ctx = canvas.getContext('2d');

  toggleInvisible('output-card', false);
  const segmentationMapData = new ImageData(segmentationMap, width, height);

  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.width = width;
  canvas.height = height;

  ctx.putImageData(segmentationMapData, 0, 0);

  // draw background: ctx === src image context
  // ctx.drawImage(maskContext.canvas, 0, 0, ctx.canvas.width, ctx.canvas.height);

  const legendList = document.getElementById('legend');

  while (legendList.firstChild) {
    legendList.removeChild(legendList.firstChild);
  }

  Object.keys(legend).forEach((label) => {
    const tag = document.createElement('span');
    const [red, green, blue] = legend[label];

    tag.innerHTML = label;
    tag.classList.add('column');
    tag.style.backgroundColor = `rgb(${red}, ${green}, ${blue})`;
    tag.style.padding = '1em';
    tag.style.margin = '1em';
    tag.style.color = '#ffffff';

    legendList.appendChild(tag);
  });

  toggleInvisible('legend-card', false);

  const inputContainer = document.getElementById('input-card');
  inputContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
};

const status = (message) => {
  const statusMessage = document.getElementById('status-message');
  statusMessage.innerText = message;
  console.log(message);
};

const runPrediction = (modelName, input, initialisationStart) => {
  deeplab.then((model) => {
    model.segment(input).then((output) => {
      displaySegmentationMap(modelName, output);
      status(`Ran in ${
        ((performance.now() - initialisationStart) / 1000).toFixed(2)} s`);
    });
  });
};

const runDeeplab = async (modelName) => {
  status(`Running the inference...`);

  const selector = document.getElementById('quantizationBytes');
  const quantizationBytes =
    Number(selector.options[selector.selectedIndex].text);

  if (state.quantizationBytes !== quantizationBytes) {
    // reset the deeplab model
    await deeplab.dispose();
    deeplab = null;

    state.quantizationBytes = quantizationBytes;
  }

  const input = document.getElementById('input-image');
  if (!input.src || !input.src.length || input.src.length === 0) {
    status('Failed! Please load an image first.');
    return;
  }

  toggleInvisible('input-card', false);

  // if the model is not yet loaded, do it now
  if (!deeplab) {
    await reloadModal();
  }

  const predictionStart = performance.now();

  if (input.complete && input.naturalHeight !== 0) {
    runPrediction(modelName, input, predictionStart);
    return;
  }

  input.onload = () => {
    // if the image is still loading while the model is hit run,
    // queue the prediction on load complete
    runPrediction(modelName, input, predictionStart);
  };
};

window.onload = initializeModels;
