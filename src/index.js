import 'bulma/css/bulma.css';

import { load } from '@tensorflow-models/deeplab';
import * as tf from '@tensorflow/tfjs-core';

// const modelNames = ['pascal', 'cityscapes', 'ade20k'];
const modalName = 'pascal';
const deeplab = {};
const state = {};

const toggleInvisible = (elementId, force = undefined) => {
  const outputContainer = document.getElementById(elementId);
  outputContainer.classList.toggle('is-invisible', force);
};

const initializeModels = async () => {
  const selector = document.getElementById('quantizationBytes');
  const quantizationBytes =
    Number(selector.options[selector.selectedIndex].text);

  state.quantizationBytes = quantizationBytes;
  deeplab[modalName] = load({ base: modalName, quantizationBytes });

  const runner = document.getElementById(`run-${modalName}`);

  runner.onclick = async () => {
    toggleInvisible('output-card', true);
    toggleInvisible('legend-card', true);
    await tf.nextFrame();
    await runDeeplab(modalName);
  };

  const uploader = document.getElementById('upload-image');
  uploader.addEventListener('change', processImages);

  status('Initialised models, waiting for input...');
};

const reloadModal = () => {
  const selector = document.getElementById('quantizationBytes');
  const quantizationBytes =
    Number(selector.options[selector.selectedIndex].text);

  state.quantizationBytes = quantizationBytes;
  deeplab[modalName] = load({ base: modalName, quantizationBytes });
};

const setImage = (src) => {
  toggleInvisible('output-card', true);
  toggleInvisible('legend-card', true);

  const image = document.getElementById('input-image');
  image.src = src;

  toggleInvisible('input-card', false);
  status('Waiting until the model is picked...');
};

const processImage = (file) => {
  if (!file.type.match('image.*')) {
    return;
  }

  const reader = new FileReader();

  reader.onload = (event) => {
    setImage(event.target.result);
  };

  reader.readAsDataURL(file);
};

const processImages = (event) => {
  const files = event.target.files;
  Array.from(files).forEach(processImage);
};

const displaySegmentationMap = (modelName, deeplabOutput) => {
  // console.log("ready to display image ... ");

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
  deeplab[modelName].then((model) => {
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
    // for (const base of modelNames) {
    //   if (deeplab[base]) {
    //     (await deeplab[base]).dispose();
    //     deeplab[base] = undefined;
    //   }
    // };

    deeplab.dispose();
    deeplab = {};

    state.quantizationBytes = quantizationBytes;
  }

  const input = document.getElementById('input-image');

  if (!input.src || !input.src.length || input.src.length === 0) {
    status('Failed! Please load an image first.');
    return;
  }

  toggleInvisible('input-card', false);

  if (!deeplab[modelName]) {
    status('Loading the model...');
    const loadingStart = performance.now();
    deeplab[modelName] = load({ base: modelName, quantizationBytes });
    await deeplab[modelName];
    status(`Loaded the model in ${
      ((performance.now() - loadingStart) / 1000).toFixed(2)} s`);
  }

  const predictionStart = performance.now();

  if (input.complete && input.naturalHeight !== 0) {
    runPrediction(modelName, input, predictionStart);
  } else {
    input.onload = () => {
      runPrediction(modelName, input, predictionStart);
    };
  }
};

window.onload = initializeModels;
