// Copyright (c) 2018 ml5
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

/*
Image Classifier using pre-trained networks
*/

import * as tf from '@tensorflow/tfjs';

const DEFAULTS = {
  modelPath: 'https://raw.githubusercontent.com/zaidalyafeai/HostedModels/master/unet-128/model.json',
  imageSize: 128,
  returnTensors: false,
}

const callCallback = (promise, callback) => {
  if (callback) {
    promise
      .then((result) => {
        callback(undefined, result);
        return result;
      })
      .catch((error) => {
        callback(error);
        return error;
      });
  }

  return promise;
}

// Converts a tf to DOM img
const array3DToImage = (tensor) => {
  const [imgHeight, imgWidth] = tensor.shape;
  const data = tensor.dataSync();

  const canvas = document.createElement('canvas');
  canvas.width = imgWidth;
  canvas.height = imgHeight;

  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < imgWidth * imgHeight; i += 1) {
    const j = i * 4;
    const k = i * 3;

    imageData.data[j + 0] = Math.floor(256 * data[k + 0]);
    imageData.data[j + 1] = Math.floor(256 * data[k + 1]);
    imageData.data[j + 2] = Math.floor(256 * data[k + 2]);
    imageData.data[j + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);

  // Create img HTML element from canvas
  const dataUrl = canvas.toDataURL();

  const outputImg = document.createElement('img');
  outputImg.src = dataUrl;
  outputImg.style.width = imgWidth;
  outputImg.style.height = imgHeight;

  tensor.dispose();

  return outputImg;
};

class UNET {
  /**
   * Create UNET class.
   * @param {Object} options - Optional. A set of options.
   * @param {function} callback - Optional. A callback function that is called once the model has loaded. If no callback is provided, it will return a promise that will be resolved once the model has loaded.
   */
  constructor(options, callback) {
    this.modelReady = false;
    this.isPredicting = false;

    this.config = {
      modelPath: typeof options.modelPath !== 'undefined' ? options.modelPath : DEFAULTS.modelPath,
      imageSize: typeof options.imageSize !== 'undefined' ? options.imageSize : DEFAULTS.imageSize,
      returnTensors: typeof options.returnTensors !== 'undefined' ? options.returnTensors : DEFAULTS.returnTensors,
    };

    this.ready = callCallback(this.loadModel(), callback);
  }

  async loadModel() {
    this.model = await tf.loadLayersModel(this.config.modelPath);
    this.modelReady = true;

    return this;
  }

  async segment(input, cb) {
    await this.ready;
    let imgToPredict;
    let callback = cb;

    if (
      input instanceof HTMLImageElement ||
      input instanceof HTMLVideoElement ||
      input instanceof HTMLCanvasElement ||
      input instanceof ImageData
    ) {
      imgToPredict = input;
      return callCallback(this.segmentInternal(imgToPredict), callback);
    }

    throw new Error('[error] unknown image type:', typeof input);
  }

  static dataURLtoBlob(dataurl) {
    const arr = dataurl.split(',');
    const mime = arr[0].match(/:(.*?);/)[1];
    const bstr = atob(arr[1]);

    let n = bstr.length;
    const u8arr = new Uint8Array(n);

    while (n) {
      u8arr[n] = bstr.charCodeAt(n);
      n -= 1;
    }

    return new Blob([u8arr], {
      type: mime
    });
  }

  async segmentInternal(imgToPredict) {
    // Wait for the model to be ready
    await this.ready;
    // skip asking for next frame if it's not video
    if (imgToPredict instanceof HTMLVideoElement) {
      await tf.nextFrame();
    }
    this.isPredicting = true;

    const {
      featureMask,
      backgroundMask,
      segmentation
    } = tf.tidy(() => {
      // preprocess the input image
      const tfImage = tf.browser.fromPixels(imgToPredict).toFloat();
      const resizedImg = tf.image.resizeBilinear(tfImage, [this.config.imageSize, this.config.imageSize]);
      let normTensor = resizedImg.div(tf.scalar(255));
      const batchedImage = normTensor.expandDims(0);
      // get the segmentation
      const pred = this.model.predict(batchedImage);

      // add back the alpha channel to the normalized input image
      const alpha = tf.ones([128, 128, 1]).tile([1, 1, 1])
      normTensor = normTensor.concat(alpha, 2)

      // TODO: optimize these redundancies below, e.g. repetitive squeeze() etc
      // get the background mask;
      let maskBackgroundInternal = pred.squeeze([0]);
      maskBackgroundInternal = maskBackgroundInternal.tile([1, 1, 4]);
      maskBackgroundInternal = maskBackgroundInternal.sub(0.3).sign().relu().neg().add(1);
      const featureMaskInternal = maskBackgroundInternal.mul(normTensor);

      // get the feature mask;
      let maskFeature = pred.squeeze([0]);
      maskFeature = maskFeature.tile([1, 1, 4]);
      maskFeature = maskFeature.sub(0.3).sign().relu();
      const backgroundMaskInternal = maskFeature.mul(normTensor);

      const alpha255 = tf.ones([128, 128, 1]).tile([1, 1, 1]).mul(255);
      let newpred = pred.squeeze([0]);
      newpred = tf.cast(newpred.tile([1, 1, 3]).sub(0.3).sign().relu().mul(255), 'int32')
      newpred = newpred.concat(alpha255, 2)

      return {
        featureMask: featureMaskInternal,
        backgroundMask: backgroundMaskInternal,
        segmentation: newpred
      };
    });

    this.isPredicting = false;

    // these come first because array3DToImage() will dispose of the input tensor
    const maskFeat = await tf.browser.toPixels(featureMask);
    const maskBg = await tf.browser.toPixels(backgroundMask);
    const mask = await tf.browser.toPixels(segmentation);

    const maskFeatDom = array3DToImage(featureMask);
    const maskBgDom = array3DToImage(backgroundMask);
    const maskFeatBlob = UNET.dataURLtoBlob(maskFeatDom.src);
    const maskBgBlob = UNET.dataURLtoBlob(maskBgDom.src);


    let pFeatureMask;
    let pBgMask;
    let pMask;

    // if (p5Utils.checkP5()) {
    //   pFeatureMask = await this.convertToP5Image(maskFeat);
    //   pBgMask = await this.convertToP5Image(maskBg)
    //   pMask = await this.convertToP5Image(mask)
    // }

    if (!this.config.returnTensors) {
      featureMask.dispose();
      backgroundMask.dispose();
      segmentation.dispose();
    }

    return {
      segmentation: mask,
      blob: {
        featureMask: maskFeatBlob,
        backgroundMask: maskBgBlob
      },
      tensor: {
        featureMask,
        backgroundMask,
      },
      raw: {
        featureMask: maskFeat,
        backgroundMask: maskBg
      },
      // returns if p5 is available
      featureMask: pFeatureMask,
      backgroundMask: pBgMask,
      mask: pMask
    };
  }
}

const uNet = (videoOr, optionsOr, cb) => {
  let video = null;
  let options = {};
  let callback = cb;

  if (videoOr instanceof HTMLVideoElement) {
    video = videoOr;
  } else if (typeof videoOr === 'object' && videoOr.elt instanceof HTMLVideoElement) {
    video = videoOr.elt; // Handle p5.js image
  } else if (typeof videoOr === 'function') {
    callback = videoOr;
  } else if (typeof videoOr === 'object') {
    options = videoOr;
  }

  if (typeof optionsOr === 'object') {
    options = optionsOr;
  } else if (typeof optionsOr === 'function') {
    callback = optionsOr;
  }
  return new UNET(video, options, callback);
};

export default uNet;
