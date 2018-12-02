class ControllerDataset {
  constructor(numClasses) {
    this.numClasses = numClasses;
  }
  addExample(example, label) {
    const y = tf.tidy(() => tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses))
    if (this.xs == null) {
      this.xs = tf.keep(example);
      this.ys = tf.keep(y);
    } else {
      const oldX = this.xs;
      this.xs = tf.keep(oldX.concat(example, 0));

      const oldY = this.ys;
      this.ys = tf.keep(oldY.concat(y, 0));

      oldX.dispose();
      oldY.dispose();
      y.dispose();
    }
  }
}

const controllerDataset = new ControllerDataset(10);
let model
let mobilenet
let form_html
let original_form_html
let videoElement = document.createElement("video");
videoElement.setAttribute("playsInline", true)
videoElement.setAttribute("muted", true)
videoElement.setAttribute("autoPlay", true)
videoElement.setAttribute("width", "224")
videoElement.setAttribute("height", "224")

async function loadMobilenet () {
  const mobilenet = await tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

function capture(webcamElement) {
  return tf.tidy(() => {
    const webcamImage = tf.fromPixels(webcamElement);
    const croppedImage = cropImage(webcamImage);
    const batchedImage = croppedImage.expandDims(0);
    return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
  });
}

function cropImage(img) {
  const size = Math.min(img.shape[0], img.shape[1]);
  const centerHeight = img.shape[0] / 2;
  const beginHeight = centerHeight - (size / 2);
  const centerWidth = img.shape[1] / 2;
  const beginWidth = centerWidth - (size / 2);
  return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}

function predict (videoElement) {
  return new Promise((solve, reject) => {
    setTimeout(async () => {
      const predictedClass = tf.tidy(() => {
        const img = capture(videoElement);
        const activation = mobilenet.predict(img);
        const predictions = model.predict(activation);
        return predictions.as1D().argMax();
      });
      const classId = (await predictedClass.data())[0];
      solve(classId)
    }, 1000);
  })
}

function predicting() {
  const buttonField = document.getElementById("images-for-facepass")
  buttonField.value = "Ready!!!"

  const navigatorAny = navigator;
  navigator.getUserMedia = navigator.getUserMedia || navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia || navigatorAny.msGetUserMedia
  if (navigator.getUserMedia) {
    navigator.getUserMedia(
      {video: true},
      async stream => {
        videoElement.srcObject = stream
        let checking = true
        while (checking) {
          const classId = await predict(videoElement)
          console.log(classId)
          if (classId === 1) {
            buttonField.value = "Succeeded"
            buttonField.classList.remove("btn-danger")
            buttonField.classList.remove("btn-secondary")
            buttonField.classList.add("btn-success")
            checking = false
            setTimeout(() => {
              form_html.innerHTML = original_form_html
            }, 1000)
          } else {
            buttonField.value = "Failed"
            buttonField.classList.remove("btn-secondary")
            buttonField.classList.add("btn-danger")
            

          }
        }
      },
      error => console.log(error)
    )
  }
}

function training() {
  const buttonField = document.getElementById("images-for-facepass")
  buttonField.value = "training images now..."

  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      tf.layers.dense({
        units: 100,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true
      }),
      tf.layers.dense({
        units: 10,
        kernelInitializer: 'varianceScaling',
        useBias: false,
        activation: 'softmax'
      })
    ]
  })

  const optimizer = tf.train.adam(0.0001);
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

  const batchSize = Math.floor(controllerDataset.xs.shape[0] * 0.4);
  if (!(batchSize > 0)) {
    throw new Error(`Batch size is 0 or NaN. Please choose a non-zero fraction.`);
  }

  model.fit(controllerDataset.xs, controllerDataset.ys, {
    batchSize,
    epochs: 20,
    callbacks: {
      onBatchEnd: async () => await tf.nextFrame()
    }
  })
  predicting()
}

async function fetch_images(){
  const username = document.getElementById("username-for-facepass").value
  if (!username) {
    const buttonField = document.getElementById("images-for-facepass")
    buttonField.value = "Please Enter Label"
    return
  }
  const buttonField = document.getElementById("images-for-facepass")
  buttonField.setAttribute("disabled", "")
  buttonField.value = "fetching images now..."

  
  
  
  const preResponse = await fetch(`https://face-pass-keijun.herokuapp.com/api/get_all_images_for_extentioins?email=${username}`)
  const response = await preResponse.json()

  const {images, fake_images} = response
  const image_string = images.map((item) => item.x_data)
  const fake_image_string = fake_images.map((item) => item.x_data)

  image_string.forEach((value) => {
    const image_tensor = tf.tensor1d(value.split(','))
    controllerDataset.addExample(image_tensor.reshape([1, 7, 7, 256]), 1)
  })

  fake_image_string.forEach((value) => {
    const image_tensor = tf.tensor1d(value.split(','))
    controllerDataset.addExample(image_tensor.reshape([1, 7, 7, 256]), Math.floor(Math.random() * 9 ) + 2)
  })
  training()
}

async function init() {
  mobilenet = await loadMobilenet()
  form_html = document.getElementsByClassName('auth-form-body')[0]
  original_form_html = form_html.innerHTML

  const text = '<p>このサイトはFacePassによりロックされています</p>'
  const label = '<label for="login_field">Label</label>'
  const input = '<input type="text" name="login" id="username-for-facepass" class="form-control input-block" tabindex="1" autocapitalize="off" autocorrect="off" autofocus="autofocus">'
  const button = '<input id="images-for-facepass" name="commit" value="Fetch Images from FacePass" tabindex="3" class="btn btn-secondary btn-block" data-disable-with="Signing in…"></input>'
  form_html.innerHTML = text + label + input + button
  
  document.getElementById("images-for-facepass").addEventListener("click", fetch_images); 
}

init()