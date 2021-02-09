const inp_label = document.getElementById("input-label");
const btn_collect = document.getElementById("btn-record");
const btn_add = document.getElementById("btn-add-label");
const btn_remove = document.getElementById("btn-delete-label");
const btn_train = document.getElementById("train");
const btn_listen = document.getElementById("listen");



let collect_label_arr = [];
let extra_label_arr = [];

let recognizer;
let label_counter = 0;

let current_examples_length = 0;


inp_label.addEventListener("input", function () {

  if (inp_label.value !== "") btn_collect.disabled = false;
  else btn_collect.disabled = true;
});


// One frame is ~23ms of audio.
const NUM_FRAMES = 10;

let tmp_examples = [];
let examples = [];

const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;

async function app() {
  recognizer = speechCommands.create("BROWSER_FFT");
  await recognizer.ensureModelLoaded();
}

app();

function addLabel() {
  //falls gefunden push dazu
  //falls nichts gefunden push neu
  examples = examples.concat(tmp_examples);
  tmp_examples = [];
  let current_input = inp_label.value;

  if (extra_label_arr.includes(current_input)) {
    let indexOfLabel = extra_label_arr.indexOf(current_input);
    collect_label_arr[indexOfLabel][1] = current_examples_length;
  }

  else {
    collect_label_arr.push([current_input, current_examples_length]);
    extra_label_arr.push(current_input);
    label_counter++;
  }

  
  current_examples_length = 0;

  inp_label.value = "";
  document.querySelector("#console").value = `0 Aufnahmen`;
  updateLabelList();

  toggleThisButtons(true, true, true, true, true);


  if (label_counter >= 2) {
    toggleThisButtons(true, true, true, false, true);
  }

}

function collect() {
  if (recognizer.isListening()) {
    document.getElementById("btn-record").textContent = "Aufnahme starten";

    toggleThisButtons(false, false, false, true, true);
    console.log("jetzt sollten save und remove enabled sein");
    return recognizer.stopListening();
  }

  //wenn input != ""
  if (inp_label.value !== "") {
    document.getElementById("btn-record").textContent = "Aufnahme stoppen";
    recognizer.listen(
      async ({ spectrogram: { frameSize, data } }) => {
        let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));

        let current_input = inp_label.value;

        if (extra_label_arr.includes(current_input)) {
          let label_counter = extra_label_arr.indexOf(current_input);
          tmp_examples.push({ vals, label_counter });
        }

        else tmp_examples.push({ vals, label_counter });

        current_examples_length++;

        document.querySelector("#console").value = `${current_examples_length} Aufnahmen`;
      },
      {
        overlapFactor: 0.999,
        includeSpectrogram: true,
        invokeCallbackOnNoiseAndUnknown: true
      }
    );
  }
  else alert("Bitte gib deiner Aufnahme zuerst ein Label");
}

function normalize(x) {
  const mean = -100;
  const std = 10;
  return x.map(x => (x - mean) / std);
}

async function train() {

  switchConsoleColor("#7FFF00");

  buildModel();
  toggleInputLabel(true);
  toggleThisButtons(true, true, true, true, true);
  const ys = tf.oneHot(examples.map(e => e.label_counter), label_counter);
  const xsShape = [examples.length, ...INPUT_SHAPE];

  const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);

  await model.fit(xs, ys, {
    batchSize: 16,
    epochs: 10,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        document.querySelector("#console").value = `Genauig.: ${(logs.acc * 100).toFixed(1)}% Epo.: ${epoch + 1}`;
      }
    }
  });
  tf.dispose([xs, ys]);
  toggleInputLabel(false);
  toggleThisButtons(true, true, true, true, false);
}

function buildModel() {
  model = tf.sequential();
  model.add(
    tf.layers.depthwiseConv2d({
      depthMultiplier: 8,
      kernelSize: [NUM_FRAMES, 3],
      activation: "relu",
      inputShape: INPUT_SHAPE
    })
  );
  model.add(tf.layers.maxPooling2d({ poolSize: [1, 2], strides: [2, 2] }));
  model.add(tf.layers.flatten());
  model.add(tf.layers.dense({ units: label_counter, activation: "softmax" }));
  const optimizer = tf.train.adam(0.01);
  model.compile({
    optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"]
  });
}

function flatten(tensors) {
  const size = tensors[0].length;
  const result = new Float32Array(tensors.length * size);
  tensors.forEach((arr, i) => result.set(arr, i * size));
  return result;
}

async function giveValue(labelTensor) {
  const label = (await labelTensor.data())[0];
  document.getElementById("console").value = "predict: " + collect_label_arr[label][0];
  //console.log(collect_label_arr[label][0]);
  //socket.emit("label", collect_label_arr[label][0]);
};


function listen() {
  switchConsoleColor("#00FFFF");
  if (recognizer.isListening()) {
    recognizer.stopListening();
    document.getElementById("listen").textContent = "Listen";
    return;
  }

  document.getElementById("listen").textContent = "Erkennung stoppen";
  
  recognizer.listen(
    async ({ spectrogram: { frameSize, data } }) => {
      const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
      const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
      const probs = model.predict(input);
      const predLabel = probs.argMax(1);
      await giveValue(predLabel);
      tf.dispose([input, probs, predLabel]);
    },
    {
      overlapFactor: 0.999,
      includeSpectrogram: true,
      invokeCallbackOnNoiseAndUnknown: true
    }
  );
}

//save model
async function save() {
  await model.save("downloads://speech-commands-model");
  arr2JSON(collect_label_arr, "collected-labels.json"); //word list für abgleich in meeting js
  arr2JSON(examples, "examples.json");                  //wird benötigt um weiter zu trainieren
}

let arr2JSON = (function () {
  let a = document.createElement("a");
  document.body.appendChild(a);
  a.style = "display: none";
  return function (data, fileName) {
    let json = JSON.stringify(data),
      blob = new Blob([json], { type: "octet/stream" }),
      url = window.URL.createObjectURL(blob);
    a.href = url;
    a.download = fileName;
    a.click();
    window.URL.revokeObjectURL(url);
  };
})();


function load() {
  //const model = await tf.loadLayersModel('file://speech-commands-model.json');
  document.getElementById("fileUpload").click();
}

function updateLabelList() {
  let label_list = "";
  let sort_arr = [...collect_label_arr].sort(function (a, b) { return a[0] > b[0] ? 1 : -1; });
  for (let i = 0; i < sort_arr.length; i++) {
    label_list += "<tr class='real-tr' onclick='updateGUI(this)'><td class='real-td' style='padding-left: 8px;'>" + sort_arr[i][0] + "</td><td style='text-align: center';>" + sort_arr[i][1] + "</td></tr>";
  }
  if (sort_arr.length < 6) {
    for (let i = 0; i < (6 - sort_arr.length); i++) {
      label_list += "<tr><td class='unreal-td'>\xa0</td><td>\xa0</td></tr>"
    }
  }
  document.getElementById("label-list").innerHTML = label_list;
}

function updateGUI(e) {
  inp_label.value = e.cells[0].textContent;
  //console.log(e.cells[1].textContent);
  toggleThisButtons(false, true, true, true, true);
  switchConsoleColor("#ff8c00");


  document.querySelector("#console").value = e.cells[1].textContent + " Aufnahmen";
  current_examples_length = parseInt(e.cells[1].textContent);
}

function deleteTmpExamples() {
  tmp_examples = [];
  current_examples_length = 0;
  document.querySelector("#console").value = `0 Aufnahmen`;

  toggleThisButtons(false, true, true, true, true);

}

function switchConsoleColor(color) {
  document.getElementById("console").style.color = color;
  document.getElementById("console").style.border = "1px solid" + color;
}


/**
 * 7. wenn aufngenommen wird soll add this = Training speichern
 * 8. Exampels so einbetten, dass reload sie neu lädt damit user ohne bedenken damit rum spielen können
 */

function toggleThisButtons(record, save, remove, train, listen) {

  //übergeben wird true oder false
  btn_collect.disabled = record;
  btn_add.disabled = save;
  btn_remove.disabled = remove;
  
  btn_listen.disabled = listen;


  if(label_counter >= 2 && !train){
    //dann darf training button disbaled = false gesetzt werden;
    btn_train.disabled = train;
  }

}

function toggleInputLabel(disabledState){
  inp_label.disabled = disabledState;
}

/*
set timeout für ein paar sekunden dann muss nicht stop gedrückt werden

*/