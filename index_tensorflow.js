const inp_label = document.getElementById("input-label");
const btn_collect = document.getElementById("btn-record");
const btn_add = document.getElementById("btn-add-label");
const btn_remove = document.getElementById("btn-delete-label");
const btn_train = document.getElementById("train");
const btn_listen = document.getElementById("listen");

let start_collect_state = false;
let collect_label_arr = [];
let extra_label_arr = [];
let recognizer;
let label_counter = 0;
let current_examples_length = 0;
let exist_label_counter = null;
let time;
let timeCounter = 2;

inp_label.addEventListener("input", function () {
  switchConsoleColor("#ff8c00");
  if (inp_label.value !== "") {

    // check
    if (extra_label_arr.includes(inp_label.value)) {
      //ändere exist_label_counter
      let index = extra_label_arr.indexOf(inp_label.value);

      //[index][1] sind die examples diese in console eintragen
      current_examples_length = collect_label_arr[index][1];
      document.querySelector("#console").value = collect_label_arr[index][1] + ` Aufnahmen`;

    }
    else {
      current_examples_length = 0;
      document.querySelector("#console").value = `0 Aufnahmen`;
    }
    btn_collect.disabled = false;
  }
  else {
    current_examples_length = 0;
    document.querySelector("#console").value = `0 Aufnahmen`;
    btn_collect.disabled = true;
  }
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
  examples = examples.concat(tmp_examples);
  tmp_examples = [];
  let current_input = inp_label.value;

//soll++ bei neu oder bei gemachtem eintrag
//collect_label_arr = ["test", 1]
//collect_label_arr = ["test", 2]


/*
logik:
- drückt start => nimmt paar samples
- wird gestoppt => aufnahmen_counter++
- drückt add => collect_label_arr =[index][1] = aufnahme_counter
*/

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
  toggleLabelListItems(false);
  toggleThisButtons(true, true, true, true, true);
  if (label_counter >= 2) {
    toggleThisButtons(true, true, true, false, true);
  }
}

/*

prüfe ob input selber wie davor ist, wenn ja dann counter nicht = 0 setzen

*/

let last_label = "";

function collect() {
  if (recognizer.isListening()) {
    last_label = inp_label.value;
    current_examples_length++;
    document.getElementById("btn-record").textContent = "Aufnahme starten";
    document.querySelector("#console").value = `${current_examples_length} Aufnahmen`;
    toggleLabelListItems(true);
    toggleThisButtons(false, false, false, true, true);
    start_collect_state = false;
    exist_label_counter = null;
    return recognizer.stopListening();
  }
  if (inp_label.value !== "") {
toggleThisButtons(1,1,1,1,1);

    document.getElementById("btn-record").textContent = "Aufnahme läuft";
    recognizer.listen(
      async ({ spectrogram: { frameSize, data } }) => {
        let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
        let current_input = inp_label.value;

        //wenn nicht gestartet
        if (!start_collect_state) {
          time2out();
          start_collect_state = true;
          //wenn label vorhanden

          if (extra_label_arr.includes(current_input)) {
            //ändere exist_label_counter
            exist_label_counter = extra_label_arr.indexOf(current_input);
            let label_counter = exist_label_counter;
            tmp_examples.push({ vals, label_counter });
            
          }
          else if(inp_label.value !== last_label) {
            current_examples_length = 0;
          } 
        }
        else {
          if (exist_label_counter !== null) {
            let label_counter = exist_label_counter;
            tmp_examples.push({ vals, label_counter });
            
          }
          else {
            tmp_examples.push({ vals, label_counter });
            
          }
        }
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
  toggleLabelListItems(true);
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
  toggleLabelListItems(false);
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
  //socket.emit("label", collect_label_arr[label][0]);
};

function listen() {
  switchConsoleColor("#00FFFF");
  if (recognizer.isListening()) {
    recognizer.stopListening();
    document.getElementById("listen").textContent = "Zuhören starten";
    toggleInputLabel(false);
    toggleLabelListItems(false);
    toggleThisButtons(true, true, true, true, false);
    return;
  }
  toggleInputLabel(true);
  toggleLabelListItems(true);
  toggleThisButtons(true, true, true, true, false);
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
  toggleThisButtons(false, true, true, true, true);
  switchConsoleColor("#ff8c00");
  document.querySelector("#console").value = e.cells[1].textContent + " Aufnahmen";
  current_examples_length = parseInt(e.cells[1].textContent);
}

function deleteTmpExamples() {
  tmp_examples = [];
  current_examples_length = 0;
  document.querySelector("#console").value = `0 Aufnahmen`;
  toggleLabelListItems(false);
  toggleThisButtons(true, true, true, true, true);
  inp_label.value = "";
}

function switchConsoleColor(color) {
  document.getElementById("console").style.color = color;
  document.getElementById("console").style.border = "1px solid" + color;
}

function toggleThisButtons(record, save, remove, train, listen) {
  btn_collect.disabled = record;
  btn_add.disabled = save;
  btn_remove.disabled = remove;
  btn_listen.disabled = listen;
  if (label_counter >= 2) {
    btn_train.disabled = train;
  }
}

function toggleInputLabel(disabledState) {
  inp_label.disabled = disabledState;
}

function toggleLabelListItems(disabledState) {
  let rows = document.getElementsByClassName("real-tr");
  let labels = document.getElementsByClassName("real-td");
  if (disabledState) {
    for (let j = 0; j < rows.length; j++) {
      rows[j].removeAttribute("onclick");
    }
    for (let index = 0; index < labels.length; index++) {
      labels[index].className = "real-td dis";
    }
  }
  else {
    updateLabelList();
  }
}


function time2out(){
  let time = setInterval(function(){
    if(timeCounter > 0){
      document.querySelector("#console").value = timeCounter +  ` Sekunden`;
      timeCounter--;
    }
    else{
      collect();
      timeCounter = 2;
      clearInterval(time);
    }
    
  }, 1000);
}
