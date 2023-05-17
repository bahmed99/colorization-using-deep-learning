import React, { useState, useEffect } from 'react'
import { useNavigate } from "react-router-dom";
import Navbar from '../components/Navbar'
import axios from 'axios'
import PropagateLoader from "react-spinners/PropagateLoader"
import Swal from 'sweetalert2'
import '../assets/css/Colorize.css'
import InfoBulle from '../components/InfoBulle'
import ModalReference from '../components/ModalReference';
import ModalScribble from '../components/ModalScribble';
/**
 * Page for colorizing images
 */
export default function Colorize() {
  const [loading, setLoading] = useState(false)
  const [datasets, setDatasets] = useState([])
  const [dataset, setDataset] = useState("")
  const [models, setModels] = useState([])
  const [modelsSelected, setModelsSelected] = useState([])
  const [images, setImages] = useState([])
  const [dir, setDir] = useState([])
  /* extract minute and seconds from the current time and add a 0 to the minute or seconds if it's composed by only one number*/
  const now = new Date();
  const hours = now.getHours().toString().padStart(2, '0')
  const minutes = now.getMinutes().toString().padStart(2, '0')
  const year = now.getFullYear()
  const month = (now.getMonth() + 1).toString().padStart(2, '0')
  const day = (now.getDate()).toString().padStart(2, '0')

  const [colorizationName, setColorizationName] = useState(`Colorisation_${year}-${month}-${day}_${hours}:${minutes}`)
  const [pathfolder, setPathFolder] = useState([])
  const [displayUpload, setDisplayUpload] = useState("")
  const [displayPath, setDisplayPath] = useState("")
  const [displayDataset, setDisplayDataset] = useState("")
  const [checkColorizationName, setCheckColorizationName] = useState(false)

  const [showModal, setShowModal] = useState(false);
  const [reference, setReference] = useState("")
  const [scribble, setScribble] = useState("")
  const [imageReference, setImageReference] = useState([])
  const [model, setModel] = useState("")


  const Navigate = useNavigate();
  let formData = new FormData()

  useEffect(() => {
    axios.get("http://localhost:5000/image/getModels").then((data) => {
      setModels(data.data);
    }).catch(err => {
      console.log(err)
    })

    axios.get("http://localhost:5000/image/dataset")
      .then(res => {
        setDatasets(res.data)

      }).catch(err => {
        console.log(err)
      })

  }, [])

  function UploadImages(e) {
    setDisplayPath("none")
    setDisplayDataset("none")
    const files = e.target.files;
    setImages(prevState => [...prevState, ...files])
  }

  function UploadFolder(e) {
    setDisplayPath("none")
    setDisplayDataset("none")
    const files = e.target.files;
    setDir(prevState => [...prevState, ...files]);
  }

  function Colorize_Images() {

    if(getMetricsChecked().length !== 0){
      Swal.fire({
        title: 'Error',
        text: "You can't compute metrics for one shot colorization",
        icon: 'error',
        confirmButtonText: 'Ok'
      })
      return
    }


    axios.post("http://localhost:5000/image/checkColorizationName", { "colorization_name": colorizationName }).then(res => {
      setCheckColorizationName(res.data.message)
    })

    if (checkColorizationName) {
      Swal.fire({
        title: 'Error',
        text: 'Colorization name already exists',
        icon: 'error',
        confirmButtonText: 'Ok'
      })
    }

    else {
      for (let i = 0; i < images.length; i++) {
        formData.append('file[]', images[i]);
      }
      for (let i = 0; i < dir.length; i++) {
        formData.append('dir[]', dir[i]);
      }

      for (let i = 0; i < modelsSelected.length; i++) {
        formData.append('model[]', modelsSelected[i]);

      }
      formData.append('colorization_name', colorizationName)

      formData.append('path', pathfolder)
      
      formData.append('scribble', scribble)

      for (let i = 0; i < imageReference.length; i++) {
        formData.append('reference[]', imageReference[i]);
      }

    

      let listMetrics = getMetricsChecked(); // get the selected metrics 
      console.log(listMetrics);
      for (let i = 0; i < listMetrics.length; i++) { //add the selected metrics to formData in order to send and treat them in backend
        formData.append('metrics[]', listMetrics[i]);
      }


      if ((formData.get('file[]') || formData.get('dir[]') || formData.get("path")) && formData.get('model[]') && formData.get("colorization_name")) {

        setLoading(true)
        axios.post("http://localhost:5000/image/colorize", formData)
          .then(res => {
            if (res.data.message === "success") {
              setLoading(false)
              setDisplayDataset("")
              setDisplayPath("")
              setDisplayUpload("")
              setColorizationName("")
              Swal.fire({
                title: 'Colorization completes successfully',
                text: `You can find the result in the directory /uploads/Results/${res.data.name}`,
                icon: 'success',
                showCancelButton: true,
                confirmButtonColor: '#3085d6',
                cancelButtonColor: '#d33',
                confirmButtonText: 'Visualize the results'
              }).then((result) => {
                if (result.isConfirmed) {
                  Navigate(`/visualize/${res.data.name}`)
                }

              })
            }
            else {
              setLoading(false)
              Swal.fire({
                title: 'Error',
                text: res.data.text,
                icon: 'error',
                confirmButtonText: 'OK'
              })
            }
          }).catch(err => {
            setLoading(false)
            console.log(err)
          })
      }
    }
  }

  function HandleChange(e) {
    if (e.target.value != "") {
      setDisplayPath("none")
      setDisplayUpload("none")
      setDataset(e.target.value)
    }
    else {
      setDisplayPath("")
      setDisplayUpload("")
    }
  }
  function HandleChangeModel(e) {
    if (e.target.value != "") {
      if (!modelsSelected.includes(e.target.value)) {
        if ((e.target.value === "SuperUnetV2" || e.target.value === "SuperUnetV1") && reference === "") {
          setModel(e.target.value)
          setShowModal(true)
        }
        setModelsSelected(modelsSelected.concat(e.target.value))

      }
    }
  }

  function removeSelectedModel(model) {

    if (modelsSelected.includes(model)) {
      setModelsSelected(modelsSelected.filter((m) => m !== model));
      if (model === "SuperUnetV1" || model === "SuperUnetV2") {
        setReference("")
      }
    }
  }

  function ColorizeDataset() {
    axios.post("http://localhost:5000/image/checkColorizationName", { "colorization_name": colorizationName }).then(res => {
      setCheckColorizationName(res.data.message)

    })

    if (checkColorizationName) {
      Swal.fire({
        title: 'Error',
        text: 'Colorization name already exists',
        icon: 'error',
        confirmButtonText: 'Ok'
      })
    }

    else {




      if (dataset !== "" && modelsSelected.length !== 0 && colorizationName !== "") {


        setLoading(true)
        axios.post("http://localhost:5000/image/colorizeDataset",
          {
            "dataset": dataset, "models": modelsSelected,
            "colorization_name": colorizationName, "reference": reference, "metrics[]": getMetricsChecked(),
            "scribble": scribble
          },
          {
            headers: {
              'Content-Type': 'application/json'
            }
          })
          .then(res => {
            if (res.data.message === "success") {
              setLoading(false)
              setDisplayDataset("")
              setDisplayPath("")
              setDisplayUpload("")
              setColorizationName("")
              Swal.fire({
                title: 'Colorization completes successfully',
                text: `You can find the result in the directory /uploads/Results/${res.data.name}`,
                icon: 'success',
                showCancelButton: true,
                confirmButtonColor: '#3085d6',
                cancelButtonColor: '#d33',
                confirmButtonText: 'Visualize the results'
              }).then((result) => {
                if (result.isConfirmed) {
                  Navigate(`/visualize/${res.data.name}`)
                }

              })
            }
            else {
              setLoading(false)
              Swal.fire({
                title: 'Error',
                text: res.data.text,
                icon: 'error',
                confirmButtonText: 'OK'
              })
            }
          }).catch(err => {
            setLoading(false)
            console.log(err)
          })
      }
    }
  }

  function HandleChangeName(e) {

    setColorizationName(e.target.value)

  }
  function HandleChangeNamePath(e) {
    if (e.target.value !== "") {
      setDisplayUpload("none")
      setDisplayDataset("none")
    }
    else {
      setDisplayUpload("")
      setDisplayDataset("")

    }
    setPathFolder(e.target.value)

  }

  function Send() {
    if (dataset === "") {
      Colorize_Images()
    }
    else {
      ColorizeDataset()
    }
  }

  /**
   * returns the metrics that were checked by the user
   */
  function getMetricsChecked() {
    let listMetrics = []
    let checkboxes = document.getElementsByClassName("checkboxMetrics"); // get all elements of the class
    for (let i = 0; i < checkboxes.length; ++i) { // for every checkbox in the class
      let checkbox = checkboxes[i];
      if (checkbox.checked) { // if it is checked
        listMetrics.push(checkbox.id); // add its id to the list of checked metricss
      }
    }
    return listMetrics;
  }

  if (loading) {
    return <PropagateLoader cssOverride={{
      "display": "flex",
      "justifyContent": "center", "alignItems": "center", "height": "100vh"
    }}
      color="#36d7b7" />
  }

  function ChooseModels(e) {
    const value = e.target.value;
    if (value !== "") {
      if (!modelsSelected.includes(value)) {
        if ((value === "SuperUnetV2" || value === "SuperUnetV1") && reference === "") {
          setModel(value)
          setShowModal(true)
        }
        if (value === "UniColor") {
          setModel(value)
          setShowModal(true)

        }
        setModelsSelected([...modelsSelected, value]);
      } else {
        /*the case of the method already exists on the methods list so we delete it from the list and if the method is by reference, the function clear the reference*/
        setModelsSelected(modelsSelected.filter(model => model !== value));
        if (model === "SuperUnetV1" || model === "SuperUnetV2") {
          setReference("")
        }
        else if (model === "UniColor") {
          setScribble("")
        }
      }
    }
  }

  function getMethodType(method) {
    switch (method) {
      case "SuperUnetV1":
        return "Reference";
      case "SuperUnetV2":
        return "Reference";
      case "ChromaGAN":
        return "Automatic";
      case "ECCV16":
        return "Automatic";
      case "SIGGRAPH":
        return "Automatic";
      case "OptiScribbles":
        return "Scribbles";
      case "UniColor":
        return "Scribbles";
      default:
        return "Undefined";
    }
  }
  return (
    <div className='colorize_page'>
      <Navbar />
      <h2>Name your colorization</h2>
      <input type="text" value={colorizationName} onChange={(e) => HandleChangeName(e)} placeholder="Name your colorization" className='input_colorize' />

      <div>
      </div>
      <div className='container_choices'>
        <div className='image_input' style={{ "display": displayUpload }}>
          <h2>Load a dataset for one shot colorization </h2>

          <label >Single files : </label>
          <input className="file_upload" type="file" multiple onChange={(e) => UploadImages(e)} />
          <label >Folders : </label>
          <input className="file_upload" type="file" webkitdirectory="" onChange={(e) => UploadFolder(e)} />
        </div>
        {/* <div className='path_container' style={{ "display": displayPath }}>
          <h2>Put path of folder</h2>
          <input type="text" onChange={(e) => HandleChangeNamePath(e)} placeholder="folder_path" className='input_colorize_path' />
          <InfoBulle  />
        </div> */}
        <div style={{ "display": displayDataset }} className='path_container'>
          <h2>Choose an existing dataset to colorize</h2>
          <div className='select_container'>
            <select onChange={(e) => HandleChange(e)} className="select_models">
              <option className='options' value=''>Choose a dataset</option>
              {datasets.map((item, index) => {
                return <option key={index} value={item}>{item}</option>

              })}
            </select>
          </div>
        </div>
      </div>
      <div>



        <section className="method-list" >
          <header className="method-list-header">
            <h2 className="method-list-title">Choose a method of colorization</h2>
          </header>
          <div className="method-list-container">
            <span className="method-list">
              {
                models.map((item, index) => (
                  <label className="method-list-item" key={index}>
                    <input
                      type="checkbox"
                      value={item}
                      className="method-list-cb"
                      onChange={ChooseModels}
                    />
                    <span className="method-list-mark"></span>
                    <span className="method-list-desc">{item} : {getMethodType(item)}</span>
                  </label>
                ))}
            </span>
          </div>
        </section>

        {(() => {

          if (modelsSelected.includes("SuperUnetV2") || modelsSelected.includes("SuperUnetV1")) {
            let modelref = "- "
            {
              modelsSelected.filter(modelsSelected => modelsSelected.includes("SuperUnetV1") || modelsSelected.includes("SuperUnetV2")).map(filteredName => (
                modelref += filteredName + " - "
              ))
            }
            return (
              <div>
                <h2>Reference</h2>
                <ModalReference imageReference={imageReference} setImageReference={setImageReference} reference={reference} dataset={dataset} setReference={setReference} model={modelref} />
              </div>
            )
          }
        })()}
        {(() => {

          if (modelsSelected.includes("UniColor")) {
            let modelScribble = "- "
            {
              modelsSelected.filter(modelsSelected => modelsSelected.includes("UniColor")).map(filteredName => (
                modelScribble += filteredName + " - "
              ))
            }
            return (
              <div>
                <h2>Scribbles</h2>
                <ModalScribble  scribble={scribble} dataset={dataset} setScribble={setScribble} model={modelScribble} />
              </div>
            )
          }
        })()}
      </div>
      <div>
        <h2>Choose metrics to compute</h2>
        <div id="checkboxes">
          <ul id="checkbox-list">
            <li> Peak Signal to Noise Ratio (PSNR) <input type="checkbox" id="PSNR" className="checkboxMetrics" ></input> </li>
            <li> Structural Similarity Index Measure (SSIM) <input type="checkbox" id="SSIM" className="checkboxMetrics" ></input> </li>
            <li> Mean Squared Error (MSE) <input type="checkbox" id="MSE" className="checkboxMetrics" ></input> </li>
            <li> Mean Absolute Error (MAE) <input type="checkbox" id="MAE" className="checkboxMetrics" ></input></li>
          </ul>
        </div>
      </div>

      <input type="button" className='colorize_btn' onClick={Send} value="Colorize" />
      {/* {showModal && <ModalReference imageReference={imageReference} setImageReference={setImageReference} reference={reference} dataset={dataset} setShowModal={setShowModal} setReference={setReference} model={model} />}     */}
    </div>
  )
}