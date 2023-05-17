import React, { useState, useEffect } from 'react'
import { useNavigate } from "react-router-dom";
import Navbar from '../components/Navbar'
import axios from 'axios'
import InfoBulle from '../components/InfoBulle'

export default function ScribbleSelection() {
  const Navigate = useNavigate();
  const [datasets, setDatasets] = useState([])
  const [dataset, setDataset] = useState("")
  const [displayUpload, setDisplayUpload] = useState("")
  const [displayPath, setDisplayPath] = useState("")
  const [displayDataset, setDisplayDataset] = useState("")
  const now = new Date();
  const hours = now.getHours().toString().padStart(2, '0')
  const minutes = now.getMinutes().toString().padStart(2, '0')
  const year = now.getFullYear()
  const month = (now.getMonth() + 1).toString().padStart(2, '0')
  const day = (now.getDate()).toString().padStart(2, '0')

  const [colorizationName, setColorizationName] = useState(`Scribbles_${year}-${month}-${day}_${hours}:${minutes}`)
  const [pathfolder, setPathFolder] = useState([])
  const [images, setImages] = useState([])
  const [dir, setDir] = useState([])

  let formData = new FormData()

  useEffect(() => {
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

  function Load_Images() {
    for (let i = 0; i < images.length; i++) {
      formData.append('file[]', images[i]);
    }
    for (let i = 0; i < dir.length; i++) {
      formData.append('dir[]', dir[i]);
    }
    formData.append('colorization_name', colorizationName)
    formData.append('path', pathfolder)
    if (formData.get('file[]') || formData.get('dir[]') || formData.get('colorization_name') || formData.get('path')) {
      axios.post("http://localhost:5000/image/loadImagesOriginForScribbles", formData)
        .then(res => {
          Navigate(`/scribble/${res.data.name}`)
        }).catch(err => {
          console.log(err)
        })
    }
  }

  function UseDataset() {
    if (dataset !== "") {
      axios.post("http://localhost:5000/image/useKnownDatasetForScribbles", { "dataset": dataset, "colorization_name": colorizationName }, {
        headers: {
          'Content-Type': 'application/json'
        }
      }).then(res => {
        Navigate(`/scribble/${res.data.name}`)
      }).catch(err => {
        console.log(err)
      })
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
    if (dataset !== "") {
      UseDataset()
    }
    else {
      Load_Images()
    }
  }
  return (


    <div className='colorize_page'>
      <Navbar />
      <h2>Name your Scribbles</h2>
      <input type="text" value={colorizationName} onChange={(e) => HandleChangeName(e)} placeholder="Name your scribbles" className='input_colorize' />
      <div className='container_choices'>
        <div className='image_input' style={{ "display": displayUpload }}>
          <h2>Load a dataset for one shot scribbles </h2>

          <label >Single files : </label>
          <input className="file_upload" type="file" multiple onChange={(e) => UploadImages(e)} />
          <label >Folders : </label>
          <input className="file_upload" type="file" webkitdirectory="" onChange={(e) => UploadFolder(e)} />
        </div>
        {/* <div className='path_container' style={{ "display": displayPath }}>
          <h2>Put path of folder</h2>
          <input type="text" onChange={(e) => HandleChangeNamePath(e)} placeholder="folder_path" className='input_colorize_path' />
          <InfoBulle message="Entrez ici votre valeur." />
        </div> */}
        <div className='path_container'>
          <div style={{ "display": displayDataset }} className='path_container'>
            <h2>Choose an existing dataset</h2>
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
      </div>

      <input type="button" className='colorize_btn' value="Add scribbles" onClick={Send} />

    </div>
  )
}
