import React, { useEffect, useState, useRef } from "react";
import { useParams } from "react-router-dom";
import Axios from "axios";
import "../assets/css/Visualizeimage.css";
import Navbar from "../components/Navbar";
import SearchBar from "../components/SearchBar";
import CheckboxList from "../components/CheckboxList";
import ImageDisplay from "../components/ImageDisplay"

import "../assets/css/Scribble.css";
import "../assets/css/icofont/icofont.min.css";

import { TransformWrapper, TransformComponent } from "react-zoom-pan-pinch";

/**
 * Page for visualizing images of a colorization
 */
export default function VisualizeImages() {
  const { colorisation } = useParams();
  // const [info, setInfo] = useState([]) //Save the information received by the JSON file. // Commented because not used
  const [imagesorigin, setImagesorigin] = useState([]); //Save the paths of the original images.
  const [imagescolorized, setImagescolorized] = useState([]); //Save the paths of the colorized images.
  const [imagesgroundTruth, setImagesGroundTruth] = useState([]); //Save the paths of the ground truth images.
  const [selectedImageIndex, setSelectedImageIndex] = useState(-1); // -1 to indicate that no image is selected.
  const [selectedImageGroundTruth, setSelectedImageGroundTruth] = useState(-1); // Groundtruth image index to display
  const [references, setReferences] = useState([]); //Save the paths of the references images.
  const [displayImageReference, setDisplayImageReference] = useState(true); //Save the paths of the references images.
  const [metrics, setMetrics] = useState(null) //Save metrics in order to display them for each colorized image

  const calculatedMetricsRef = useRef([])
  const [enabledMetrics, setEnabledMetrics] = useState({}) // Save metrics displaying status

  // Define if we click for the first time on an image
  const [firstClick, setFirstClick] = useState(true);

  // This ref is used by the search bar in order to keep the original list of images
  const imagesOriginRef = useRef(null);

  /********************************* */
  const transformRefs = useRef([]); // create a reference to store TransformWrapper instances
  transformRefs.current[0] = transformRefs.current[0] || React.createRef(); // create a reference for the black and white image
  transformRefs.current[1] = transformRefs.current[1] || React.createRef(); // create a reference for the groundTruth image
  transformRefs.current[2] = transformRefs.current[2] || React.createRef(); // create a reference for the reference image
  const selectedImageIndexRef = useRef(0);
  /********************************* */

  useEffect(() => {
    let unmounted = false;
    if (!unmounted) {
      Axios.get(`http://localhost:5000/image/getImages/${colorisation}`).then(
        (data) => {
          // setInfo(data.data) // Commented because not used
          setImagesorigin(data.data.images_origin);
          imagesOriginRef.current = data.data.images_origin;
          setImagescolorized(data.data.images_colorized);
          setImagesGroundTruth(data.data.images_ground_truth);
          setReferences(data.data.reference);
          setMetrics(data.data.metrics)

          calculatedMetricsRef.current = Object.keys(Object.entries(Object.entries(data.data.metrics)[0][1])[0][1])
          var tmp = {}
          for (let i = 0; i < calculatedMetricsRef.current.length; i++) {
            tmp[calculatedMetricsRef.current[i]] = true
          }
          setEnabledMetrics(tmp)
        }
      );
    }
    Array.from(document.getElementsByClassName("selectedMethod")).forEach(
      (e) => (e.checked = true)
    );

    return () => {
      unmounted = true;
    };
  }, []);

  // Event manager to select an image.
  const handleImageClick = (imagePath) => {
    // Since imagesorigin is updated accordingly to the search bar value,
    // the only array that save the index of each image path is the imagesOriginRef
    setSelectedImageIndex(imagesOriginRef.current.indexOf(imagePath));

    // Check if ground truth image exists for the selected image
    const groundTruthImageIndex = findGroundTruthIndex(imagesOriginRef.current.indexOf(imagePath));
    setSelectedImageGroundTruth(groundTruthImageIndex)
  };

  /**
   * Updates the size of the displayed images according to number of images per row
   * @param {*} e
   */
  function handleNumberImagesRow(load) {
    if (load == true) {
      document.getElementById("nbImages").value = 3 // default value 
    }
    // get DOM elements
    let imgs = document.getElementsByClassName("full_view");
    let value = document.getElementById("nbImages").value;

    // change the size for every image
    Array.from(imgs).forEach((im) => {
      im.style.width = Math.floor((window.innerWidth - 160) / value) + "px";
      im.style.height = "auto";
    });
  }

  /**
   * Updates the images visualization according to the selected methods
   * @param {*} event
   */
  function UpdateMethodChoices(event) {
    document.getElementById(event.currentTarget.name).style.display = event
      .currentTarget.checked
      ? "block"
      : "none";
  }

  function UpdateMetricsChoices(event) {
    var tmp = {...enabledMetrics}
    tmp[event.currentTarget.name] = !enabledMetrics[event.currentTarget.name]
    setEnabledMetrics(tmp)
  }

  /**
   * Get information from the DOM
   * Returns true if simultaneous zoom is activated ; false otherwise
   */
  function isZoomSimultaneous() {
    let button = document.getElementById("simultaneous_zoom_visualize");
    return button.checked;
  }

  /**
   * updates zoom and panning status of the images.
   * @param {*} stats Information on the TransformWrapper instance manipulated
   */
  function update(stats) {
    if (isZoomSimultaneous()) {
      // applies the transformation to all images
      transformRefs.current.forEach(
        (ref) =>
          ref.current &&
          ref.current.setTransform(
            stats.state.positionX,
            stats.state.positionY,
            stats.state.scale
          )
      );
    } else {
      // applies the transformation only to the instance manipulated;
      stats.setTransform(
        stats.state.positionX,
        stats.state.positionY,
        stats.state.scale
      );
    }
  }

  /**
   * reinitializes the slider as well as the zoom when the page is reloaded
   */
  function reinitialize() {
    handleNumberImagesRow(true);
    transformRefs.current.forEach(
      (ref) => ref.current && ref.current.resetTransform()
    );
  }


  function findGroundTruthIndex(selectedImageIndex) {
    if (!imagesgroundTruth)
      return -1

    const selectedImageName = imagesOriginRef.current[selectedImageIndex].split('/').pop();

    for (let i = 0; i < imagesgroundTruth.length; i++) {
      const groundTruthName = imagesgroundTruth[i].split('/').pop();

      if (selectedImageName === groundTruthName) {
        return i;
      }
    }

    return -1;
  }

  const imageContext = { enabledMetrics, transformRefs, update, imagesOriginRef, selectedImageIndex, selectedImageIndexRef, reinitialize, metrics }

  return (
    <div id="Visualise-image-page">
      <Navbar />
      <div id="Visualise-Body">
        <ul className="Visualise-dataset-list">
          <SearchBar
            elements={imagesOriginRef.current}
            setElements={setImagesorigin}
            criteriaGetter={(element) => element}
          />
          <div className="Visualise-image-scroller">
            {imagesorigin.map((value, index) => {
              // value contains the path to the symbolic link of the original image.
              return (
                <li key={index} onClick={() => handleImageClick(value)}>
                  <img src={`${value}`} alt="image" />
                  <h6>{value.split("/").pop()}</h6>
                </li>
              );
            })}
          </div>
        </ul>
        <div>
          <div className="topToolMenu">
            <CheckboxList
              id="scrolling_menu"
              label={"Metrics"}
              onChange={UpdateMetricsChoices}
              elementList={calculatedMetricsRef.current}
              elementClassName={"selectedMethod"}
            />

            <CheckboxList
              id="scrolling_menu"
              label={"Methods"}
              onChange={UpdateMethodChoices}
              elementList={
                ["Grayscale", "GroundTruth"]
                  .concat(references.length !== 0 ? ["Reference"] : [])
                  .concat(imagescolorized.map(methodImages => Object.keys(methodImages)[0]))
              }
              elementClassName={"selectedMethod"}
            />

            {/* checkbox to activate or deactivate simultaneous zoom */}
            <input
              id="simultaneous_zoom_visualize"
              className="simultaneous_zoom_checkbox"
              type="checkbox"
              defaultChecked="true"
            />
            <label for="simultaneous_zoom_visualize">Simultaneous zoom</label>

            <label for="nbImages">Number of images per row:</label>
            <input type="number" id="nbImages" name="nbImages"
              min="1" max="10" onClick={() => handleNumberImagesRow(false)} onLoadStart={handleNumberImagesRow} />

            {/* max input = window width - (image list width + 10px margin on each side)*/}
          </div>
          {selectedImageIndex !== -1 && (
            <div className="display_images">
              <ImageDisplay context={imageContext}
                id="Grayscale" title="Black and white image" transformRefsIndec={0}
                src={`${imagesOriginRef.current[selectedImageIndex]}`}
              />

              {references.length !== 0 && displayImageReference && (
                <ImageDisplay context={imageContext}
                  id="Reference" title="Image reference" transformRefsIndec={1}
                  src={`${references[selectedImageIndex]}`}
                />
              )}

              {/*displaying ground truth image if it exists */}
              {selectedImageGroundTruth !== -1 && (
                <ImageDisplay context={imageContext}
                  id="GroundTruth" title="Ground Truth Image" transformRefsIndec={2}
                  src={`${imagesgroundTruth[selectedImageGroundTruth]}`}
                />
              )}

              {/*displaying colorized images */}
              {imagescolorized.map((methodImages, index) => {
                const method = Object.keys(methodImages)[0]; // name of the method.
                const imageSrc = methodImages[method][selectedImageIndex]; // Retrieval of the corresponding colored image.
                // create a new ref for the current TransformWrapper instance
                transformRefs.current[index + 3] =
                  transformRefs.current[index + 3] || React.createRef();

                return (
                  <ImageDisplay context={imageContext}
                    key={index} id={method} title={method} transformRefsIndec={index + 3}
                    src={`${imageSrc}`}
                  />
                )
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
