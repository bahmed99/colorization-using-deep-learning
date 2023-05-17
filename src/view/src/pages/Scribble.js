import React, { useEffect, useState, useRef } from 'react'
import { useParams } from "react-router-dom";
import Axios from 'axios';
import SearchBar from '../components/SearchBar'
import Navbar from '../components/Navbar'
import { ChromePicker } from 'react-color';
import '../assets/css/Scribble.css'
import '../assets/css/icofont/icofont.min.css'
export default function Scribble() {
    const [imagesorigin, setImagesorigin] = useState([]) //Save the paths of the original images.
    const { colorization_name } = useParams()
    const [mouseDown, setMouseDown] = useState(false)
    const [canvas, setCanvas] = useState([])
    const [ctx, setCtx] = useState([])
    const [unmounted, setMount] = useState(false)
    const [currentImg, setCurrentImg] = useState([])
    const [lineWidth, setLineWidth] = useState(5)
    const [color, setColor] = useState([])
    const [eraserBrush, setEraserBrush] = useState([])
    const [eraseAll, setEraseAll] = useState([])
    const [scribbleList, setScribbleList] = useState([])
    // scales are used to resize the canvas and draw correctly
    const [scaleX, setScaleX] = useState(1)
    const [scaleY, setScaleY] = useState(1)
    let erase = false;

    // This ref is used by the search bar in order to keep the original list of images
    const imagesOriginRef = useRef(null)

    /**
     * Save Scribble Method
     */
    function handleSave() {
        //const canvasImg = canvasRef.current;
        const dataURL = canvas.toDataURL('image/png');
        let fileName = currentImg.split('/').pop()
        Axios.post(`http://localhost:5000/image/saveImage/${colorization_name}/${fileName}`, JSON.stringify({ dataURL }), {
            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(res => {
                setScribbleList(res.data)
                const dataURL = document.getElementById('drawing-board').toDataURL('image/png');
                localStorage.setItem( fileName, dataURL );
            })
            .catch(err => {
            });
    }

    useEffect(() => {

        if (!unmounted) {
            let scribbles = window.location.href.split('/')
            let scribble = scribbles[scribbles.length - 1]
            Axios.get(`http://localhost:5000/image/getImagesOriginForScribbles/${scribble}`).then(
                (res) => {
                    setImagesorigin(res.data['images_origin'])
                    imagesOriginRef.current = res.data['images_origin']
                    setScribbleList(res.data['scribbles'])
                }
            )
        }
        setCanvas(document.getElementById('drawing-board'))
        setCtx(document.getElementById('drawing-board').getContext('2d'))
        setEraserBrush(document.getElementById('eraser-brush'))
        setEraseAll(document.getElementById('erase'))
        setMount(true)
        let brushSize = 5
        setLineWidth(brushSize)
        document.getElementById("lineWidth").value = brushSize
    }, [])

    /**
     * Updates the displayed image
     * @param {String} name 
     */
    function viewImage(name) {
        if(currentImg.length===0){
            localStorage.clear()
        } 
        setCurrentImg(name)
    }

    /**
     * Initialize the canvas
     * @param {*} width 
     * @param {*} height 
     * @param {*} top 
     */
    function initialiseCanvas(width, height, top) {
        const canvas = document.getElementById("drawing-board")
        canvas.width = width;
        canvas.height = height;
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = color;
        ctx.lineCap = 'round';
        canvas.style.position = 'absolute'
        canvas.style.top = top + 'px'
        canvas.style.left = 120 + (window.innerWidth - 300 - width) / 2 + 'px'
        let fileName = currentImg.split('/').pop()
        if(localStorage.getItem(fileName)){
            const pic = new Image()
                pic.src = localStorage.getItem( fileName)
                pic.onload = function(){
                    ctx.drawImage(pic, 0, 0,width,height)
                }
        } 
        else {
        let elem = fileName.split('.')
        fileName = elem[0]+".png"
        scribbleList.map(s=>{
            if(s.split('/').pop()==fileName){
                const pic = new Image()
                pic.src = s
                pic.onload = function(){
                    ctx.drawImage(pic, 0, 0,width,height)
                }
        }})
        
        }
    }

    /**
     * Make the canvas responsive
     */
    function updateCanvas() {
        const canvas = document.getElementById("drawing-board")
        const img_scribble = document.getElementById("img_scribble")
        const ctx = document.getElementById('drawing-board').getContext('2d')
        canvas.style.top = img_scribble.style.top + 'px'
        canvas.style.left = 120 + (window.innerWidth - 300 - img_scribble.width) / 2 + 'px'
        canvas.style.width = img_scribble.width + 'px';
        canvas.style.height = img_scribble.height + 'px';
        setScaleX(img_scribble.width / canvas.width)
        setScaleY(img_scribble.height / canvas.height)
    }

    /**
     * Listen for page size change
     */
    window.onresize = updateCanvas

    function handlerMouseDown() {
        setMouseDown(true)
        ctx.beginPath();
    }

    function handlerMouseUp() {
        setMouseDown(false)
    }

    /**
     * Draw a line if the mouse is down
     * @param {*} e 
     */
    function handlerMouseMove(e) {
        if (mouseDown) {
            ctx.stroke();
            ctx.lineWidth = lineWidth;
            ctx.lineTo((e.pageX - canvas.offsetLeft) / scaleX, (e.pageY - canvas.offsetTop) / scaleY);
        }
    }

    /**
     * Draw a point
     * @param {*} e 
     */
    function handlerMouseClick(e) {
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.arc((e.pageX - canvas.offsetLeft) / scaleX, (e.pageY - canvas.offsetTop) / scaleY, lineWidth / 2, 0, 2 * Math.PI)
        ctx.fill()
    }

    /**
     * Change brush color
     */
    function handlerColor(color) {
        setColor(color.hex);
        ctx.strokeStyle = color.hex
    }

    /**
     * Change brush width
     * @param {*} e 
     */
    function handlerlineWidth(e) {
        setLineWidth(e.target.value)
        ctx.lineWidth = lineWidth;
    }

    /**
     * Change brush and eraser button properties
     * @param {String} backgroundColor 
     * @param {String} color 
     * @param {String} ctxOperation 
     */
    function eraserBrushChange(backgroundColor, color, ctxOperation) {
        eraserBrush.style.backgroundColor = backgroundColor;
        eraserBrush.style.color = color;
        ctx.globalCompositeOperation = ctxOperation;
    }

    /**
     * handle the eraser brush
     */
    function handlerEraserBrush() {
        erase = !erase;
        if (erase)
            eraserBrushChange("black", "white", "destination-out");
        else
            eraserBrushChange("transparent", "black", "source-over");
    }

    /**
     * Erase all drawings
     */
    function handlerEraseAll() {
        ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    }

    /**
     * load a new canvas
     */
    function loadCanvas() {
        const img_scribble = document.getElementById("img_scribble")
        initialiseCanvas(img_scribble.width, img_scribble.height, img_scribble.style.top)
    }

    return (
        <div>

            <Navbar />
            <div className="scribble_container">

                <ul className='dataset-list'>
                    <SearchBar elements={imagesOriginRef.current} setElements={setImagesorigin} criteriaGetter={(element) => element} />
                    <div className='scribble-image-scroller'>
                        {imagesorigin.map(name => (
                            <li key={name} onClick={() => viewImage(name)}>
                                <img src={name} />
                                <h6>{name.split('/').pop()}</h6>
                            </li>
                        ))}
                    </div>
                </ul>
                <img id="img_scribble" src={currentImg} onLoad={loadCanvas} />
                <canvas id="drawing-board" onMouseMove={handlerMouseMove} onMouseDown={handlerMouseDown} onMouseUp={handlerMouseUp} onClick={handlerMouseClick}></canvas>
                <div id="toolbar">
                    <label>Pick color</label>
                    <ChromePicker id='color' width='140px' disableAlpha={true}
                        color={color}
                        onChange={(color) => handlerColor(color)}
                    />
                    <label>Brush size</label>
                    <input id="lineWidth" type="range" min="1" max="25" onClick={handlerlineWidth}/>
                    <button id="eraser-brush" className="icofont-eraser" onClick={handlerEraserBrush}></button>
                    <button id="erase" className="material-symbols-outlined" onClick={handlerEraseAll}> delete </button>
                    <button id="save" onSubmit={(event) => event.preventDefault()} onClick={handleSave} type="button" className="material-symbols-outlined">save</button>
                </div>
            </div>

        </div>
    )
}
