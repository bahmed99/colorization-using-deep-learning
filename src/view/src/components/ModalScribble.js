import React, { useEffect, useState } from 'react'
import "../assets/css/ModalReference.css"
import axios from 'axios'


/**
 * Component used to choose a scribble for a model
 * @param {Object} props
 */
export default function ModalScribble(props) {


    const [scribbles, setScribbles] = useState([])


    useEffect(() => {
        axios.get("http://localhost:5000/image/getScribblesList").then((res) => {
            setScribbles(res.data.data);
        }).catch(err => {
            console.log(err)
        })
    }, [])


     /**
     * Choose a scribble for a model
     
     * @param {Object} e
     * 
     */
    const HandleChange = (e) => {
        props.setScribble(e.target.value)
    };

    /**
     * Validate the scribble
     *  
     *  
    **/
    const Validate = () => {
        if (props.scribble !== "" ) {
            
            props.setShowModal(false)
        }
    }

   

    return (
        <div className="modal">
            <div className="modal-content">

              
                    <div>
                        <h3>Choose a scribbles for model(s) {props.model}  </h3>
                        <select onChange={(e) => HandleChange(e)} className="select_models">
                            <option className='options' value=''>Choose a scribble</option>
                            {scribbles.map((item, index) => {
                                return <option key={index} value={item}>{item}</option>
                            })}
                        </select>
                    </div> 
            </div>
        </div>
    )
}
