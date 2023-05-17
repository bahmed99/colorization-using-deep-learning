import React, { useEffect, useState } from 'react'
import "../assets/css/ModalReference.css"
import axios from 'axios'


/**
 * Component used to choose a reference for a model
 * @param {Object} props
 */
export default function ModalReference(props) {


    const [references, setReferences] = useState([])


    useEffect(() => {
        axios.get("http://localhost:5000/image/getReferences").then((res) => {
            setReferences(res.data.data);
        }).catch(err => {
            console.log(err)
        })
    }, [])


     /**
     * Choose a reference for a model
     
     * @param {Object} e
     * 
     */
    const HandleChange = (e) => {
        props.setReference(e.target.value)
    };

    /**
     * Validate the reference
     *  
     *  
    **/
    const Validate = () => {
        if (props.reference !== "" || props.imageReference.length > 0) {
            
            props.setShowModal(false)
        }
    }

    /**
     * Upload images references for a model
     * @param {Object} e
     *  
    **/
    function UploadImages(e) {
        const files = e.target.files;
        props.setImageReference(prevState => [...prevState, ...files])
      }

    return (
        <div className="modal">
            <div className="modal-content">

                {props.dataset !== "" ?
                    <div>
                        <h3>Choose a reference for model(s) {props.model}  </h3>
                        <select onChange={(e) => HandleChange(e)} className="select_models">
                            <option className='options' value=''>Choose a  reference</option>
                            {references.map((item, index) => {
                                return <option key={index} value={item}>{item}</option>
                            })}
                        </select>
                    </div> :
                    <div>
                        <h3>Upload images references for model {props.model}  </h3>
                        <input className="file_upload" type="file" multiple onChange={(e) => UploadImages(e)}/>
                    </div>
                }
            </div>
        </div>
    )
}
