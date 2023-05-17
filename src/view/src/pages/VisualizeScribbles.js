import React, { useEffect, useState } from 'react'
import axios from 'axios'
import Navbar from '../components/Navbar'
import "../assets/css/Visualize.css"
import { useNavigate } from "react-router-dom";
import VisualizeCards from '../components/VisualizeCards';
/**
 * Page for visualizing history of colorizations
 */
export default function VisualizeScribbles() {
  const [scribbles, setScribbles] = useState([]) //save the colorizations

  useEffect(() => {
    let unmounted = false
    if (!unmounted) {
      axios.get("http://localhost:5000/image/getScribbles").then(
        (data) => {
          setScribbles(data.data);
        }
      )
    }
    return () => { unmounted = true }
  }, [])

  const Navigate = useNavigate();

  const redirectNewScribbles = () => {
    Navigate(`/scribbleSelection`)
  }

  const redirect = (e) => {
    Navigate(`/scribble/${e}`)
  }


  return (
    <div className="Visualize-container">
      <Navbar />
      <button className='scribble_btn' onClick={redirectNewScribbles}> New scribbles configuration </button>
      <VisualizeCards list={scribbles} redirect={redirect} />
    </div>
  )
}
