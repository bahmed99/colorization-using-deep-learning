import React, { useEffect, useState } from 'react'
import axios from 'axios'
import Navbar from '../components/Navbar'
import "../assets/css/Visualize.css"
import VisualizeCards from '../components/VisualizeCards';
import { useNavigate } from "react-router-dom";
/**
 * Page for visualizing history of colorizations
 */
export default function VisualizeColorisations() {
  const [colorisations, setColorisations] = useState([]) //save the colorizations
  const Navigate = useNavigate();

  useEffect(() => {
    let unmounted = false
    if (!unmounted) {
      axios.get("http://localhost:5000/image/getColorizations").then(
        (data) => {
          setColorisations(data.data);
        }
      )
    }
    return () => { unmounted = true }
  }, [])

  //function to redirect the user to a visualization page based on the selected colorization
  const redirect = (e) => {
    Navigate(`/visualize/${e}`)
  }

  return (
    <div className="Visualize-container">
      <Navbar />
      <VisualizeCards list={colorisations} redirect={redirect} />
    </div>
  )
}
