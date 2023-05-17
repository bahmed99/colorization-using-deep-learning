import React from 'react'
import { useNavigate } from "react-router-dom";
export default function VisualizeCards({ list, redirect }) {
const Navigate = useNavigate();


  // //function to redirect the user to a visualization page based on the selected colorization
  // const redirect = (e) => {
  //   Navigate(`/visualize/${e}`)
  // }

  return (
    
    <div className="Visualize-cards-container">
      {list.sort((a, b) => -a.date.localeCompare(b.date)).map((element, index) => {
        return (
          <div className="Visualize-card" key={index} onClick={() => redirect(element.name)}>
            <h3>{element.name}</h3>
            <p>Dataset length: <span className='fields_values'>{element.length_images}</span></p>
            <p className='fields_params' >Models used: <span className='fields_values' >{element.models.join(" - ")}</span></p>
            <p>Date: <span className='fields_values'>{element.date}</span></p>
            <button>See results</button>
          </div>
        )
      })}
    </div>
 
  )
}
