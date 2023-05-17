import React, { useState } from "react";
import "../assets/css/Navbar.css";
import { NavLink } from "react-router-dom";

/**
 * Component for navbar
 */
function Navbar() {
  const [click, setClick] = React.useState(false);

  const handleClick = () => setClick(!click);
  const Close = () => setClick(false);

  return (

    <div>
     <div className={click ? "main-container" : ""}  onClick={()=>Close()} />
      <nav className="navbar" onClick={e => e.stopPropagation()}>
        <div className="nav-container">
          <NavLink exact to="/" className="nav-logo">
          Chrominance Lens
            <i className="fa fa-code"></i>
          </NavLink>
          <ul className={click ? "nav-menu active" : "nav-menu"}>
            <li className="nav-item">
              <NavLink
                exact
                to="/"
                activeClassName="active"
                className="nav-links"
                onClick={click ? handleClick : null}
              >
                Home
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink
                exact
                to="/colorize"
                activeClassName="active"
                className="nav-links"
                onClick={click ? handleClick : null}
              >
                Colorize
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink
                exact
                to="/visualize"
                activeClassName="active"
                className="nav-links"
                onClick={click ? handleClick : null}
              >
                Visualize
              </NavLink>
            </li>
            <li className="nav-item">
              <NavLink
                exact
                to="/visualizeScribbles"
                activeClassName="active"
                className="nav-links"
               onClick={click ? handleClick : null}
              >
                Scribbles
              </NavLink>
            </li>
          </ul>
          <div className="nav-icon" onClick={handleClick}>
            <i className={click ? "fa fa-times" : "fa fa-bars"}></i>
          </div>
        </div>
      </nav>
    </ div>
  );
}

export default Navbar;