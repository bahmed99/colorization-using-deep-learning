import React from "react";
import ParticlesBg from "particles-bg";
import Navbar from "./Navbar";
import "../assets/css/Header.css"

function Header() {
    return (
        <div id="home">

            <ParticlesBg color="#E29578" type="cobweb" bg={true} num={150} />
            <Navbar />
            <div className="header-wraper">
                <div className="main-info">
                    <div className="title-welcome">
                        <h1>Welcome to <span className="name">our application</span></h1>
                    </div>
                    <div className="description">
                        <p>
                            This application enables users to colorize black and white images
                            using various deep learning-based colorization methods such as
                            reference-based, scribble-based, and automatic colorization.
                            Additionally, this application provides a visual
                            representation of the results to allow for easy comparison between models.
                        </p>
                    </div>

                </div>
            </div>
        </div>
    );
}

export default Header;