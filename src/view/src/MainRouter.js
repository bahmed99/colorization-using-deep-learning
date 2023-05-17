import React from 'react'
import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Colorize from "./pages/Colorize";
import VisualizeColorisations from "./pages/VisualizeColorisations";
import VisualizeImages from "./pages/VisualizeImages";
import ScribbleSelection from "./pages/ScribbleSelection";
import VisualizeScribbles from './pages/VisualizeScribbles';
import Scribble from "./pages/Scribble";

export default function MainRouter() {
  return (
    <div>
        <Routes>
            <Route path={"/"} element={<Home/>} />
            <Route path={"/colorize"} element={<Colorize/>} />
            <Route path={"/visualize"} element={<VisualizeColorisations/>} />
            <Route path={"/visualizeScribbles"} element={<VisualizeScribbles/>} />
            <Route path={"/visualize/:colorisation"} element={<VisualizeImages/>} />
            <Route path={"/scribbleSelection"} element={<ScribbleSelection/>} />
            <Route path={"/scribble/:colorization_name"} element={<Scribble/>} />
        </Routes>
    </div>
  )
}
