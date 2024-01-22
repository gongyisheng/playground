import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";

// import Home from './pages/Home';
import SignUp from "./pages/Signup";
import LogIn from "./pages/Login";
import Playground from "./pages/Playground";

const Main = () => {
  return (
    <Router>
      <Routes>
        <Route path="/playground" element={<Playground />} />
        <Route path="/signup" element={<SignUp />} />
        <Route path="/login" element={<LogIn />} />
      </Routes>
    </Router>
  );
};

export default Main;
