import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";

import { SESSION_COOKIE_NAME } from "../constants";
import { getCookie } from "../utils/cookie";

const Home = () => {
  const navigate = useNavigate();

  const redirect = () => {
    let cookie = getCookie(SESSION_COOKIE_NAME);
    console.log("home page cookie: ", cookie);
    // if cookie exists
    if (cookie !== null) {
      navigate("/playground");
    }
    else {
      navigate("/signin");
    }
  }

  useEffect(() => {
    redirect();
  }, []);

  return (
    <div></div>
  );
};

export default Home;
