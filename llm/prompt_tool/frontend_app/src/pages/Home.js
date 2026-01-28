import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";

import { ENDPOINT_AUTH } from "../constants";

const Home = () => {
  const navigate = useNavigate();
  const [sessionState, setSessionState] = useState(0);
  const [checkSessionStateEvent, setCheckSessionStateEvent] = useState(0);

  const handleSessionStateChange = (state) => {
    setSessionState(state);
    setCheckSessionStateEvent(checkSessionStateEvent + 1);
  };

  const checkSessionState = async () => {
    try {
      const response = await fetch(ENDPOINT_AUTH, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include", // <-- includes cookies in the request
      });
      if (response.status === 200) {
        handleSessionStateChange(1);
      } else if (response.status === 401) {
        handleSessionStateChange(-1);
      }
    } catch (error) {
      console.error("Failed to check session state:", error);
    }
  }

  useEffect(() => {
    if (sessionState > 0) {
      navigate("/playground");
    } else if (sessionState < 0) {
      navigate("/signin");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [checkSessionStateEvent]);

  useEffect(() => {
    checkSessionState();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div></div>
  );
};

export default Home;
