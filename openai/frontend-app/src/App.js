// App.js
import React, { useState, useEffect } from "react";

import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import Paper from "@mui/material/Paper";
import { styled } from "@mui/material/styles";

const BASE_URL = "http://localhost:5600";
const ENDPOINT_META = BASE_URL + "/meta";
const ENDPOINT_CHAT = BASE_URL + "/chat";

const Item = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.mode === "dark" ? "#1A2027" : "#fff",
  ...theme.typography.body2,
  padding: theme.spacing(1),
  textAlign: "left",
  color: theme.palette.text.secondary,
}));

function App() {
  // Metadata of backend
  const [model, setModel] = useState("fetching...");

  async function getMeta() {
    try {
      const response = await fetch(ENDPOINT_META, {
        method: "GET",
      });
      var _model = await response.json().then((data) => data.model);
      setModel(_model);
    } catch (error) {
      console.error("error fetching meta:", error);
      setModel("unknown");
    }
  }
  getMeta();

  // UUID of current chat session
  const [uuid, setUUID] = useState("");

  function handleUUIDChange(newUUID) {
    // Ensure that newUuid is a string
    if (typeof newUUID === "string") {
      setUUID(newUUID);
    } else {
      console.error("received invalid uuid:", newUUID);
    }
  }

  // Handle input text change
  const [inputText, setInputText] = useState("");

  function handleInputChange(e) {
    setInputText(e.target.value);
  }

  // Handle button click
  async function handleButtonClick() {
    try {
      const response = await fetch(ENDPOINT_CHAT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: inputText }),
      });

      const uuid = await response.json().then((data) => data.uuid);
      handleUUIDChange(uuid);
    } catch (error) {
      console.error("error fetching UUID:", error);
    }
  }

  // Get SSE data from backend
  const [sseData, setSSEData] = useState("");

  useEffect(() => {
    setSSEData("");
    if (uuid === "") {
      return;
    }

    const sse = new EventSource(`${ENDPOINT_CHAT}?uuid=${uuid}`);

    // Event listener for SSE messages
    sse.onmessage = (event) => {
      setSSEData((prev) => prev + JSON.parse(event.data).content);
    };

    sse.onerror = (event) => {
      sse.close();
    };

    return () => sse.close();
  }, [uuid]);

  return (
    <div>
      <Container maxWidth="false">
        <Grid container spacing={2}>
          <Grid item xs={12}>
            <Item sx={{ height: "10vh", textAlign: "center" }}>
              <h1>OpenAI Chatbot [{model}]</h1>
            </Item>
          </Grid>
          <Grid item xs={8}>
            <Item sx={{ height: "80vh" }}>
              <div>
                <h2>Answer</h2>
                <div>{sseData}</div>
              </div>
            </Item>
          </Grid>
          <Grid item xs={4}>
            <Item sx={{ height: "80vh" }}>
              <h2>Question</h2>
              <div>
                <input
                  type="text"
                  value={inputText}
                  onChange={handleInputChange}
                />
                <button onClick={handleButtonClick}>submit</button>
              </div>
            </Item>
          </Grid>
        </Grid>
      </Container>
    </div>
  );
}

export default App;
