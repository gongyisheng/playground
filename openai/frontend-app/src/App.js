// App.js
import React, { useState, useEffect } from "react";

import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import TextField from "@mui/material/TextField";
import { styled } from "@mui/material/styles";

const BASE_URL = "http://localhost:5600";
const ENDPOINT_LIST_MODELS = BASE_URL + "/list_models";
const ENDPOINT_CHAT = BASE_URL + "/chat";

const Item = styled(Paper)(({ theme }) => ({
  backgroundColor: theme.palette.mode === "dark" ? "#1A2027" : "#fff",
  ...theme.typography.body2,
  padding: theme.spacing(1),
  textAlign: "left",
  color: theme.palette.text.secondary,
}));

function App() {
  // Supported models
  const [supportedModels, setSupportedModels] = useState([]);

  async function getSupportedModels() {
    try {
      const response = await fetch(ENDPOINT_LIST_MODELS, {
        method: "GET",
      });
      var _models = await response.json().then((data) => data.models);
      setSupportedModels(_models);
    } catch (error) {
      console.error("error fetching supported models:", error);
      setSupportedModels([]);
    }
  }
  getSupportedModels();

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

  // Handle model change
  const [model, setModel] = useState("gpt-3.5-turbo");

  function handleModelChange(e) {
    setModel(e.target.value);
  }

  // Handle system message change
  const [systemMessage, setSystemMessage] = useState("");

  function handleSystemMessageChange(e) {
    setSystemMessage(e.target.value);
  }

  // Handle user message change
  const [userMessage, setUserMessage] = useState("");

  function handleUserMessageChange(e) {
    setUserMessage(e.target.value);
  }

  // Handle button click
  async function handleSubmit() {
    try {
      const response = await fetch(ENDPOINT_CHAT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          systemMessage: systemMessage,
          userMessage: userMessage,
        }),
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

    const sse = new EventSource(`${ENDPOINT_CHAT}?uuid=${uuid}&model=${model}`);

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
              <h1>Prompt Tester</h1>
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
                <TextField
                  id="outlined-multiline-flexible"
                  label="Model"
                  fullWidth
                  select
                  value={model}
                  onChange={handleModelChange}
                  sx={{ margin: 1 }}
                >
                  {supportedModels.map((model) => (
                    <MenuItem value={model}>{model}</MenuItem>
                  ))}
                </TextField>
                <TextField
                  id="outlined-multiline-flexible"
                  label="System Input"
                  fullWidth
                  multiline
                  maxRows={5}
                  placeholder="You are a helpful assistant."
                  onChange={handleSystemMessageChange}
                  sx={{ margin: 1 }}
                />
                <TextField
                  id="outlined-multiline-flexible"
                  label="User Input"
                  fullWidth
                  multiline
                  maxRows={5}
                  placeholder="Message ChatGPT..."
                  onChange={handleUserMessageChange}
                  sx={{ margin: 1 }}
                />
                <Button
                  variant="contained"
                  onClick={handleSubmit}
                  sx={{ margin: 1 }}
                >
                  Submit
                </Button>
              </div>
            </Item>
          </Grid>
        </Grid>
      </Container>
    </div>
  );
}

export default App;
