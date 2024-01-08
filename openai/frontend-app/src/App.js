// App.js
import React, { useState, useEffect } from "react";

import Button from "@mui/material/Button";
import Container from "@mui/material/Container";
import Grid from "@mui/material/Grid";
import MenuItem from "@mui/material/MenuItem";
import Paper from "@mui/material/Paper";
import TextField from "@mui/material/TextField";
import { styled } from "@mui/material/styles";

const BASE_URL = "http://172.1.1.13:5600";
const ENDPOINT_LIST_MODELS = BASE_URL + "/list_models";
const ENDPOINT_CHAT = BASE_URL + "/chat";
const ENDPOINT_PROMPT = BASE_URL + "/prompt";

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
  useEffect(() => {
    getSupportedModels();
  }, []);

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
  const [model, setModel] = useState("");

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

  // Handle save prompt
  const [promptName, setPromptName] = useState("");
  const [promptVersion, setPromptVersion] = useState("");

  async function handleSavePrompt() {
    try {
      if (promptName === "" || promptVersion === "") {
        return;
      }
      const response = await fetch(ENDPOINT_PROMPT, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: promptName,
          version: promptVersion,
          systemMessage: systemMessage,
        }),
      });
      const status = response.status;
      if (status === 200) {
        alert("Save prompt success");
      } else {
        alert("Save prompt failed");
      }
    } catch (error) {
      alert("Save prompt failed")
      console.error("error save prompt", error);
    }
  }

  // Handle reserved prompt

  const [reservedPrompts, setReservedPrompts] = useState({});

  function handleSelectReservedPromptChange(e) {
    const label = e.target.value;
    if (label === "Empty") {
      setPromptName("");
      setPromptVersion("");
      setSystemMessage("");
      return;
    } else {
      const promptName = reservedPrompts[label]["name"];
      const promptVersion = reservedPrompts[label]["version"];
      const systemMessage = reservedPrompts[label]["systemMessage"];

      setPromptName(promptName);
      setPromptVersion(promptVersion);
      setSystemMessage(systemMessage);
    }
  }

  async function handleReservedPromptRefresh() {
    try {
      const response = await fetch(ENDPOINT_PROMPT, {
        method: "GET",
      });
      const reservedPrompts = await response.json().then((data) => data.prompts);
      setReservedPrompts(reservedPrompts);
    } catch (error) {
      console.error("error fetching reserved prompts:", error);
      setReservedPrompts({});
    }
  }

  useEffect(() => {
    handleReservedPromptRefresh();
  }, []);

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
            <Item sx={{ minHeight: "80vh" }}>
              <div>
                <h2>Answer</h2>
                <div>{sseData}</div>
              </div>
            </Item>
          </Grid>
          <Grid item xs={4}>
            <Item sx={{ minHeight: "80vh" }}>
              <h2>Question</h2>
              <div>
                <TextField
                  id="select-model"
                  label="Model"
                  fullWidth
                  select
                  value={model}
                  defaultValue="gpt-3.5-turbo"
                  onChange={handleModelChange}
                  sx={{ marginTop: 1 }}
                >
                  {supportedModels.map((model) => (
                    <MenuItem key={model} value={model}>{model}</MenuItem>
                  ))}
                </TextField>
                <TextField
                  id="select-prompt"
                  label="Prompt"
                  fullWidth
                  select
                  defaultValue="Empty"
                  onChange={handleSelectReservedPromptChange}
                  sx={{ marginTop: 1 }}
                >
                  <MenuItem value={"Empty"}>{"Empty"}</MenuItem>
                  {Object.keys(reservedPrompts).map((key) => (
                    <MenuItem key={key} value={key}>{key}</MenuItem>
                  ))}
                </TextField>
                <TextField
                  id="system-input"
                  label="System Input"
                  fullWidth
                  multiline
                  maxRows={5}
                  value={systemMessage}
                  placeholder="You are a helpful assistant."
                  onChange={handleSystemMessageChange}
                  sx={{ marginTop: 2 }}
                />
                <TextField
                  id="user-input"
                  label="User Input"
                  fullWidth
                  multiline
                  maxRows={5}
                  placeholder="Message ChatGPT..."
                  onChange={handleUserMessageChange}
                  sx={{ marginTop: 2 }}
                />
                <Button
                  id="submit-button"
                  variant="contained"
                  onClick={handleSubmit}
                  sx={{ marginTop: 2 }}
                >
                  Submit
                </Button>
              </div>
              <div>
                <TextField
                  label="Prompt Name"
                  fullWidth
                  value={promptName}
                  onChange={(e) => setPromptName(e.target.value)}
                  sx={{ marginTop: 2 }}
                />
                <TextField
                  label="Version"
                  fullWidth
                  value={promptVersion}
                  onChange={(e) => setPromptVersion(e.target.value)}
                  sx={{ marginTop: 2 }}
                />
                <Button
                  variant="contained"
                  onClick={handleSavePrompt}
                  sx={{ marginTop: 2 }}
                >
                  Save Prompt
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
